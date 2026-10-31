import { buildCandidateIndexes, normalizeStringExact, promoteResolvedNode, resolveWithSimilarity } from "./dedup";
import { createBatch, generateResponse } from "./llm";
import {
  type EntityEdge,
  type EntityNode,
  type EpisodicNode,
  entitiesByIds,
  entitiesByNames,
  episodesByIds,
  episodesMentioning,
  retrieveEpisodes,
  saveEdge,
  saveEntity,
  saveMention,
} from "./model";
import { customExtractionInstructions, entityTypes } from "./ontology";
import {
  dedupeNodes,
  type EntityTypeContext,
  edgeDuplicate,
  edgeTimestamps,
  extractEdges as extractEdgesPrompt,
  extractedEdges,
  extractedEntities,
  extractEntitySummariesFromEpisodes,
  extractJson,
  extractMessage,
  extractSummariesBatch,
  extractText,
  extractTimestamps,
  maxSummaryChars,
  nodeResolutions,
  type PreviousEpisode,
  resolveEdge,
  summarizedEntities,
} from "./prompts";
import { pyIsoformat } from "./pyformat";
import { nodeSimilaritySearch, search, searchRecipes } from "./search";
import { idList, inIds, type Store, type Write } from "./store";
import { applyTemporalInvalidation, normalizeFutureTemporalBounds, parseTimestamp } from "./temporal";

const relevantSchemaLimit = 10;
const nodeDedupCandidateLimit = 15;
const nodeDedupCosineMinScore = 0.6;
const maxNodesPerSummaryCall = 30;
const summaryRebuildBatchSize = 10;
const semaphoreLimit = 20;
const dedupeCandidateSummaryChars = 120;
const defaultEntityTypeDescription =
  "A specific, identifiable entity that does not fit any of the other listed types. Must still be a concrete, meaningful thing — " +
  "specific enough to be uniquely identifiable. GOOD: a named entity not covered by the other types. " +
  'BAD: "luck", "ideas", "tomorrow", "things", "them", "everybody", "a sense of wonder", "great times". ' +
  "When in doubt, do not extract the entity.";

export interface IngestResult {
  episode: { id: string; createdAt: string };
  nodes: { id: string; name: string; labels: string[]; created: boolean }[];
  edges: {
    id: string;
    sourceId: string;
    targetId: string;
    name: string;
    fact: string;
    status: "new" | "duplicate";
    validAt: string | null;
    invalidAt: string | null;
    expiredAt: string | null;
  }[];
  invalidated: { id: string; fact: string; invalidAt: string | null; expiredAt: string | null }[];
}

interface ResolvedNodes {
  nodes: EntityNode[];
  uuidMap: Map<string, string>;
}

interface ResolvedEdges {
  resolvedEdges: EntityEdge[];
  invalidatedEdges: EntityEdge[];
  newEdges: EntityEdge[];
}

interface DedupedEdge {
  resolved: EntityEdge;
  candidates: EntityEdge[] | null;
}

function previousEpisodeContext(episodes: readonly EpisodicNode[]): PreviousEpisode[] {
  return episodes.map((episode) => ({ content: episode.content, timestamp: pyIsoformat(episode.validAt) }));
}

function concatenateEpisodes(episodes: readonly EpisodicNode[]): string {
  const [only] = episodes;
  if (episodes.length === 1 && only !== undefined) {
    return only.content;
  }
  return episodes.map((episode, index) => `[Episode ${index}] (timestamp: ${pyIsoformat(episode.validAt)})\n${episode.content}`).join("\n\n");
}

function truncateAtSentence(text: string, maxChars: number): string {
  const characters = [...text];
  if (text === "" || characters.length <= maxChars) {
    return text;
  }
  const truncated = characters.slice(0, maxChars).join("");
  const last = [...truncated.matchAll(/[.!?](?:\s|$)/g)].at(-1);
  return last === undefined ? truncated.trimEnd() : text.slice(0, last.index + last[0].length).trimEnd();
}

function uniqueById<T extends { id: string }>(items: readonly T[]): T[] {
  const seen = new Set<string>();
  return items.filter((item) => {
    if (seen.has(item.id)) {
      return false;
    }
    seen.add(item.id);
    return true;
  });
}

async function semaphoreGather<T>(tasks: readonly (() => Promise<T>)[]): Promise<T[]> {
  const results = new Array<T>(tasks.length);
  let next = 0;
  const worker = async (): Promise<void> => {
    while (next < tasks.length) {
      const index = next;
      next += 1;
      const task = tasks[index];
      if (task !== undefined) {
        results[index] = await task();
      }
    }
  };
  await Promise.all(Array.from({ length: Math.min(semaphoreLimit, tasks.length) }, worker));
  return results;
}

async function embedFacts(ai: Ai, edges: readonly EntityEdge[]): Promise<void> {
  const withFacts = edges.filter((edge) => edge.fact !== "");
  if (withFacts.length === 0) {
    return;
  }
  const vectors = await createBatch(
    ai,
    withFacts.map((edge) => edge.fact),
  );
  withFacts.forEach((edge, index) => {
    edge.factEmbedding = vectors[index] ?? null;
  });
}

function entityTypesContext(): EntityTypeContext[] {
  return [
    { entity_type_id: 0, entity_type_name: "Entity", entity_type_description: defaultEntityTypeDescription },
    ...Object.entries(entityTypes).map(([name, description], index) => ({
      entity_type_id: index + 1,
      entity_type_name: name,
      entity_type_description: description,
    })),
  ];
}

function entityTypeDescription(labels: readonly string[]): string {
  const typeName = labels.find((label) => entityTypes[label] !== undefined);
  return (typeName === undefined ? undefined : entityTypes[typeName]) ?? "Default Entity Type";
}

function newEntityNode(name: string, typeName: string, nowIso: string): EntityNode {
  return {
    id: crypto.randomUUID(),
    name,
    labels: typeName === "Entity" ? ["Entity"] : ["Entity", typeName],
    summary: "",
    nameEmbedding: null,
    createdAt: nowIso,
  };
}

function specificLabelCount(node: EntityNode): number {
  return node.labels.filter((label) => label !== "Entity").length;
}

function collapseExactDuplicateExtractedNodes(nodes: readonly EntityNode[]): EntityNode[] {
  const canonicalByName = new Map<string, EntityNode>();
  for (const node of nodes) {
    const key = normalizeStringExact(node.name);
    const existing = canonicalByName.get(key);
    if (
      existing === undefined ||
      specificLabelCount(node) > specificLabelCount(existing) ||
      (specificLabelCount(node) === specificLabelCount(existing) && node.name.trim().length > existing.name.trim().length)
    ) {
      canonicalByName.set(key, node);
    }
  }
  return [...canonicalByName.values()];
}

async function extractNodes(ai: Ai, episode: EpisodicNode, previousEpisodes: readonly EpisodicNode[], nowIso: string): Promise<EntityNode[]> {
  const typesContext = entityTypesContext();
  const context = {
    episodeContent: episode.content,
    previousEpisodes: previousEpisodeContext(previousEpisodes),
    customExtractionInstructions,
    entityTypes: typesContext,
    sourceDescription: episode.sourceDescription,
  };
  const prompt =
    episode.source === "message" ? extractMessage(context) : episode.source === "json" ? extractJson(context) : extractText(context);
  const response = await generateResponse(ai, prompt, extractedEntities);
  const nodes = response.extracted_entities
    .filter((entity) => entity.name.trim() !== "")
    .map((entity) => newEntityNode(entity.name, typesContext[entity.entity_type_id]?.entity_type_name ?? "Entity", nowIso));
  return collapseExactDuplicateExtractedNodes(nodes);
}

async function collectCandidateNodes(ai: Ai, store: Store, extracted: readonly EntityNode[]): Promise<EntityNode[][]> {
  const exactByName = new Map<string, EntityNode[]>();
  const names = [...new Set(extracted.map((node) => node.name).filter((name) => name !== ""))].sort();
  for (const candidate of names.length === 0 ? [] : entitiesByNames(store, names)) {
    const key = normalizeStringExact(candidate.name);
    if (key !== "") {
      exactByName.set(key, [...(exactByName.get(key) ?? []), candidate]);
    }
  }
  const queryIndices = extracted.flatMap((node, index) => (node.name.replaceAll("\n", " ").trim() === "" ? [] : [index]));
  const queries = queryIndices.map((index) => extracted[index]?.name.replaceAll("\n", " ") ?? "");
  const semantic = extracted.map((): EntityNode[] => []);
  if (queries.length > 0) {
    const vectors = await createBatch(ai, queries);
    if (vectors.length !== queries.length) {
      throw new Error(`embedder returned ${vectors.length} vectors for ${queries.length} queries`);
    }
    queryIndices.forEach((nodeIndex, queryIndex) => {
      const vector = vectors[queryIndex];
      if (vector !== undefined) {
        semantic[nodeIndex] = nodeSimilaritySearch(store, vector, nodeDedupCandidateLimit, nodeDedupCosineMinScore);
      }
    });
  }
  return extracted.map((node, index) =>
    uniqueById([...(exactByName.get(normalizeStringExact(node.name)) ?? []), ...(semantic[index] ?? [])]),
  );
}

async function resolveWithLlm(
  ai: Ai,
  extracted: readonly EntityNode[],
  unresolvedIndices: readonly number[],
  candidates: readonly EntityNode[],
  episode: EpisodicNode,
  previousEpisodes: readonly EpisodicNode[],
  resolved: (EntityNode | null)[],
): Promise<void> {
  const llmNodes = unresolvedIndices.flatMap((index) => {
    const node = extracted[index];
    return node === undefined ? [] : [node];
  });
  const response = await generateResponse(
    ai,
    dedupeNodes({
      extractedNodes: llmNodes.map((node, id) => ({
        id,
        name: node.name,
        entity_type: node.labels,
        entity_type_description: entityTypeDescription(node.labels),
      })),
      existingNodes: candidates.map((candidate, candidateId) => ({
        candidate_id: candidateId,
        name: candidate.name,
        entity_types: candidate.labels,
        summary: [...candidate.summary].slice(0, dedupeCandidateSummaryChars).join(""),
      })),
      episodeContent: episode.content,
      previousEpisodes: previousEpisodeContext(previousEpisodes),
    }),
    nodeResolutions,
  );
  const processed = new Set<number>();
  for (const resolution of response.entity_resolutions) {
    const originalIndex = unresolvedIndices[resolution.id];
    const node = originalIndex === undefined ? undefined : extracted[originalIndex];
    if (originalIndex === undefined || node === undefined || processed.has(resolution.id)) {
      continue;
    }
    processed.add(resolution.id);
    const duplicate = resolution.duplicate_candidate_id < 0 ? undefined : candidates[resolution.duplicate_candidate_id];
    resolved[originalIndex] = duplicate === undefined ? node : promoteResolvedNode(node, duplicate);
  }
}

async function resolveExtractedNodes(
  ai: Ai,
  store: Store,
  extracted: readonly EntityNode[],
  episode: EpisodicNode,
  previousEpisodes: readonly EpisodicNode[],
): Promise<ResolvedNodes> {
  const candidatesByNode = await collectCandidateNodes(ai, store, extracted);
  const resolved: (EntityNode | null)[] = extracted.map(() => null);
  const unresolvedIndices: number[] = [];
  extracted.forEach((node, index) => {
    const candidates = candidatesByNode[index] ?? [];
    if (candidates.length === 0) {
      return;
    }
    const match = resolveWithSimilarity(node, buildCandidateIndexes(candidates));
    if (match === null) {
      unresolvedIndices.push(index);
      return;
    }
    resolved[index] = match;
  });
  if (unresolvedIndices.length > 0) {
    const llmCandidates = uniqueById(unresolvedIndices.flatMap((index) => candidatesByNode[index] ?? []));
    await resolveWithLlm(ai, extracted, unresolvedIndices, llmCandidates, episode, previousEpisodes, resolved);
  }
  const uuidMap = new Map<string, string>();
  const nodes = extracted.map((node, index) => {
    const match = resolved[index] ?? node;
    uuidMap.set(node.id, match.id);
    return match;
  });
  return { nodes, uuidMap };
}

async function extractEdges(
  ai: Ai,
  episode: EpisodicNode,
  nodes: EntityNode[],
  previousEpisodes: readonly EpisodicNode[],
  nowIso: string,
): Promise<{ edges: EntityEdge[]; materialized: EntityNode[] }> {
  const initialIds = new Set(nodes.map((node) => node.id));
  const nameToNode = new Map(nodes.map((node) => [normalizeStringExact(node.name), node]));
  const response = await generateResponse(
    ai,
    extractEdgesPrompt({
      episodeContent: episode.content,
      nodes: nodes.map((node) => ({ name: node.name, entity_types: node.labels })),
      previousEpisodes: previousEpisodeContext(previousEpisodes),
      referenceTime: episode.validAt,
      customExtractionInstructions,
    }),
    extractedEdges,
  );
  const endpoint = (name: string): EntityNode => {
    const key = normalizeStringExact(name);
    const known = nameToNode.get(key);
    if (known !== undefined) {
      return known;
    }
    const materialized = newEntityNode(name.trim(), "Entity", nowIso);
    nodes.push(materialized);
    nameToNode.set(key, materialized);
    return materialized;
  };
  const edges: EntityEdge[] = [];
  for (const edgeData of response.edges) {
    const sourceName = edgeData.source_entity_name.trim();
    const targetName = edgeData.target_entity_name.trim();
    if (sourceName === "" || targetName === "" || edgeData.fact.trim() === "") {
      continue;
    }
    const source = endpoint(sourceName);
    const target = endpoint(targetName);
    const [validAt, invalidAt] = normalizeFutureTemporalBounds(
      edgeData.fact,
      episode.validAt,
      parseTimestamp(edgeData.valid_at),
      parseTimestamp(edgeData.invalid_at),
    );
    edges.push({
      id: crypto.randomUUID(),
      sourceId: source.id,
      targetId: target.id,
      name: edgeData.relation_type,
      fact: edgeData.fact,
      factEmbedding: null,
      episodes: [episode.id],
      validAt,
      invalidAt,
      expiredAt: null,
      createdAt: nowIso,
    });
  }
  const referenced = new Set(edges.flatMap((edge) => [edge.sourceId, edge.targetId]));
  const survivors = nodes.filter((node) => initialIds.has(node.id) || referenced.has(node.id));
  nodes.splice(0, nodes.length, ...survivors);
  return { edges, materialized: survivors.filter((node) => !initialIds.has(node.id)) };
}

async function mergeMaterializedEndpointNodes(
  ai: Ai,
  store: Store,
  materialized: readonly EntityNode[],
  episode: EpisodicNode,
  previousEpisodes: readonly EpisodicNode[],
  resolved: ResolvedNodes,
): Promise<void> {
  const nodesByName = new Map(resolved.nodes.map((node) => [normalizeStringExact(node.name), node]));
  const knownIds = new Set(resolved.nodes.map((node) => node.id));
  const unresolved: EntityNode[] = [];
  for (const node of materialized) {
    if (knownIds.has(node.id)) {
      continue;
    }
    const existing = nodesByName.get(normalizeStringExact(node.name));
    if (existing === undefined) {
      unresolved.push(node);
    } else {
      resolved.uuidMap.set(node.id, existing.id);
    }
  }
  if (unresolved.length === 0) {
    return;
  }
  const merged = await resolveExtractedNodes(ai, store, unresolved, episode, previousEpisodes);
  merged.uuidMap.forEach((target, source) => resolved.uuidMap.set(source, target));
  for (const node of merged.nodes) {
    if (!knownIds.has(node.id)) {
      resolved.nodes.push(node);
      knownIds.add(node.id);
    }
  }
}

function resolveEdgePointers(edges: readonly EntityEdge[], uuidMap: ReadonlyMap<string, string>): EntityEdge[] {
  return edges.filter((edge) => {
    const originalSource = edge.sourceId;
    const originalTarget = edge.targetId;
    edge.sourceId = uuidMap.get(originalSource) ?? originalSource;
    edge.targetId = uuidMap.get(originalTarget) ?? originalTarget;
    return originalSource === originalTarget || edge.sourceId !== edge.targetId;
  });
}

async function extractEdgeTimestamps(ai: Ai, edge: EntityEdge, episode: EpisodicNode): Promise<void> {
  if (edge.validAt !== null || edge.invalidAt !== null) {
    return;
  }
  try {
    const timestamps = await generateResponse(
      ai,
      extractTimestamps({ fact: edge.fact, referenceTime: pyIsoformat(episode.validAt) }),
      edgeTimestamps,
    );
    [edge.validAt, edge.invalidAt] = normalizeFutureTemporalBounds(
      edge.fact,
      episode.validAt,
      parseTimestamp(timestamps.valid_at) ?? edge.validAt,
      parseTimestamp(timestamps.invalid_at) ?? edge.invalidAt,
    );
  } catch (error) {
    console.warn(JSON.stringify({ event: "edge_timestamps_failed", edge: edge.id, error: String(error) }));
  }
}

async function dedupeExtractedEdge(
  ai: Ai,
  extracted: EntityEdge,
  related: readonly EntityEdge[],
  existing: readonly EntityEdge[],
  episode: EpisodicNode,
): Promise<DedupedEdge> {
  const appendEpisodeProvenance = (resolved: EntityEdge) => {
    for (const episodeId of [...extracted.episodes, episode.id]) {
      if (!resolved.episodes.includes(episodeId)) {
        resolved.episodes.push(episodeId);
      }
    }
  };
  if (related.length === 0 && existing.length === 0) {
    await extractEdgeTimestamps(ai, extracted, episode);
    return { resolved: extracted, candidates: null };
  }
  const normalizedFact = normalizeStringExact(extracted.fact);
  const verbatim = related.find(
    (edge) =>
      edge.sourceId === extracted.sourceId && edge.targetId === extracted.targetId && normalizeStringExact(edge.fact) === normalizedFact,
  );
  if (verbatim !== undefined) {
    appendEpisodeProvenance(verbatim);
    return { resolved: verbatim, candidates: null };
  }
  const offset = related.length;
  const response = await generateResponse(
    ai,
    resolveEdge({
      existingEdges: related.map((edge, idx) => ({ idx, fact: edge.fact })),
      edgeInvalidationCandidates: existing.map((edge, index) => ({ idx: offset + index, fact: edge.fact })),
      newEdge: extracted.fact,
    }),
    edgeDuplicate,
  );
  const duplicateIds = response.duplicate_facts.filter((idx) => idx >= 0 && idx < related.length);
  const duplicate = duplicateIds[0] === undefined ? undefined : related[duplicateIds[0]];
  const resolved = duplicate ?? extracted;
  if (duplicate !== undefined) {
    appendEpisodeProvenance(duplicate);
  }
  const candidates = response.contradicted_facts.flatMap((idx) => {
    const candidate = idx < offset ? related[idx] : existing[idx - offset];
    return idx < 0 || candidate === undefined ? [] : [candidate];
  });
  if (resolved === extracted) {
    await extractEdgeTimestamps(ai, extracted, episode);
  }
  return { resolved, candidates };
}

function dedupeWithinExtraction(edges: readonly EntityEdge[]): EntityEdge[] {
  const seen = new Map<string, EntityEdge>();
  const kept: EntityEdge[] = [];
  for (const edge of edges) {
    const fact = normalizeStringExact(edge.fact);
    const key = `${edge.sourceId}\n${edge.targetId}\n${fact}`;
    if (seen.has(key) || seen.get(`${edge.targetId}\n${edge.sourceId}\n${fact}`)?.name === edge.name) {
      continue;
    }
    seen.set(key, edge);
    kept.push(edge);
  }
  return kept;
}

async function resolveExtractedEdges(ai: Ai, store: Store, extracted: readonly EntityEdge[], episode: EpisodicNode, nowIso: string): Promise<ResolvedEdges> {
  const edges = dedupeWithinExtraction(extracted);
  await embedFacts(ai, edges);
  const config = searchRecipes.edge_hybrid_search_rrf;
  const relatedResults = await semaphoreGather(
    edges.map((edge) => () => {
      const pair = [edge.sourceId, edge.targetId];
      return search(ai, store, {
        query: edge.fact,
        config,
        filters: { edgeSourceNodeIds: pair, edgeTargetNodeIds: pair },
        queryVector: edge.factEmbedding ?? undefined,
      });
    }),
  );
  const relatedLists = edges.map((edge, index) =>
    (relatedResults[index]?.edges ?? []).filter(
      (candidate) =>
        (candidate.sourceId === edge.sourceId && candidate.targetId === edge.targetId) ||
        (candidate.sourceId === edge.targetId && candidate.targetId === edge.sourceId),
    ),
  );
  const invalidationResults = await semaphoreGather(
    edges.map((edge) => () => search(ai, store, { query: edge.fact, config, queryVector: edge.factEmbedding ?? undefined })),
  );
  const invalidationLists = edges.map((_, index) => {
    const relatedIds = new Set((relatedLists[index] ?? []).map((candidate) => candidate.id));
    return (invalidationResults[index]?.edges ?? []).filter((candidate) => !relatedIds.has(candidate.id));
  });
  const deduped = await semaphoreGather(
    edges.map((edge, index) => () => dedupeExtractedEdge(ai, edge, relatedLists[index] ?? [], invalidationLists[index] ?? [], episode)),
  );
  const resolvedEdges = deduped.map((result) => result.resolved);
  const newEdges = resolvedEdges.filter((resolved, index) => resolved.id === edges[index]?.id);
  const invalidatedEdges = deduped.flatMap((result) =>
    result.candidates === null ? [] : applyTemporalInvalidation(result.resolved, result.candidates, nowIso),
  );
  await Promise.all([embedFacts(ai, resolvedEdges), embedFacts(ai, invalidatedEdges)]);
  return { resolvedEdges, invalidatedEdges, newEdges };
}

async function summarizeFlight(
  ai: Ai,
  nodes: readonly EntityNode[],
  episodeContent: string,
  previousEpisodes: readonly EpisodicNode[],
  useEpisodePrompt: boolean,
): Promise<void> {
  const context = {
    entities: nodes.map((node) => ({ name: node.name, summary: node.summary, entity_types: node.labels, attributes: {} })),
    episodeContent,
    previousEpisodes: previousEpisodeContext(previousEpisodes),
    entityTypeDescriptions: {},
  };
  const response = await generateResponse(
    ai,
    useEpisodePrompt ? extractEntitySummariesFromEpisodes(context) : extractSummariesBatch(context),
    summarizedEntities,
  );
  for (const summarized of response.summaries) {
    const summary = truncateAtSentence(summarized.summary, maxSummaryChars);
    for (const node of nodes) {
      if (node.name.toLowerCase() === summarized.name.toLowerCase()) {
        node.summary = summary;
      }
    }
  }
}

async function extractEntitySummaries(
  ai: Ai,
  nodes: readonly EntityNode[],
  episode: EpisodicNode,
  previousEpisodes: readonly EpisodicNode[],
  edges: readonly EntityEdge[],
): Promise<void> {
  const edgesByNode = new Map<string, EntityEdge[]>();
  for (const edge of edges) {
    for (const nodeId of [edge.sourceId, edge.targetId]) {
      edgesByNode.set(nodeId, [...(edgesByNode.get(nodeId) ?? []), edge]);
    }
  }
  const needingLlm: EntityNode[] = [];
  for (const node of nodes) {
    const nodeEdges = edgesByNode.get(node.id) ?? [];
    let summaryWithEdges = node.summary;
    if (nodeEdges.length > 0) {
      const edgeFacts = nodeEdges
        .filter((edge) => edge.fact !== "")
        .map((edge) => edge.fact)
        .join("\n");
      summaryWithEdges = `${summaryWithEdges}\n${edgeFacts}`.trim();
    }
    if (summaryWithEdges !== "" && [...summaryWithEdges].length <= maxSummaryChars * 2) {
      node.summary = summaryWithEdges;
      continue;
    }
    needingLlm.push(node);
  }
  const flights: EntityNode[][] = [];
  for (let start = 0; start < needingLlm.length; start += maxNodesPerSummaryCall) {
    flights.push(needingLlm.slice(start, start + maxNodesPerSummaryCall));
  }
  await semaphoreGather(flights.map((flight) => () => summarizeFlight(ai, flight, episode.content, previousEpisodes, false)));
}

function persistedEntityIds(store: Store, ids: readonly string[]): Set<string> {
  return new Set(store.all<{ id: string }>(`SELECT id FROM entities WHERE id ${inIds}`, idList(ids)).map((row) => row.id));
}

export async function addEpisode(ai: Ai, store: Store, episode: EpisodicNode, nowIso: string): Promise<IngestResult> {
  const previousEpisodes = retrieveEpisodes(store, episode.validAt, relevantSchemaLimit, episode.source, episode.id);
  const extractedNodes = await extractNodes(ai, episode, previousEpisodes, nowIso);
  const resolved = await resolveExtractedNodes(ai, store, extractedNodes, episode, previousEpisodes);
  const { edges: extractedEdgeList, materialized } = await extractEdges(ai, episode, extractedNodes, previousEpisodes, nowIso);
  if (materialized.length > 0) {
    await mergeMaterializedEndpointNodes(ai, store, materialized, episode, previousEpisodes, resolved);
  }
  const edges = resolveEdgePointers(extractedEdgeList, resolved.uuidMap);
  const { resolvedEdges, invalidatedEdges, newEdges } = await resolveExtractedEdges(ai, store, edges, episode, nowIso);
  const { nodes } = resolved;
  await extractEntitySummaries(ai, nodes, episode, previousEpisodes, newEdges);
  const namedNodes = nodes.filter((node) => node.name !== "");
  if (namedNodes.length > 0) {
    const nameVectors = await createBatch(
      ai,
      namedNodes.map((node) => node.name),
    );
    namedNodes.forEach((node, index) => {
      node.nameEmbedding = nameVectors[index] ?? null;
    });
  }

  const persisted = persistedEntityIds(
    store,
    nodes.map((node) => node.id),
  );
  const entityEdges = [...resolvedEdges, ...invalidatedEdges];
  store.commit([
    ...nodes.map(saveEntity),
    ...nodes.map((node) => saveMention(episode.id, node.id, nowIso)),
    ...entityEdges.flatMap(saveEdge),
    ["UPDATE memories SET extracted_at = ? WHERE id = ?", nowIso, episode.id],
  ]);

  const newEdgeIds = new Set(newEdges.map((edge) => edge.id));
  return {
    episode: { id: episode.id, createdAt: episode.createdAt },
    nodes: uniqueById(nodes).map((node) => ({ id: node.id, name: node.name, labels: node.labels, created: !persisted.has(node.id) })),
    edges: uniqueById(resolvedEdges).map((edge) => ({
      id: edge.id,
      sourceId: edge.sourceId,
      targetId: edge.targetId,
      name: edge.name,
      fact: edge.fact,
      status: newEdgeIds.has(edge.id) ? "new" : "duplicate",
      validAt: edge.validAt,
      invalidAt: edge.invalidAt,
      expiredAt: edge.expiredAt,
    })),
    invalidated: uniqueById(invalidatedEdges).map((edge) => ({
      id: edge.id,
      fact: edge.fact,
      invalidAt: edge.invalidAt,
      expiredAt: edge.expiredAt,
    })),
  };
}

async function rebuildNodeSummary(ai: Ai, store: Store, node: EntityNode): Promise<void> {
  const previousSummary = node.summary;
  try {
    const remaining = episodesMentioning(store, node.id).sort((left, right) => Date.parse(left.validAt) - Date.parse(right.validAt));
    node.summary = "";
    for (let start = 0; start < remaining.length; start += summaryRebuildBatchSize) {
      const batch = remaining.slice(start, start + summaryRebuildBatchSize);
      await summarizeFlight(ai, [node], concatenateEpisodes(batch), [], true);
    }
  } catch (error) {
    node.summary = previousSummary;
    console.error(JSON.stringify({ event: "summary_rebuild_failed", entity: node.id, error: String(error) }));
  }
}

export async function removeEpisode(ai: Ai, store: Store, episodeId: string): Promise<boolean> {
  if (episodesByIds(store, [episodeId]).length === 0) {
    return false;
  }
  const edgeIdsToDelete = store
    .all<{ fact_id: string }>(
      "SELECT fact_id FROM fact_episodes WHERE memory_id = ?1 AND fact_id NOT IN (SELECT fact_id FROM fact_episodes WHERE memory_id <> ?1)",
      episodeId,
    )
    .map((row) => row.fact_id);
  const mentioned = store.all<{ entity_id: string; mentions: number }>(
    "SELECT DISTINCT entity_id, (SELECT count(*) FROM episode_entities other WHERE other.entity_id = mention.entity_id) AS mentions " +
      "FROM episode_entities mention WHERE memory_id = ?",
    episodeId,
  );
  const nodeIdsToDelete = mentioned.filter((row) => row.mentions === 1).map((row) => row.entity_id);
  const surviving = entitiesByIds(
    store,
    mentioned.filter((row) => row.mentions !== 1).map((row) => row.entity_id),
  );
  const deletes: Write[] = [
    [`DELETE FROM facts WHERE id ${inIds}`, idList(edgeIdsToDelete)],
    [`DELETE FROM entities WHERE id ${inIds}`, idList(nodeIdsToDelete)],
    ["DELETE FROM memories WHERE id = ?", episodeId],
    ["DELETE FROM jobs WHERE episode_id = ?", episodeId],
  ];
  store.commit(deletes);
  await semaphoreGather(surviving.map((node) => () => rebuildNodeSummary(ai, store, node)));
  store.commit(surviving.map(saveEntity));
  return true;
}
