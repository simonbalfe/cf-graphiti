import { fulltextQuery } from "./fts";
import { create, rank } from "./llm";
import { edgesByIds, type EntityEdge, type EntityNode, type EpisodicNode, entitiesByIds, episodesByIds } from "./model";
import { idList, inIds, type Store, type VectorTable } from "./store";

const defaultSearchLimit = 10;
const defaultMinScore = 0.6;
const defaultMmrLambda = 0.5;
const maxSearchDepth = 3;
const pythonWhitespace = new Set([
  0x09, 0x0a, 0x0b, 0x0c, 0x0d, 0x1c, 0x1d, 0x1e, 0x1f, 0x20, 0x85, 0xa0, 0x1680, 0x2000, 0x2001, 0x2002, 0x2003, 0x2004, 0x2005, 0x2006,
  0x2007, 0x2008, 0x2009, 0x200a, 0x2028, 0x2029, 0x202f, 0x205f, 0x3000,
]);

export type SearchMethod = "cosine_similarity" | "bm25" | "breadth_first_search";
export type Reranker = "reciprocal_rank_fusion" | "node_distance" | "episode_mentions" | "mmr" | "cross_encoder";

export interface ScopeConfig {
  searchMethods: SearchMethod[];
  reranker: Reranker;
  simMinScore: number;
  mmrLambda: number;
  bfsMaxDepth: number;
}

export interface SearchConfig {
  edgeConfig: ScopeConfig | null;
  nodeConfig: ScopeConfig | null;
  episodeConfig: ScopeConfig | null;
  limit: number;
  rerankerMinScore: number;
}

export interface SearchFilters {
  edgeSourceNodeIds?: readonly string[];
  edgeTargetNodeIds?: readonly string[];
}

export interface SearchRequest {
  query: string;
  config: SearchConfig;
  filters?: SearchFilters;
  centerNodeId?: string;
  bfsOriginNodeIds?: readonly string[];
  queryVector?: readonly number[];
}

export interface SearchResults {
  edges: EntityEdge[];
  edgeRerankerScores: number[];
  nodes: EntityNode[];
  nodeRerankerScores: number[];
  episodes: EpisodicNode[];
  episodeRerankerScores: number[];
}

type Ranked = [ids: string[], scores: number[]];

function scope(searchMethods: SearchMethod[], reranker: Reranker, mmrLambda = defaultMmrLambda): ScopeConfig {
  return { searchMethods, reranker, simMinScore: defaultMinScore, mmrLambda, bfsMaxDepth: maxSearchDepth };
}

function recipe(scopes: Partial<Pick<SearchConfig, "edgeConfig" | "nodeConfig" | "episodeConfig">>): SearchConfig {
  return { edgeConfig: null, nodeConfig: null, episodeConfig: null, limit: defaultSearchLimit, rerankerMinScore: 0, ...scopes };
}

const hybrid: SearchMethod[] = ["bm25", "cosine_similarity"];
const hybridWithBfs: SearchMethod[] = ["bm25", "cosine_similarity", "breadth_first_search"];

export const searchRecipes = {
  combined_hybrid_search_rrf: recipe({
    edgeConfig: scope(hybrid, "reciprocal_rank_fusion"),
    nodeConfig: scope(hybrid, "reciprocal_rank_fusion"),
    episodeConfig: scope(["bm25"], "reciprocal_rank_fusion"),
  }),
  combined_hybrid_search_mmr: recipe({
    edgeConfig: scope(hybrid, "mmr", 1),
    nodeConfig: scope(hybrid, "mmr", 1),
    episodeConfig: scope(["bm25"], "reciprocal_rank_fusion"),
  }),
  combined_hybrid_search_cross_encoder: recipe({
    edgeConfig: scope(hybridWithBfs, "cross_encoder"),
    nodeConfig: scope(hybridWithBfs, "cross_encoder"),
    episodeConfig: scope(["bm25"], "cross_encoder"),
  }),
  edge_hybrid_search_rrf: recipe({ edgeConfig: scope(hybrid, "reciprocal_rank_fusion") }),
  edge_hybrid_search_mmr: recipe({ edgeConfig: scope(hybrid, "mmr") }),
  edge_hybrid_search_node_distance: recipe({ edgeConfig: scope(hybrid, "node_distance") }),
  edge_hybrid_search_episode_mentions: recipe({ edgeConfig: scope(hybrid, "episode_mentions") }),
  edge_hybrid_search_cross_encoder: recipe({ edgeConfig: scope(hybridWithBfs, "cross_encoder") }),
  node_hybrid_search_rrf: recipe({ nodeConfig: scope(hybrid, "reciprocal_rank_fusion") }),
  node_hybrid_search_mmr: recipe({ nodeConfig: scope(hybrid, "mmr") }),
  node_hybrid_search_node_distance: recipe({ nodeConfig: scope(hybrid, "node_distance") }),
  node_hybrid_search_episode_mentions: recipe({ nodeConfig: scope(hybrid, "episode_mentions") }),
  node_hybrid_search_cross_encoder: recipe({ nodeConfig: scope(hybridWithBfs, "cross_encoder") }),
} satisfies Record<string, SearchConfig>;

export type RecipeName = keyof typeof searchRecipes;
export const recipeNames = Object.keys(searchRecipes) as [RecipeName, ...RecipeName[]];

export function rrf(results: readonly (readonly string[])[], minScore = 0, rankConst = 1): Ranked {
  const scores = new Map<string, number>();
  for (const result of results) {
    result.forEach((id, index) => scores.set(id, (scores.get(id) ?? 0) + 1 / (index + rankConst)));
  }
  const sorted = [...scores.entries()].sort((left, right) => right[1] - left[1]).filter(([, score]) => score >= minScore);
  return [sorted.map(([id]) => id), sorted.map(([, score]) => score)];
}

function normalizeL2(vector: ArrayLike<number>): number[] {
  const values = Array.from(vector);
  const norm = Math.sqrt(values.reduce((sum, value) => sum + value * value, 0));
  return norm === 0 ? values : values.map((value) => value / norm);
}

function dot(left: ArrayLike<number>, right: ArrayLike<number>): number {
  let total = 0;
  for (let index = 0; index < left.length; index += 1) {
    total += (left[index] ?? 0) * (right[index] ?? 0);
  }
  return total;
}

function maximalMarginalRelevance(
  queryVector: readonly number[],
  candidates: ReadonlyMap<string, ArrayLike<number>>,
  mmrLambda: number,
  minScore: number,
): Ranked {
  const ids = [...candidates.keys()];
  const normalized = ids.map((id) => normalizeL2(candidates.get(id) ?? []));
  const maxSimilarity = ids.map(() => 0);
  normalized.forEach((left, row) => {
    normalized.slice(0, row).forEach((right, column) => {
      const similarity = dot(left, right);
      maxSimilarity[row] = Math.max(maxSimilarity[row] ?? 0, similarity);
      maxSimilarity[column] = Math.max(maxSimilarity[column] ?? 0, similarity);
    });
  });
  const scores = new Map(
    ids.map((id, index) => [
      id,
      mmrLambda * dot(queryVector, normalized[index] ?? []) + (mmrLambda - 1) * (maxSimilarity[index] ?? 0),
    ]),
  );
  const sorted = ids.sort((left, right) => (scores.get(right) ?? 0) - (scores.get(left) ?? 0)).filter((id) => (scores.get(id) ?? 0) >= minScore);
  return [sorted, sorted.map((id) => scores.get(id) ?? 0)];
}

function embeddingsFor(store: Store, table: VectorTable, ids: readonly string[]): Map<string, Float32Array> {
  return new Map(
    ids.flatMap((id): [string, Float32Array][] => {
      const vector = store.vector(table, id);
      return vector === undefined ? [] : [[id, vector]];
    }),
  );
}

function nodeDistanceReranker(store: Store, nodeIds: readonly string[], centerNodeId: string, minScore: number): Ranked {
  const filtered = nodeIds.filter((id) => id !== centerNodeId);
  const connected = new Set(
    store
      .all<{ id: string }>(
        "SELECT CASE WHEN source_id = ?1 THEN target_id ELSE source_id END AS id FROM facts " +
          "WHERE (source_id = ?1 AND target_id IN (SELECT value FROM json_each(?2))) " +
          "OR (target_id = ?1 AND source_id IN (SELECT value FROM json_each(?2)))",
        centerNodeId,
        idList(filtered),
      )
      .map((row) => row.id),
  );
  const scores = new Map<string, number>(filtered.map((id) => [id, connected.has(id) ? 1 : Number.POSITIVE_INFINITY]));
  const ordered = filtered.sort((left, right) => (scores.get(left) ?? 0) - (scores.get(right) ?? 0));
  if (nodeIds.includes(centerNodeId)) {
    scores.set(centerNodeId, 0.1);
    ordered.unshift(centerNodeId);
  }
  const kept = ordered.filter((id) => 1 / (scores.get(id) ?? 0) >= minScore);
  return [kept, kept.map((id) => 1 / (scores.get(id) ?? 0))];
}

function episodeMentionsReranker(store: Store, results: readonly (readonly string[])[], minScore: number): Ranked {
  const [sorted] = rrf(results);
  const counts = new Map(
    store
      .all<{ id: string; mentions: number }>(
        `SELECT entity_id AS id, count(*) AS mentions FROM episode_entities WHERE entity_id ${inIds} GROUP BY entity_id`,
        idList(sorted),
      )
      .map((row) => [row.id, row.mentions]),
  );
  const ordered = sorted.sort((left, right) => (counts.get(right) ?? 0) - (counts.get(left) ?? 0)).filter((id) => (counts.get(id) ?? 0) >= minScore);
  return [ordered, ordered.map((id) => counts.get(id) ?? 0)];
}

function fulltextIds(store: Store, table: "facts" | "entities" | "memories", query: string, limit: number, filter = "", params: string[] = []): string[] {
  const match = fulltextQuery(query);
  if (match === null) {
    return [];
  }
  return store
    .all<{ id: string }>(
      `SELECT t.id AS id FROM (SELECT rowid, rank FROM ${table}_fts WHERE ${table}_fts MATCH ? ORDER BY rank LIMIT ?) hits ` +
        `JOIN ${table} t ON t.rowid = hits.rowid${filter} ORDER BY hits.rank LIMIT ?`,
      match,
      limit,
      ...params,
      limit,
    )
    .map((row) => row.id);
}

function edgeFilter(filters: SearchFilters): { sql: string; params: string[] } {
  const clauses: string[] = [];
  const params: string[] = [];
  if (filters.edgeSourceNodeIds !== undefined) {
    clauses.push(`t.source_id ${inIds}`);
    params.push(idList(filters.edgeSourceNodeIds));
  }
  if (filters.edgeTargetNodeIds !== undefined) {
    clauses.push(`t.target_id ${inIds}`);
    params.push(idList(filters.edgeTargetNodeIds));
  }
  return { sql: clauses.length === 0 ? "" : ` WHERE ${clauses.join(" AND ")}`, params };
}

function allowedEdges(store: Store, filters: SearchFilters): Set<string> | undefined {
  const { sql, params } = edgeFilter(filters);
  return sql === "" ? undefined : new Set(store.all<{ id: string }>(`SELECT t.id AS id FROM facts t${sql}`, ...params).map((row) => row.id));
}

function edgeFulltextSearch(store: Store, query: string, filters: SearchFilters, limit: number): EntityEdge[] {
  const { sql, params } = edgeFilter(filters);
  return edgesByIds(store, fulltextIds(store, "facts", query, limit, sql, params));
}

function edgeSimilaritySearch(store: Store, vector: readonly number[], filters: SearchFilters, limit: number, minScore: number): EntityEdge[] {
  const found = store.nearest("facts", vector, limit, minScore, allowedEdges(store, filters));
  return edgesByIds(
    store,
    found.map((item) => item.id),
  );
}

interface Hop {
  id: string;
  targetId: string;
  relatesTo: boolean;
}

function walkPaths(store: Store, origins: readonly string[], maxDepth: number, visit: (hop: Hop) => boolean): void {
  const outgoing = new Map<string, Hop[]>();
  const hopsFrom = (nodeId: string): Hop[] => {
    const cached = outgoing.get(nodeId);
    if (cached !== undefined) {
      return cached;
    }
    const relations = store
      .all<{ id: string; target_id: string }>("SELECT id, target_id FROM facts WHERE source_id = ? ORDER BY rowid", nodeId)
      .map((row): Hop => ({ id: row.id, targetId: row.target_id, relatesTo: true }));
    const mentions = store
      .all<{ id: string; entity_id: string }>("SELECT id, entity_id FROM episode_entities WHERE memory_id = ? ORDER BY rowid", nodeId)
      .map((row): Hop => ({ id: row.id, targetId: row.entity_id, relatesTo: false }));
    const hops = [...relations, ...mentions];
    outgoing.set(nodeId, hops);
    return hops;
  };
  const used = new Set<string>();
  const expand = (nodeId: string, depth: number): boolean => {
    for (const hop of hopsFrom(nodeId)) {
      if (used.has(hop.id)) {
        continue;
      }
      used.add(hop.id);
      const stop = visit(hop) || (depth < maxDepth && expand(hop.targetId, depth + 1));
      used.delete(hop.id);
      if (stop) {
        return true;
      }
    }
    return false;
  };
  for (const origin of origins) {
    if (expand(origin, 1)) {
      return;
    }
  }
}

function edgeBfsSearch(store: Store, origins: readonly string[], maxDepth: number, filters: SearchFilters, limit: number): EntityEdge[] {
  if (origins.length === 0) {
    return [];
  }
  const allowed = allowedEdges(store, filters);
  const edgeIds = new Set<string>();
  walkPaths(store, origins, maxDepth, (hop) => {
    if (edgeIds.size >= limit) {
      return true;
    }
    if (hop.relatesTo && (allowed === undefined || allowed.has(hop.id))) {
      edgeIds.add(hop.id);
    }
    return false;
  });
  return edgesByIds(store, [...edgeIds]);
}

function nodeBfsSearch(store: Store, origins: readonly string[], maxDepth: number, limit: number): EntityNode[] {
  if (origins.length === 0 || maxDepth < 1) {
    return [];
  }
  const rows: string[] = [];
  walkPaths(store, origins, maxDepth, (hop) => {
    if (rows.length >= limit) {
      return true;
    }
    rows.push(hop.targetId);
    return false;
  });
  const nodeById = new Map(entitiesByIds(store, rows).map((node) => [node.id, node]));
  return rows.flatMap((id) => {
    const node = nodeById.get(id);
    return node === undefined ? [] : [node];
  });
}

function nodeFulltextSearch(store: Store, query: string, limit: number): EntityNode[] {
  return entitiesByIds(store, fulltextIds(store, "entities", query, limit));
}

export function nodeSimilaritySearch(store: Store, vector: readonly number[], limit: number, minScore: number): EntityNode[] {
  return entitiesByIds(
    store,
    store.nearest("entities", vector, limit, minScore).map((item) => item.id),
  );
}

function episodeFulltextSearch(store: Store, query: string, limit: number): EpisodicNode[] {
  return episodesByIds(store, fulltextIds(store, "memories", query, limit));
}

async function crossEncoderRank(ai: Ai, query: string, idByPassage: ReadonlyMap<string, string>, minScore: number): Promise<Ranked> {
  const ranked = (await rank(ai, query, [...idByPassage.keys()])).filter(([, score]) => score >= minScore);
  return [ranked.map(([passage]) => idByPassage.get(passage) ?? ""), ranked.map(([, score]) => score)];
}

function byId<T extends { id: string }>(results: readonly T[][]): Map<string, T> {
  return new Map(results.flat().map((item) => [item.id, item]));
}

function idsOf(results: readonly { id: string }[][]): string[][] {
  return results.map((result) => result.map((item) => item.id));
}

interface ScopeSearch {
  query: string;
  vector: readonly number[];
  filters: SearchFilters;
  centerNodeId: string | undefined;
  bfsOriginNodeIds: readonly string[] | undefined;
  limit: number;
  minScore: number;
}

async function edgeSearch(ai: Ai, store: Store, config: ScopeConfig, request: ScopeSearch): Promise<[EntityEdge[], number[]]> {
  const candidateLimit = 2 * request.limit;
  const results: EntityEdge[][] = [];
  if (config.searchMethods.includes("bm25")) {
    results.push(edgeFulltextSearch(store, request.query, request.filters, candidateLimit));
  }
  if (config.searchMethods.includes("cosine_similarity")) {
    results.push(edgeSimilaritySearch(store, request.vector, request.filters, candidateLimit, config.simMinScore));
  }
  if (config.searchMethods.includes("breadth_first_search")) {
    results.push(edgeBfsSearch(store, request.bfsOriginNodeIds ?? [], config.bfsMaxDepth, request.filters, candidateLimit));
    if (request.bfsOriginNodeIds === undefined) {
      const origins = results.flat().map((edge) => edge.sourceId);
      results.push(edgeBfsSearch(store, origins, config.bfsMaxDepth, request.filters, candidateLimit));
    }
  }
  const edgeById = byId(results);
  let ranked: Ranked = [[], []];
  if (config.reranker === "reciprocal_rank_fusion" || config.reranker === "episode_mentions") {
    ranked = rrf(idsOf(results), request.minScore);
  } else if (config.reranker === "mmr") {
    ranked = maximalMarginalRelevance(request.vector, embeddingsFor(store, "facts", [...edgeById.keys()]), config.mmrLambda, request.minScore);
  } else if (config.reranker === "cross_encoder") {
    const [fused] = rrf(idsOf(results), request.minScore);
    const idByFact = new Map(fused.slice(0, candidateLimit).map((id) => [edgeById.get(id)?.fact ?? "", id]));
    ranked = await crossEncoderRank(ai, request.query, idByFact, request.minScore);
  } else if (config.reranker === "node_distance") {
    if (request.centerNodeId === undefined) {
      throw new Error("No center node provided for Node Distance reranker");
    }
    const [fused] = rrf(idsOf(results), request.minScore);
    const edgesBySource = new Map<string, string[]>();
    for (const id of fused) {
      const sourceId = edgeById.get(id)?.sourceId ?? "";
      edgesBySource.set(sourceId, [...(edgesBySource.get(sourceId) ?? []), id]);
    }
    const [nodeIds, scores] = nodeDistanceReranker(store, [...edgesBySource.keys()], request.centerNodeId, request.minScore);
    ranked = [nodeIds.flatMap((nodeId) => edgesBySource.get(nodeId) ?? []), scores];
  }
  const edges = ranked[0].flatMap((id) => {
    const edge = edgeById.get(id);
    return edge === undefined ? [] : [edge];
  });
  if (config.reranker === "episode_mentions") {
    edges.sort((left, right) => right.episodes.length - left.episodes.length);
  }
  return [edges.slice(0, request.limit), ranked[1].slice(0, request.limit)];
}

async function nodeSearch(ai: Ai, store: Store, config: ScopeConfig, request: ScopeSearch): Promise<[EntityNode[], number[]]> {
  const candidateLimit = 2 * request.limit;
  const results: EntityNode[][] = [];
  if (config.searchMethods.includes("bm25")) {
    results.push(nodeFulltextSearch(store, request.query, candidateLimit));
  }
  if (config.searchMethods.includes("cosine_similarity")) {
    results.push(nodeSimilaritySearch(store, request.vector, candidateLimit, config.simMinScore));
  }
  if (config.searchMethods.includes("breadth_first_search")) {
    results.push(nodeBfsSearch(store, request.bfsOriginNodeIds ?? [], config.bfsMaxDepth, candidateLimit));
    if (request.bfsOriginNodeIds === undefined) {
      results.push(nodeBfsSearch(store, results.flat().map((node) => node.id), config.bfsMaxDepth, candidateLimit));
    }
  }
  const nodeById = byId(results);
  const resultIds = idsOf(results);
  let ranked: Ranked = [[], []];
  if (config.reranker === "reciprocal_rank_fusion") {
    ranked = rrf(resultIds, request.minScore);
  } else if (config.reranker === "mmr") {
    ranked = maximalMarginalRelevance(request.vector, embeddingsFor(store, "entities", [...nodeById.keys()]), config.mmrLambda, request.minScore);
  } else if (config.reranker === "cross_encoder") {
    const idByName = new Map([...nodeById.values()].map((node) => [node.name, node.id]));
    ranked = await crossEncoderRank(ai, request.query, idByName, request.minScore);
  } else if (config.reranker === "episode_mentions") {
    ranked = episodeMentionsReranker(store, resultIds, request.minScore);
  } else if (config.reranker === "node_distance") {
    if (request.centerNodeId === undefined) {
      throw new Error("No center node provided for Node Distance reranker");
    }
    ranked = nodeDistanceReranker(store, rrf(resultIds, request.minScore)[0], request.centerNodeId, request.minScore);
  }
  const nodes = ranked[0].flatMap((id) => {
    const node = nodeById.get(id);
    return node === undefined ? [] : [node];
  });
  return [nodes.slice(0, request.limit), ranked[1].slice(0, request.limit)];
}

async function episodeSearch(ai: Ai, store: Store, config: ScopeConfig, request: ScopeSearch): Promise<[EpisodicNode[], number[]]> {
  const results = [episodeFulltextSearch(store, request.query, 2 * request.limit)];
  const episodeById = byId(results);
  let ranked: Ranked = [[], []];
  if (config.reranker === "reciprocal_rank_fusion") {
    ranked = rrf(idsOf(results), request.minScore);
  } else if (config.reranker === "cross_encoder") {
    const [fused] = rrf(idsOf(results), request.minScore);
    const idByContent = new Map(fused.slice(0, request.limit).map((id) => [episodeById.get(id)?.content ?? "", id]));
    ranked = await crossEncoderRank(ai, request.query, idByContent, request.minScore);
  }
  const episodes = ranked[0].flatMap((id) => {
    const episode = episodeById.get(id);
    return episode === undefined ? [] : [episode];
  });
  return [episodes.slice(0, request.limit), ranked[1].slice(0, request.limit)];
}

function needsQueryVector(config: SearchConfig): boolean {
  return [config.edgeConfig, config.nodeConfig].some(
    (scopeConfig) => scopeConfig !== null && (scopeConfig.searchMethods.includes("cosine_similarity") || scopeConfig.reranker === "mmr"),
  );
}

export async function search(ai: Ai, store: Store, request: SearchRequest): Promise<SearchResults> {
  const empty: SearchResults = { edges: [], edgeRerankerScores: [], nodes: [], nodeRerankerScores: [], episodes: [], episodeRerankerScores: [] };
  if ([...request.query].every((char) => pythonWhitespace.has(char.codePointAt(0) ?? 0))) {
    return empty;
  }
  const { config } = request;
  const vector = request.queryVector ?? (needsQueryVector(config) ? await create(ai, request.query.replaceAll("\n", " ")) : []);
  const scopeSearch: ScopeSearch = {
    query: request.query,
    vector,
    filters: request.filters ?? {},
    centerNodeId: request.centerNodeId,
    bfsOriginNodeIds: request.bfsOriginNodeIds,
    limit: config.limit,
    minScore: config.rerankerMinScore,
  };
  const [[edges, edgeRerankerScores], [nodes, nodeRerankerScores], [episodes, episodeRerankerScores]] = await Promise.all([
    config.edgeConfig === null ? ([[], []] as [EntityEdge[], number[]]) : edgeSearch(ai, store, config.edgeConfig, scopeSearch),
    config.nodeConfig === null ? ([[], []] as [EntityNode[], number[]]) : nodeSearch(ai, store, config.nodeConfig, scopeSearch),
    config.episodeConfig === null ? ([[], []] as [EpisodicNode[], number[]]) : episodeSearch(ai, store, config.episodeConfig, scopeSearch),
  ]);
  return { edges, edgeRerankerScores, nodes, nodeRerankerScores, episodes, episodeRerankerScores };
}
