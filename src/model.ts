import { idList, inIds, type Store, vectorBlob, type Write } from "./store";

export type EpisodeType = "message" | "text" | "json";

export interface EpisodicNode {
  id: string;
  content: string;
  source: EpisodeType;
  sourceDescription: string;
  validAt: string;
  createdAt: string;
}

export interface EntityNode {
  id: string;
  name: string;
  labels: string[];
  summary: string;
  nameEmbedding: number[] | null;
  createdAt: string;
}

export interface EntityEdge {
  id: string;
  sourceId: string;
  targetId: string;
  name: string;
  fact: string;
  factEmbedding: number[] | null;
  episodes: string[];
  validAt: string | null;
  invalidAt: string | null;
  expiredAt: string | null;
  createdAt: string;
}

type EntityRow = {
  id: string;
  name: string;
  labels: string;
  summary: string;
  created_at: string;
};

type EdgeRow = {
  id: string;
  source_id: string;
  target_id: string;
  name: string;
  fact: string;
  episodes: string;
  valid_at: string | null;
  invalid_at: string | null;
  expired_at: string | null;
  created_at: string;
};

type EpisodeRow = {
  id: string;
  text: string;
  source_type: EpisodeType;
  source_description: string;
  reference_at: string;
  created_at: string;
};

const entitySelect = "SELECT id, name, labels, summary, created_at FROM entities";
const edgeSelect =
  "SELECT f.id AS id, f.source_id AS source_id, f.target_id AS target_id, f.name AS name, f.fact AS fact, " +
  "f.valid_at AS valid_at, f.invalid_at AS invalid_at, f.expired_at AS expired_at, f.created_at AS created_at, " +
  "(SELECT json_group_array(memory_id) FROM fact_episodes WHERE fact_id = f.id) AS episodes FROM facts f";
export const episodeSelect =
  "SELECT id, text, source_type, source_description, reference_at, created_at FROM memories";

function toEntity(row: EntityRow): EntityNode {
  return {
    id: row.id,
    name: row.name,
    labels: JSON.parse(row.labels) as string[],
    summary: row.summary,
    nameEmbedding: null,
    createdAt: row.created_at,
  };
}

function toEdge(row: EdgeRow): EntityEdge {
  return {
    id: row.id,
    sourceId: row.source_id,
    targetId: row.target_id,
    name: row.name,
    fact: row.fact,
    factEmbedding: null,
    episodes: JSON.parse(row.episodes) as string[],
    validAt: row.valid_at,
    invalidAt: row.invalid_at,
    expiredAt: row.expired_at,
    createdAt: row.created_at,
  };
}

export function toEpisode(row: EpisodeRow): EpisodicNode {
  return {
    id: row.id,
    content: row.text,
    source: row.source_type,
    sourceDescription: row.source_description,
    validAt: row.reference_at,
    createdAt: row.created_at,
  };
}

function inOrder<T extends { id: string }>(items: T[], ids: readonly string[]): T[] {
  const byId = new Map(items.map((item) => [item.id, item]));
  return [...new Set(ids)].flatMap((id) => {
    const item = byId.get(id);
    return item === undefined ? [] : [item];
  });
}

export function entitiesByIds(store: Store, ids: readonly string[]): EntityNode[] {
  return inOrder(store.all<EntityRow>(`${entitySelect} WHERE id ${inIds}`, idList(ids)).map(toEntity), ids);
}

export function entitiesByNames(store: Store, names: readonly string[]): EntityNode[] {
  return store.all<EntityRow>(`${entitySelect} WHERE name ${inIds}`, idList(names)).map(toEntity);
}

export function edgesByIds(store: Store, ids: readonly string[]): EntityEdge[] {
  return inOrder(store.all<EdgeRow>(`${edgeSelect} WHERE f.id ${inIds}`, idList(ids)).map(toEdge), ids);
}

export function episodesByIds(store: Store, ids: readonly string[]): EpisodicNode[] {
  return inOrder(store.all<EpisodeRow>(`${episodeSelect} WHERE id ${inIds}`, idList(ids)).map(toEpisode), ids);
}

export function retrieveEpisodes(store: Store, referenceTime: string, lastN: number, source: EpisodeType, excludeId: string): EpisodicNode[] {
  return store
    .all<EpisodeRow>(
      `${episodeSelect} WHERE reference_at <= ? AND source_type = ? AND id <> ? AND extracted_at IS NOT NULL ` +
        "ORDER BY reference_at DESC, created_at DESC LIMIT ?",
      referenceTime,
      source,
      excludeId,
      lastN,
    )
    .map(toEpisode)
    .reverse();
}

export function episodesMentioning(store: Store, entityId: string): EpisodicNode[] {
  return store
    .all<EpisodeRow>(`${episodeSelect} WHERE id IN (SELECT memory_id FROM episode_entities WHERE entity_id = ?)`, entityId)
    .map(toEpisode);
}

export function saveEntity(node: EntityNode): Write {
  return [
    "INSERT INTO entities (id, name, labels, summary, name_embedding, created_at) VALUES (?, ?, ?, ?, ?, ?) " +
      "ON CONFLICT (id) DO UPDATE SET name = excluded.name, labels = excluded.labels, summary = excluded.summary, " +
      "name_embedding = coalesce(excluded.name_embedding, entities.name_embedding)",
    node.id,
    node.name,
    JSON.stringify(node.labels),
    node.summary,
    vectorBlob(node.nameEmbedding),
    node.createdAt,
  ];
}

export function saveEdge(edge: EntityEdge): Write[] {
  return [
    [
      "INSERT INTO facts (id, source_id, target_id, name, fact, fact_embedding, valid_at, invalid_at, expired_at, created_at) " +
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?) ON CONFLICT (id) DO UPDATE SET source_id = excluded.source_id, " +
        "target_id = excluded.target_id, name = excluded.name, fact = excluded.fact, " +
        "fact_embedding = coalesce(excluded.fact_embedding, facts.fact_embedding), valid_at = excluded.valid_at, " +
        "invalid_at = excluded.invalid_at, expired_at = excluded.expired_at",
      edge.id,
      edge.sourceId,
      edge.targetId,
      edge.name,
      edge.fact,
      vectorBlob(edge.factEmbedding),
      edge.validAt,
      edge.invalidAt,
      edge.expiredAt,
      edge.createdAt,
    ],
    ["DELETE FROM fact_episodes WHERE fact_id = ?", edge.id],
    ...edge.episodes.map(
      (episodeId): Write => ["INSERT INTO fact_episodes (fact_id, memory_id) VALUES (?, ?) ON CONFLICT DO NOTHING", edge.id, episodeId],
    ),
  ];
}

export function saveMention(episodeId: string, entityId: string, createdAt: string): Write {
  return ["INSERT INTO episode_entities (id, memory_id, entity_id, created_at) VALUES (?, ?, ?, ?)", crypto.randomUUID(), episodeId, entityId, createdAt];
}
