import type { EntityEdge, EntityNode, EpisodicNode } from "./model";
import type { SearchResults } from "./search";
import { idList, inIds, type Store } from "./store";

export type GraphEntity = {
  id: string;
  name: string;
  labels: string[];
  summary: string;
  createdAt: string;
};

export type GraphFact = {
  id: string;
  sourceId: string;
  sourceName: string;
  targetId: string;
  targetName: string;
  name: string;
  fact: string;
  episodes: string[];
  validAt: string | null;
  invalidAt: string | null;
  expiredAt: string | null;
  createdAt: string;
};

export type Episode = {
  id: string;
  text: string;
  source: string;
  sourceId: string;
  revision: string;
  kind: string;
  subject: string | null;
  referenceAt: string;
  createdAt: string;
  extractedAt: string | null;
};

type EntityEpisode = {
  id: string;
  source: string;
  sourceId: string;
  subject: string | null;
  text: string;
  referenceAt: string;
};

export interface Counts {
  episodes: number;
  unextracted: number;
  entities: number;
  facts: number;
  currentFacts: number;
}

export interface EntityDetail {
  entity: GraphEntity;
  facts: GraphFact[];
  episodes: EntityEpisode[];
  neighbours: GraphEntity[];
}

export interface SearchView {
  edges: (GraphFact & { score: number | null })[];
  nodes: (GraphEntity & { score: number | null })[];
  episodes: (Episode & { score: number | null })[];
}

type EntityRow = Omit<GraphEntity, "labels"> & { labels: string };
type FactRow = Omit<GraphFact, "episodes"> & { episodes: string };

const entitySelect = "SELECT id, name, labels, summary, created_at AS createdAt FROM entities";
const factSelect =
  "SELECT f.id AS id, f.source_id AS sourceId, s.name AS sourceName, f.target_id AS targetId, t.name AS targetName, " +
  "f.name AS name, f.fact AS fact, f.valid_at AS validAt, f.invalid_at AS invalidAt, f.expired_at AS expiredAt, " +
  "f.created_at AS createdAt, (SELECT json_group_array(memory_id) FROM fact_episodes WHERE fact_id = f.id) AS episodes " +
  "FROM facts f JOIN entities s ON s.id = f.source_id JOIN entities t ON t.id = f.target_id";
const episodeSelect =
  "SELECT id, text, source, source_id AS sourceId, revision, kind, subject, reference_at AS referenceAt, " +
  "created_at AS createdAt, extracted_at AS extractedAt FROM memories";

function toEntity(row: EntityRow): GraphEntity {
  return { ...row, labels: JSON.parse(row.labels) as string[] };
}

function toFact(row: FactRow): GraphFact {
  return { ...row, episodes: JSON.parse(row.episodes) as string[] };
}

function entities(store: Store, where: string, ...params: SqlStorageValue[]): GraphEntity[] {
  return store.all<EntityRow>(`${entitySelect} ${where}`, ...params).map(toEntity);
}

function facts(store: Store, where: string, ...params: SqlStorageValue[]): GraphFact[] {
  return store.all<FactRow>(`${factSelect} ${where}`, ...params).map(toFact);
}

export function counts(store: Store): Counts {
  return (
    store.first<Counts & Record<string, number>>(
      "SELECT (SELECT count(*) FROM memories) AS episodes, " +
        "(SELECT count(*) FROM memories WHERE extracted_at IS NULL) AS unextracted, " +
        "(SELECT count(*) FROM entities) AS entities, " +
        "(SELECT count(*) FROM facts) AS facts, " +
        "(SELECT count(*) FROM facts WHERE expired_at IS NULL) AS currentFacts",
    ) ?? { episodes: 0, unextracted: 0, entities: 0, facts: 0, currentFacts: 0 }
  );
}

export function listEpisodes(store: Store, limit: number): Episode[] {
  return store.all<Episode>(`${episodeSelect} ORDER BY reference_at DESC LIMIT ?`, limit);
}

export function wholeGraph(store: Store, includeExpired: boolean): { entities: GraphEntity[]; facts: GraphFact[] } {
  return {
    entities: entities(store, "ORDER BY created_at DESC"),
    facts: facts(store, `${includeExpired ? "" : "WHERE f.expired_at IS NULL "}ORDER BY f.created_at DESC`),
  };
}

export function findEntity(store: Store, nameOrId: string): GraphEntity | null {
  return (
    entities(store, "WHERE id = ?", nameOrId)[0] ??
    entities(store, "WHERE lower(name) = lower(?) ORDER BY created_at", nameOrId)[0] ??
    entities(store, "WHERE instr(lower(name), lower(?)) > 0 ORDER BY created_at DESC", nameOrId)[0] ??
    null
  );
}

export function entityDetail(store: Store, entity: GraphEntity): EntityDetail {
  const entityFacts = facts(store, "WHERE f.source_id = ?1 OR f.target_id = ?1 ORDER BY f.created_at DESC", entity.id);
  const episodes = store.all<EntityEpisode>(
    "SELECT m.id AS id, m.source AS source, m.source_id AS sourceId, m.subject AS subject, m.text AS text, " +
      "m.reference_at AS referenceAt FROM episode_entities ee " +
      "JOIN memories m ON m.id = ee.memory_id WHERE ee.entity_id = ? ORDER BY m.reference_at DESC",
    entity.id,
  );
  const neighbourIds = entityFacts.map((fact) => (fact.sourceId === entity.id ? fact.targetId : fact.sourceId)).filter((id) => id !== entity.id);
  return {
    entity,
    facts: entityFacts,
    episodes,
    neighbours: entities(store, `WHERE id ${inIds}`, idList(neighbourIds)),
  };
}

function scored<T>(items: T[], scores: readonly number[]): (T & { score: number | null })[] {
  return items.map((item, index) => ({ ...item, score: scores[index] ?? null }));
}

export function searchView(store: Store, results: SearchResults): SearchView {
  const ordered = <T extends { id: string }>(rows: T[], items: readonly { id: string }[]): T[] => {
    const byId = new Map(rows.map((row) => [row.id, row]));
    return items.flatMap((item) => {
      const row = byId.get(item.id);
      return row === undefined ? [] : [row];
    });
  };
  const idsOf = (items: readonly (EntityEdge | EntityNode | EpisodicNode)[]) => idList(items.map((item) => item.id));
  return {
    edges: scored(ordered(facts(store, `WHERE f.id ${inIds}`, idsOf(results.edges)), results.edges), results.edgeRerankerScores),
    nodes: scored(ordered(entities(store, `WHERE id ${inIds}`, idsOf(results.nodes)), results.nodes), results.nodeRerankerScores),
    episodes: scored(ordered(store.all<Episode>(`${episodeSelect} WHERE id ${inIds}`, idsOf(results.episodes)), results.episodes), results.episodeRerankerScores),
  };
}
