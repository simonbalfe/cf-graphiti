import { DurableObject } from "cloudflare:workers";
import { addEpisode, type IngestResult, removeEpisode } from "./ingest";
import { isRateLimit } from "./llm";
import { type EpisodeType, type EpisodicNode, episodesByIds } from "./model";
import type { Kind, NormalizedEvent } from "./normalize";
import {
  type Counts,
  counts,
  type EntityDetail,
  entityDetail,
  type Episode as StoredEpisode,
  findEntity,
  type GraphEntity,
  type GraphFact,
  listEpisodes,
  type SearchView,
  searchView,
  wholeGraph,
} from "./queries";
import { type RecipeName, search, searchRecipes } from "./search";
import { Store } from "./store";

const maxAttempts = 8;
const baseDelayMs = 30_000;
const maxDelayMs = 3_600_000;
const rateLimitPauseMs = 90_000;

const episodeTypes: Record<Kind, EpisodeType> = {
  message: "message",
  transcript: "message",
  note: "text",
  document: "text",
  record: "json",
};

export type RememberResult =
  | (IngestResult & { status: "extracted" })
  | { status: "duplicate"; episode: { id: string } }
  | { status: "pending"; episode: { id: string; createdAt: string }; error: string; retryAt: string };

export interface Job {
  episodeId: string;
  attempts: number;
  nextAt: string;
  error: string | null;
  failed: boolean;
}

export interface SearchParams {
  query: string;
  recipe: RecipeName;
  limit: number;
  centerNodeId?: string;
  bfsOriginNodeIds?: string[];
}

type JobRow = {
  episode_id: string;
  attempts: number;
  next_at: number;
  error: string | null;
  failed: number;
};

async function eventHash(event: NormalizedEvent): Promise<string> {
  const text = `${event.source}\n${event.sourceId}\n${event.revision}`;
  const digest = new Uint8Array(await crypto.subtle.digest("SHA-256", new TextEncoder().encode(text)));
  return [...digest].map((byte) => byte.toString(16).padStart(2, "0")).join("");
}

export class MemoryGraph extends DurableObject<Env> {
  readonly #store: Store;
  #queue: Promise<unknown> = Promise.resolve();

  constructor(ctx: DurableObjectState, env: Env) {
    super(ctx, env);
    this.#store = new Store(ctx.storage);
    void ctx.blockConcurrencyWhile(() => this.#armAlarm());
  }

  remember(event: NormalizedEvent, raw: string): Promise<RememberResult> {
    return this.#serial(async () => {
      const hash = await eventHash(event);
      const existing = this.#store.first<{ id: string }>("SELECT id FROM memories WHERE hash = ?", hash);
      if (existing !== null) {
        return { status: "duplicate", episode: { id: existing.id } };
      }
      const episode: EpisodicNode = {
        id: crypto.randomUUID(),
        content: event.text,
        source: episodeTypes[event.kind],
        sourceDescription: `${event.source} ${event.kind}`,
        validAt: event.occurredAt,
        createdAt: new Date().toISOString(),
      };
      const pausedUntil = this.#pausedUntil();
      const paused = pausedUntil > Date.now();
      const queuedAt = Math.max(Date.now(), pausedUntil);
      this.#store.commit([
        [
          "INSERT INTO memories (id, source, source_id, revision, kind, subject, text, raw, hash, reference_at, created_at, " +
            "source_type, source_description) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
          episode.id,
          event.source,
          event.sourceId,
          event.revision,
          event.kind,
          event.subject,
          episode.content,
          raw,
          hash,
          episode.validAt,
          episode.createdAt,
          episode.source,
          episode.sourceDescription,
        ],
        ["INSERT INTO jobs (episode_id, next_at, error) VALUES (?, ?, ?)", episode.id, queuedAt, paused ? "queued: Workers AI rate limit pause" : null],
      ]);
      if (paused) {
        await this.#armAlarm();
        return {
          status: "pending",
          episode: { id: episode.id, createdAt: episode.createdAt },
          error: "queued behind a Workers AI rate limit",
          retryAt: new Date(queuedAt).toISOString(),
        };
      }
      return this.#process(episode);
    });
  }

  forget(id: string): Promise<boolean> {
    return this.#serial(() => removeEpisode(this.env.AI, this.#store, id));
  }

  deleteFact(id: string): Promise<boolean> {
    return this.#serial(async () => this.#store.run("DELETE FROM facts WHERE id = ? RETURNING id", id).length > 0);
  }

  status(): Counts & { jobs: Job[] } {
    return { ...counts(this.#store), jobs: this.#jobs() };
  }

  episodes(limit: number): StoredEpisode[] {
    return listEpisodes(this.#store, limit);
  }

  graph(includeExpired: boolean): { entities: GraphEntity[]; facts: GraphFact[] } {
    return wholeGraph(this.#store, includeExpired);
  }

  entity(nameOrId: string): EntityDetail | null {
    const entity = findEntity(this.#store, nameOrId);
    return entity === null ? null : entityDetail(this.#store, entity);
  }

  async search(params: SearchParams): Promise<SearchView> {
    const results = await search(this.env.AI, this.#store, {
      query: params.query,
      config: { ...searchRecipes[params.recipe], limit: params.limit },
      centerNodeId: params.centerNodeId,
      bfsOriginNodeIds: params.bfsOriginNodeIds,
    });
    return searchView(this.#store, results);
  }

  override async alarm(): Promise<void> {
    await this.#serial(async () => {
      const due = this.#store.all<JobRow>("SELECT * FROM jobs WHERE failed = 0 AND next_at <= ? ORDER BY next_at", Date.now());
      for (const job of due) {
        const [episode] = episodesByIds(this.#store, [job.episode_id]);
        if (episode === undefined) {
          this.#store.run("DELETE FROM jobs WHERE episode_id = ?", job.episode_id);
          continue;
        }
        await this.#process(episode);
        if (this.#pausedUntil() > Date.now()) {
          break;
        }
      }
      await this.#armAlarm();
    });
  }

  #jobs(): Job[] {
    return this.#store.all<JobRow>("SELECT * FROM jobs ORDER BY next_at").map((row) => ({
      episodeId: row.episode_id,
      attempts: row.attempts,
      nextAt: new Date(row.next_at).toISOString(),
      error: row.error,
      failed: row.failed === 1,
    }));
  }

  async #process(episode: EpisodicNode): Promise<RememberResult> {
    try {
      const result = await addEpisode(this.env.AI, this.#store, episode, new Date().toISOString());
      this.#store.run("DELETE FROM jobs WHERE episode_id = ?", episode.id);
      return { status: "extracted", ...result };
    } catch (error) {
      return this.#failed(episode, error);
    }
  }

  async #failed(episode: EpisodicNode, error: unknown): Promise<RememberResult> {
    const message = error instanceof Error ? error.message : String(error);
    const job = this.#store.first<JobRow>("SELECT * FROM jobs WHERE episode_id = ?", episode.id);
    const rateLimited = isRateLimit(error);
    const attempts = (job?.attempts ?? 0) + (rateLimited ? 0 : 1);
    const nextAt = rateLimited
      ? Date.now() + rateLimitPauseMs + Math.random() * 30_000
      : Date.now() + Math.min(baseDelayMs * 2 ** (attempts - 1), maxDelayMs);
    this.#store.run(
      "UPDATE jobs SET attempts = ?, next_at = ?, error = ?, failed = ? WHERE episode_id = ?",
      attempts,
      nextAt,
      message,
      attempts >= maxAttempts ? 1 : 0,
      episode.id,
    );
    if (rateLimited) {
      this.ctx.storage.kv.put("pausedUntil", nextAt);
      this.#store.run("UPDATE jobs SET next_at = max(next_at, ?) WHERE failed = 0", nextAt);
    }
    console.error(JSON.stringify({ event: "ingest_failed", episode: episode.id, attempts, rateLimited, error: message }));
    await this.#armAlarm();
    return {
      status: "pending",
      episode: { id: episode.id, createdAt: episode.createdAt },
      error: message,
      retryAt: new Date(nextAt).toISOString(),
    };
  }

  #pausedUntil(): number {
    return this.ctx.storage.kv.get<number>("pausedUntil") ?? 0;
  }

  async #armAlarm(): Promise<void> {
    const next = this.#store.first<{ next_at: number | null }>("SELECT min(next_at) AS next_at FROM jobs WHERE failed = 0")?.next_at;
    if (next === null || next === undefined) {
      await this.ctx.storage.deleteAlarm();
      return;
    }
    await this.ctx.storage.setAlarm(next);
  }

  #serial<T>(task: () => Promise<T>): Promise<T> {
    const run = this.#queue.then(task, task);
    this.#queue = run.catch(() => undefined);
    return run;
  }
}
