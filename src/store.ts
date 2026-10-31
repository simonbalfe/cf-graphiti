import { migrations } from "./schema";

export type Row = Record<string, SqlStorageValue>;
export type Write = readonly [query: string, ...params: SqlStorageValue[]];
export type VectorTable = "entities" | "facts";

export interface Scored {
  id: string;
  score: number;
}

const vectorQueries: Record<VectorTable, string> = {
  entities: "SELECT id, name_embedding AS embedding FROM entities WHERE name_embedding IS NOT NULL",
  facts: "SELECT id, fact_embedding AS embedding FROM facts WHERE fact_embedding IS NOT NULL",
};

export const inIds = "IN (SELECT value FROM json_each(?))";

export function idList(ids: readonly string[]): string {
  return JSON.stringify([...new Set(ids)]);
}

export function vectorBlob(values: readonly number[] | null): ArrayBuffer | null {
  return values === null ? null : new Float32Array(values).buffer;
}

function normalizedCosine(left: ArrayLike<number>, right: ArrayLike<number>): number | null {
  let dot = 0;
  let leftNorm = 0;
  let rightNorm = 0;
  for (let index = 0; index < left.length; index += 1) {
    const a = left[index] ?? 0;
    const b = right[index] ?? 0;
    dot += a * b;
    leftNorm += a * a;
    rightNorm += b * b;
  }
  return leftNorm === 0 || rightNorm === 0 ? null : (1 + dot / Math.sqrt(leftNorm * rightNorm)) / 2;
}

export class Store {
  readonly #storage: DurableObjectStorage;
  readonly #vectors = new Map<VectorTable, Map<string, Float32Array>>();

  constructor(storage: DurableObjectStorage) {
    this.#storage = storage;
    const version = storage.kv.get<number>("schemaVersion") ?? 0;
    if (version < migrations.length) {
      storage.transactionSync(() => {
        for (const step of migrations.slice(version)) {
          storage.sql.exec(step);
        }
        storage.kv.put("schemaVersion", migrations.length);
      });
    }
  }

  all<T extends Row>(query: string, ...params: SqlStorageValue[]): T[] {
    return this.#storage.sql.exec<T>(query, ...params).toArray();
  }

  first<T extends Row>(query: string, ...params: SqlStorageValue[]): T | null {
    return this.all<T>(query, ...params)[0] ?? null;
  }

  run<T extends Row = Row>(query: string, ...params: SqlStorageValue[]): T[] {
    const rows = this.all<T>(query, ...params);
    this.#vectors.clear();
    return rows;
  }

  commit(writes: readonly Write[]): void {
    if (writes.length === 0) {
      return;
    }
    this.#storage.transactionSync(() => {
      for (const [query, ...params] of writes) {
        this.#storage.sql.exec(query, ...params);
      }
    });
    this.#vectors.clear();
  }

  vector(table: VectorTable, id: string): Float32Array | undefined {
    return this.#loaded(table).get(id);
  }

  nearest(table: VectorTable, query: ArrayLike<number>, limit: number, minScore: number, allowed?: ReadonlySet<string>): Scored[] {
    const scored: Scored[] = [];
    for (const [id, vector] of this.#loaded(table)) {
      if (allowed !== undefined && !allowed.has(id)) {
        continue;
      }
      const score = normalizedCosine(query, vector);
      if (score !== null && score > minScore) {
        scored.push({ id, score });
      }
    }
    return scored.sort((left, right) => right.score - left.score).slice(0, limit);
  }

  #loaded(table: VectorTable): Map<string, Float32Array> {
    const cached = this.#vectors.get(table);
    if (cached !== undefined) {
      return cached;
    }
    const loaded = new Map(
      this.all<{ id: string; embedding: ArrayBuffer }>(vectorQueries[table]).map((row) => [row.id, new Float32Array(row.embedding)]),
    );
    this.#vectors.set(table, loaded);
    return loaded;
  }
}
