import type { EntityNode } from "./model";

const nameEntropyThreshold = 1.5;
const minNameLength = 6;
const minTokenCount = 2;
const fuzzyJaccardThreshold = 0.9;
const minhashPermutations = 32;
const minhashBandSize = 4;
const shingleCacheSize = 512;
const blake2bIv = new Uint32Array([
  0xf3bcc908, 0x6a09e667, 0x84caa73b, 0xbb67ae85, 0xfe94f82b, 0x3c6ef372, 0x5f1d36f1, 0xa54ff53a, 0xade682d1, 0x510e527f, 0x2b3e6c1f,
  0x9b05688c, 0xfb41bd6b, 0x1f83d9ab, 0x137e2179, 0x5be0cd19,
]);
const blake2bSigma = [
  0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 14, 10, 4, 8, 9, 15, 13, 6, 1, 12, 0, 2, 11, 7, 5, 3, 11, 8, 12, 0, 5, 2, 15, 13, 10,
  14, 3, 6, 7, 1, 9, 4, 7, 9, 3, 1, 13, 12, 11, 14, 2, 6, 5, 10, 4, 0, 15, 8, 9, 0, 5, 7, 2, 4, 10, 15, 14, 1, 11, 12, 6, 8, 3, 13, 2, 12, 6,
  10, 0, 11, 8, 3, 4, 13, 7, 5, 15, 14, 1, 9, 12, 5, 1, 15, 14, 13, 4, 10, 0, 7, 6, 3, 9, 2, 8, 11, 13, 11, 7, 14, 12, 1, 3, 9, 5, 0, 15, 4, 8,
  6, 2, 10, 6, 15, 14, 9, 11, 3, 0, 8, 12, 2, 13, 7, 1, 4, 10, 5, 10, 2, 8, 4, 7, 6, 1, 5, 15, 11, 9, 14, 3, 12, 13, 0, 0, 1, 2, 3, 4, 5, 6, 7,
  8, 9, 10, 11, 12, 13, 14, 15, 14, 10, 4, 8, 9, 15, 13, 6, 1, 12, 0, 2, 11, 7, 5, 3,
].map((index) => index * 2);
const utf8 = new TextEncoder();
const shingleCache = new Map<string, Set<string>>();

export function normalizeStringExact(name: string): string {
  return name.toLowerCase().replace(/\s+/g, " ").trim();
}

function normalizeNameForFuzzy(name: string): string {
  return normalizeStringExact(name)
    .replace(/[^a-z0-9' ]/g, " ")
    .trim()
    .replace(/\s+/g, " ");
}

function nameEntropy(normalizedName: string): number {
  const counts = new Map<string, number>();
  for (const char of normalizedName.replaceAll(" ", "")) {
    counts.set(char, (counts.get(char) ?? 0) + 1);
  }
  const total = [...counts.values()].reduce((sum, count) => sum + count, 0);
  if (total === 0) {
    return 0;
  }
  let entropy = 0;
  for (const count of counts.values()) {
    const probability = count / total;
    entropy -= probability * Math.log2(probability);
  }
  return entropy;
}

function hasHighEntropy(normalizedName: string): boolean {
  const tokenCount = normalizedName.split(/\s+/).filter((token) => token !== "").length;
  if (normalizedName.length < minNameLength && tokenCount < minTokenCount) {
    return false;
  }
  return nameEntropy(normalizedName) >= nameEntropyThreshold;
}

function cachedShingles(normalizedName: string): Set<string> {
  const cached = shingleCache.get(normalizedName);
  if (cached !== undefined) {
    shingleCache.delete(normalizedName);
    shingleCache.set(normalizedName, cached);
    return cached;
  }
  const computed = shingles(normalizedName);
  shingleCache.set(normalizedName, computed);
  if (shingleCache.size > shingleCacheSize) {
    const oldest = shingleCache.keys().next();
    if (oldest.done !== true) {
      shingleCache.delete(oldest.value);
    }
  }
  return computed;
}

function shingles(normalizedName: string): Set<string> {
  const cleaned = normalizedName.replaceAll(" ", "");
  const characters = [...cleaned];
  if (characters.length < 2) {
    return cleaned === "" ? new Set() : new Set([cleaned]);
  }
  const result = new Set<string>();
  for (let index = 0; index < characters.length - 2; index += 1) {
    result.add(characters.slice(index, index + 3).join(""));
  }
  return result;
}

function add64(v: Uint32Array, a: number, low: number, high: number): void {
  const sum = (v[a] ?? 0) + low;
  v[a + 1] = (v[a + 1] ?? 0) + high + (sum >= 0x100000000 ? 1 : 0);
  v[a] = sum;
}

function blake2bMix(v: Uint32Array, m: Uint32Array, a: number, b: number, c: number, d: number, ix: number, iy: number): void {
  add64(v, a, v[b] ?? 0, v[b + 1] ?? 0);
  add64(v, a, m[ix] ?? 0, m[ix + 1] ?? 0);
  let low = (v[d] ?? 0) ^ (v[a] ?? 0);
  let high = (v[d + 1] ?? 0) ^ (v[a + 1] ?? 0);
  v[d] = high;
  v[d + 1] = low;
  add64(v, c, v[d] ?? 0, v[d + 1] ?? 0);
  low = (v[b] ?? 0) ^ (v[c] ?? 0);
  high = (v[b + 1] ?? 0) ^ (v[c + 1] ?? 0);
  v[b] = (low >>> 24) ^ (high << 8);
  v[b + 1] = (high >>> 24) ^ (low << 8);
  add64(v, a, v[b] ?? 0, v[b + 1] ?? 0);
  add64(v, a, m[iy] ?? 0, m[iy + 1] ?? 0);
  low = (v[d] ?? 0) ^ (v[a] ?? 0);
  high = (v[d + 1] ?? 0) ^ (v[a + 1] ?? 0);
  v[d] = (low >>> 16) ^ (high << 16);
  v[d + 1] = (high >>> 16) ^ (low << 16);
  add64(v, c, v[d] ?? 0, v[d + 1] ?? 0);
  low = (v[b] ?? 0) ^ (v[c] ?? 0);
  high = (v[b + 1] ?? 0) ^ (v[c + 1] ?? 0);
  v[b] = (high >>> 31) ^ (low << 1);
  v[b + 1] = (low >>> 31) ^ (high << 1);
}

function blake2bCompress(h: Uint32Array, block: Uint8Array, counter: number, last: boolean): void {
  const v = new Uint32Array(32);
  const m = new Uint32Array(32);
  for (let index = 0; index < 16; index += 1) {
    v[index] = h[index] ?? 0;
    v[index + 16] = blake2bIv[index] ?? 0;
  }
  v[24] = (v[24] ?? 0) ^ counter;
  v[25] = (v[25] ?? 0) ^ Math.floor(counter / 0x100000000);
  if (last) {
    v[28] = ~(v[28] ?? 0);
    v[29] = ~(v[29] ?? 0);
  }
  for (let index = 0; index < 32; index += 1) {
    const offset = index * 4;
    m[index] = (block[offset] ?? 0) ^ ((block[offset + 1] ?? 0) << 8) ^ ((block[offset + 2] ?? 0) << 16) ^ ((block[offset + 3] ?? 0) << 24);
  }
  for (let round = 0; round < 12; round += 1) {
    const s = round * 16;
    const sigma = (index: number): number => blake2bSigma[s + index] ?? 0;
    blake2bMix(v, m, 0, 8, 16, 24, sigma(0), sigma(1));
    blake2bMix(v, m, 2, 10, 18, 26, sigma(2), sigma(3));
    blake2bMix(v, m, 4, 12, 20, 28, sigma(4), sigma(5));
    blake2bMix(v, m, 6, 14, 22, 30, sigma(6), sigma(7));
    blake2bMix(v, m, 0, 10, 20, 30, sigma(8), sigma(9));
    blake2bMix(v, m, 2, 12, 22, 24, sigma(10), sigma(11));
    blake2bMix(v, m, 4, 14, 16, 26, sigma(12), sigma(13));
    blake2bMix(v, m, 6, 8, 18, 28, sigma(14), sigma(15));
  }
  for (let index = 0; index < 16; index += 1) {
    h[index] = (h[index] ?? 0) ^ (v[index] ?? 0) ^ (v[index + 16] ?? 0);
  }
}

function byteSwap32(word: number): number {
  return (((word & 0xff) << 24) | (((word >>> 8) & 0xff) << 16) | (((word >>> 16) & 0xff) << 8) | ((word >>> 24) & 0xff)) >>> 0;
}

function hashShingle(shingle: string, seed: number): bigint {
  const input = utf8.encode(`${seed}:${shingle}`);
  const h = new Uint32Array(blake2bIv);
  h[0] = (h[0] ?? 0) ^ 0x01010008;
  const blockCount = Math.max(1, Math.ceil(input.length / 128));
  for (let blockIndex = 0; blockIndex < blockCount; blockIndex += 1) {
    const start = blockIndex * 128;
    const block = new Uint8Array(128);
    block.set(input.subarray(start, start + 128));
    const last = blockIndex === blockCount - 1;
    blake2bCompress(h, block, last ? input.length : start + 128, last);
  }
  return (BigInt(byteSwap32(h[0] ?? 0)) << 32n) | BigInt(byteSwap32(h[1] ?? 0));
}

function minhashSignature(shingleSet: ReadonlySet<string>): bigint[] {
  if (shingleSet.size === 0) {
    return [];
  }
  const signature: bigint[] = [];
  for (let seed = 0; seed < minhashPermutations; seed += 1) {
    let min: bigint | null = null;
    for (const shingle of shingleSet) {
      const hash = hashShingle(shingle, seed);
      if (min === null || hash < min) {
        min = hash;
      }
    }
    signature.push(min ?? 0n);
  }
  return signature;
}

function lshBands(signature: readonly bigint[]): string[] {
  const bands: string[] = [];
  for (let start = 0; start + minhashBandSize <= signature.length; start += minhashBandSize) {
    bands.push(`${start / minhashBandSize}:${signature.slice(start, start + minhashBandSize).join(",")}`);
  }
  return bands;
}

function jaccardSimilarity(left: ReadonlySet<string>, right: ReadonlySet<string>): number {
  if (left.size === 0 && right.size === 0) {
    return 1;
  }
  if (left.size === 0 || right.size === 0) {
    return 0;
  }
  let intersection = 0;
  for (const item of left) {
    if (right.has(item)) {
      intersection += 1;
    }
  }
  const union = left.size + right.size - intersection;
  return union === 0 ? 0 : intersection / union;
}

export interface CandidateIndexes {
  existingNodes: EntityNode[];
  nodesById: Map<string, EntityNode>;
  normalizedExisting: Map<string, EntityNode[]>;
  shinglesByCandidate: Map<string, Set<string>>;
  lshBuckets: Map<string, string[]>;
}

export function buildCandidateIndexes(existingNodes: EntityNode[]): CandidateIndexes {
  const indexes: CandidateIndexes = {
    existingNodes,
    nodesById: new Map(),
    normalizedExisting: new Map(),
    shinglesByCandidate: new Map(),
    lshBuckets: new Map(),
  };
  for (const candidate of existingNodes) {
    const normalized = normalizeStringExact(candidate.name);
    indexes.normalizedExisting.set(normalized, [...(indexes.normalizedExisting.get(normalized) ?? []), candidate]);
    indexes.nodesById.set(candidate.id, candidate);
    const candidateShingles = cachedShingles(normalizeNameForFuzzy(candidate.name));
    indexes.shinglesByCandidate.set(candidate.id, candidateShingles);
    for (const band of lshBands(minhashSignature(candidateShingles))) {
      indexes.lshBuckets.set(band, [...(indexes.lshBuckets.get(band) ?? []), candidate.id]);
    }
  }
  return indexes;
}

export function promoteResolvedNode(extracted: EntityNode, resolved: EntityNode): EntityNode {
  const resolvedSpecific = resolved.labels.filter((label) => label !== "Entity");
  const extractedSpecific = extracted.labels.filter((label) => label !== "Entity");
  const extractedSet = new Set(extractedSpecific);
  const isStrictSubset = resolvedSpecific.every((label) => extractedSet.has(label)) && extractedSet.size > new Set(resolvedSpecific).size;
  if (!isStrictSubset) {
    return resolved;
  }
  resolved.labels = [...new Set(["Entity", ...resolvedSpecific, ...extractedSpecific])];
  return resolved;
}

export function resolveWithSimilarity(node: EntityNode, indexes: CandidateIndexes): EntityNode | null {
  const existingMatches = indexes.normalizedExisting.get(normalizeStringExact(node.name)) ?? [];
  const [onlyMatch] = existingMatches;
  if (existingMatches.length === 1 && onlyMatch !== undefined) {
    return promoteResolvedNode(node, onlyMatch);
  }
  if (existingMatches.length > 1) {
    return null;
  }
  const normalizedFuzzy = normalizeNameForFuzzy(node.name);
  if (!hasHighEntropy(normalizedFuzzy)) {
    return null;
  }
  const nodeShingles = cachedShingles(normalizedFuzzy);
  const candidateIds = new Set<string>();
  for (const band of lshBands(minhashSignature(nodeShingles))) {
    for (const id of indexes.lshBuckets.get(band) ?? []) {
      candidateIds.add(id);
    }
  }
  let bestCandidate: EntityNode | undefined;
  let bestScore = 0;
  for (const id of candidateIds) {
    const score = jaccardSimilarity(nodeShingles, indexes.shinglesByCandidate.get(id) ?? new Set());
    if (score > bestScore) {
      bestScore = score;
      bestCandidate = indexes.nodesById.get(id);
    }
  }
  if (bestCandidate !== undefined && bestScore >= fuzzyJaccardThreshold) {
    return promoteResolvedNode(node, bestCandidate);
  }
  return null;
}
