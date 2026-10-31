import type { EntityNode } from "./model";

const nameEntropyThreshold = 1.5;
const minNameLength = 6;
const minTokenCount = 2;
const fuzzyJaccardThreshold = 0.9;
const minhashPermutations = 32;
const minhashBandSize = 4;
const fnvOffset = 0xcbf29ce484222325n;
const fnvPrime = 0x100000001b3n;
const uint64 = 0xffffffffffffffffn;

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

function shingles(normalizedName: string): Set<string> {
  const cleaned = normalizedName.replaceAll(" ", "");
  if (cleaned.length < 2) {
    return cleaned === "" ? new Set() : new Set([cleaned]);
  }
  const result = new Set<string>();
  for (let index = 0; index < cleaned.length - 2; index += 1) {
    result.add(cleaned.slice(index, index + 3));
  }
  return result;
}

function hashShingle(shingle: string, seed: number): bigint {
  let hash = fnvOffset;
  for (const byte of new TextEncoder().encode(`${seed}:${shingle}`)) {
    hash = ((hash ^ BigInt(byte)) * fnvPrime) & uint64;
  }
  return hash;
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
    const candidateShingles = shingles(normalizeNameForFuzzy(candidate.name));
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
  const nodeShingles = shingles(normalizedFuzzy);
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
