import type { EntityEdge } from "./model";

const futureAssertion =
  /\b(will|shall|is going to|are going to|going to|plans? to|planned to|planning to|intends? to|intended to|expects? to|expected to|promises? to|promised to|scheduled to|is scheduled to|are scheduled to|set to|aims? to|aimed to|committed to)\b|[a-z]+'ll\b/i;
const deadline = /\b(by|before|no later than|deadline|due by|due on)\b/i;
const ongoingStateEnd = /\b(until|through|thru|expires?|expired|ends?|ended|ending|stops?|stopped|ceases?|ceased|no longer)\b/i;

function time(value: string | null): number | null {
  return value === null ? null : Date.parse(value);
}

export function parseTimestamp(value: string | null): string | null {
  if (value === null || value.trim() === "") {
    return null;
  }
  const text = value.trim().replace(/Z$/, "+00:00");
  const hasZone = /(?:[+-]\d{2}:?\d{2})$/.test(text);
  const hasTime = text.includes("T") || text.includes(" ");
  const parsed = Date.parse(hasTime && !hasZone ? `${text.replace(" ", "T")}Z` : text);
  return Number.isNaN(parsed) ? null : new Date(parsed).toISOString();
}

export function normalizeFutureTemporalBounds(
  fact: string,
  referenceTime: string | null,
  validAt: string | null,
  invalidAt: string | null,
): [validAt: string | null, invalidAt: string | null] {
  const reference = time(referenceTime);
  if (reference === null || !futureAssertion.test(fact)) {
    return [validAt, invalidAt];
  }
  const valid = time(validAt);
  const normalizedValidAt = valid === null || valid > reference ? referenceTime : validAt;
  const invalid = time(invalidAt);
  const dropsInvalidAt = invalid !== null && invalid > reference && deadline.test(fact) && !ongoingStateEnd.test(fact);
  return [normalizedValidAt, dropsInvalidAt ? null : invalidAt];
}

function resolveEdgeContradictions(resolved: EntityEdge, candidates: readonly EntityEdge[], nowIso: string): EntityEdge[] {
  const invalidated: EntityEdge[] = [];
  const resolvedValid = time(resolved.validAt);
  const resolvedInvalid = time(resolved.invalidAt);
  for (const edge of candidates) {
    const edgeInvalid = time(edge.invalidAt);
    const edgeValid = time(edge.validAt);
    if (
      (edgeInvalid !== null && resolvedValid !== null && edgeInvalid <= resolvedValid) ||
      (edgeValid !== null && resolvedInvalid !== null && resolvedInvalid <= edgeValid)
    ) {
      continue;
    }
    if (edgeValid !== null && resolvedValid !== null && edgeValid < resolvedValid) {
      edge.invalidAt = resolved.validAt;
      edge.expiredAt = edge.expiredAt ?? nowIso;
      invalidated.push(edge);
    }
  }
  return invalidated;
}

export function applyTemporalInvalidation(resolved: EntityEdge, candidates: EntityEdge[], nowIso: string): EntityEdge[] {
  if (resolved.invalidAt !== null && resolved.expiredAt === null) {
    resolved.expiredAt = nowIso;
  }
  if (resolved.expiredAt === null) {
    candidates.sort((left, right) => {
      const leftValid = time(left.validAt);
      const rightValid = time(right.validAt);
      if (leftValid === null || rightValid === null) {
        return Number(leftValid === null) - Number(rightValid === null);
      }
      return leftValid - rightValid;
    });
    const resolvedValid = time(resolved.validAt);
    for (const candidate of candidates) {
      const candidateValid = time(candidate.validAt);
      if (candidateValid !== null && resolvedValid !== null && candidateValid > resolvedValid) {
        resolved.invalidAt = candidate.validAt;
        resolved.expiredAt = nowIso;
        break;
      }
    }
  }
  if (candidates.length === 0) {
    return [];
  }
  return resolveEdgeContradictions(resolved, candidates, nowIso);
}
