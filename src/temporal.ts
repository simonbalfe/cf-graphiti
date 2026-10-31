import type { EntityEdge } from "./model";

const futureAssertion =
  /\b(will|shall|is going to|are going to|going to|plans? to|planned to|planning to|intends? to|intended to|expects? to|expected to|promises? to|promised to|scheduled to|is scheduled to|are scheduled to|set to|aims? to|aimed to|committed to)\b|[a-z]+'ll\b/i;
const deadline = /\b(by|before|no later than|deadline|due by|due on)\b/i;
const ongoingStateEnd = /\b(until|through|thru|expires?|expired|ends?|ended|ending|stops?|stopped|ceases?|ceased|no longer)\b/i;
const isoTimestamp =
  /^(?:(\d{4})-(\d{2})-(\d{2})|(\d{4})(\d{2})(\d{2}))(?:[^\d](\d{2})(?::?(\d{2})(?::?(\d{2})(?:[.,](\d+))?)?)?(?:([+-])(\d{2})(?::?(\d{2})(?::?(\d{2})(?:\.\d+)?)?)?)?)?$/;

function time(value: string | null): number | null {
  return value === null ? null : Date.parse(value);
}

export function parseTimestamp(value: string | null): string | null {
  if (value === null || value === "") {
    return null;
  }
  const match = isoTimestamp.exec(value.replaceAll("Z", "+00:00"));
  if (match === null) {
    return null;
  }
  const year = Number(match[1] ?? match[4]);
  const month = Number(match[2] ?? match[5]);
  const day = Number(match[3] ?? match[6]);
  const hour = Number(match[7] ?? 0);
  const minute = Number(match[8] ?? 0);
  const second = Number(match[9] ?? 0);
  const millisecond = Number((match[10] ?? "").slice(0, 3).padEnd(3, "0"));
  const offsetHour = Number(match[12] ?? 0);
  const offsetMinute = Number(match[13] ?? 0);
  const offsetSecond = Number(match[14] ?? 0);
  const daysInMonth = new Date(Date.UTC(2000, month, 0)).getUTCDate() - (month === 2 && !(year % 4 === 0 && (year % 100 !== 0 || year % 400 === 0)) ? 1 : 0);
  if (
    year < 1 ||
    month < 1 ||
    month > 12 ||
    day < 1 ||
    day > daysInMonth ||
    hour > 23 ||
    minute > 59 ||
    second > 59 ||
    offsetHour > 23 ||
    offsetMinute > 59 ||
    offsetSecond > 59
  ) {
    return null;
  }
  const date = new Date(Date.UTC(2000, month - 1, day, hour, minute, second, millisecond));
  date.setUTCFullYear(year);
  const offset = (match[11] === "-" ? -1 : 1) * ((offsetHour * 60 + offsetMinute) * 60 + offsetSecond) * 1000;
  return new Date(date.getTime() - offset).toISOString();
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
