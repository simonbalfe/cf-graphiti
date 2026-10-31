const maxQueryLength = 128;
const groupIdCount = 1;
const tokenPattern = /[\p{L}\p{N}\p{Co}]+/gu;

export function fulltextQuery(query: string): string | null {
  if (query.split(" ").length + groupIdCount >= maxQueryLength) {
    return null;
  }
  const terms = query.match(tokenPattern) ?? [];
  return terms.length === 0 ? null : terms.map((term) => `"${term}"`).join(" OR ");
}
