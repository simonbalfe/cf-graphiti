const maxQueryLength = 128;
const tokenPattern = /[\p{L}\p{M}\p{N}]+/gu;

export function fulltextQuery(query: string): string | null {
  if (query.split(" ").length >= maxQueryLength) {
    return null;
  }
  const terms = [...new Set(query.toLowerCase().match(tokenPattern) ?? [])];
  return terms.length === 0 ? null : terms.map((term) => `"${term}"`).join(" OR ");
}
