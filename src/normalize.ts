import { z } from "zod";

export const kinds = ["message", "transcript", "note", "document", "record"] as const;
export type Kind = (typeof kinds)[number];

const maxRenderedChars = 200_000;
const futureToleranceMs = 5 * 60_000;

const json: z.ZodType<unknown> = z.lazy(() =>
  z.union([z.string(), z.number(), z.boolean(), z.null(), z.array(json), z.record(z.string(), json)]),
);

const oneLine = (max: number) =>
  z
    .string()
    .transform((value) => value.replace(/\s+/g, " ").trim())
    .pipe(z.string().min(1).max(max));

export const eventSchema = z
  .object({
    source: z.string().regex(/^[a-z][a-z0-9-]{0,31}$/, "source must be a lowercase slug like gmail or hubspot"),
    sourceId: oneLine(300),
    revision: oneLine(100),
    kind: z.enum(kinds),
    occurredAt: z.iso.datetime({ offset: true }),
    participants: z.array(z.object({ name: oneLine(120), role: oneLine(60).optional() })).max(50).optional(),
    subject: oneLine(300).optional(),
    body: z.union([z.string(), z.record(z.string(), json)]),
  })
  .superRefine((event, context) => {
    if (event.kind === "record" && typeof event.body === "string") {
      context.addIssue({ code: "custom", path: ["body"], message: "a record body must be a JSON object" });
    }
    if (event.kind !== "record" && typeof event.body !== "string") {
      context.addIssue({ code: "custom", path: ["body"], message: `a ${event.kind} body must be text` });
    }
  });

export type Event = z.output<typeof eventSchema>;

export interface NormalizedEvent {
  source: string;
  sourceId: string;
  revision: string;
  kind: Kind;
  occurredAt: string;
  subject: string | null;
  text: string;
}

const entities: Record<string, string> = { amp: "&", lt: "<", gt: ">", quot: '"', apos: "'", nbsp: " " };

function decodeEntities(text: string): string {
  return text.replace(/&(#x[0-9a-f]+|#\d+|[a-z]+);/gi, (whole, name: string) => {
    if (name.startsWith("#x") || name.startsWith("#X")) {
      return String.fromCodePoint(Number.parseInt(name.slice(2), 16));
    }
    if (name.startsWith("#")) {
      return String.fromCodePoint(Number.parseInt(name.slice(1), 10));
    }
    return entities[name.toLowerCase()] ?? whole;
  });
}

function stripHtml(text: string): string {
  if (!/<\/?[a-z][^>]*>/i.test(text)) {
    return text;
  }
  return text
    .replace(/<(script|style|head)[^>]*>[\s\S]*?<\/\1>/gi, "")
    .replace(/<br\s*\/?>/gi, "\n")
    .replace(/<\/(p|div|li|tr|h[1-6]|blockquote)>/gi, "\n")
    .replace(/<li[^>]*>/gi, "- ")
    .replace(/<[^>]+>/g, "");
}

function dropReplyHistory(text: string): string {
  const lines = text.split("\n");
  const kept: string[] = [];
  for (const line of lines) {
    const trimmed = line.trim();
    if (/^-- ?$/.test(line) || /^On .{4,200} wrote:$/.test(trimmed) || /^-{2,} ?Original Message ?-{2,}$/i.test(trimmed)) {
      break;
    }
    if (/^From: .+$/.test(trimmed) && kept.length > 0 && kept[kept.length - 1]?.trim() === "") {
      break;
    }
    if (!trimmed.startsWith(">")) {
      kept.push(line);
    }
  }
  return kept.join("\n");
}

export function cleanText(text: string, kind: Kind): string {
  let clean = decodeEntities(stripHtml(text.replace(/\r\n?/g, "\n")))
    .replace(/[\u200B-\u200D\uFEFF]/g, "")
    .replace(/\u00A0/g, " ");
  if (kind === "message") {
    clean = dropReplyHistory(clean);
  }
  return clean
    .split("\n")
    .map((line) => line.replace(/[ \t]+/g, " ").trimEnd())
    .join("\n")
    .replace(/\n{3,}/g, "\n\n")
    .trim();
}

function prunedObject(value: object): unknown {
  const entries = Object.entries(value)
    .map(([key, item]): [string, unknown] => [key, pruned(item)])
    .filter(([, item]) => item !== undefined)
    .sort(([left], [right]) => left.localeCompare(right));
  return entries.length === 0 ? undefined : Object.fromEntries(entries);
}

function pruned(value: unknown): unknown {
  if (Array.isArray(value)) {
    const items = value.map(pruned).filter((item) => item !== undefined);
    return items.length === 0 ? undefined : items;
  }
  if (value !== null && typeof value === "object") {
    return prunedObject(value);
  }
  if (typeof value === "string") {
    const trimmed = value.replace(/\s+/g, " ").trim();
    return trimmed === "" ? undefined : trimmed;
  }
  return value ?? undefined;
}

export function canonicalRecord(body: Record<string, unknown>): string {
  return JSON.stringify(pruned(body) ?? {}, null, 2);
}

export type NormalizeResult = { ok: true; event: NormalizedEvent } | { ok: false; reason: string };

export function normalizeEvent(event: Event, nowMs: number): NormalizeResult {
  const occurred = Date.parse(event.occurredAt);
  if (occurred > nowMs + futureToleranceMs) {
    return { ok: false, reason: "occurredAt is in the future" };
  }
  const body = typeof event.body === "string" ? cleanText(event.body, event.kind) : canonicalRecord(event.body);
  if (body === "" || body === "{}") {
    return { ok: false, reason: "body is empty after cleaning" };
  }
  const occurredAt = new Date(occurred).toISOString();
  const seen = new Set<string>();
  const participants = (event.participants ?? []).filter((person) => {
    const key = person.name.toLowerCase();
    if (seen.has(key)) {
      return false;
    }
    seen.add(key);
    return true;
  });
  const header = [
    `Source: ${event.source} ${event.sourceId} (revision ${event.revision})`,
    `Kind: ${event.kind}`,
    `Occurred: ${occurredAt}`,
    participants.length > 0
      ? `Participants: ${participants.map((person) => (person.role === undefined ? person.name : `${person.name} (${person.role})`)).join(", ")}`
      : null,
    event.subject === undefined ? null : `Subject: ${event.subject}`,
  ].filter((line): line is string => line !== null);
  const text = `${header.join("\n")}\n---\n${body}`;
  if (text.length > maxRenderedChars) {
    return { ok: false, reason: `event is ${text.length} characters after cleaning; split it into events under ${maxRenderedChars}` };
  }
  return {
    ok: true,
    event: {
      source: event.source,
      sourceId: event.sourceId,
      revision: event.revision,
      kind: event.kind,
      occurredAt,
      subject: event.subject ?? null,
      text,
    },
  };
}
