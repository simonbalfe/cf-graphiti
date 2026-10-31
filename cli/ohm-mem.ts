#!/usr/bin/env bun
import { createHash } from "node:crypto";
import { readFileSync } from "node:fs";
import { homedir } from "node:os";
import { join } from "node:path";
import { parseArgs } from "node:util";
import { z } from "zod";

const configDir = join(homedir(), ".config", "ohm-mem");
const defaultProfile = "default";
const timeoutMs = 300_000;
const kinds = ["message", "transcript", "note", "document", "record"] as const;

const usage = `ohm-mem: query and write the cf-graphiti memory graph

Usage: ohm-mem <command> [args] [--profile p] [--json]

Commands
  search <query...> [--recipe r] [--limit n] [--center entity-id] [--origins id,id]
                                                         graphiti search_ (default recipe combined_hybrid_search_cross_encoder)
  graph [entity...] [--all]                              every entity, or one entity's facts and episodes
  episodes [--limit n]                                   stored events, newest first
  status                                                 counts and pending ingest jobs
  remember [text...|-] [options]                         add one event (text from args, --file or stdin)
      --source s        where it came from (default cli)
      --id id           stable id in that source (default the --file path or a content hash)
      --revision r      version of that item (default a content hash, so edits are new revisions)
      --kind k          message, transcript, note, document or record (default note; record with --record)
      --at iso          when it happened (default now)
      --subject s       subject line
      --participant p   name or name:role, repeatable
      --file path       read the body from a file
      --record          body is a JSON object
  forget <episode-id>                                    graphiti remove_episode: delete an event, facts and entities only it produced
  drop-fact <fact-id>                                    delete one fact

Global
  --profile p   memory profile (default $OHM_MEM_PROFILE, ~/.config/ohm-mem/profile or ${defaultProfile})
  --json        print the raw API response
  --url u       API base (default $OHM_MEM_URL or ~/.config/ohm-mem/url)

Auth: $OHM_MEM_TOKEN, else ~/.config/ohm-mem/token.
`;

const options = {
  profile: { type: "string" },
  json: { type: "boolean" },
  url: { type: "string" },
  help: { type: "boolean", short: "h" },
  recipe: { type: "string" },
  center: { type: "string" },
  origins: { type: "string" },
  limit: { type: "string" },
  all: { type: "boolean" },
  source: { type: "string" },
  id: { type: "string" },
  revision: { type: "string" },
  kind: { type: "string" },
  at: { type: "string" },
  subject: { type: "string" },
  participant: { type: "string", multiple: true },
  file: { type: "string" },
  record: { type: "boolean" },
} as const;

interface Flags {
  profile?: string;
  json?: boolean;
  url?: string;
  help?: boolean;
  recipe?: string;
  center?: string;
  origins?: string;
  limit?: string;
  all?: boolean;
  source?: string;
  id?: string;
  revision?: string;
  kind?: string;
  at?: string;
  subject?: string;
  participant?: string[];
  file?: string;
  record?: boolean;
}

class UsageError extends Error {}

const date = z.string().nullable();
const errorBody = z.object({ error: z.string() });

const fact = z.object({
  id: z.string(),
  sourceId: z.string(),
  sourceName: z.string(),
  targetId: z.string(),
  targetName: z.string(),
  name: z.string(),
  fact: z.string(),
  episodes: z.array(z.string()),
  validAt: date,
  invalidAt: date,
  expiredAt: date,
  createdAt: z.string(),
});
const entity = z.object({ id: z.string(), name: z.string(), labels: z.array(z.string()), summary: z.string() });
const score = z.number().nullable();
const searchResponse = z.object({
  edges: z.array(fact.extend({ score })),
  nodes: z.array(entity.extend({ score })),
  episodes: z.array(z.object({ id: z.string(), source: z.string(), sourceId: z.string(), subject: z.string().nullable(), text: z.string(), referenceAt: z.string(), score })),
});
const graphResponse = z.object({ entities: z.array(entity), facts: z.array(fact) });
const entityResponse = z.object({
  entity,
  facts: z.array(fact),
  episodes: z.array(z.object({ id: z.string(), source: z.string(), sourceId: z.string(), referenceAt: z.string() })),
});
const episodesResponse = z.object({
  episodes: z.array(
    z.object({
      id: z.string(),
      source: z.string(),
      sourceId: z.string(),
      revision: z.string(),
      kind: z.string(),
      subject: z.string().nullable(),
      referenceAt: z.string(),
      extractedAt: date,
    }),
  ),
});
const statusResponse = z.object({
  episodes: z.number(),
  unextracted: z.number(),
  entities: z.number(),
  facts: z.number(),
  currentFacts: z.number(),
  jobs: z.array(z.object({ episodeId: z.string(), attempts: z.number(), nextAt: z.string(), error: z.string().nullable(), failed: z.boolean() })),
});
const rememberResponse = z.discriminatedUnion("status", [
  z.object({ status: z.literal("duplicate"), episode: z.object({ id: z.string() }) }),
  z.object({ status: z.literal("pending"), episode: z.object({ id: z.string() }), error: z.string(), retryAt: z.string() }),
  z.object({
    status: z.literal("extracted"),
    episode: z.object({ id: z.string() }),
    nodes: z.array(z.object({ name: z.string(), labels: z.array(z.string()), created: z.boolean() })),
    edges: z.array(z.object({ name: z.string(), fact: z.string(), status: z.string() })),
    invalidated: z.array(z.object({ fact: z.string() })),
  }),
]);
const deleteResponse = z.object({ id: z.string(), deleted: z.boolean() });

interface Context {
  base: string;
  prefix: string;
  json: boolean;
}

function configFile(name: string): string | undefined {
  try {
    return readFileSync(join(configDir, name), "utf8").trim() || undefined;
  } catch {
    return undefined;
  }
}

function token(): string {
  const value = process.env.OHM_MEM_TOKEN?.trim() || configFile("token");
  if (value === undefined) {
    throw new Error("no token: set OHM_MEM_TOKEN or write ~/.config/ohm-mem/token");
  }
  return value;
}

function apiBase(flag: string | undefined): string {
  const value = flag ?? process.env.OHM_MEM_URL ?? configFile("url");
  if (value === undefined) {
    throw new UsageError("no API base: pass --url, set OHM_MEM_URL or write ~/.config/ohm-mem/url");
  }
  return value.replace(/\/+$/, "");
}

async function request<Schema extends z.ZodType>(
  context: Context,
  schema: Schema,
  method: string,
  path: string,
  body?: unknown,
): Promise<{ raw: unknown; data: z.output<Schema> }> {
  const response = await fetch(`${context.base}${context.prefix}${path}`, {
    method,
    headers: { authorization: `Bearer ${token()}`, "content-type": "application/json", "user-agent": "ohm-mem" },
    body: body === undefined ? undefined : JSON.stringify(body),
    signal: AbortSignal.timeout(timeoutMs),
  });
  const text = await response.text();
  let raw: unknown;
  try {
    raw = JSON.parse(text);
  } catch {
    throw new Error(`HTTP ${response.status}: ${text.slice(0, 200)}`);
  }
  if (!response.ok) {
    const failure = errorBody.safeParse(raw);
    throw new Error(`${failure.success ? failure.data.error : "request failed"} (HTTP ${response.status})`);
  }
  return { raw, data: schema.parse(raw) };
}

function print(context: Context, raw: unknown, text: string): void {
  process.stdout.write(context.json ? `${JSON.stringify(raw, null, 2)}\n` : text);
}

const day = (value: string | null) => value?.slice(0, 10) ?? "?";

const typeOf = (labels: string[]) => labels.find((label) => label !== "Entity") ?? "Entity";
const scoreText = (value: number | null) => (value === null ? "-" : value.toFixed(4)).padEnd(8);

function factLine(item: z.output<typeof fact>): string {
  const window = item.invalidAt === null ? `since ${day(item.validAt ?? item.createdAt)}` : `${day(item.validAt)} to ${day(item.invalidAt)}`;
  return `  ${item.name}  ${item.fact}  (${window})${item.expiredAt === null ? "" : " [expired]"}`;
}

function query(params: Record<string, string | undefined>): string {
  const search = new URLSearchParams();
  for (const [key, value] of Object.entries(params)) {
    if (value !== undefined) {
      search.set(key, value);
    }
  }
  const text = search.toString();
  return text === "" ? "" : `?${text}`;
}

async function searchCommand(context: Context, words: string[], flags: Flags): Promise<void> {
  if (words.length === 0) {
    throw new UsageError("search: missing query");
  }
  const { raw, data } = await request(
    context,
    searchResponse,
    "GET",
    `/search${query({ q: words.join(" "), recipe: flags.recipe, limit: flags.limit, center: flags.center, origins: flags.origins })}`,
  );
  const sections = [
    ...(data.edges.length === 0 ? [] : ["edges", ...data.edges.map((item) => `${scoreText(item.score)}${factLine(item).trimStart()}`)]),
    ...(data.nodes.length === 0 ? [] : ["nodes", ...data.nodes.map((item) => `${scoreText(item.score)}${item.name} (${typeOf(item.labels)})  ${item.id}`)]),
    ...(data.episodes.length === 0
      ? []
      : ["episodes", ...data.episodes.map((item) => `${scoreText(item.score)}${day(item.referenceAt)}  ${item.subject ?? `${item.source} ${item.sourceId}`}  ${item.id}`)]),
  ];
  print(context, raw, sections.length === 0 ? "no matches\n" : `${sections.join("\n")}\n`);
}

async function graphCommand(context: Context, words: string[], flags: Flags): Promise<void> {
  const name = words.join(" ").trim();
  if (name !== "") {
    const { raw, data } = await request(context, entityResponse, "GET", `/entities/${encodeURIComponent(name)}`);
    const shown = data.facts.filter((item) => flags.all === true || item.expiredAt === null);
    const text = [
      `${data.entity.name} (${typeOf(data.entity.labels)})  ${data.entity.id}`,
      data.entity.summary === "" ? "" : `\n${data.entity.summary}`,
      `\nfacts${flags.all === true ? "" : " (current; --all for ended)"}:`,
      ...shown.map(factLine),
      "\nepisodes:",
      ...data.episodes.map((episode) => `  ${day(episode.referenceAt)}  ${episode.source} ${episode.sourceId}  ${episode.id}`),
      "",
    ].join("\n");
    print(context, raw, text);
    return;
  }
  const { raw, data } = await request(context, graphResponse, "GET", `/graph${query({ includeExpired: flags.all === true ? "1" : undefined })}`);
  const degree = new Map<string, number>();
  for (const item of data.facts) {
    degree.set(item.sourceId, (degree.get(item.sourceId) ?? 0) + 1);
    degree.set(item.targetId, (degree.get(item.targetId) ?? 0) + 1);
  }
  const rows = data.entities
    .map((item) => ({ item, count: degree.get(item.id) ?? 0 }))
    .sort((left, right) => right.count - left.count)
    .map(({ item, count }) => `${String(count).padStart(3)}  ${item.name} (${typeOf(item.labels)})`);
  print(context, raw, `${data.entities.length} entities, ${data.facts.length} facts\n${rows.join("\n")}\n`);
}

async function episodesCommand(context: Context, flags: Flags): Promise<void> {
  const { raw, data } = await request(context, episodesResponse, "GET", `/episodes${query({ limit: flags.limit })}`);
  const lines = data.episodes.map(
    (episode) =>
      `${day(episode.referenceAt)}  ${episode.kind}  ${episode.source} ${episode.sourceId} (rev ${episode.revision})` +
      `${episode.extractedAt === null ? " [not extracted]" : ""}\n  ${episode.id}${episode.subject === null ? "" : `  ${episode.subject}`}`,
  );
  print(context, raw, `${lines.join("\n")}\n`);
}

async function statusCommand(context: Context): Promise<void> {
  const { raw, data } = await request(context, statusResponse, "GET", "/status");
  const lines = [
    `episodes ${data.episodes} (unextracted ${data.unextracted})  entities ${data.entities}  facts ${data.facts} (current ${data.currentFacts})`,
    ...data.jobs.map(
      (job) => `  job ${job.episodeId}  attempts ${job.attempts}  next ${job.nextAt}${job.failed ? "  FAILED" : ""}  ${job.error ?? ""}`,
    ),
  ];
  print(context, raw, `${lines.join("\n")}\n`);
}

function participant(value: string): { name: string; role?: string } {
  const [name = "", role] = value.split(":");
  return role === undefined || role.trim() === "" ? { name: name.trim() } : { name: name.trim(), role: role.trim() };
}

async function readBody(words: string[], file: string | undefined): Promise<string> {
  const joined = words.join(" ");
  if (file !== undefined && joined !== "") {
    throw new UsageError("remember: pass text or --file, not both");
  }
  if (file !== undefined) {
    return readFileSync(file, "utf8").trim();
  }
  return (joined === "" || joined === "-" ? await Bun.stdin.text() : joined).trim();
}

function parseKind(value: string | undefined, record: boolean): (typeof kinds)[number] {
  const kind = value ?? (record ? "record" : "note");
  const found = kinds.find((item) => item === kind);
  if (found === undefined) {
    throw new UsageError(`remember: --kind must be one of ${kinds.join(", ")}`);
  }
  return found;
}

function parseRecord(text: string): unknown {
  try {
    return JSON.parse(text);
  } catch {
    throw new UsageError("remember: --record body is not valid JSON");
  }
}

async function rememberCommand(context: Context, words: string[], flags: Flags): Promise<void> {
  const text = await readBody(words, flags.file);
  if (text === "") {
    throw new UsageError("remember: missing text (args, --file or stdin)");
  }
  const record = flags.record === true;
  const at = flags.at ?? new Date().toISOString();
  if (Number.isNaN(Date.parse(at))) {
    throw new UsageError("remember: --at must be an ISO datetime");
  }
  const hash = createHash("sha256").update(text).digest("hex").slice(0, 16);
  const event = {
    source: flags.source ?? "cli",
    sourceId: flags.id ?? flags.file ?? `note-${hash}`,
    revision: flags.revision ?? hash,
    kind: parseKind(flags.kind, record),
    occurredAt: new Date(at).toISOString(),
    participants: (flags.participant ?? []).map(participant),
    subject: flags.subject,
    body: record ? parseRecord(text) : text,
  };
  const { raw, data } = await request(context, rememberResponse, "POST", "/remember", event);
  print(context, raw, rememberText(data));
}

function rememberText(data: z.output<typeof rememberResponse>): string {
  if (data.status === "duplicate") {
    return `already stored ${data.episode.id}\n`;
  }
  if (data.status === "pending") {
    return `stored ${data.episode.id}, extraction queued until ${data.retryAt}: ${data.error}\n`;
  }
  return [
    `remembered ${data.episode.id}`,
    `entities: ${data.nodes.map((item) => `${item.name} (${typeOf(item.labels)}${item.created ? ", new" : ""})`).join(", ") || "none"}`,
    ...data.edges.map((item) => `  ${item.name}  ${item.fact}${item.status === "duplicate" ? "  (already known)" : ""}`),
    ...data.invalidated.map((item) => `  expired  ${item.fact}`),
    "",
  ].join("\n");
}

async function deleteCommand(context: Context, kind: "episodes" | "facts", words: string[]): Promise<void> {
  const [id] = words;
  if (id === undefined) {
    throw new UsageError("missing id");
  }
  const { raw, data } = await request(context, deleteResponse, "DELETE", `/${kind}/${encodeURIComponent(id)}`);
  print(context, raw, `deleted ${data.id}\n`);
}

async function main(argv: string[]): Promise<number> {
  const { values: flags, positionals } = parseArgs({ args: argv, allowPositionals: true, options });
  const [command, ...words] = positionals;
  if (command === undefined || flags.help === true) {
    process.stdout.write(usage);
    return command === undefined && flags.help !== true ? 2 : 0;
  }
  const context: Context = {
    base: apiBase(flags.url),
    prefix: `/profiles/${encodeURIComponent(flags.profile ?? process.env.OHM_MEM_PROFILE ?? configFile("profile") ?? defaultProfile)}`,
    json: flags.json === true,
  };
  const commands: Record<string, () => Promise<void>> = {
    search: () => searchCommand(context, words, flags),
    graph: () => graphCommand(context, words, flags),
    episodes: () => episodesCommand(context, flags),
    status: () => statusCommand(context),
    remember: () => rememberCommand(context, words, flags),
    forget: () => deleteCommand(context, "episodes", words),
    "drop-fact": () => deleteCommand(context, "facts", words),
  };
  const run = commands[command];
  if (run === undefined) {
    throw new UsageError(`unknown command ${command}`);
  }
  await run();
  return 0;
}

try {
  process.exitCode = await main(process.argv.slice(2));
} catch (error) {
  process.stderr.write(`ohm-mem: ${error instanceof Error ? error.message : String(error)}\n`);
  process.exitCode = error instanceof UsageError ? 2 : 1;
}
