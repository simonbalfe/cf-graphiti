import { Hono } from "hono";
import { HTTPException } from "hono/http-exception";
import { z } from "zod";
import { eventSchema, normalizeEvent } from "./normalize";
import { recipeNames } from "./search";

export { MemoryGraph } from "./graph";

type App = { Bindings: Env; Variables: { profile: string } };

const profileName = z.string().regex(/^[a-z0-9][a-z0-9-]{0,63}$/, "profile must be lowercase letters, digits and dashes, 1..64");

const searchQuery = z.object({
  q: z.string().trim().min(1).max(500),
  recipe: z.enum(recipeNames).default("combined_hybrid_search_cross_encoder"),
  limit: z.coerce.number().int().min(1).max(50).default(10),
  center: z.string().min(1).optional(),
  origins: z
    .string()
    .optional()
    .transform((value) => value?.split(",").filter((id) => id !== "")),
});

const graphQuery = z.object({ includeExpired: z.enum(["0", "1"]).default("0") });
const listQuery = z.object({ limit: z.coerce.number().int().min(1).max(200).default(50) });

function parse<Schema extends z.ZodType>(schema: Schema, value: unknown): z.output<Schema> {
  const result = schema.safeParse(value);
  if (!result.success) {
    const message = result.error.issues
      .map((issue) => (issue.path.length > 0 ? `${issue.path.join(".")}: ${issue.message}` : issue.message))
      .join("; ");
    throw new HTTPException(400, { message });
  }
  return result.data;
}

function graph(env: Env, profile: string) {
  return env.GRAPH.get(env.GRAPH.idFromName(profile));
}

const app = new Hono<App>();

app.onError((error, c) => {
  if (error instanceof HTTPException) {
    return c.json({ error: error.message }, error.status);
  }
  console.error(error);
  return c.json({ error: "Internal error" }, 500);
});

app.get("/health", (c) => c.text("ok\n"));

app.use("/profiles/*", async (c, next) => {
  if (c.req.header("authorization") !== `Bearer ${c.env.API_TOKEN}`) {
    throw new HTTPException(401, { message: "Missing or wrong bearer token" });
  }
  await next();
});

const profile = new Hono<App>();

profile.use("*", async (c, next) => {
  c.set("profile", parse(profileName, c.req.param("profile")));
  await next();
});

profile.post("/remember", async (c) => {
  const raw: unknown = await c.req.json();
  const normalized = normalizeEvent(parse(eventSchema, raw), Date.now());
  if (!normalized.ok) {
    throw new HTTPException(422, { message: normalized.reason });
  }
  const result = await graph(c.env, c.var.profile).remember(normalized.event, JSON.stringify(raw));
  return c.json(result, result.status === "duplicate" ? 200 : result.status === "pending" ? 202 : 201);
});

profile.get("/search", async (c) => {
  const params = parse(searchQuery, c.req.query());
  const results = await graph(c.env, c.var.profile).search({
    query: params.q,
    recipe: params.recipe,
    limit: params.limit,
    centerNodeId: params.center,
    bfsOriginNodeIds: params.origins,
  });
  return c.json({ query: params.q, recipe: params.recipe, ...results });
});

profile.get("/status", async (c) => {
  return c.json(await graph(c.env, c.var.profile).status());
});

profile.get("/graph", async (c) => {
  const params = parse(graphQuery, c.req.query());
  return c.json(await graph(c.env, c.var.profile).graph(params.includeExpired === "1"));
});

profile.get("/entities/:entity", async (c) => {
  const detail = await graph(c.env, c.var.profile).entity(c.req.param("entity"));
  if (detail === null) {
    throw new HTTPException(404, { message: "Entity not found" });
  }
  return c.json(detail);
});

profile.get("/episodes", async (c) => {
  const params = parse(listQuery, c.req.query());
  return c.json({ episodes: await graph(c.env, c.var.profile).episodes(params.limit) });
});

profile.delete("/episodes/:id", async (c) => {
  const id = c.req.param("id");
  if (!(await graph(c.env, c.var.profile).forget(id))) {
    throw new HTTPException(404, { message: "Episode not found" });
  }
  return c.json({ id, deleted: true });
});

profile.delete("/facts/:id", async (c) => {
  const id = c.req.param("id");
  if (!(await graph(c.env, c.var.profile).deleteFact(id))) {
    throw new HTTPException(404, { message: "Fact not found" });
  }
  return c.json({ id, deleted: true });
});

app.route("/profiles/:profile", profile);

export default app;
