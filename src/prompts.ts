import { z } from "zod";
import { pyDatetime, pyRepr, toPromptJson } from "./pyformat";

export const maxSummaryChars = 1000;

export interface Message {
  role: "system" | "user";
  content: string;
}

export interface ResponseModel<Schema extends z.ZodType> {
  name: string;
  schema: Schema;
  jsonSchema: Record<string, unknown>;
}

function strictJsonSchema(node: unknown): unknown {
  if (Array.isArray(node)) {
    return node.map(strictJsonSchema);
  }
  if (node === null || typeof node !== "object") {
    return node;
  }
  const out: Record<string, unknown> = {};
  for (const [key, value] of Object.entries(node)) {
    if (key === "$schema" || key === "default") {
      continue;
    }
    if ((key === "minimum" || key === "maximum") && Math.abs(Number(value)) === Number.MAX_SAFE_INTEGER) {
      continue;
    }
    out[key] = strictJsonSchema(value);
  }
  const properties = out["properties"];
  if (out["type"] === "object" && properties !== null && typeof properties === "object") {
    out["additionalProperties"] = false;
    out["required"] = Object.keys(properties);
  }
  return out;
}

function responseModel<Schema extends z.ZodType>(name: string, schema: Schema): ResponseModel<Schema> {
  const jsonSchema = strictJsonSchema(z.toJSONSchema(schema, { io: "input" }));
  return { name, schema, jsonSchema: jsonSchema !== null && typeof jsonSchema === "object" ? { ...jsonSchema } : {} };
}

export interface PreviousEpisode {
  content: string;
  timestamp: string | null;
}

export interface EntityTypeContext {
  entity_type_id: number;
  entity_type_name: string;
  entity_type_description: string;
}

export interface ExtractNodesContext {
  episodeContent: string;
  previousEpisodes: PreviousEpisode[];
  customExtractionInstructions: string;
  entityTypes: EntityTypeContext[];
  sourceDescription: string;
}

export interface DedupeNodesContext {
  extractedNodes: { id: number; name: string; entity_type: string[]; entity_type_description: string }[];
  existingNodes: { candidate_id: number; name: string; entity_types: string[]; summary: string }[];
  episodeContent: string;
  previousEpisodes: PreviousEpisode[];
}

export interface ExtractEdgesContext {
  episodeContent: string;
  nodes: { name: string; entity_types: string[] }[];
  previousEpisodes: PreviousEpisode[];
  referenceTime: string;
  customExtractionInstructions: string;
}

export interface ExtractTimestampsContext {
  fact: string;
  referenceTime: string;
}

export interface ResolveEdgeContext {
  existingEdges: { idx: number; fact: string }[];
  edgeInvalidationCandidates: { idx: number; fact: string }[];
  newEdge: string;
}

export interface SummariesContext {
  entities: { name: string; summary: string; entity_types: string[]; attributes: Record<string, unknown> }[];
  episodeContent: string;
  previousEpisodes: PreviousEpisode[];
  entityTypeDescriptions: Record<string, string>;
}

export const extractedEntities = responseModel(
  "ExtractedEntities",
  z.object({
    extracted_entities: z
      .array(
        z.object({
          name: z.string().describe("Name of the extracted entity"),
          entity_type_id: z
            .number()
            .int()
            .describe("ID of the classified entity type. Must be one of the provided entity_type_id integers."),
          episode_indices: z
            .array(z.number().int())
            .default([0])
            .describe(
              "List of episode numbers (0-indexed) this entity was extracted from. When processing a single episode, this should be [0].",
            ),
        }),
      )
      .describe("List of extracted entities"),
  }),
);

export const nodeResolutions = responseModel(
  "NodeResolutions",
  z.object({
    entity_resolutions: z
      .array(
        z.object({
          id: z.number().int().describe("integer id of the entity"),
          name: z
            .string()
            .describe(
              "Name of the entity. Should be the most complete and descriptive name of the entity. Do not include any JSON formatting in the Entity name such as {}.",
            ),
          duplicate_candidate_id: z
            .number()
            .int()
            .describe("candidate_id of the matching EXISTING ENTITY, or -1 if no duplicate exists."),
        }),
      )
      .describe("List of resolved nodes"),
  }),
);

export const extractedEdges = responseModel(
  "ExtractedEdges",
  z.object({
    edges: z.array(
      z.object({
        source_entity_name: z.string().describe("The name of the source entity from the ENTITIES list"),
        target_entity_name: z.string().describe("The name of the target entity from the ENTITIES list"),
        relation_type: z
          .string()
          .describe(
            "The type of relationship between the entities, in SCREAMING_SNAKE_CASE (e.g., WORKS_AT, LIVES_IN, IS_FRIENDS_WITH)",
          ),
        fact: z
          .string()
          .describe("A natural language description of the relationship between the entities, paraphrased from the source text"),
        valid_at: z
          .string()
          .nullable()
          .default(null)
          .describe(
            "The date and time when the relationship or assertion described by the edge fact became true or was established. Future-tense plans/promises use the statement reference time, not the future occurrence date. Use ISO 8601 format (YYYY-MM-DDTHH:MM:SS.SSSSSSZ)",
          ),
        invalid_at: z
          .string()
          .nullable()
          .default(null)
          .describe(
            "The date and time when the relationship described by the edge fact stopped being true or ended. Soft deadlines are not invalid_at values. Use ISO 8601 format (YYYY-MM-DDTHH:MM:SS.SSSSSSZ)",
          ),
        episode_indices: z
          .array(z.number().int())
          .default([0])
          .describe(
            "List of episode numbers (0-indexed) that this fact was derived from. When processing a single episode, this should be [0].",
          ),
      }),
    ),
  }),
);

export const edgeTimestamps = responseModel(
  "EdgeTimestamps",
  z
    .object({
      valid_at: z
        .string()
        .nullable()
        .default(null)
        .describe(
          "When the fact became true. Future-tense plans/promises use the reference time, not the future occurrence date. ISO 8601 with Z suffix (e.g., 2025-04-30T00:00:00Z)",
        ),
      invalid_at: z
        .string()
        .nullable()
        .default(null)
        .describe(
          "When the fact stopped being true. Soft deadlines are not invalid_at values. ISO 8601 with Z suffix (e.g., 2025-04-30T00:00:00Z)",
        ),
    })
    .describe("Temporal bounds extracted from a fact."),
);

export const edgeDuplicate = responseModel(
  "EdgeDuplicate",
  z.object({
    duplicate_facts: z
      .array(z.number().int())
      .describe("List of idx values of duplicate facts (only from EXISTING FACTS range). Empty list if none."),
    contradicted_facts: z
      .array(z.number().int())
      .describe("List of idx values of contradicted facts (from full idx range). Empty list if none."),
  }),
);

export const summarizedEntities = responseModel(
  "SummarizedEntities",
  z.object({
    summaries: z
      .array(
        z.object({
          name: z.string().describe("Name of the entity being summarized"),
          summary: z.string().describe("Updated summary for the entity"),
        }),
      )
      .describe("List of entity summaries. Only include entities that need summary updates."),
  }),
);

const summaryInstructions = `Guidelines:
        1. Output only factual content. Never explain what you're doing, why, or mention limitations or constraints.
        2. Only use the provided messages, entity, and entity context to set attribute values.
        3. Keep the summary information-dense and entity-specific. STATE FACTS DIRECTLY IN UNDER 1000 CHARACTERS.
        4. Preserve all materially relevant names, roles, places, dates, counts, and temporal qualifiers that are explicitly supported.
        5. Prefer compact factual sentences over vague thematic phrasing or meta-language.
        6. When the durable fact is the content of what was said, state the content directly instead of narrating that it was said.
        7. Use communication verbs only when the act of speaking, asking, sharing, presenting, announcing, or telling is itself the important fact.
        8. Never use filler verbs like "mentioned", "described", "stated", "reported", "noted", "discussed", "referenced", or "indicated" unless the communication act itself is the fact.
        9. Include temporal anchors when the messages provide them and they help ground the fact.
        10. Begin with the entity name or a direct fact, not with "A", "An", "The", or "This is" unless that wording is part of the entity name.

        Example summary:
        BAD: "The context shows John ordered pizza. Due to length constraints, other details are omitted from this summary."
        GOOD: "John ordered pepperoni pizza from Mario's at 7:30 PM and had it delivered to the office."
        `;

export function extractMessage(context: ExtractNodesContext): Message[] {
  return [
    { role: "system", content: "You are an entity extraction specialist for conversational messages. NEVER extract abstract concepts, feelings, or generic words." },
    { role: "user", content: `
NEVER extract any of the following:
- Pronouns (you, me, I, he, she, they, we, us, it, them, him, her, this, that, those)
- Abstract concepts or feelings (joy, balance, growth, resilience, happiness, passion, motivation)
- Generic common nouns or bare object words (day, life, people, work, stuff, things, food, time,
  way, tickets, supplies, clothes, keys, gear)
- Generic media/content nouns unless uniquely identified in the node name itself (photo, pic, picture,
  image, video, post, story)
- Generic event/activity nouns unless uniquely identified in the node name itself (event, game, meeting,
  class, workshop, competition)
- Broad institutional nouns unless explicitly named or uniquely qualified (government, school, company,
  team, office)
- Ambiguous bare nouns whose meaning depends on sentence context rather than the node name itself
- Sentence fragments or clauses ("what you really care about", "results of that effort")
- Adjectives or descriptive phrases ("amazing", "something different", "new hair color")
- Duplicate references to the same real-world entity. Extract each entity at most once per message,
  even if it appears multiple times or both as a speaker label and in the body text.
- Bare relational or kinship terms (dad, mom, mother, father, sister, brother, husband, wife,
  spouse, son, daughter, uncle, aunt, cousin, grandma, grandpa, friend, boss, teacher, neighbor,
  roommate) and bare animal/pet words (dog, cat, pet, puppy, kitten). These are too generic on
  their own. Instead, qualify them with the possessor: extract "Nisha's dad" not "dad",
  "Jordan's dog" not "dog".
- Bare generic objects that cannot be meaningfully qualified with a possessor, brand, or
  distinguishing detail (e.g., NEVER extract "supplies" from "I picked up some supplies")

Your task is to extract **entity nodes** that are **explicitly** mentioned in the CURRENT MESSAGE.
Pronoun references such as he/she/they or this/that/those should be disambiguated to the names of the
reference entities. Only extract distinct entities from the CURRENT MESSAGE.

<ENTITY TYPES>
${pyRepr(context.entityTypes)}
</ENTITY TYPES>

<PREVIOUS MESSAGES>
${toPromptJson(context.previousEpisodes)}
</PREVIOUS MESSAGES>

<CURRENT MESSAGE>
${context.episodeContent}
</CURRENT MESSAGE>

1. **Speaker Extraction**: Always extract the speaker (the part before the colon \`:\` in each dialogue line) as the first entity node.
   - If the speaker is mentioned again in the message, treat both mentions as a **single entity**.

2. **Entity Identification**:
   - Extract named entities and specific, concrete things that are **explicitly** mentioned in the CURRENT MESSAGE.
   - Only extract entities that are specific enough to be uniquely identifiable. Ask: "Could this have its own Wikipedia article or database entry, OR is it specific enough to distinguish from other items of the same category within this conversation?"
   - For objects, possessions, and physical items, extract when they are specific enough
     to distinguish from other items of the same category. SHOULD be extracted:
     - Brand-named items ("Gamecube", "Ford Mustang", "Moen faucet")
     - Qualified items ("wool coat", "red and purple lighting", "cracked windshield",
       "dog leash")
     - Items with a concrete distinguishing descriptor (color, material, size, model,
       owner, specific use)
     Should NOT be extracted:
     - Bare head nouns alone ("car", "coat", "game", "lighting", "windshield")
   - When a speaker or named person refers to a relative, pet, or associate using a bare term
     (e.g., "my dad", "his cat"), extract the entity qualified with the possessor's name
     (e.g., "Nisha's dad", "Jordan's cat"). Do NOT extract the bare term alone.
   - **Exclude** entities mentioned only in the PREVIOUS MESSAGES (they are for context only).

3. **Entity Classification**:
   - Use the descriptions in ENTITY TYPES to classify each extracted entity.
   - Assign the appropriate \`entity_type_id\` for each one.

4. **Exclusions**:
   - Do NOT extract entities representing relationships or actions.
   - Do NOT extract dates, times, or other temporal information — these will be handled separately.
   - When in doubt, do NOT extract.

5. **Specificity**:
   - Always use the **most specific form** mentioned in the message. If the message says "road cycling",
     extract "road cycling" not "cycling". If it says "wool coat", extract "wool coat" not "coat".
   - When context makes an object's type clear, include that context in the name. For example, if the
     message mentions forgetting a leash while discussing a dog walk, extract "dog leash" not "leash".
   - If a phrase would not be distinguishable when read alone later, do NOT extract it.

6. **Formatting**:
   - Be **explicit and unambiguous** in naming entities (e.g., use full names when available).

<EXAMPLE>
Message: "Jordan: We just moved to Denver last month. My spouse started a new role at Lockheed Martin and I enrolled in a ceramics workshop at the Belmont Arts Center."
Good extractions: "Jordan" (speaker), "Denver" (Location), "Lockheed Martin" (Organization), "Belmont Arts Center" (Location), "ceramics" (Topic)
Do NOT extract: "spouse" (generic reference — extract only if named), "new role" (not an entity), "last month" (temporal), "we" (pronoun)
</EXAMPLE>

<EXAMPLE>
Message: "Nisha: My dad is visiting next week. He loves walking his dogs in Riverside Park."
Good extractions: "Nisha" (speaker), "Nisha's dad" (Person), "Riverside Park" (Location)
Do NOT extract: "dad" (bare relational term — qualify as "Nisha's dad"), "dogs" (bare animal word — no specific identity), "next week" (temporal)
</EXAMPLE>

<EXAMPLE>
Message: "Mary: I forgot Trigger's leash so I couldn't take him on a dog walk. After that I went road cycling in my new wool coat."
Good extractions: "Mary" (speaker), "Trigger" (animal name), "dog leash" (Object), "road cycling" (Topic), "wool coat" (Object)
Do NOT extract: "leash" (too generic — use "dog leash"), "cycling" (too generic — use "road cycling"), "coat" (too generic — use "wool coat"), "dog walk" (activity, not an entity)
</EXAMPLE>

<EXAMPLE>
Message: "Nate: My gaming room has red and purple lighting and I mostly play on a Gamecube. Last week the windshield on my Mustang got cracked."
Good extractions: "Nate" (speaker), "gaming room" (Object), "red and purple lighting" (Object), "Gamecube" (Object), "Mustang" (Object), "cracked windshield" (Object)
Do NOT extract: "lighting" (bare head noun — use "red and purple lighting"), "windshield" (bare head noun — use "cracked windshield"), "week" (temporal)
</EXAMPLE>

<EXAMPLE>
Message: "Alex: I shared a pic from the game after the event."
Good extractions: "Alex" (speaker)
Do NOT extract: "pic" (generic media noun), "game" (generic event noun), "event" (generic event noun)
</EXAMPLE>

<EXAMPLE>
Message: "Jordan: We won by a tight score. Scoring that last basket felt incredible."
Good extractions: "Jordan" (speaker)
Do NOT extract: "basket" (ambiguous bare noun that depends on sentence context)
</EXAMPLE>

${context.customExtractionInstructions}
` },
  ];
}

export function extractJson(context: ExtractNodesContext): Message[] {
  return [
    { role: "system", content: "You are an entity extraction specialist for JSON data. NEVER extract abstract concepts, dates, or generic field values." },
    { role: "user", content: `
NEVER extract:
- Date, time, or timestamp values
- Abstract concepts or generic field values (e.g., "true", "active", "pending")
- Numeric IDs or codes that are not meaningful entity names
- Bare relational or kinship terms (e.g., "spouse", "parent", "pet") — only extract if qualified
  with a possessor name
- Bare generic objects or common nouns (e.g., "supplies", "tickets", "gear") — only extract if
  qualified with a distinguishing detail
- Generic media/content nouns unless uniquely identified in the value itself (photo, pic, picture,
  image, video, post, story)
- Generic event/activity nouns unless uniquely identified in the value itself (event, game, meeting,
  class, workshop, competition)
- Broad institutional nouns unless explicitly named or uniquely qualified (government, school, company,
  team, office)
- Ambiguous bare nouns whose meaning depends on surrounding text rather than the extracted value itself

Extract entities from the JSON and classify each using the ENTITY TYPES above.

<ENTITY TYPES>
${pyRepr(context.entityTypes)}
</ENTITY TYPES>

<SOURCE DESCRIPTION>
${context.sourceDescription}
</SOURCE DESCRIPTION>

<JSON>
${context.episodeContent}
</JSON>

Guidelines:
1. Extract the primary entity the JSON represents (e.g., a "name" or "user" field).
2. Extract named entities referenced in other properties throughout the JSON structure.
3. Only extract entities specific enough to be uniquely identifiable.
4. Be explicit in naming entities — use full names when available.
5. Use the most specific form present in the data (e.g., "road cycling" not "cycling").
6. If a value would not be meaningful and distinguishable when read alone later, do NOT extract it.

${context.customExtractionInstructions}

<EXAMPLE>
JSON: {"user": "Jordan Lee", "company": "Acme Corp", "role": "engineer", "start_date": "2024-01-15", "location": "Denver", "active": true}
Good extractions: "Jordan Lee" (Person), "Acme Corp" (Organization), "Denver" (Location)
Do NOT extract: "engineer" (role, not an entity), "2024-01-15" (date), "true" (field value)
</EXAMPLE>

<EXAMPLE>
JSON: {"author": "Alex", "attachment_type": "photo", "event_name": "event", "agency": "government"}
Good extractions: "Alex" (Person)
Do NOT extract: "photo" (generic media noun), "event" (generic event noun), "government" (broad institutional noun)
</EXAMPLE>
` },
  ];
}

export function extractText(context: ExtractNodesContext): Message[] {
  return [
    { role: "system", content: "You are an entity extraction specialist for unstructured text. NEVER extract abstract concepts, feelings, or generic words." },
    { role: "user", content: `
NEVER extract:
- Pronouns (you, me, he, she, they, it, them, him, her, we, us, this, that, those)
- Abstract concepts (joy, balance, growth, resilience, passion, motivation)
- Generic common nouns or bare object words (day, life, people, work, stuff, things, food, time,
  tickets, supplies, clothes, keys, gear)
- Generic media/content nouns unless uniquely identified in the node name itself (photo, pic, picture,
  image, video, post, story)
- Generic event/activity nouns unless uniquely identified in the node name itself (event, game, meeting,
  class, workshop, competition)
- Broad institutional nouns unless explicitly named or uniquely qualified (government, school, company,
  team, office)
- Ambiguous bare nouns whose meaning depends on sentence context rather than the node name itself
- Sentence fragments or clauses as entity names
- Bare relational or kinship terms (dad, mom, sister, brother, spouse, friend, boss, pet, dog,
  cat) unless qualified with a possessor (e.g., "Nisha's dad" is acceptable, "dad" alone is not)
- Bare generic objects that cannot be meaningfully qualified with a possessor, brand, or
  distinguishing detail (e.g., NEVER extract "supplies" from "I picked up some supplies")

Extract entities from the TEXT that are **explicitly mentioned**.
For each entity, classify it using the ENTITY TYPES above.
Only extract entities specific enough to be uniquely identifiable — ask: "Could this have its own Wikipedia article or database entry?"

<ENTITY TYPES>
${pyRepr(context.entityTypes)}
</ENTITY TYPES>

<TEXT>
${context.episodeContent}
</TEXT>

Guidelines:
1. Extract named entities and specific, concrete things.
2. Do not create nodes for relationships or actions.
3. Do not create nodes for temporal information like dates, times or years.
4. Be explicit in node names, using full names and avoiding abbreviations.
5. Always use the most specific form from the text (e.g., "road cycling" not "cycling",
   "wool coat" not "coat"). Include qualifying context when it's clear from the text.
6. When the text refers to a person's relative, pet, or associate by a bare term, qualify the
   entity with the possessor's name (e.g., "Dr. Osei's colleague" not "colleague").
7. If a phrase would not be meaningful and distinguishable when read alone later, do NOT extract it.
8. When in doubt, do NOT extract.

${context.customExtractionInstructions}

<EXAMPLE>
Text: "Dr. Amara Osei presented her migraine study results at the AAN conference. The study tracked 340 patients using a new CGRP combination protocol."
Good extractions: "Dr. Amara Osei" (Person), "AAN" (Organization), "migraine study" (Topic), "CGRP combination protocol" (Object)
Do NOT extract: "results" (generic noun), "340" (number), "patients" (generic noun), "conference" (generic without a specific name)
</EXAMPLE>

<EXAMPLE>
Text: "Alex shared a pic after the event and said scoring the last basket felt incredible."
Good extractions: "Alex" (Person)
Do NOT extract: "pic" (generic media noun), "event" (generic event noun), "basket" (ambiguous bare noun)
</EXAMPLE>
` },
  ];
}

export function dedupeNodes(context: DedupeNodesContext): Message[] {
  return [
    { role: "system", content: "You are an entity deduplication assistant. NEVER fabricate entity names or mark distinct entities as duplicates." },
    { role: "user", content: `
<PREVIOUS MESSAGES>
${toPromptJson(context.previousEpisodes)}
</PREVIOUS MESSAGES>

<CURRENT MESSAGE>
${context.episodeContent}
</CURRENT MESSAGE>

<ENTITIES>
${toPromptJson(context.extractedNodes)}
</ENTITIES>

<EXISTING ENTITIES>
${toPromptJson(context.existingNodes)}
</EXISTING ENTITIES>

Each of the above ENTITIES was extracted from the CURRENT MESSAGE.
For each entity, determine if it is a duplicate of any EXISTING ENTITY.
Entities should only be considered duplicates if they refer to the *same real-world object or concept*.

NEVER mark entities as duplicates if:
- They are related but distinct.
- They have similar names or purposes but refer to separate instances or concepts.

Task:
ENTITIES contains ${context.extractedNodes.length} entities with IDs 0 through ${context.extractedNodes.length - 1}.
Your response MUST include EXACTLY ${context.extractedNodes.length} resolutions with IDs 0 through ${context.extractedNodes.length - 1}. Do not skip or add IDs.

For every entity, provide:
- \`id\`: integer id from ENTITIES
- \`name\`: the best full name for the entity (preserve the original name unless a duplicate has a more complete name)
- \`duplicate_candidate_id\`: the \`candidate_id\` of the EXISTING ENTITY that is the best duplicate match, or -1 if there is no duplicate

<EXAMPLE>
ENTITY: "Sam" (Person)
EXISTING ENTITIES: [{"candidate_id": 0, "name": "Sam", "entity_types": ["Person"], "summary": "Sam enjoys hiking and photography"}]
Result: duplicate_candidate_id = 0 (same person referenced in conversation)

ENTITY: "NYC"
EXISTING ENTITIES: [{"candidate_id": 0, "name": "New York City", "entity_types": ["Location"]}, {"candidate_id": 1, "name": "New York Knicks", "entity_types": ["Organization"]}]
Result: duplicate_candidate_id = 0 (same location, abbreviated name)

ENTITY: "Java" (programming language)
EXISTING ENTITIES: [{"candidate_id": 0, "name": "Java", "entity_types": ["Location"], "summary": "An island in Indonesia"}]
Result: duplicate_candidate_id = -1 (same name but distinct real-world things)

ENTITY: "Marco's car"
EXISTING ENTITIES: [{"candidate_id": 0, "name": "Marco's vehicle", "entity_types": ["Entity"], "summary": "Marco drives a red sedan."}]
Result: duplicate_candidate_id = 0 (synonym — "car" and "vehicle" refer to the same thing, same possessor)
</EXAMPLE>
` },
  ];
}

export function extractEdges(context: ExtractEdgesContext): Message[] {
  return [
    { role: "system", content: "You are an expert fact extractor that extracts fact triples from text. 1. Extracted fact triples should also be extracted with relevant date information. 2. The CURRENT_MESSAGE may contain multiple episodes, each with its own timestamp. Use each episode's timestamp to resolve temporal references within that episode. REFERENCE_TIME is a fallback for when no per-episode timestamp is available." },
    { role: "user", content: `
<PREVIOUS_MESSAGES>
${toPromptJson(context.previousEpisodes)}
</PREVIOUS_MESSAGES>

<CURRENT_MESSAGE>
${context.episodeContent}
</CURRENT_MESSAGE>

<ENTITIES>
${toPromptJson(context.nodes)}
</ENTITIES>

<REFERENCE_TIME>
${pyDatetime(context.referenceTime)}  # ISO 8601 (UTC); used to resolve relative time mentions
</REFERENCE_TIME>

# TASK
Extract all factual relationships between the given ENTITIES based on the CURRENT MESSAGE.
Only extract facts that:
- involve two DISTINCT ENTITIES from the ENTITIES list,
- are clearly stated or unambiguously implied in the CURRENT MESSAGE,
    and can be represented as edges in a knowledge graph.
- Facts should include entity names rather than pronouns whenever possible.

You may use information from the PREVIOUS MESSAGES only to disambiguate references or support continuity.


${context.customExtractionInstructions}

# EXTRACTION RULES

1. **Entity Name Validation**: \`source_entity_name\` and \`target_entity_name\` must use only the \`name\` values from the ENTITIES list provided above.
   - **CRITICAL**: Using names not in the list will cause the edge to be rejected
2. Each fact must involve two **distinct** entities — \`source_entity_name\` and \`target_entity_name\` NEVER refer to the same entity.
3. Prefer facts that involve two distinct entities from the ENTITIES list. When a sentence describes a specific, concrete detail about a single entity (a brand name, a specific item, a physical description, a quantity, a location, a named activity), do NOT drop it. Instead, look for a second entity in the ENTITIES list that the detail relates to and form a proper triple (e.g., Entity -> OWNS -> item-entity, Entity -> LIVES_IN -> place-entity, Entity -> HAS_ATTRIBUTE -> detail-entity). Only skip the fact when no second entity in the ENTITIES list can anchor the detail.
   - BAD: "Alice feels happy" (vague single-entity state with no concrete detail — what is Alice happy about?)
   - GOOD: "Alice feels happy about Bob's promotion" → Alice -> FEELS_HAPPY_ABOUT -> Bob's promotion
   - GOOD: "Nate plays games on a Gamecube" → Nate -> PLAYS_GAMES_ON -> Gamecube (when "Gamecube" is in ENTITIES)
   - GOOD: "Alice congratulated Bob" (relationship between two entities), "Alice lives in Paris" (relationship between entity and place)
4. Do not emit semantically redundant facts, even across episodes within the CURRENT_MESSAGE. However, if a later episode adds specific details to a previously stated fact (e.g., adding a brand name, a count, a color, a location, or any concrete attribute), extract the more detailed version as a NEW fact — it is NOT a duplicate. Only treat facts as duplicates when they convey the same specificity.
   - NOT a duplicate: "user plays video games" (Episode 0) vs. "user plays games on a Gamecube" (Episode 1) → extract the second, more detailed fact.
   - IS a duplicate: "user plays games on a Gamecube" (Episode 0) vs. "user plays Gamecube games" (Episode 1) → extract once, list both episodes in \`episode_indices\`.
5. The \`fact\` MUST preserve all specific details from the source text: proper nouns, brand names, product names, model numbers, quantities, counts, colors, materials, physical descriptions, specific items, named locations, and named activities. Paraphrase the sentence structure but NEVER generalize:
   - NEVER generalize "Gamecube" to "gaming console", "Ford Mustang" to "car", "wool coat" to "coat", "red and purple lighting" to "lighting", "cracked windshield" to "car damage", or "three screenplays" to "several screenplays".
   - Do not verbatim quote the original text, but every concrete noun, number, and descriptor in the source should survive into the \`fact\`.
6. Use \`REFERENCE_TIME\` to resolve vague or relative temporal expressions (e.g., "last week"). When the CURRENT_MESSAGE contains multiple episodes with per-episode timestamps, prefer the timestamp of the specific episode the fact originates from.
7. Do **not** hallucinate or infer temporal bounds from unrelated events.

# DATETIME RULES

- Use ISO 8601 with "Z" suffix (UTC) (e.g., 2025-04-30T00:00:00Z).
- If the fact is ongoing (present tense), set \`valid_at\` to the timestamp of the episode the fact originates from. If no per-episode timestamp is available, use REFERENCE_TIME.
- If the fact is a future-tense plan, promise, scheduled event, or deadline, set \`valid_at\` to the episode timestamp / REFERENCE_TIME. Do NOT set \`valid_at\` to the future occurrence date.
- Preserve future occurrence dates and deadlines in the \`fact\` text. Do NOT use a soft deadline ("by March 2026", "by the end of Q3 2025") as \`invalid_at\`.
- Use a future date as \`invalid_at\` only when the fact describes an ongoing state that ends then (for example, "is unavailable until December 25, 2024" or "works at Acme until January 15, 2027").
- If a change/termination is expressed, set \`invalid_at\` to the relevant timestamp.
- Leave both fields \`null\` if no explicit or resolvable time is stated.
- If only a date is mentioned (no time), assume 00:00:00.
- If only a year is mentioned, use January 1st at 00:00:00.
- Examples with REFERENCE_TIME 2020-01-01T00:00:00Z:
  - "Emma will relocate to Tokyo on December 25, 2022" -> valid_at 2020-01-01T00:00:00Z, invalid_at null.
  - "Alice will complete the project by July 30, 2024" -> valid_at 2020-01-01T00:00:00Z, invalid_at null.
  - "The team will ship version 2.0 by the end of Q3 2025" -> valid_at 2020-01-01T00:00:00Z, invalid_at null.
        ` },
  ];
}

export function extractTimestamps(context: ExtractTimestampsContext): Message[] {
  return [
    { role: "system", content: "You extract temporal bounds from facts. NEVER hallucinate dates." },
    { role: "user", content: `Given a FACT and its REFERENCE TIME, determine when the fact became true
(valid_at) and when it stopped being true (invalid_at).

Rules:
- Resolve relative expressions ("last week", "2 years ago", "yesterday") using REFERENCE TIME.
- If the fact is ongoing (present tense), set valid_at to REFERENCE TIME.
- If the fact is a future-tense plan, promise, scheduled event, or deadline, set valid_at to REFERENCE TIME. Do NOT set valid_at to the future occurrence date.
- Preserve future occurrence dates and deadlines in the fact text. Do NOT use a soft deadline ("by March 2026", "by the end of Q3 2025") as invalid_at.
- Use a future date as invalid_at only when the fact describes an ongoing state that ends then (for example, "is unavailable until December 25, 2024" or "works at Acme until January 15, 2027").
- If a change or end is expressed, set invalid_at to the relevant time.
- Leave both null if no time is stated or resolvable.
- If only a date is mentioned (no time), assume 00:00:00.
- Use ISO 8601 with Z suffix (e.g., 2025-04-30T00:00:00Z).
- Do NOT hallucinate or infer dates from unrelated events.
- Examples with REFERENCE_TIME 2020-01-01T00:00:00Z:
  - "Emma will relocate to Tokyo on December 25, 2022" -> valid_at 2020-01-01T00:00:00Z, invalid_at null.
  - "Alice will complete the project by July 30, 2024" -> valid_at 2020-01-01T00:00:00Z, invalid_at null.

<FACT>
${context.fact}
</FACT>

<REFERENCE TIME>
${context.referenceTime}
</REFERENCE TIME>
` },
  ];
}

export function resolveEdge(context: ResolveEdgeContext): Message[] {
  return [
    { role: "system", content: "You are a fact deduplication assistant. NEVER mark facts as contradicted unless the new fact explicitly makes the old fact false." },
    { role: "user", content: `
NEVER mark facts as duplicates if they have key differences, particularly around numeric values, dates, or key qualifiers.

Only mark a fact as contradicted when the NEW FACT and the existing fact cannot both be true.
Do NOT treat additional activity, progress, status, or project-update facts as contradictions of role,
ownership, responsibility, membership, or assignment facts. A person doing work on a project is
compatible with that person remaining responsible for, owning, leading, or being the DRI for the project.

IMPORTANT constraints:
- duplicate_facts: ONLY idx values from EXISTING FACTS (NEVER include FACT INVALIDATION CANDIDATES)
- contradicted_facts: idx values from EITHER list (EXISTING FACTS or FACT INVALIDATION CANDIDATES)
- The idx values are continuous across both lists (INVALIDATION CANDIDATES start where EXISTING FACTS end)

<EXISTING FACTS>
${pyRepr(context.existingEdges)}
</EXISTING FACTS>

<FACT INVALIDATION CANDIDATES>
${pyRepr(context.edgeInvalidationCandidates)}
</FACT INVALIDATION CANDIDATES>

<NEW FACT>
${context.newEdge}
</NEW FACT>

You will receive TWO lists of facts with CONTINUOUS idx numbering across both lists.
EXISTING FACTS are indexed first, followed by FACT INVALIDATION CANDIDATES.

1. DUPLICATE DETECTION:
   - If the NEW FACT represents identical factual information as any fact in EXISTING FACTS, return those idx values in duplicate_facts.
   - If no duplicates, return an empty list for duplicate_facts.

2. CONTRADICTION DETECTION:
   - Determine which facts the NEW FACT directly contradicts from either list.
   - A contradiction requires explicit evidence that the old fact stopped being true or was replaced,
     such as clear termination, replacement, transfer, or mutually exclusive current-state language.
   - Do NOT infer contradiction from recency alone, episode dates, meeting dates, weekly snapshots,
     section headings such as "updates" or "status changes", or a new event involving the same entities.
   - Do NOT mark a role/ownership/responsibility/assignment fact as contradicted merely because the
     NEW FACT describes work performed, updates shipped, documents created, progress made, blockers,
     next steps, or collaboration involving the same person and project.
   - A fact from EXISTING FACTS can be both a duplicate AND contradicted only when the same assertion
     is restated with an explicit incompatible change.
   - Return all contradicted idx values in contradicted_facts.
   - If no contradictions, return an empty list for contradicted_facts.

<EXAMPLE>
EXISTING FACT: idx=0, "Alice joined Acme Corp in 2020"
NEW FACT: "Alice joined Acme Corp in 2020"
Result: duplicate_facts=[0], contradicted_facts=[] (identical factual information)

EXISTING FACT: idx=1, "Alice works at Acme Corp as a software engineer"
NEW FACT: "Alice works at Acme Corp as a senior engineer"
Result: duplicate_facts=[], contradicted_facts=[1] (same relationship but updated title — contradiction, NOT a duplicate)

EXISTING FACT: idx=2, "Bob ran 5 miles on Tuesday"
NEW FACT: "Bob ran 3 miles on Wednesday"
Result: duplicate_facts=[], contradicted_facts=[] (different events on different days — neither duplicate nor contradiction)
</EXAMPLE>
` },
  ];
}

function entityTypeDescriptionsSection(descriptions: Record<string, string>): string {
  if (Object.keys(descriptions).length === 0) {
    return "";
  }
  return `
<ENTITY_TYPE_DESCRIPTIONS>
${toPromptJson(descriptions)}
</ENTITY_TYPE_DESCRIPTIONS>
When an entity's type appears in ENTITY_TYPE_DESCRIPTIONS, use the description to decide which facts are most relevant to that entity type. NEVER mention the entity type, type description, or classification in the summary text itself.
`;
}

export function extractSummariesBatch(context: SummariesContext): Message[] {
  return [
    { role: "system", content: "You are a helpful assistant that generates concise entity summaries from provided context." },
    { role: "user", content: `
Given the MESSAGES and a list of ENTITIES, generate an updated summary for each entity that needs one.
Each summary must be under ${maxSummaryChars} characters.

${summaryInstructions}

<MESSAGES>
${toPromptJson(context.previousEpisodes)}
${toPromptJson(context.episodeContent)}
</MESSAGES>
${entityTypeDescriptionsSection(context.entityTypeDescriptions)}
<ENTITIES>
${toPromptJson(context.entities)}
</ENTITIES>

For each entity, combine relevant information from the MESSAGES with any existing summary content.
Only return summaries for entities that have meaningful information to summarize.
If an entity has no relevant information in the messages and no existing summary, you may skip it.
` },
  ];
}

const entityEpisodeSummarySystemPrompt = `You maintain detailed, information-dense entity memories from episode text.

Use ONLY facts explicitly stated in EPISODES and durable facts already present in EXISTING_SUMMARY.
NEVER infer beyond what is directly supported.

Primary goal:
Write a dense factual summary of the entity that preserves as many supported details as possible while staying coherent and durable.

When the input includes entity_type_descriptions, use them to decide which facts are most relevant to the entity type. NEVER mention the entity type, type description, or classification in the summary text itself.

What to capture:
- Stable facts about the entity
- All materially relevant named people, organizations, places, events, documents, objects, and other entities linked to it
- Explicit actions, roles, responsibilities, relationships, and outcomes
- Counts, sequences, and repeated patterns when the evidence supports them
- Temporal details at the highest fidelity available: dates, months, years, ordering, and changes over time
- Current state over superseded state when newer episodes clearly update older information

Rules:
- Be exhaustive within the evidence. Prefer retaining a supported concrete detail over omitting it for brevity.
- NEVER infer preferences, habits, recurrence, frequency, causality, intent, importance, or category from a name, a single mention, or weak evidence.
- Only describe something as recurring, preferred, typical, habitual, or ongoing when multiple episodes explicitly support that claim or one episode states it directly.
- Include all materially relevant named participants that appear in the evidence.
- Include temporal qualifiers whenever they are available.
- Mention counts when they are directly supported and meaningful. Prefer direct factual phrasing over meta phrasing.
- When the durable fact is the content of what was said, state the content directly instead of describing that it was said.
- Use communication verbs only when the act of speaking, asking, sharing, presenting, announcing, or telling is itself the important fact.
- NEVER manufacture pattern language from a single occurrence. A single mention can support a fact, but not a trend, habit, or preference unless the text states that directly.
- If the evidence is insufficient or ambiguous, omit the claim.
- NEVER mention the source material or summarization process.
- NEVER mention episodes, messages, prompts, summaries, memory, graphs, nodes, labels, node types, ontology, schema, or categorization.
- NEVER output phrases like "the summary", "the entity", "categorized as", "tagged as", "suggests", "implies", "appears to", or "recorded interaction".
- NEVER use "the entity" as a pronoun. Use the entity's actual name or a natural pronoun (he, she, it, they).
- NEVER use meta-language verbs like "mentioned", "described", "stated", "noted", "discussed", "referenced", "indicated", or "reported". State the fact directly instead of describing how it was communicated.
- NEVER begin the summary with "A ", "An ", or "This is". If the entity's name starts with "The" (e.g. "The Washington Post"), that is acceptable; otherwise NEVER lead with "The ". Lead with the entity's name or a concrete fact.
- When newer episode text conflicts with older summary content, prefer the newer explicit fact.
- If the new episodes add no durable fact, return the existing summary unchanged.
- The summary should read like a compact brief, not a tagline.
- Write 2-6 dense sentences in third person.
- Return only the summary text.

<EXAMPLES>
Input: {"name": "Jordan Lee", "existing_summary": "Jordan Lee works at Belmont Arts Center.", "episodes": [{"content": "Mina: Jordan Lee presented a ceramics workshop at Belmont Arts Center on March 3, 2025. The workshop had 24 attendees and focused on wheel-thrown bowls.\\nOwen: After the session, Jordan announced a second April workshop for returning students."}, {"content": "Mina: Jordan shared that the new kiln room opened last month and that Jordan now supervises two studio assistants.\\nOwen: Jordan still teaches beginner ceramics on Wednesday evenings."}]}
GOOD: "Jordan Lee works at Belmont Arts Center. Jordan presented a ceramics workshop there on March 3, 2025 for 24 attendees focused on wheel-thrown bowls, and later announced a second April workshop for returning students. Jordan supervises two studio assistants, teaches beginner ceramics on Wednesday evenings, and works out of the new kiln room that opened the previous month."
BAD: "Jordan Lee seems interested in ceramics. Jordan mentioned teaching and was described as busy at the arts center."
</EXAMPLES>`;

export function extractEntitySummariesFromEpisodes(context: SummariesContext): Message[] {
  return [
    { role: "system", content: entityEpisodeSummarySystemPrompt },
    { role: "user", content: `NEVER include meta-language about the summarization process. Use ONLY facts from the provided EPISODES.
Each summary must be under ${maxSummaryChars} characters. Write 2-6 dense sentences in third person. Preserve all material names, roles, dates, counts, and changes over time that are explicitly supported.

For each entity below, generate an updated summary using ONLY the provided EPISODES and any existing summary already on the entity.

<EPISODES>
${toPromptJson(context.previousEpisodes)}
${toPromptJson(context.episodeContent)}
</EPISODES>
${entityTypeDescriptionsSection(context.entityTypeDescriptions)}
<ENTITIES>
${toPromptJson(context.entities)}
</ENTITIES>

Only return summaries for entities that have meaningful information to summarize.
If an entity has no relevant information in the episodes and no existing summary, you may skip it.
` },
  ];
}
