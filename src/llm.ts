import type { z } from "zod";
import type { Message, ResponseModel } from "./prompts";

export const graphModel = "@cf/deepseek-ai/deepseek-v4-flash-0731";
export const embeddingModel = "@cf/google/embeddinggemma-300m";
export const rerankModel = "@cf/baai/bge-reranker-base";
export const defaultMaxTokens = 16384;
const temperature = 1;
const maxApplicationRetries = 2;
const transportRetries = 2;
const embeddingDim = 1024;
const embedBatch = 100;
const rerankChars = 2000;
const maxConcurrentCalls = 3;
const retryDelaysMs = [2_000, 5_000, 12_000, 25_000];
const zeroWidth = /[\u200b\u200c\u200d\ufeff\u2060]/g;
const loneSurrogate = /[\ud800-\udbff](?![\udc00-\udfff])|(?<![\ud800-\udbff])[\udc00-\udfff]/g;
const controlCharacters = /[\u0000-\u0008\u000b\u000c\u000e-\u001f]/g;
const transportFailure = /\b5\d\d\b|timed out|timeout|network|connection|internal server/i;
const extractionLanguageInstruction =
  "\n\nAny extracted information should be returned in the same language as it was written in. " +
  "Only output non-English text when the user has written full sentences or phrases in that non-English language. " +
  "Otherwise, output English.";

export class RateLimitedError extends Error {}

export class RefusalError extends Error {}

class TransportError extends Error {}

export function isRateLimit(error: unknown): boolean {
  const message = error instanceof Error ? error.message : String(error);
  return error instanceof RateLimitedError || /\b3021\b|rate limit|too many requests|\b429\b/i.test(message);
}

let activeCalls = 0;
const waitingCalls: (() => void)[] = [];

function delay(ms: number): Promise<void> {
  const { promise, resolve } = Promise.withResolvers<void>();
  setTimeout(resolve, ms);
  return promise;
}

async function withSlot<T>(task: () => Promise<T>): Promise<T> {
  if (activeCalls >= maxConcurrentCalls) {
    const { promise, resolve } = Promise.withResolvers<void>();
    waitingCalls.push(resolve);
    await promise;
  }
  activeCalls += 1;
  try {
    return await task();
  } finally {
    activeCalls -= 1;
    waitingCalls.shift()?.();
  }
}

async function runAi<T>(task: () => Promise<T>): Promise<T> {
  let transportAttempt = 0;
  for (let attempt = 0; ; attempt += 1) {
    try {
      return await withSlot(task);
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      if (isRateLimit(error)) {
        const wait = retryDelaysMs[attempt];
        if (wait === undefined) {
          throw new RateLimitedError(message);
        }
        await delay(wait + Math.random() * 1_000);
        continue;
      }
      if (!transportFailure.test(message)) {
        throw error;
      }
      if (transportAttempt >= transportRetries) {
        throw new TransportError(message);
      }
      await delay(Math.min(500 * 2 ** transportAttempt, 8_000) * (1 - 0.25 * Math.random()));
      transportAttempt += 1;
    }
  }
}

async function completeOnce<Schema extends z.ZodType>(
  ai: Ai,
  messages: readonly Message[],
  model: ResponseModel<Schema>,
  maxTokens: number,
): Promise<z.output<Schema>> {
  const output = await runAi(() =>
    ai.run(graphModel, {
      messages: messages.map((message) => ({ role: message.role, content: message.content })),
      response_format: { type: "json_schema", json_schema: { name: model.name, schema: model.jsonSchema, strict: true } },
      chat_template_kwargs: { enable_thinking: false },
      reasoning_effort: "none",
      temperature,
      max_completion_tokens: maxTokens,
    }),
  );
  const choice = output.choices[0];
  if (choice?.finish_reason === "length") {
    throw new Error(`Output length exceeded max tokens ${defaultMaxTokens}: ${choice.message.content ?? ""}`);
  }
  const content: unknown = choice?.message.content;
  if (content !== null && typeof content === "object") {
    return model.schema.parse(content);
  }
  if (typeof content === "string" && content !== "") {
    return model.schema.parse(JSON.parse(content));
  }
  if (choice?.message.refusal) {
    throw new RefusalError(choice.message.refusal);
  }
  throw new Error(`Invalid response from LLM: ${JSON.stringify(output)}`);
}

export async function generateResponse<Schema extends z.ZodType>(
  ai: Ai,
  prompt: readonly Message[],
  model: ResponseModel<Schema>,
  maxTokens: number = defaultMaxTokens,
): Promise<z.output<Schema>> {
  const messages: Message[] = prompt.map((message, index) =>
    index === 0 ? { ...message, content: message.content + extractionLanguageInstruction } : { ...message },
  );
  for (let retry = 0; ; retry += 1) {
    for (const message of messages) {
      message.content = message.content.replace(loneSurrogate, "").replace(zeroWidth, "").replace(controlCharacters, "");
    }
    try {
      return await completeOnce(ai, messages, model, maxTokens);
    } catch (error) {
      if (
        error instanceof RateLimitedError ||
        error instanceof RefusalError ||
        error instanceof TransportError ||
        retry >= maxApplicationRetries
      ) {
        throw error;
      }
      const name = error instanceof Error ? error.constructor.name : "Error";
      const details = error instanceof Error ? error.message : String(error);
      messages.push({
        role: "user",
        content:
          `The previous response attempt was invalid. Error type: ${name}. Error details: ${details}. ` +
          "Please try again with a valid response, ensuring the output matches the expected format and constraints.",
      });
    }
  }
}

export async function createBatch(ai: Ai, texts: readonly string[]): Promise<number[][]> {
  const vectors: number[][] = [];
  for (let start = 0; start < texts.length; start += embedBatch) {
    const group = texts.slice(start, start + embedBatch);
    const { data } = await runAi(() => ai.run(embeddingModel, { text: [...group] }));
    if (data.length !== group.length) {
      throw new Error(`${embeddingModel} returned ${data.length} vectors for ${group.length} inputs`);
    }
    vectors.push(...data.map((vector) => vector.slice(0, embeddingDim)));
  }
  return vectors;
}

export async function create(ai: Ai, text: string): Promise<number[]> {
  const { data } = await runAi(() => ai.run(embeddingModel, { text: [text] }));
  const [vector] = data;
  if (vector === undefined) {
    throw new Error(`${embeddingModel} returned no vector`);
  }
  return vector.slice(0, embeddingDim);
}

export async function rank(ai: Ai, query: string, passages: readonly string[]): Promise<[string, number][]> {
  if (passages.length === 0) {
    return [];
  }
  const input: Ai_Cf_Baai_Bge_Reranker_Base_Input & { query: string } = {
    query,
    contexts: passages.map((passage) => ({ text: passage.slice(0, rerankChars) })),
    top_k: passages.length,
  };
  const { response } = await runAi(() => ai.run(rerankModel, input));
  const scores = new Map((response ?? []).map((entry) => [entry.id, entry.score ?? 0]));
  return passages
    .map((passage, index): [string, number] => [passage, scores.get(index) ?? 0])
    .sort((left, right) => right[1] - left[1]);
}
