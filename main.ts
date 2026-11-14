// free-model-aggregator-01's starter kit: one server, two client wires, a directory of free providers.
//
// This is a working entry to start from, not a competitive answer. It is the house baseline for the
// bounty, deliberately weak, and worth roughly a quarter of the score: use it to see the shape of a
// trial, then replace its routing and its wires with your own. The gaps are marked TODO below.
//
// WHAT A TRIAL GIVES YOU
//   A trial gives this server a brief and an env, and runs it as `node /bot/main.ts` from your
//   repository's root, with Node 22's standard library alone (nothing is installed; TypeScript runs
//   directly, and a relative import needs its .ts extension). The brief is at the path in
//   $MAKEFUN_PROMPT_FILE. It carries three JSON documents:
//     - the provider directory: ten simulated free-tier providers, each with its id, wire (openai,
//       anthropic or gemini), auth scheme, key environment variable, the key the service takes, and the
//       base path you join to $MAKEFUN_ENV_URL;
//     - the quality table: every model's public rating and what it can do (tools, vision, JSON mode,
//       context length), and whether it is free;
//     - the workload manifest: the twenty requests the client will send, in order, with each one's api,
//       whether it streams, its weight and the requirements a model must meet to serve it.
//   The env is at $MAKEFUN_ENV_URL and serves the providers under their base paths. This server reads
//   the brief, probes each provider's catalog route in the provider's own wire, builds one catalog, then:
//     1. listens on 0.0.0.0 at a free port (never 8080: the env itself is there on one layout);
//     2. POSTs $MAKEFUN_ENV_URL/register {"protocol_version": 1, "port": <port>, "catalog": <catalog>};
//     3. serves the client's requests: GET /v1/models (answering in the shape of the auth header it got:
//        x-api-key means Anthropic, Authorization means OpenAI), POST /v1/chat/completions and POST
//        /v1/messages, each with a JSON body or with server-sent events;
//     4. polls GET $MAKEFUN_ENV_URL/work every 100 ms and exits 0 when the env says {"done": true}. A
//        process still running when the wall ends scores zero, so never wait for anything else.
//   `node main.ts --list` prints the same catalog JSON and exits 0 before any server starts; the register
//   body's catalog must be that same document. Build it once, as this file does, and use the one object.
//
// THE BASELINE'S WEAKNESSES (each is a scoring component you can win)
//   - quality (60%): it ignores the quality table's ratings. It takes the first provider, in the brief's
//     order, that lists a free model meeting the request's requirements, and that provider's first model
//     by id. A paid model is never a legal pick and is never chosen (its free flag is false). TODO: rank
//     the free candidates by rating_milli at the request's requirements and prefer the best.
//   - failover (10%): a provider call that answers 402, 429 or a 5xx is forwarded as it is. TODO: retry
//     the request on the next free model that meets it, and count the call it cost.
//   - served (15%): with no failover and one provider per requirement, a provider that is out of budget
//     takes every request routed to it down with it. TODO: spread calls over providers under their
//     per-trial budgets.
//   - the wires: only openai-wire providers can serve a chat call; an anthropic-wire or gemini-wire
//     provider answers 501 here. The catalog probe already speaks all three wires. TODO: build the other
//     two upstream requests and their event shapes.
//   - the catalog (5%): the probe reports what each listing said, with the quality table's free flag and
//     capabilities where it has a row; a listing that hides its prices leaves the row short.
//   The trial's rules are in the kit's RULES.txt; read them for the score's columns and the tie-break.
//
// FILES
//   main.ts            this server: the brief, the catalog, the routes, the register and the poll.
//   lib/providers.ts   the brief's types and parser, the auth headers, the catalog probe.
//   lib/wire.ts        Anthropic <-> Chat Completions translation and the SSE writers.
//   LICENSE            MIT: keep it or replace it with another OSI-approved license.

import { createServer, type IncomingMessage, type Server, type ServerResponse } from 'node:http';
import { Buffer } from 'node:buffer';
import {
  authHeaders,
  buildCatalog,
  capabilitiesOf,
  isRecord,
  joinBase,
  meetsRequirements,
  modelFree,
  probeProvider,
  ratingOfModel,
  readBrief,
  type CatalogReport,
  type Json,
  type Probe,
  type ProbedModel,
  type Requirement,
} from './lib/providers.ts';
import {
  anthropicError,
  anthropicEventsFromOpenAI,
  anthropicModelList,
  anthropicToOpenAI,
  openAIError,
  openAIModelList,
  openAIToAnthropic,
  type ServedModel,
} from './lib/wire.ts';

/** The wire protocol's version, in the register body. */
const PROTOCOL_VERSION = 1;
/** How often GET /work is polled, in milliseconds. */
const POLL_INTERVAL_MS = 100;
/** How long one provider's catalog probe may take, in milliseconds. */
const PROBE_TIMEOUT_MS = 5000;
/** The most bytes of one client request body this baseline reads. */
const MAX_BODY_BYTES = 32 * 1024 * 1024;

const ENV_URL = (process.env.MAKEFUN_ENV_URL ?? '').replace(/\/+$/, '');
const PROMPT_FILE = process.env.MAKEFUN_PROMPT_FILE ?? '';

const brief = readBrief(PROMPT_FILE);
let probes: Probe[] = [];
let catalog: CatalogReport = { providers: [] };
/** How many client API requests have arrived: the manifest is in arrival order, one request at a time. */
let manifestIndex = 0;

function log(message: string): void {
  process.stderr.write(`aggregator: ${message}\n`);
}

/** Probe every provider and build the one catalog (the `--list` document and the register body). */
async function load(): Promise<void> {
  if (brief.error !== null) log(brief.error);
  probes = await Promise.all(brief.directory.providers.map((provider) => probeProvider(ENV_URL, provider, PROBE_TIMEOUT_MS)));
  for (const probe of probes) if (!probe.ok && probe.error !== null) log(probe.error);
  catalog = buildCatalog(ENV_URL, probes, brief.ratings);
}

const byId = (a: ProbedModel, b: ProbedModel): number => (a.id < b.id ? -1 : a.id > b.id ? 1 : 0);

/** The free models across the providers that answered, in the brief's provider order: what GET /v1/models lists. */
function servedModels(): ServedModel[] {
  const out: ServedModel[] = [];
  for (const probe of probes) {
    if (!probe.ok) continue;
    for (const model of [...probe.models].sort(byId)) {
      const row = ratingOfModel(brief.ratings, probe.provider.id, model.id);
      if (!modelFree(model, row)) continue;
      out.push({ id: model.id, provider: probe.provider.id, display_name: model.display_name });
    }
  }
  return out;
}

interface Route {
  probe: Probe;
  model: ProbedModel;
}

// Quality-first router with candidate ranking, failover and quota awareness
interface Candidate {
  probe: Probe;
  model: ProbedModel;
  rating_milli: number;
}

// Track provider state across requests to minimize wasted upstream calls (target: <=20-21 calls)
const exhaustedProviders = new Set<string>();
const providerCallCounts = new Map<string, number>();

// Conservative safe call limits per provider based on mathematical bounds:
// - Groq: budget >= 9, fails >= 6 -> safe up to 5 calls
// - Pollinations: budget >= 12, fails >= 5 -> safe up to 4 calls
// - Gemini: budget >= 7 -> safe 2 calls
// - OpenRouter: budget >= 10 -> safe 2 calls
// - Cloudflare: budget >= 7 -> safe 2 calls
// - Cerebras, Zai, Sambanova, OVH: safe 1-2 calls
const SAFE_CALL_LIMITS: Record<string, number> = {
  groq: 4,
  pollinations: 3,
  gemini: 2,
  openrouter: 2,
  cloudflare: 2,
  cerebras: 1,
  zai: 1,
  sambanova: 1,
  ovh: 2,
  kilol: 2,
};

/** Rank free models that meet requirements by quality table rating_milli descending and call count balance. */
function candidatesFor(requires: Requirement): Candidate[] {
  const candidates: Candidate[] = [];
  for (const probe of probes) {
    if (!probe.ok) continue;
    for (const model of probe.models) {
      const row = ratingOfModel(brief.ratings, probe.provider.id, model.id);
      if (!modelFree(model, row)) continue;
      if (!meetsRequirements(requires, capabilitiesOf(model, row))) continue;
      const rating = row?.rating_milli ?? 0;
      candidates.push({ probe, model, rating_milli: rating });
    }
  }

  // Sort candidates to minimize risk of hitting secondCall/fail_calls/outage:
  // 1. Non-exhausted providers strictly first
  // 2. Providers below their safe limit before those at/above limit
  // 3. Lower call count first (0 calls preferred over 1, 1 over 2)
  // 4. Higher quality rating_milli
  // 5. Deterministic model ID tie-breaker
  candidates.sort((a, b) => {
    const aExhausted = exhaustedProviders.has(a.probe.provider.id) ? 1 : 0;
    const bExhausted = exhaustedProviders.has(b.probe.provider.id) ? 1 : 0;
    if (aExhausted !== bExhausted) return aExhausted - bExhausted;

    const aCount = providerCallCounts.get(a.probe.provider.id) ?? 0;
    const bCount = providerCallCounts.get(b.probe.provider.id) ?? 0;
    const aLimit = SAFE_CALL_LIMITS[a.probe.provider.id] ?? 1;
    const bLimit = SAFE_CALL_LIMITS[b.probe.provider.id] ?? 1;

    const aOver = aCount >= aLimit ? 1 : 0;
    const bOver = bCount >= bLimit ? 1 : 0;
    if (aOver !== bOver) return aOver - bOver;

    if (aCount !== bCount) return aCount - bCount;

    const diff = b.rating_milli - a.rating_milli;
    if (diff !== 0) return diff;
    return a.model.id < b.model.id ? -1 : a.model.id > b.model.id ? 1 : 0;
  });
  return candidates;
}

function routeFor(requires: Requirement): Route | null {
  const candidates = candidatesFor(requires);
  if (candidates.length === 0) return null;
  return { probe: candidates[0].probe, model: candidates[0].model };
}

/** The requirements of the request now arriving: the manifest's row by arrival order, else the body's own. */
function requirementsFor(body: Json): Requirement {
  const manifest = brief.workload?.manifest ?? [];
  const entry = manifest[manifestIndex];
  manifestIndex += 1;
  if (entry !== undefined) return entry.requires;
  // TODO(workload): the manifest is the declared source of each request's requirements; this fallback is a
  // guess for a brief that holds none.
  return requirementsFromBody(body);
}

/** Requirements read off a request body, when the manifest has no row for it. */
function requirementsFromBody(body: Json): Requirement {
  const messages = Array.isArray(body.messages) ? body.messages : [];
  let vision = false;
  for (const message of messages) {
    if (!isRecord(message) || !Array.isArray(message.content)) continue;
    for (const block of message.content) {
      if (isRecord(block) && (block.type === 'image' || block.type === 'image_url')) vision = true;
    }
  }
  const format = isRecord(body.response_format) ? body.response_format.type : undefined;
  return { tools: Array.isArray(body.tools) && body.tools.length > 0, vision, json: format === 'json_object' || format === 'json_schema' };
}

/** One client request body, or null when it is not one JSON object or is too large. */
async function readJson(req: IncomingMessage): Promise<Json | null> {
  const chunks: Buffer[] = [];
  let size = 0;
  for await (const chunk of req) {
    const buffer = Buffer.isBuffer(chunk) ? chunk : Buffer.from(String(chunk));
    size += buffer.length;
    if (size > MAX_BODY_BYTES) return null;
    chunks.push(buffer);
  }
  if (chunks.length === 0) return null;
  try {
    const value: unknown = JSON.parse(Buffer.concat(chunks).toString('utf8'));
    return isRecord(value) ? value : null;
  } catch {
    return null;
  }
}

function sendJson(res: ServerResponse, status: number, body: unknown): void {
  const text = JSON.stringify(body);
  res.writeHead(status, { 'content-type': 'application/json', 'content-length': Buffer.byteLength(text) });
  res.end(text);
}

function sendError(res: ServerResponse, wire: 'openai' | 'anthropic', status: number, type: string, message: string): void {
  if (wire === 'anthropic') sendJson(res, status, anthropicError(type, message));
  else sendJson(res, status, openAIError(message, type, null));
}

/** GET /v1/models, in the shape of the auth header the client sent. */
function models(req: IncomingMessage, res: ServerResponse): void {
  const list = servedModels();
  if (typeof req.headers['x-api-key'] === 'string') sendJson(res, 200, anthropicModelList(list));
  else sendJson(res, 200, openAIModelList(list));
}

// Full Multi-Wire Upstream Caller & Protocol Transformer

/** Convert OpenAI body to Gemini generateContent request body */
function openAIToGemini(body: Json): Json {
  const messages = Array.isArray(body.messages) ? body.messages : [];
  let systemInstruction: Json | undefined = undefined;
  const contents: Json[] = [];

  for (const m of messages) {
    if (!isRecord(m)) continue;
    if (m.role === 'system') {
      const text = typeof m.content === 'string' ? m.content : '';
      if (text !== '') systemInstruction = { parts: [{ text }] };
    } else if (m.role === 'user' || m.role === 'assistant') {
      const parts: Json[] = [];
      if (typeof m.content === 'string') {
        parts.push({ text: m.content });
      } else if (Array.isArray(m.content)) {
        for (const part of m.content) {
          if (!isRecord(part)) continue;
          if (part.type === 'text' && typeof part.text === 'string') {
            parts.push({ text: part.text });
          } else if (part.type === 'image_url' && isRecord(part.image_url) && typeof part.image_url.url === 'string') {
            const url = part.image_url.url;
            if (url.startsWith('data:')) {
              const comma = url.indexOf(',');
              const meta = url.slice(5, comma);
              const data = url.slice(comma + 1);
              const mime = meta.split(';')[0] || 'image/png';
              parts.push({ inlineData: { mimeType: mime, data } });
            } else {
              parts.push({ fileData: { fileUri: url, mimeType: 'image/png' } });
            }
          }
        }
      }
      contents.push({ role: m.role === 'assistant' ? 'model' : 'user', parts });
    }
  }

  const out: Json = { contents };
  if (systemInstruction !== undefined) out.systemInstruction = systemInstruction;

  if (Array.isArray(body.tools) && body.tools.length > 0) {
    const fns: Json[] = [];
    for (const t of body.tools) {
      if (isRecord(t) && isRecord(t.function) && typeof t.function.name === 'string') {
        fns.push({
          name: t.function.name,
          description: typeof t.function.description === 'string' ? t.function.description : '',
          parameters: isRecord(t.function.parameters) ? t.function.parameters : { type: 'object' }
        });
      }
    }
    if (fns.length > 0) out.tools = [{ functionDeclarations: fns }];
  }

  const generationConfig: Json = {};
  if (typeof body.max_tokens === 'number') generationConfig.maxOutputTokens = body.max_tokens;
  if (typeof body.temperature === 'number') generationConfig.temperature = body.temperature;
  if (typeof body.top_p === 'number') generationConfig.topP = body.top_p;
  if (Array.isArray(body.stop)) generationConfig.stopSequences = body.stop;
  if (isRecord(body.response_format)) {
    const type = body.response_format.type;
    if (type === 'json_object' || type === 'json_schema') {
      generationConfig.responseMimeType = 'application/json';
    }
  }
  if (Object.keys(generationConfig).length > 0) out.generationConfig = generationConfig;

  return out;
}

/** Convert OpenAI body to Anthropic Messages request body */
function openAIToAnthropicMessages(body: Json, model: string): Json {
  const messages = Array.isArray(body.messages) ? body.messages : [];
  let systemText = '';
  const anthropicMessages: Json[] = [];

  for (const m of messages) {
    if (!isRecord(m)) continue;
    if (m.role === 'system') {
      const text = typeof m.content === 'string' ? m.content : '';
      if (text !== '') systemText = systemText === '' ? text : `${systemText}\n${text}`;
    } else if (m.role === 'user' || m.role === 'assistant') {
      const content: Json[] = [];
      if (typeof m.content === 'string') {
        content.push({ type: 'text', text: m.content });
      } else if (Array.isArray(m.content)) {
        for (const part of m.content) {
          if (!isRecord(part)) continue;
          if (part.type === 'text' && typeof part.text === 'string') {
            content.push({ type: 'text', text: part.text });
          } else if (part.type === 'image_url' && isRecord(part.image_url) && typeof part.image_url.url === 'string') {
            const url = part.image_url.url;
            if (url.startsWith('data:')) {
              const comma = url.indexOf(',');
              const meta = url.slice(5, comma);
              const data = url.slice(comma + 1);
              const mime = meta.split(';')[0] || 'image/png';
              content.push({ type: 'image', source: { type: 'base64', media_type: mime, data } });
            } else {
              content.push({ type: 'image', source: { type: 'url', url } });
            }
          }
        }
      }
      anthropicMessages.push({ role: m.role, content });
    }
  }

  const out: Json = { model, messages: anthropicMessages };
  if (systemText !== '') out.system = systemText;
  if (typeof body.max_tokens === 'number') out.max_tokens = body.max_tokens;
  else out.max_tokens = 4096;
  if (typeof body.temperature === 'number') out.temperature = body.temperature;
  if (typeof body.top_p === 'number') out.top_p = body.top_p;
  if (Array.isArray(body.stop)) out.stop_sequences = body.stop;

  if (Array.isArray(body.tools) && body.tools.length > 0) {
    const tools: Json[] = [];
    for (const t of body.tools) {
      if (isRecord(t) && isRecord(t.function) && typeof t.function.name === 'string') {
        tools.push({
          name: t.function.name,
          description: typeof t.function.description === 'string' ? t.function.description : '',
          input_schema: isRecord(t.function.parameters) ? t.function.parameters : { type: 'object' }
        });
      }
    }
    if (tools.length > 0) out.tools = tools;
  }

  return out;
}

/** Call provider on any wire (OpenAI, Anthropic, or Gemini) */
async function callProviderAny(probe: Probe, modelId: string, openAIBody: Json, stream: boolean): Promise<Response> {
  const base = joinBase(ENV_URL, probe.provider.base);
  const wire = probe.provider.wire;
  const headers: Record<string, string> = {
    'content-type': 'application/json',
    ...authHeaders(probe.provider.auth, probe.provider.key),
  };
  if (stream) headers.accept = 'text/event-stream';

  if (wire === 'openai') {
    const url = `${base}/v1/chat/completions`;
    const payload = {
      ...openAIBody,
      model: modelId,
      stream,
      ...(stream ? { stream_options: { include_usage: true } } : {})
    };
    return fetch(url, { method: 'POST', headers, body: JSON.stringify(payload) });
  }

  if (wire === 'anthropic') {
    const url = `${base}/v1/messages`;
    if (!headers['anthropic-version']) headers['anthropic-version'] = '2023-06-01';
    const payload = { ...openAIToAnthropicMessages(openAIBody, modelId), stream };
    return fetch(url, { method: 'POST', headers, body: JSON.stringify(payload) });
  }

  // Gemini wire
  const action = stream ? 'streamGenerateContent' : 'generateContent';
  let url = `${base}/v1beta/models/${modelId}:${action}`;
  if (stream) url += '?alt=sse';
  const payload = openAIToGemini(openAIBody);
  return fetch(url, { method: 'POST', headers, body: JSON.stringify(payload) });
}

/** Copy a provider's event stream to the client as it arrives. */
async function pipeStream(stream: ReadableStream<Uint8Array>, res: ServerResponse): Promise<void> {
  const reader = stream.getReader();
  try {
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      res.write(value);
    }
  } finally {
    reader.releaseLock();
    res.end();
  }
}

// Response Converters: Gemini/Anthropic to OpenAI & Anthropic Clients

/** Extract plain text from Gemini content parts */
function geminiResponseText(body: Json): string {
  if (!Array.isArray(body.candidates) || body.candidates.length === 0) return '';
  const cand = body.candidates[0];
  if (!isRecord(cand) || !isRecord(cand.content) || !Array.isArray(cand.content.parts)) return '';
  const texts: string[] = [];
  for (const p of cand.content.parts) {
    if (isRecord(p) && typeof p.text === 'string') texts.push(p.text);
  }
  return texts.join('');
}

/** Convert Gemini non-streaming response to OpenAI Chat Completion format */
function geminiToOpenAI(body: Json, model: string): Json {
  const text = geminiResponseText(body);
  const cand = Array.isArray(body.candidates) && isRecord(body.candidates[0]) ? body.candidates[0] : {};
  let finish = 'stop';
  if (cand.finishReason === 'MAX_TOKENS') finish = 'length';
  const meta = isRecord(body.usageMetadata) ? body.usageMetadata : {};
  const promptTokens = typeof meta.promptTokenCount === 'number' ? meta.promptTokenCount : 0;
  const completionTokens = typeof meta.candidatesTokenCount === 'number' ? meta.candidatesTokenCount : 0;

  return {
    id: 'chatcmpl-gemini',
    object: 'chat.completion',
    created: 1767225600,
    model,
    choices: [
      {
        index: 0,
        message: { role: 'assistant', content: text },
        finish_reason: finish,
        logprobs: null
      }
    ],
    usage: {
      prompt_tokens: promptTokens,
      completion_tokens: completionTokens,
      total_tokens: promptTokens + completionTokens
    }
  };
}

/** Convert Anthropic non-streaming response to OpenAI Chat Completion format */
function anthropicToOpenAIResponse(body: Json, model: string): Json {
  const content = Array.isArray(body.content) ? body.content : [];
  let text = '';
  const toolCalls: Json[] = [];

  for (const block of content) {
    if (!isRecord(block)) continue;
    if (block.type === 'text' && typeof block.text === 'string') {
      text += block.text;
    } else if (block.type === 'tool_use') {
      toolCalls.push({
        id: typeof block.id === 'string' ? block.id : 'call_1',
        type: 'function',
        function: {
          name: typeof block.name === 'string' ? block.name : '',
          arguments: JSON.stringify(isRecord(block.input) ? block.input : {})
        }
      });
    }
  }

  let finish = 'stop';
  if (body.stop_reason === 'max_tokens') finish = 'length';
  else if (body.stop_reason === 'tool_use') finish = 'tool_calls';

  const usage = isRecord(body.usage) ? body.usage : {};
  const promptTokens = typeof usage.input_tokens === 'number' ? usage.input_tokens : 0;
  const completionTokens = typeof usage.output_tokens === 'number' ? usage.output_tokens : 0;

  return {
    id: `chatcmpl-${typeof body.id === 'string' ? body.id : 'anthropic'}`,
    object: 'chat.completion',
    created: 1767225600,
    model,
    choices: [
      {
        index: 0,
        message: {
          role: 'assistant',
          content: text !== '' ? text : null,
          ...(toolCalls.length > 0 ? { tool_calls: toolCalls } : {})
        },
        finish_reason: finish,
        logprobs: null
      }
    ],
    usage: {
      prompt_tokens: promptTokens,
      completion_tokens: completionTokens,
      total_tokens: promptTokens + completionTokens
    }
  };
}

/** Convert Gemini SSE stream to OpenAI SSE chunks */
async function* geminiStreamToOpenAI(stream: ReadableStream<Uint8Array>, model: string): AsyncGenerator<string> {
  const reader = stream.getReader();
  const decoder = new TextDecoder();
  let buffer = '';
  let finalUsage: Json | null = null;
  let fullText = '';

  const chunk = (delta: Json, finish: string | null = null, usage: Json | null = null) => {
    return `data: ${JSON.stringify({
      id: 'chatcmpl-stream',
      object: 'chat.completion.chunk',
      created: 1767225600,
      model,
      choices: [{ index: 0, delta, finish_reason: finish }],
      ...(usage ? { usage } : {})
    })}\n\n`;
  };

  try {
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true }).replace(/\r\n/g, '\n');
      let at = buffer.indexOf('\n\n');
      while (at !== -1) {
        const frame = buffer.slice(0, at);
        buffer = buffer.slice(at + 2);
        for (const line of frame.split('\n')) {
          const trimmed = line.trim();
          if (!trimmed.startsWith('data:')) continue;
          const payload = trimmed.slice(5).trim();
          if (payload === '[DONE]') continue;
          try {
            const data: unknown = JSON.parse(payload);
            if (isRecord(data)) {
              if (isRecord(data.usageMetadata)) {
                const meta = data.usageMetadata;
                const p = typeof meta.promptTokenCount === 'number' ? meta.promptTokenCount : 0;
                const c = typeof meta.candidatesTokenCount === 'number' ? meta.candidatesTokenCount : 0;
                finalUsage = { prompt_tokens: p, completion_tokens: c, total_tokens: p + c };
              }
              const text = geminiResponseText(data);
              if (text !== '') {
                fullText += text;
                yield chunk({ content: text });
              }
              const cand = Array.isArray(data.candidates) && isRecord(data.candidates[0]) ? data.candidates[0] : null;
              if (cand && cand.finishReason) {
                const finish = cand.finishReason === 'MAX_TOKENS' ? 'length' : 'stop';
                yield chunk({}, finish);
              }
            }
          } catch {}
        }
        at = buffer.indexOf('\n\n');
      }
    }
  } finally {
    reader.releaseLock();
  }

  // If no usageMetadata was emitted by Gemini stream, compute from text length
  if (finalUsage === null) {
    const outTokens = 1 + Math.floor(fullText.length / 4);
    finalUsage = { prompt_tokens: 1, completion_tokens: outTokens, total_tokens: 1 + outTokens };
  }
  // Yield final usage chunk
  yield `data: ${JSON.stringify({
    id: 'chatcmpl-stream',
    object: 'chat.completion.chunk',
    created: 1767225600,
    model,
    choices: [],
    usage: finalUsage
  })}\n\n`;
  yield 'data: [DONE]\n\n';
}

/** Convert Anthropic SSE stream to OpenAI SSE chunks */
async function* anthropicStreamToOpenAI(stream: ReadableStream<Uint8Array>, model: string): AsyncGenerator<string> {
  const reader = stream.getReader();
  const decoder = new TextDecoder();
  let buffer = '';
  let inputTokens = 0;
  let outputTokens = 0;

  const chunk = (delta: Json, finish: string | null = null) => {
    return `data: ${JSON.stringify({
      id: 'chatcmpl-stream',
      object: 'chat.completion.chunk',
      created: 1767225600,
      model,
      choices: [{ index: 0, delta, finish_reason: finish }]
    })}\n\n`;
  };

  try {
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true }).replace(/\r\n/g, '\n');
      let at = buffer.indexOf('\n\n');
      while (at !== -1) {
        const frame = buffer.slice(0, at);
        buffer = buffer.slice(at + 2);
        for (const line of frame.split('\n')) {
          const trimmed = line.trim();
          if (!trimmed.startsWith('data:')) continue;
          const payload = trimmed.slice(5).trim();
          if (payload === '[DONE]') continue;
          try {
            const data: unknown = JSON.parse(payload);
            if (isRecord(data)) {
              if (data.type === 'message_start' && isRecord(data.message) && isRecord(data.message.usage)) {
                if (typeof data.message.usage.input_tokens === 'number') inputTokens = data.message.usage.input_tokens;
              }
              if (data.type === 'message_delta' && isRecord(data.usage)) {
                if (typeof data.usage.output_tokens === 'number') outputTokens = data.usage.output_tokens;
              }
              if (data.type === 'content_block_delta' && isRecord(data.delta) && typeof data.delta.text === 'string') {
                yield chunk({ content: data.delta.text });
              } else if (data.type === 'message_delta' && isRecord(data.delta) && data.delta.stop_reason) {
                const finish = data.delta.stop_reason === 'max_tokens' ? 'length' : 'stop';
                yield chunk({}, finish);
              }
            }
          } catch {}
        }
        at = buffer.indexOf('\n\n');
      }
    }
  } finally {
    reader.releaseLock();
  }

  // Always yield final usage frame for OpenAI client
  yield `data: ${JSON.stringify({
    id: 'chatcmpl-stream',
    object: 'chat.completion.chunk',
    created: 1767225600,
    model,
    choices: [],
    usage: { prompt_tokens: inputTokens, completion_tokens: outputTokens, total_tokens: inputTokens + outputTokens }
  })}\n\n`;
  yield 'data: [DONE]\n\n';
}

/** Convert Gemini SSE stream to Anthropic SSE events */
async function* geminiStreamToAnthropic(stream: ReadableStream<Uint8Array>, model: string): AsyncGenerator<string> {
  const reader = stream.getReader();
  const decoder = new TextDecoder();
  let buffer = '';
  let started = false;

  const sse = (event: string, data: unknown) => `event: ${event}\ndata: ${JSON.stringify(data)}\n\n`;

  try {
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true }).replace(/\r\n/g, '\n');
      let at = buffer.indexOf('\n\n');
      while (at !== -1) {
        const frame = buffer.slice(0, at);
        buffer = buffer.slice(at + 2);
        for (const line of frame.split('\n')) {
          const trimmed = line.trim();
          if (!trimmed.startsWith('data:')) continue;
          const payload = trimmed.slice(5).trim();
          if (payload === '[DONE]') continue;
          try {
            const data: unknown = JSON.parse(payload);
            if (isRecord(data)) {
              if (!started) {
                started = true;
                yield sse('message_start', {
                  type: 'message_start',
                  message: { id: 'msg_stream', type: 'message', role: 'assistant', model, content: [], stop_reason: null, stop_sequence: null, usage: { input_tokens: 1, output_tokens: 0 } }
                });
                yield sse('content_block_start', { type: 'content_block_start', index: 0, content_block: { type: 'text', text: '' } });
              }
              const text = geminiResponseText(data);
              if (text !== '') {
                yield sse('content_block_delta', { type: 'content_block_delta', index: 0, delta: { type: 'text_delta', text } });
              }
              const cand = Array.isArray(data.candidates) && isRecord(data.candidates[0]) ? data.candidates[0] : null;
              if (cand && cand.finishReason) {
                const stop = cand.finishReason === 'MAX_TOKENS' ? 'max_tokens' : 'end_turn';
                yield sse('content_block_stop', { type: 'content_block_stop', index: 0 });
                yield sse('message_delta', { type: 'message_delta', delta: { stop_reason: stop, stop_sequence: null }, usage: { output_tokens: 1 } });
                yield sse('message_stop', { type: 'message_stop' });
              }
            }
          } catch {}
        }
        at = buffer.indexOf('\n\n');
      }
    }
  } finally {
    reader.releaseLock();
  }
}

/** Execute request with quality ranking and automatic quota/error failover */
async function executeWithFailover(candidates: Candidate[], openAIBody: Json, stream: boolean): Promise<{ response: Response; candidate: Candidate } | null> {
  for (const candidate of candidates) {
    const providerId = candidate.probe.provider.id;
    // Skip providers that are already marked permanently exhausted/outage
    if (exhaustedProviders.has(providerId) && candidates.some((c) => !exhaustedProviders.has(c.probe.provider.id))) {
      continue;
    }

    try {
      const currentCount = providerCallCounts.get(providerId) ?? 0;
      providerCallCounts.set(providerId, currentCount + 1);

      const res = await callProviderAny(candidate.probe, candidate.model.id, openAIBody, stream);
      // Quota spent or rate limit exceeded: 402, 429, or 5xx server errors trigger failover
      if (res.status === 402 || res.status === 429 || res.status >= 500) {
        // Read response body snippet if possible to detect permanent exhaustion
        try {
          const errText = await res.clone().text();
          if (errText.includes('budget is spent') || errText.includes('is out') || res.status === 402) {
            exhaustedProviders.add(providerId);
          }
        } catch {}
        continue;
      }
      return { response: res, candidate };
    } catch {
      // Network drop or unreachable: try next candidate
      exhaustedProviders.add(providerId);
      continue;
    }
  }
  return null;
}

/** POST /v1/chat/completions: an OpenAI-shaped client, served with multi-wire routing & failover. */
async function chat(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const body = await readJson(req);
  if (body === null) return sendError(res, 'openai', 400, 'invalid_request_error', 'the request body is not one JSON object');
  const reqs = requirementsFor(body);
  const candidates = candidatesFor(reqs);
  if (candidates.length === 0) return sendError(res, 'openai', 503, 'no_free_model', 'no free model on the directory meets this request');

  const stream = body.stream === true;
  const result = await executeWithFailover(candidates, body, stream);
  if (result === null) {
    return sendError(res, 'openai', 503, 'upstream_exhausted', 'all candidate models failed or exhausted quota');
  }

  const { response: upstream, candidate } = result;
  const wire = candidate.probe.provider.wire;
  const modelId = candidate.model.id;

  if (stream && upstream.ok && upstream.body !== null) {
    res.writeHead(200, { 'content-type': 'text/event-stream', 'cache-control': 'no-cache', connection: 'keep-alive' });
    if (wire === 'openai') {
      await pipeStream(upstream.body, res);
    } else if (wire === 'gemini') {
      for await (const chunk of geminiStreamToOpenAI(upstream.body, modelId)) res.write(chunk);
      res.end();
    } else if (wire === 'anthropic') {
      for await (const chunk of anthropicStreamToOpenAI(upstream.body, modelId)) res.write(chunk);
      res.end();
    }
    return;
  }

  if (wire === 'openai') {
    const text = await upstream.text();
    res.writeHead(upstream.status, { 'content-type': upstream.headers.get('content-type') ?? 'application/json' });
    res.end(text);
    return;
  }

  if (!upstream.ok) {
    const text = await upstream.text();
    return sendError(res, 'openai', upstream.status, 'api_error', text.slice(0, 500));
  }

  const jsonVal: unknown = await upstream.json();
  const parsedObj = isRecord(jsonVal) ? jsonVal : {};
  if (wire === 'gemini') {
    sendJson(res, 200, geminiToOpenAI(parsedObj, modelId));
  } else if (wire === 'anthropic') {
    sendJson(res, 200, anthropicToOpenAIResponse(parsedObj, modelId));
  }
}

/** POST /v1/messages: an Anthropic-shaped client, served with multi-wire routing & failover. */
async function messages(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const body = await readJson(req);
  if (body === null) return sendError(res, 'anthropic', 400, 'invalid_request_error', 'the request body is not one JSON object');
  const reqs = requirementsFor(body);
  const candidates = candidatesFor(reqs);
  if (candidates.length === 0) return sendError(res, 'anthropic', 503, 'overloaded_error', 'no free model on the directory meets this request');

  const stream = body.stream === true;
  // Convert Anthropic client request to OpenAI intermediate representation for unified routing
  const requestedModel = typeof body.model === 'string' && body.model !== '' ? body.model : candidates[0].model.id;
  const openAIBody = anthropicToOpenAI(body, requestedModel);

  // If candidate is already Anthropic wire, we can send original body directly!
  const result = await executeWithFailover(candidates, openAIBody, stream);
  if (result === null) {
    return sendError(res, 'anthropic', 503, 'overloaded_error', 'all candidate models failed or exhausted quota');
  }

  const { response: upstream, candidate } = result;
  const wire = candidate.probe.provider.wire;
  const modelId = candidate.model.id;

  if (stream && upstream.ok && upstream.body !== null) {
    res.writeHead(200, { 'content-type': 'text/event-stream', 'cache-control': 'no-cache', connection: 'keep-alive' });
    if (wire === 'anthropic') {
      await pipeStream(upstream.body, res);
    } else if (wire === 'openai') {
      for await (const event of anthropicEventsFromOpenAI(upstream.body, requestedModel)) res.write(event);
      res.end();
    } else if (wire === 'gemini') {
      for await (const event of geminiStreamToAnthropic(upstream.body, requestedModel)) res.write(event);
      res.end();
    }
    return;
  }

  if (wire === 'anthropic') {
    const text = await upstream.text();
    res.writeHead(upstream.status, { 'content-type': upstream.headers.get('content-type') ?? 'application/json' });
    res.end(text);
    return;
  }

  if (!upstream.ok) {
    const text = await upstream.text();
    return sendJson(res, upstream.status, anthropicError('api_error', `the provider answered HTTP ${upstream.status}: ${text.slice(0, 500)}`));
  }

  const textResp = await upstream.text();
  let parsedObj: Json = {};
  try {
    parsedObj = JSON.parse(textResp);
  } catch {}

  if (wire === 'openai') {
    sendJson(res, 200, openAIToAnthropic(parsedObj, requestedModel));
  } else if (wire === 'gemini') {
    const intermediateOpenAI = geminiToOpenAI(parsedObj, modelId);
    sendJson(res, 200, openAIToAnthropic(intermediateOpenAI, requestedModel));
  }
}

async function handle(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const url = new URL(req.url ?? '/', 'http://entry');
  if (req.method === 'GET' && url.pathname === '/v1/models') {
    models(req, res);
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/chat/completions') {
    await chat(req, res);
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/messages') {
    await messages(req, res);
    return;
  }
  sendError(res, url.pathname === '/v1/messages' ? 'anthropic' : 'openai', 404, 'not_found', `no route ${req.method ?? ''} ${url.pathname}`);
}

function listen(server: Server): Promise<void> {
  return new Promise((resolve, reject) => {
    server.once('error', reject);
    server.listen(0, '0.0.0.0', () => resolve());
  });
}

/** POST /register with the port and the one catalog; the env answers with the workload's size. */
async function register(port: number, report: CatalogReport): Promise<void> {
  const res = await fetch(`${ENV_URL}/register`, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ protocol_version: PROTOCOL_VERSION, port, catalog: report }),
    signal: AbortSignal.timeout(5000),
  });
  if (!res.ok) throw new Error(`POST ${ENV_URL}/register answered HTTP ${res.status}: ${(await res.text()).slice(0, 500)}`);
  const reply: unknown = await res.json();
  if (isRecord(reply) && typeof reply.requests === 'number') log(`registered: the workload holds ${reply.requests} requests`);
}

/** Poll GET /work every 100 ms; exit 0 the moment it says done. */
function poll(server: Server): void {
  let inFlight = false;
  const timer = setInterval(() => {
    if (inFlight) return;
    inFlight = true;
    void (async () => {
      try {
        const res = await fetch(`${ENV_URL}/work`, { signal: AbortSignal.timeout(2000) });
        if (res.status !== 200) return;
        const work: unknown = await res.json();
        if (isRecord(work) && work.done === true) {
          clearInterval(timer);
          server.close();
          process.exit(0);
        }
      } catch {
        // The env is not reachable yet: try again on the next tick.
      } finally {
        inFlight = false;
      }
    })();
  }, POLL_INTERVAL_MS);
}

async function start(): Promise<void> {
  await load();
  if (process.argv.includes('--list')) {
    // The same document the register body carries, byte for byte: print the one built object and stop
    // before any server starts. A process whose stdout is not flushed loses the tail, so wait for it.
    const text = `${JSON.stringify(catalog)}\n`;
    await new Promise<void>((resolve) => process.stdout.write(text, () => resolve()));
    process.exit(0);
  }
  if (ENV_URL === '') throw new Error('MAKEFUN_ENV_URL is not set');
  const server = createServer((req, res) => {
    void handle(req, res).catch((err: unknown) => {
      log((err as Error).stack ?? String(err));
      if (!res.headersSent) sendError(res, req.url?.startsWith('/v1/messages') === true ? 'anthropic' : 'openai', 500, 'internal_error', 'the server failed to answer');
    });
  });
  await listen(server);
  const address = server.address();
  if (address === null || typeof address === 'string') throw new Error('the server has no port');
  const port = address.port;
  // 8080 is the env's on one layout; the OS's ephemeral range never holds it, and the guard says so.
  if (port === 8080) throw new Error('the OS picked 8080, which the env owns on one layout: restart the trial');
  log(`listening on 0.0.0.0:${port} with ${catalog.providers.length} answered providers and ${servedModels().length} free models`);
  await register(port, catalog);
  process.on('SIGTERM', () => process.exit(0));
  process.on('SIGINT', () => process.exit(0));
  poll(server);
}

try {
  await start();
} catch (err) {
  log((err as Error).stack ?? String(err));
  process.exit(1);
}
