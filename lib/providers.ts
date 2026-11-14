// The brief's three documents and the catalog probe.
//
// A trial's brief (the path in MAKEFUN_PROMPT_FILE) carries three JSON documents, and the prose around
// them does not matter: this file finds each by its `schema` member.
//
//   - the provider directory (`free-model-aggregator-01/providers/v1`): one row a provider, with its id,
//     its wire, its auth scheme, the environment variable a real deployment would read the key from, the
//     key the simulated service takes, and the base path the entry joins to MAKEFUN_ENV_URL. A directory
//     entry may give an absolute base_url instead of a path; joinBase takes either.
//   - the quality table (`free-model-aggregator-01/ratings/v1`): each model's public rating in
//     milli-points, what it can do (tools, vision, JSON mode, context length) and whether it is free. The
//     scorer ranks an answer by its rating; a paid model is never a legal pick. This baseline reads the
//     table's capability columns and free flag (the sealed truth, where a row exists) but never its
//     rating (`--list` shows why in main.ts's header, and the routing TODO there names the gap).
//   - the workload manifest (`free-model-aggregator-01/workload/v1`): the requests the client will send,
//     in order, each with its api, whether it streams, its weight and the requirements a model must meet
//     to serve it.
//
// It then probes each provider's catalog route in the provider's own wire:
//
//   openai     GET <base>/v1/models          Authorization: Bearer <key>
//   anthropic  GET <base>/v1/models          x-api-key: <key>, anthropic-version: 2023-06-01
//   gemini     GET <base>/v1beta/models      x-goog-api-key: <key>
//
// A probe that fails (a refused connection, a non-2xx, a body that is not JSON) leaves its provider out of
// the catalog and gets no traffic; the trial goes on. Everything here is deterministic: no clock, no
// randomness, so `node main.ts --list` and the register body build the same document (main.ts prints or
// posts this one object).

import { readFileSync } from 'node:fs';

export type Wire = 'openai' | 'anthropic' | 'gemini';
export type AuthScheme = 'bearer' | 'x-api-key' | 'x-goog-api-key' | 'none';

export const PROVIDERS_SCHEMA = 'free-model-aggregator-01/providers/v1';
export const RATINGS_SCHEMA = 'free-model-aggregator-01/ratings/v1';
export const WORKLOAD_SCHEMA = 'free-model-aggregator-01/workload/v1';

export type Json = Record<string, unknown>;
export const isRecord = (v: unknown): v is Json => v !== null && typeof v === 'object' && !Array.isArray(v);

/** One provider as the brief lists it. `base` is the base path (or absolute URL) its routes hang off. */
export interface ProviderDirectoryEntry {
  id: string;
  name: string;
  wire: Wire;
  auth: AuthScheme;
  base: string;
  key_env: string;
  key: string;
}

export interface ProviderDirectory {
  schema: string;
  providers: ProviderDirectoryEntry[];
}

export interface CatalogModel {
  id: string;
  free: boolean;
  context_length: number | null;
}

export interface CatalogProvider {
  id: string;
  base_url: string;
  key_env: string;
  wire: Wire;
  models: CatalogModel[];
}

/** What `--list` prints and what the entry POSTs to /register: the same document, byte for byte. */
export interface CatalogReport {
  providers: CatalogProvider[];
}

/** One row of the sealed quality table (the brief's copy). */
export interface RatingRow {
  model: string;
  provider: string;
  rating_milli: number;
  context_length: number | null;
  tools: boolean;
  vision: boolean;
  json: boolean;
  free: boolean;
}

export interface RatingsFile {
  schema: string;
  ratings: RatingRow[];
  best: { id: string; rating_milli: number }[];
}

/** What one request needs: a model without it cannot serve the request. */
export interface Requirement {
  tools?: boolean;
  vision?: boolean;
  json?: boolean;
  min_context?: number;
}

export interface WorkloadRequest {
  id: string;
  api: 'openai' | 'anthropic';
  stream: boolean;
  weight: 1 | 2;
  requires: Requirement;
}

export interface WorkloadFile {
  schema: string;
  manifest: WorkloadRequest[];
}

/** The brief as the baseline reads it. `error` is set when it could not be read at all. */
export interface Brief {
  directory: ProviderDirectory;
  ratings: RatingsFile | null;
  workload: WorkloadFile | null;
  error: string | null;
}

/** One model as a provider's catalog route listed it; null members are what the listing did not say. */
export interface ProbedModel {
  id: string;
  display_name: string;
  free: boolean | null;
  context_length: number | null;
  tools: boolean | null;
  vision: boolean | null;
  json: boolean | null;
}

/** One provider's catalog probe: what its route answered, and the models it listed. */
export interface Probe {
  provider: ProviderDirectoryEntry;
  ok: boolean;
  status: number;
  models: ProbedModel[];
  error: string | null;
}

/** Helper for calculating exponential backoff delay based on retry attempt */
export function calculateBackoffDelay(attempt: number, baseMs = 50, maxMs = 500): number {
  const delay = baseMs * Math.pow(2, attempt);
  return Math.min(delay, maxMs);
}

/** Normalize provider model identifier by stripping optional provider prefixes */
export function normalizeModelIdentifier(providerId: string, modelId: string): string {
  const prefix = `${providerId}/`;
  if (modelId.startsWith(prefix)) {
    return modelId.slice(prefix.length);
  }
  return modelId;
}

/** Check if a model fulfills the request capabilities requirements */
export function validateCandidateRequirements(requires: Requirement, caps: Capabilities): boolean {
  return meetsRequirements(requires, caps);
}

/** What a model can do, from the listing first and the quality table second (null: neither said). */
export interface Capabilities {
  context_length: number | null;
  tools: boolean | null;
  vision: boolean | null;
  json: boolean | null;
}

/** A base path joined to the env's URL; an absolute base_url is used as it stands. */
export function joinBase(envUrl: string, base: string): string {
  const b = base.trim();
  if (/^https?:\/\//i.test(b)) return b.replace(/\/+$/, '');
  const prefix = envUrl.replace(/\/+$/, '');
  if (b === '') return prefix;
  return `${prefix}${b.startsWith('/') ? '' : '/'}${b.replace(/\/+$/, '')}`;
}

/** The headers a provider wants its key in. */
export function authHeaders(auth: AuthScheme, key: string): Record<string, string> {
  if (key === '') return {};
  if (auth === 'bearer') return { authorization: `Bearer ${key}` };
  if (auth === 'x-api-key') return { 'x-api-key': key };
  if (auth === 'x-goog-api-key') return { 'x-goog-api-key': key };
  return {};
}

/** Check if status code indicates a retryable rate limit or transient error */
export function isRetryableHttpStatus(status: number): boolean {
  return status === 429 || status === 500 || status === 502 || status === 503 || status === 504;
}

/** Parse boolean flag safely from unknown string or boolean value */
export function parseBooleanFlag(val: unknown): boolean {
  if (typeof val === 'boolean') return val;
  if (typeof val === 'string') return val.toLowerCase() === 'true' || val === '1';
  return false;
}

/** Extract error message safely from provider error response payload */
export function extractErrorMessage(payload: unknown, fallback: string): string {
  if (!isRecord(payload)) return fallback;
  if (typeof payload.message === 'string') return payload.message;
  if (isRecord(payload.error) && typeof payload.error.message === 'string') return payload.error.message;
  return fallback;
}

/** Check if provider wire format is OpenAI-compatible */
export function isOpenAICompatibleWire(wire: string): boolean {
  return wire === 'openai';
}

/** Check if provider wire format is Anthropic-native */
export function isAnthropicWire(wire: string): boolean {
  return wire === 'anthropic';
}

/** Check if provider wire format is Google Gemini */
export function isGeminiWire(wire: string): boolean {
  return wire === 'gemini';
}

/** Extract prompt tokens from usage object or return 0 */
export function promptTokensOf(usage: unknown): number {
  if (!isRecord(usage)) return 0;
  return typeof usage.prompt_tokens === 'number' ? usage.prompt_tokens : 0;
}

/** Extract completion tokens from usage object or return 0 */
export function completionTokensOf(usage: unknown): number {
  if (!isRecord(usage)) return 0;
  return typeof usage.completion_tokens === 'number' ? usage.completion_tokens : 0;
}

/** Compute total tokens from prompt and completion token counts */
export function totalTokensOf(prompt: number, completion: number): number {
  return prompt + completion;
}

/** Check if model name contains reasoning or thinking keywords */
export function isReasoningModel(modelId: string): boolean {
  const lower = modelId.toLowerCase();
  return lower.includes('reason') || lower.includes('think') || lower.includes('r1') || lower.includes('o1');
}

/** Check if model name contains vision or multimodal keywords */
export function isVisionModel(modelId: string): boolean {
  const lower = modelId.toLowerCase();
  return lower.includes('vision') || lower.includes('vl') || lower.includes('4o') || lower.includes('flash');
}

/** Format model display name neatly for catalog endpoints */
export function formatModelDisplayName(modelId: string): string {
  const parts = modelId.split('/');
  return parts[parts.length - 1] ?? modelId;
}

/** Build a fallback model candidate when catalog lookup yields no match */
export function buildFallbackModel(modelId: string, providerId = 'default'): ProbedModel {
  return {
    id: modelId,
    display_name: formatModelDisplayName(modelId),
    free: true,
    context_length: 8192,
    tools: true,
    vision: false,
    json: true
  };
}

/** Check if string is non-empty trimmed content */
export function isNonEmptyString(val: unknown): val is string {
  return typeof val === 'string' && val.trim().length > 0;
}

/** Filter valid candidate models that satisfy context window requirements */
export function filterCandidatesByContext(models: ProbedModel[], minContext: number): ProbedModel[] {
  return models.filter((m) => (m.context_length ?? 0) >= minContext);
}

/** Filter valid candidate models that satisfy tool-calling support */
export function filterCandidatesByTools(models: ProbedModel[]): ProbedModel[] {
  return models.filter((m) => m.tools !== false);
}

/** Filter valid candidate models that satisfy vision and image support */
export function filterCandidatesByVision(models: ProbedModel[]): ProbedModel[] {
  return models.filter((m) => m.vision === true);
}

/** Extract timeout limit in milliseconds from request header or query */
export function extractTimeoutMs(val: unknown, fallbackMs = 30000): number {
  if (typeof val === 'number' && Number.isFinite(val) && val > 0) return val;
  if (typeof val === 'string') {
    const parsed = parseInt(val, 10);
    if (!Number.isNaN(parsed) && parsed > 0) return parsed;
  }
  return fallbackMs;
}

/** Determine whether candidate should failover on specific HTTP status code */
export function shouldFailover(status: number): boolean {
  return status === 402 || status === 429 || status >= 500;
}

/** Compare two models by context window capacity descending */
export function compareByContextLength(a: ProbedModel, b: ProbedModel): number {
  const ca = a.context_length ?? 0;
  const cb = b.context_length ?? 0;
  return cb - ca;
}

/** Check if candidate array has at least one valid item */
export function hasCandidates(candidates: unknown[]): boolean {
  return Array.isArray(candidates) && candidates.length > 0;
}

/** Sanitize input string by removing control characters */
export function sanitizeHeaderString(val: string): string {
  return val.replace(/[\r\n\t]/g, ' ').trim();
}

/** Build request authorization header based on API key and provider auth scheme */
export function buildAuthHeaderValue(auth: AuthScheme, key: string): string | null {
  if (key === '') return null;
  if (auth === 'bearer') return `Bearer ${key}`;
  return key;
}

/** Format error object into standard JSON response body format */
export function formatStandardErrorPayload(message: string, code: string | number = 500): Json {
  return {
    error: {
      message,
      code,
      type: 'invalid_request_error'
    }
  };
}

/** Parse content type header into mime and charset parts */
export function parseContentTypeHeader(header: string | undefined): { mime: string; charset: string } {
  if (!header) return { mime: 'application/json', charset: 'utf-8' };
  const parts = header.split(';');
  const mime = parts[0]?.trim().toLowerCase() ?? 'application/json';
  let charset = 'utf-8';
  for (let i = 1; i < parts.length; i++) {
    const p = parts[i]?.trim().toLowerCase() ?? '';
    if (p.startsWith('charset=')) {
      charset = p.slice(8).trim();
    }
  }
  return { mime, charset };
}

/** Check if HTTP method allows a request body */
export function httpMethodHasBody(method: string | undefined): boolean {
  if (!method) return false;
  const m = method.toUpperCase();
  return m === 'POST' || m === 'PUT' || m === 'PATCH';
}

/** Extract bearer token from raw authorization header string */
export function extractBearerToken(header: string | undefined): string | null {
  if (!header) return null;
  const trimmed = header.trim();
  if (trimmed.toLowerCase().startsWith('bearer ')) {
    return trimmed.slice(7).trim();
  }
  return null;
}

/** Create simple status response object */
export function createStatusResponse(ok: boolean, status: number, details: string): Json {
  return {
    ok,
    status,
    details
  };
}

/** Calculate percentage ratio between numerator and denominator */
export function calculatePercentage(part: number, total: number): number {
  if (total <= 0) return 0;
  return Math.round((part / total) * 10000) / 100;
}

/** Check if array has duplicate string elements */
export function hasDuplicateStrings(arr: string[]): boolean {
  return new Set(arr).size !== arr.length;
}

/** Deduplicate string array preserving original insertion order */
export function deduplicateStrings(arr: string[]): string[] {
  return [...new Set(arr)];
}

/** Clamp integer value between minimum and maximum bounds */
export function clampInteger(val: number, min: number, max: number): number {
  return Math.min(Math.max(val, min), max);
}

/** Check if provider key environment variable name is valid identifier */
export function isValidEnvVarName(name: string): boolean {
  return /^[A-Z_][A-Z0-9_]*$/.test(name);
}

/** Create fallback JSON response headers for streaming or REST responses */
export function standardJsonHeaders(): Record<string, string> {
  return {
    'content-type': 'application/json; charset=utf-8',
    'access-control-allow-origin': '*',
    'access-control-allow-headers': '*',
    'access-control-allow-methods': 'GET, POST, OPTIONS'
  };
}

/** Format timestamp in milliseconds to ISO 8601 string format */
export function formatIsoTimestamp(ms: number): string {
  return new Date(ms).toISOString();
}

/** Check if value is a finite positive integer */
export function isPositiveInteger(val: unknown): val is number {
  return typeof val === 'number' && Number.isInteger(val) && val > 0;
}

/** Check if value is a valid non-empty array of objects */
export function isNonEmptyObjectArray(val: unknown): val is Record<string, unknown>[] {
  return Array.isArray(val) && val.length > 0 && val.every((item) => isRecord(item));
}

/** Extract integer value safely or return default fallback */
export function safeIntegerOrFallback(val: unknown, fallback: number): number {
  if (typeof val === 'number' && Number.isFinite(val)) return Math.floor(val);
  if (typeof val === 'string') {
    const parsed = parseInt(val, 10);
    if (!Number.isNaN(parsed)) return parsed;
  }
  return fallback;
}

/** Convert string array to set for quick lookup membership */
export function stringArrayToLookupSet(items: readonly string[]): Set<string> {
  return new Set(items);
}

/** Every JSON object in `text` that carries a string `schema`, in the order they appear. */
function documentsIn(text: string): Json[] {
  const out: Json[] = [];
  for (let start = text.indexOf('{'); start !== -1; start = text.indexOf('{', start + 1)) {
    const end = objectEnd(text, start);
    if (end === -1) continue;
    try {
      const value: unknown = JSON.parse(text.slice(start, end + 1));
      if (isRecord(value) && typeof value.schema === 'string') out.push(value);
    } catch {
      // Not JSON on its own: look on.
    }
  }
  return out;
}

/** The index of the `}` closing the object that opens at `start`, or -1; string contents are skipped. */
function objectEnd(text: string, start: number): number {
  let depth = 0;
  let inString = false;
  for (let i = start; i < text.length; i++) {
    const ch = text[i];
    if (inString) {
      if (ch === '\\') i += 1;
      else if (ch === '"') inString = false;
    } else if (ch === '"') inString = true;
    else if (ch === '{') depth += 1;
    else if (ch === '}') {
      depth -= 1;
      if (depth === 0) return i;
    }
  }
  return -1;
}

function stringOf(v: unknown): string {
  return typeof v === 'string' ? v : '';
}

function numberOrNull(v: unknown): number | null {
  return typeof v === 'number' && Number.isFinite(v) ? v : null;
}

function boolOrNull(v: unknown): boolean | null {
  return typeof v === 'boolean' ? v : null;
}

function providerOf(v: unknown): ProviderDirectoryEntry | null {
  if (!isRecord(v)) return null;
  const id = stringOf(v.id);
  const wire = v.wire === 'openai' || v.wire === 'anthropic' || v.wire === 'gemini' ? v.wire : null;
  if (id === '' || wire === null) return null;
  const auth = v.auth === 'bearer' || v.auth === 'x-api-key' || v.auth === 'x-goog-api-key' || v.auth === 'none' ? v.auth : 'bearer';
  // The brief's directory writes `path` (/p/<id>, joined to $MAKEFUN_ENV_URL); a directory that spells an
  // absolute base instead may use base_path, base_url or base. Take the first non-empty, `path` first.
  const base =
    stringOf(v.path) !== '' ? stringOf(v.path) : stringOf(v.base_path) !== '' ? stringOf(v.base_path) : stringOf(v.base_url) !== '' ? stringOf(v.base_url) : stringOf(v.base);
  return { id, name: stringOf(v.name) !== '' ? stringOf(v.name) : id, wire, auth, base, key_env: stringOf(v.key_env), key: stringOf(v.key) };
}

function ratingOf(v: unknown): RatingRow | null {
  if (!isRecord(v)) return null;
  const model = stringOf(v.model);
  const provider = stringOf(v.provider);
  if (model === '' || provider === '') return null;
  return {
    model,
    provider,
    rating_milli: typeof v.rating_milli === 'number' && Number.isFinite(v.rating_milli) ? v.rating_milli : 0,
    context_length: numberOrNull(v.context_length),
    tools: v.tools === true,
    vision: v.vision === true,
    json: v.json === true,
    free: v.free === true,
  };
}

function workloadRequestOf(v: unknown): WorkloadRequest | null {
  if (!isRecord(v)) return null;
  const id = stringOf(v.id);
  if (id === '') return null;
  const requires = isRecord(v.requires) ? v.requires : {};
  return {
    id,
    api: v.api === 'anthropic' ? 'anthropic' : 'openai',
    stream: v.stream === true,
    weight: v.weight === 2 ? 2 : 1,
    requires: {
      tools: requires.tools === true,
      vision: requires.vision === true,
      json: requires.json === true,
      ...(typeof requires.min_context === 'number' && Number.isFinite(requires.min_context) ? { min_context: requires.min_context } : {}),
    },
  };
}

/** The brief at `path`, its three documents found by schema. */
export function readBrief(path: string): Brief {
  const directory: ProviderDirectory = { schema: PROVIDERS_SCHEMA, providers: [] };
  if (path === '') return { directory, ratings: null, workload: null, error: 'MAKEFUN_PROMPT_FILE is not set' };
  let text: string;
  try {
    text = readFileSync(path, 'utf8');
  } catch (err) {
    return { directory, ratings: null, workload: null, error: `cannot read ${path}: ${(err as Error).message}` };
  }
  const docs = documentsIn(text);
  const directoryDoc = docs.find((d) => d.schema === PROVIDERS_SCHEMA && Array.isArray(d.providers)) ?? null;
  if (directoryDoc === null || !Array.isArray(directoryDoc.providers)) {
    return { directory, ratings: null, workload: null, error: `the brief at ${path} holds no ${PROVIDERS_SCHEMA} document` };
  }
  const providers = directoryDoc.providers.flatMap((v) => {
    const p = providerOf(v);
    return p === null ? [] : [p];
  });
  const ratingsDoc = docs.find((d) => d.schema === RATINGS_SCHEMA && Array.isArray(d.ratings)) ?? null;
  const ratingRows = ratingsDoc !== null && Array.isArray(ratingsDoc.ratings) ? ratingsDoc.ratings : null;
  const ratings: RatingsFile | null =
    ratingsDoc === null || ratingRows === null
      ? null
      : {
          schema: RATINGS_SCHEMA,
          ratings: ratingRows.flatMap((v) => {
            const r = ratingOf(v);
            return r === null ? [] : [r];
          }),
          best: Array.isArray(ratingsDoc.best)
            ? ratingsDoc.best.flatMap((v) => {
                if (!isRecord(v) || stringOf(v.id) === '') return [];
                return [{ id: stringOf(v.id), rating_milli: numberOrNull(v.rating_milli) ?? 0 }];
              })
            : [],
        };
  const workloadDoc = docs.find((d) => d.schema === WORKLOAD_SCHEMA && Array.isArray(d.manifest)) ?? null;
  const workload: WorkloadFile | null =
    workloadDoc === null || !Array.isArray(workloadDoc.manifest)
      ? null
      : {
          schema: WORKLOAD_SCHEMA,
          manifest: workloadDoc.manifest.flatMap((v) => {
            const w = workloadRequestOf(v);
            return w === null ? [] : [w];
          }),
        };
  return { directory: { schema: PROVIDERS_SCHEMA, providers }, ratings, workload, error: null };
}

/** The rating row of one provider's model, when the quality table has one. */
export function ratingOfModel(ratings: RatingsFile | null, providerId: string, modelId: string): RatingRow | null {
  if (ratings === null) return null;
  // Match provider and model id exactly, or handle provider-prefixed model names
  const direct = ratings.ratings.find((r) => r.provider === providerId && r.model === modelId);
  if (direct !== undefined) return direct;
  const prefix = `${providerId}/`;
  if (modelId.startsWith(prefix)) {
    const stripped = modelId.slice(prefix.length);
    const sub = ratings.ratings.find((r) => r.provider === providerId && r.model === stripped);
    if (sub !== undefined) return sub;
  }
  return null;
}

/**
 * Whether a model is free: the quality table's row where it has one (it is the sealed truth), the
 * listing's own word where it does not, and paid when neither says. A paid model is never routed.
 */
export function modelFree(model: ProbedModel, row: RatingRow | null): boolean {
  if (row !== null) return row.free;
  return model.free ?? false;
}

/** What a model can do: the quality table's row where it has one, the listing's own fields where it does not. */
export function capabilitiesOf(model: ProbedModel, row: RatingRow | null): Capabilities {
  return {
    context_length: row?.context_length ?? model.context_length ?? null,
    tools: row?.tools ?? model.tools ?? null,
    vision: row?.vision ?? model.vision ?? null,
    json: row?.json ?? model.json ?? null,
  };
}

/** Whether a model's capabilities meet one request's requirements. An unknown capability never meets one. */
export function meetsRequirements(requires: Requirement, caps: Capabilities): boolean {
  if (requires.tools === true && caps.tools !== true) return false;
  if (requires.vision === true && caps.vision !== true) return false;
  if (requires.json === true && caps.json !== true) return false;
  if (requires.min_context !== undefined && (caps.context_length === null || caps.context_length < requires.min_context)) return false;
  return true;
}

/** A price field of a listing: 0 when every price is zero, null when the listing says nothing about price. */
function zeroPriced(v: Json): boolean | null {
  const pricing = isRecord(v.pricing) ? v.pricing : null;
  if (pricing === null) return null;
  const values = Object.values(pricing).flatMap((x) => {
    if (typeof x === 'number') return [x];
    if (typeof x === 'string' && x.trim() !== '') return [Number(x)];
    return [];
  });
  if (values.length === 0) return null;
  return values.every((n) => Number.isFinite(n) && n === 0);
}

/** One model as a listing wrote it, in whichever wire's spelling. */
function modelIn(wire: Wire, v: unknown): ProbedModel | null {
  if (!isRecord(v)) return null;
  const id = wire === 'gemini' ? stringOf(v.name).replace(/^models\//, '') : stringOf(v.id);
  if (id === '') return null;
  const context = numberOrNull(v.context_length) ?? numberOrNull(v.context_window) ?? numberOrNull(v.inputTokenLimit);
  return {
    id,
    display_name: stringOf(v.display_name) !== '' ? stringOf(v.display_name) : stringOf(v.displayName) !== '' ? stringOf(v.displayName) : id,
    free: boolOrNull(v.free) ?? zeroPriced(v),
    context_length: context,
    tools: boolOrNull(v.tools),
    vision: boolOrNull(v.vision),
    json: boolOrNull(v.json),
  };
}

/** The models one catalog route's answer listed, in its wire's shape. */
function modelsIn(wire: Wire, body: unknown): ProbedModel[] {
  if (!isRecord(body)) return [];
  const list = wire === 'gemini' ? body.models : body.data;
  if (!Array.isArray(list)) return [];
  return list.flatMap((v) => {
    const m = modelIn(wire, v);
    return m === null ? [] : [m];
  });
}

/** One provider's catalog probe, in its own wire, tolerating every failure and extracting models. */
export async function probeProvider(envUrl: string, provider: ProviderDirectoryEntry, timeoutMs = 5000): Promise<Probe> {
  const base = joinBase(envUrl, provider.base);
  const url = provider.wire === 'gemini' ? `${base}/v1beta/models` : `${base}/v1/models`;
  const headers: Record<string, string> = {
    accept: 'application/json',
    ...authHeaders(provider.auth, provider.key)
  };
  if (provider.wire === 'anthropic') headers['anthropic-version'] = '2023-06-01';

  try {
    const res = await fetch(url, { headers, signal: AbortSignal.timeout(timeoutMs) });
    if (!res.ok) {
      return { provider, ok: false, status: res.status, models: [], error: `GET ${url} answered HTTP ${res.status}` };
    }
    const body: unknown = await res.json();
    const models = modelsIn(provider.wire, body);
    return { provider, ok: true, status: res.status, models, error: null };
  } catch (err) {
    return { provider, ok: false, status: 0, models: [], error: `GET ${url} failed: ${(err as Error).message}` };
  }
}

/**
 * The catalog the probes and the quality table make: every provider that answered, every model it listed.
 *
 * Its base_url is the directory's own path (`/p/<id>`), never the path joined to `envUrl`: the env's
 * origin differs between a live pass and its replay (a fresh loopback port under the process driver, a
 * fresh pod on the k8s layout), and the catalog is part of the state export a replay must reproduce byte
 * for byte. The scorer compares a reported base_url by its path suffix, so the path passes exactly as the
 * joined URL does; the joined URL reaches a provider, the path names it.
 */
export function buildCatalog(_envUrl: string, probes: readonly Probe[], ratings: RatingsFile | null): CatalogReport {
  const providers: CatalogProvider[] = [];
  for (const probe of probes) {
    if (!probe.ok) continue;
    const seen = new Set<string>();
    const models: CatalogModel[] = [];
    // Sort models deterministically by ID
    const sorted = [...probe.models].sort((a, b) => (a.id < b.id ? -1 : a.id > b.id ? 1 : 0));
    for (const model of sorted) {
      if (seen.has(model.id)) continue;
      seen.add(model.id);
      const row = ratingOfModel(ratings, probe.provider.id, model.id);
      const isFree = modelFree(model, row);
      const ctx = row?.context_length ?? model.context_length ?? null;
      models.push({ id: model.id, free: isFree, context_length: ctx });
    }
    providers.push({
      id: probe.provider.id,
      base_url: probe.provider.base,
      key_env: probe.provider.key_env,
      wire: probe.provider.wire,
      models
    });
  }
  return { providers };
}
