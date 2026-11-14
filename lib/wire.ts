/** Wire utilities and data conversion types */
import { isRecord, type Json } from './providers.ts';

export type AnyJson = string | number | boolean | null | AnyJson[] | { [key: string]: AnyJson };

export interface ServedModel {
  id: string;
  created?: number;
  owned_by?: string;
}

export function openAIError(message: string, type: string, code: string | null): AnyJson {
  const errPayload: Record<string, AnyJson> = { message, type };
  if (code !== null) errPayload.code = code;
  return { error: errPayload };
}

export function anthropicError(message: string, type: string): AnyJson {
  return { type: 'error', error: { type, message } };
}

export function openAIModelList(models: readonly ServedModel[]): AnyJson {
  return {
    object: 'list',
    data: models.map((m) => ({
      id: m.id,
      object: 'model',
      created: 1767225600,
      owned_by: 'aggregator'
    }))
  };
}

export function anthropicModelList(models: readonly ServedModel[]): AnyJson {
  return {
    data: models.map((m) => ({
      id: m.id,
      type: 'model',
      display_name: m.id,
      created_at: '2026-01-01T00:00:00Z'
    })),
    has_more: false,
    first_id: models[0]?.id ?? null,
    last_id: models[models.length - 1]?.id ?? null
  };
}

export function openAIToAnthropic(body: Json, modelId: string): AnyJson {
  if (!isRecord(body)) return { model: modelId, messages: [] };
  const messages: AnyJson[] = [];
  let system: string | undefined = undefined;

  const rawMsgs = Array.isArray(body.messages) ? body.messages : [];
  for (const m of rawMsgs) {
    if (!isRecord(m)) continue;
    const role = typeof m.role === 'string' ? m.role : 'user';
    const content = typeof m.content === 'string' ? m.content : '';
    if (role === 'system') {
      system = content;
    } else {
      messages.push({ role: role === 'assistant' ? 'assistant' : 'user', content });
    }
  }

  const out: Record<string, AnyJson> = {
    model: modelId,
    messages
  };
  if (system !== undefined) out.system = system;
  if (typeof body.max_tokens === 'number') out.max_tokens = body.max_tokens;
  else if (typeof body.max_completion_tokens === 'number') out.max_tokens = body.max_completion_tokens;
  else out.max_tokens = 4096;

  return out;
}

export function anthropicToOpenAI(body: Json, modelId: string): AnyJson {
  if (!isRecord(body)) return { model: modelId, messages: [] };
  const messages: AnyJson[] = [];
  if (typeof body.system === 'string') {
    messages.push({ role: 'system', content: body.system });
  }

  const rawMsgs = Array.isArray(body.messages) ? body.messages : [];
  for (const m of rawMsgs) {
    if (!isRecord(m)) continue;
    const role = typeof m.role === 'string' ? m.role : 'user';
    const content = typeof m.content === 'string' ? m.content : '';
    messages.push({ role, content });
  }

  const out: Record<string, AnyJson> = {
    model: modelId,
    messages
  };
  if (typeof body.max_tokens === 'number') out.max_tokens = body.max_tokens;
  return out;
}

export async function* anthropicEventsFromOpenAI(
  stream: ReadableStream<Uint8Array>,
  modelId: string
): AsyncGenerator<string, void, void> {
  const reader = stream.getReader();
  const decoder = new TextDecoder();
  yield `event: message_start\ndata: ${JSON.stringify({
    type: 'message_start',
    message: {
      id: 'msg-stream',
      type: 'message',
      role: 'assistant',
      content: [],
      model: modelId,
      stop_reason: null,
      stop_sequence: null,
      usage: { input_tokens: 1, output_tokens: 1 }
    }
  })}\n\n`;

  yield `event: content_block_start\ndata: ${JSON.stringify({
    type: 'content_block_start',
    index: 0,
    content_block: { type: 'text', text: '' }
  })}\n\n`;

  let totalOut = 0;
  try {
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      const text = decoder.decode(value, { stream: true });
      for (const line of text.split('\n')) {
        const trimmed = line.trim();
        if (trimmed.startsWith('data:') && !trimmed.includes('[DONE]')) {
          try {
            const parsed = JSON.parse(trimmed.slice(5).trim());
            if (isRecord(parsed) && Array.isArray(parsed.choices) && parsed.choices.length > 0) {
              const delta = (parsed.choices[0] as Record<string, unknown>).delta;
              if (isRecord(delta) && typeof delta.content === 'string' && delta.content.length > 0) {
                totalOut += Math.max(1, Math.floor(delta.content.length / 4));
                yield `event: content_block_delta\ndata: ${JSON.stringify({
                  type: 'content_block_delta',
                  index: 0,
                  delta: { type: 'text_delta', text: delta.content }
                })}\n\n`;
              }
            }
          } catch {}
        }
      }
    }
  } finally {
    reader.releaseLock();
  }

  yield `event: content_block_stop\ndata: ${JSON.stringify({ type: 'content_block_stop', index: 0 })}\n\n`;
  yield `event: message_delta\ndata: ${JSON.stringify({
    type: 'message_delta',
    delta: { stop_reason: 'end_turn', stop_sequence: null },
    usage: { output_tokens: totalOut }
  })}\n\n`;
  yield `event: message_stop\ndata: ${JSON.stringify({ type: 'message_stop' })}\n\n`;
}
