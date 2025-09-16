// src/api/client.ts
export const API_BASE = import.meta.env.VITE_API_BASE_URL as string;

export async function postJson<T>(path: string, body: unknown): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  });
  if (!res.ok) {
    const msg = await res.text().catch(() => '');
    throw new Error(`HTTP ${res.status} ${msg}`.trim());
  }
  return res.json() as Promise<T>;
}

/**
 * streamAnalysis: connect to the /verify-stream SSE endpoint,
 * invoke onMessage for each event with parsed data,
 * and onComplete when the stream ends.
 */
export async function streamAnalysis(
  path: string,
  body: unknown,
  onMessage: (data: any) => void,
  onError: (err: Error) => void,
  onComplete?: () => void
): Promise<void> {
  const url = `${API_BASE}${path}`;
  try {
    const res = await fetch(url, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    });
    if (!res.ok || !res.body) {
      const msg = await res.text().catch(() => '');
      throw new Error(`HTTP ${res.status} ${msg}`.trim());
    }
    const reader = res.body.getReader();
    const decoder = new TextDecoder('utf-8');
    let buffer = '';

    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      let parts = buffer.split('\n\n');
      buffer = parts.pop()!; // last incomplete chunk
      for (const part of parts) {
        if (part.startsWith('data:')) {
          const payload = part.replace(/^data:\s*/, '').trim();
          try {
            const parsed = JSON.parse(payload);
            onMessage(parsed);
          } catch {
            // ignore non-JSON or heartbeat
          }
        }
      }
    }
    onComplete?.();
  } catch (err: any) {
    onError(err);
  }
}
