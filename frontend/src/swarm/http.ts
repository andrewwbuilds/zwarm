// The real backend over HTTP.
//
// Browsers allow six connections per host, and queue the rest. If the backend hangs, a queued
// E-STOP would sit behind whatever is stuck. So: every request has a timeout, each poller is
// one request at a time, and pollers share lanes that run one request each. Safety commands
// skip the lanes and go to localhost and 127.0.0.1 at once. Those count as two hosts, so the
// second path is free even if the first one is full.

import {
  parseChat, parseEvents, parseFindings, parseMapStatus, parsePoints, parseState,
} from "./parse";
import type { Api, FindingStatus, SafetyAction } from "./types";

const DEFAULT_API = "http://localhost:8010";

const STATE_TIMEOUT_MS = 1500;
const POLL_TIMEOUT_MS = 2500;
const VIDEO_TIMEOUT_MS = 1500;
const PHOTO_TIMEOUT_MS = 4000;
const POINTS_TIMEOUT_MS = 20000;
const CHAT_TIMEOUT_MS = 60000;
const ACTION_TIMEOUT_MS = 5000;
const SAFETY_TIMEOUT_MS = 1000;

export function resolveApiBase(param: string | null): string {
  if (param && /^\d+$/.test(param)) return `http://localhost:${param}`;
  if (param && /^https?:\/\//.test(param)) return param.replace(/\/+$/, "");
  return DEFAULT_API;
}

// The same backend under its other local name, or null if it isn't local.
export function alternateBase(base: string): string | null {
  const url = new URL(base);
  if (url.hostname === "localhost") url.hostname = "127.0.0.1";
  else if (url.hostname === "127.0.0.1") url.hostname = "localhost";
  else return null;
  return url.origin;
}

class Lane {
  private tail: Promise<unknown> = Promise.resolve();

  run<T>(task: () => Promise<T>): Promise<T> {
    const next = this.tail.then(task, task);
    this.tail = next.catch(() => undefined);
    return next;
  }
}

function describeFailure(error: unknown, path: string, timeoutMs: number): Error {
  if (error instanceof DOMException && (error.name === "TimeoutError" || error.name === "AbortError")) {
    return new Error(`no answer from ${path} within ${(timeoutMs / 1000).toFixed(1)} s`);
  }
  if (error instanceof TypeError) return new Error(`can't connect (${path})`);
  return error instanceof Error ? error : new Error(String(error));
}

async function readDetail(response: Response): Promise<string> {
  try {
    const body: unknown = await response.json();
    if (typeof body === "object" && body !== null && "detail" in body) {
      const { detail } = body;
      if (typeof detail === "string") return detail;
      if (Array.isArray(detail)) return detail.map(String).join(" ");
    }
  } catch {
    // Not JSON. The status code below is all there is to say.
  }
  return `HTTP ${response.status}`;
}

interface Call {
  path: string;
  timeoutMs: number;
  method?: "GET" | "POST";
  body?: unknown;
}

async function send(base: string, { path, timeoutMs, method = "GET", body }: Call): Promise<Response> {
  try {
    const response = await fetch(`${base}${path}`, {
      method,
      headers: body === undefined ? undefined : { "Content-Type": "application/json" },
      body: body === undefined ? undefined : JSON.stringify(body),
      signal: AbortSignal.timeout(timeoutMs),
    });
    if (!response.ok && response.status !== 404) throw new Error(await readDetail(response));
    return response;
  } catch (error) {
    throw describeFailure(error, path, timeoutMs);
  }
}

async function sendOk(base: string, call: Call): Promise<Response> {
  const response = await send(base, call);
  if (!response.ok) throw new Error(`${call.path} isn't on the backend (HTTP ${response.status})`);
  return response;
}

// A POST with no headers and no body needs no CORS preflight, so it's one round trip.
async function postSafety(base: string, action: SafetyAction): Promise<void> {
  const response = await fetch(`${base}/${action}`, {
    method: "POST",
    keepalive: true,
    signal: AbortSignal.timeout(SAFETY_TIMEOUT_MS),
  });
  if (!response.ok) throw new Error(await readDetail(response));
}

// Resolves on the first success, rejects only when every attempt has failed.
function firstSuccess(attempts: Promise<void>[]): Promise<void> {
  return new Promise((resolve, reject) => {
    let failures = 0;
    for (const attempt of attempts) {
      attempt.then(resolve, (error: unknown) => {
        failures += 1;
        if (failures === attempts.length) reject(error);
      });
    }
  });
}

export function httpApi(base: string): Api {
  const background = new Lane();
  const videoLane = new Lane();
  const other = alternateBase(base);

  const getJson = async (call: Call): Promise<unknown> => {
    const response = await sendOk(base, call);
    return response.json();
  };
  const post = (path: string, body?: unknown, timeoutMs = ACTION_TIMEOUT_MS) =>
    sendOk(base, { path, method: "POST", body, timeoutMs });

  return {
    label: base,
    state: async () => parseState(await getJson({ path: "/zwarm/state", timeoutMs: STATE_TIMEOUT_MS })),
    events: (after) =>
      background.run(async () => parseEvents(await getJson({ path: `/zwarm/events?after=${after}`, timeoutMs: POLL_TIMEOUT_MS }))),
    chat: async (text) => {
      const response = await post("/zwarm/chat", { text }, CHAT_TIMEOUT_MS);
      return parseChat(await response.json());
    },
    approve: async (id) => {
      await post(`/zwarm/plan/${encodeURIComponent(id)}/approve`);
    },
    reject: async (id) => {
      await post(`/zwarm/plan/${encodeURIComponent(id)}/reject`);
    },
    findings: () =>
      background.run(async () => parseFindings(await getJson({ path: "/zwarm/findings", timeoutMs: POLL_TIMEOUT_MS }))),
    setFinding: async (id, status: FindingStatus) => {
      await post(`/zwarm/findings/${encodeURIComponent(id)}`, { status });
    },
    photo: (id) =>
      background.run(async () => {
        const path = `/zwarm/findings/${encodeURIComponent(id)}/photo`;
        const response = await sendOk(base, { path, timeoutMs: PHOTO_TIMEOUT_MS });
        return response.blob();
      }),
    video: (drone) =>
      videoLane.run(async () => {
        const path = `/zwarm/video/${encodeURIComponent(drone)}.jpg`;
        const response = await send(base, { path, timeoutMs: VIDEO_TIMEOUT_MS });
        return response.status === 404 ? null : response.blob();
      }),
    mapStatus: () =>
      background.run(async () => parseMapStatus(await getJson({ path: "/zwarm/map", timeoutMs: POLL_TIMEOUT_MS }))),
    buildMap: async () => {
      await post("/zwarm/map/build");
    },
    points: async () => parsePoints(await getJson({ path: "/zwarm/map/points", timeoutMs: POINTS_TIMEOUT_MS })),
    safety: async (action) => {
      const bases = other ? [base, other] : [base];
      try {
        await firstSuccess(bases.map((b) => postSafety(b, action)));
      } catch (error) {
        throw describeFailure(error, `/${action}`, SAFETY_TIMEOUT_MS);
      }
    },
  };
}
