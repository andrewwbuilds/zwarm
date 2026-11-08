// The console's live data. Each poller waits for its answer before scheduling the next one,
// so a slow backend means slower polling and never a pile of requests.

import { useCallback, useEffect, useRef, useState } from "react";
import type { Api, Finding, FindingStatus, LogEvent, MapStatus, ZwarmState } from "./types";

const STATE_PERIOD_MS = 200;
const EVENTS_PERIOD_MS = 400;
const FINDINGS_PERIOD_MS = 1500;
const MAP_PERIOD_MS = 2000;
const DOWN_AFTER_MS = 1500;
const EVENTS_KEPT = 400;

export function poll(task: () => Promise<void>, periodMs: number): () => void {
  let stopped = false;
  let timer: number | undefined;
  const step = async () => {
    const started = performance.now();
    try {
      await task();
    } catch {
      // The task reports its own failure. The loop carries on.
    }
    if (!stopped) timer = window.setTimeout(step, Math.max(0, periodMs - (performance.now() - started)));
  };
  void step();
  return () => {
    stopped = true;
    window.clearTimeout(timer);
  };
}

export interface Link {
  down: boolean;
  lastOkAt: number | null;
  error: string | null;
}

export interface ConsoleData {
  state: ZwarmState | null;
  link: Link;
  events: LogEvent[];
  findings: Finding[];
  mapStatus: MapStatus | null;
  setFindingStatus: (id: string, status: FindingStatus) => Promise<void>;
}

const messageOf = (error: unknown) => (error instanceof Error ? error.message : String(error));

export function useConsole(api: Api): ConsoleData {
  const [state, setState] = useState<ZwarmState | null>(null);
  const [link, setLink] = useState<Link>({ down: false, lastOkAt: null, error: null });
  const [events, setEvents] = useState<LogEvent[]>([]);
  const [findings, setFindings] = useState<Finding[]>([]);
  const [mapStatus, setMapStatus] = useState<MapStatus | null>(null);
  const lastEvent = useRef(0);
  // Findings the operator just judged, so a poll already on its way can't flip them back.
  const judged = useRef(new Map<string, FindingStatus>());

  useEffect(() => {
    const opened = performance.now();
    let lastOk: number | null = null;
    const stopState = poll(async () => {
      try {
        const next = await api.state();
        lastOk = performance.now();
        setState(next);
        setLink({ down: false, lastOkAt: Date.now(), error: null });
      } catch (error) {
        const silentMs = performance.now() - (lastOk ?? opened);
        setLink((old) => ({ down: old.down || silentMs >= DOWN_AFTER_MS, lastOkAt: old.lastOkAt, error: messageOf(error) }));
      }
    }, STATE_PERIOD_MS);

    const stopEvents = poll(async () => {
      const fresh = await api.events(lastEvent.current);
      if (fresh.length === 0) return;
      lastEvent.current = Math.max(lastEvent.current, ...fresh.map((e) => e.n));
      setEvents((old) => {
        const seen = new Set(old.map((e) => e.n));
        const added = fresh.filter((e) => !seen.has(e.n));
        return added.length ? [...old, ...added].slice(-EVENTS_KEPT) : old;
      });
    }, EVENTS_PERIOD_MS);

    const stopFindings = poll(async () => {
      const next = await api.findings();
      setFindings(next.map((f) => {
        const mine = judged.current.get(f.id);
        if (mine === f.status) judged.current.delete(f.id);
        return mine && mine !== f.status ? { ...f, status: mine } : f;
      }));
    }, FINDINGS_PERIOD_MS);

    const stopMap = poll(async () => {
      setMapStatus(await api.mapStatus());
    }, MAP_PERIOD_MS);

    return () => {
      stopState();
      stopEvents();
      stopFindings();
      stopMap();
    };
  }, [api]);

  const setFindingStatus = useCallback(async (id: string, status: FindingStatus) => {
    judged.current.set(id, status);
    setFindings((old) => old.map((f) => (f.id === id ? { ...f, status } : f)));
    try {
      await api.setFinding(id, status);
    } catch (error) {
      judged.current.delete(id);
      throw error;
    }
  }, [api]);

  return { state, link, events, lastEvent: lastEvent.current, findings, mapStatus, setFindingStatus };
}
