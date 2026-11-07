import { useCallback, useEffect, useState } from "react";

const API_BASE_URL = "http://localhost:8000";

type Vec = [number, number, number];

export interface SimState {
  t: number;
  scene: string;
  firmware: string;
  state: string;
  leds: string;
  position: Vec;
  velocity: Vec;
  attitude: Vec;
  height: number;
  speed: number;
  battery: number;
  voltage: number;
  current: number;
  link: boolean;
  rc_rate: number;
  accepted: number;
  rejected: number;
  flow: boolean;
  rssi: number;
  distance: number;
  video: boolean;
  wind: { speed: number; direction: number; gust: number };
  events: [number, string][];
  scenes: string[];
  firmwares: Record<string, string>;
}

export function useSim() {
  const [ready, setReady] = useState(false);
  const [url, setUrl] = useState<string | null>(null);
  const [state, setState] = useState<SimState | null>(null);

  useEffect(() => {
    fetch(`${API_BASE_URL}/capabilities`)
      .then((r) => (r.ok ? r.json() : null))
      .then((data) => setUrl(data?.sim_url ?? null))
      .catch(() => setUrl(null))
      .finally(() => setReady(true));
  }, []);

  useEffect(() => {
    if (!url) return;
    let stop = false;
    const poll = async () => {
      try {
        const r = await fetch(`${url}/state`);
        if (r.ok && !stop) setState(await r.json());
      } catch {
        if (!stop) setState(null);
      }
    };
    poll();
    const id = setInterval(poll, 250);
    return () => {
      stop = true;
      clearInterval(id);
    };
  }, [url]);

  const post = useCallback(
    async (path: string, body: object) => {
      if (!url) return;
      const r = await fetch(`${url}${path}`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });
      if (r.ok) setState(await r.json());
    },
    [url],
  );

  const reset = useCallback((body: { scene?: string; firmware?: string }) => post("/reset", body), [post]);
  const wind = useCallback((speed: number) => post("/wind", { speed }), [post]);

  return { ready, active: Boolean(url), state, reset, wind };
}
