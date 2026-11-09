// Which backend this tab talks to.
//
// `?backend=8002` (a local port) or `?backend=http://host:port` picks one per tab, so
// several consoles can run side by side, one per drone. Without it, VITE_BACKEND_URL
// applies, then http://localhost:8000.

export interface FleetDrone {
  name: string;
  port: number;
}

// Local tunnel ports opened by tools/pi/fleet.sh up. `?ports=8011,8012` points the
// fleet page at other backends, such as simulators, without touching the real drones.
const DEFAULT_PORTS = [8001, 8002, 8003];
const portsParam = new URLSearchParams(window.location.search).get("ports");
const fleetPorts = portsParam
  ? portsParam.split(",").map(Number).filter((p) => Number.isInteger(p) && p > 0)
  : DEFAULT_PORTS;
export const FLEET: FleetDrone[] = fleetPorts.map((port, i) => ({ name: `Drone ${i + 1}`, port }));

function resolve(param: string | null): string {
  if (param && /^\d+$/.test(param)) return `http://localhost:${param}`;
  if (param && /^https?:\/\//.test(param)) return param.replace(/\/+$/, "");
  const fromEnv = import.meta.env.VITE_BACKEND_URL as string | undefined;
  return (fromEnv || "http://localhost:8000").replace(/\/+$/, "");
}

export const API_BASE_URL = resolve(new URLSearchParams(window.location.search).get("backend"));
export const WS_BASE_URL = API_BASE_URL.replace(/^http/, "ws");

export function apiUrlForPort(port: number): string {
  return `http://localhost:${port}`;
}

// What the backend believes, or null if it doesn't track it or didn't answer in time.
export async function presumedAirborne(base: string): Promise<boolean | null> {
  try {
    const r = await fetch(`${base}/capabilities`, { signal: AbortSignal.timeout(500) });
    const data = await r.json();
    return typeof data?.airborne === "boolean" ? data.airborne : null;
  } catch {
    return null;
  }
}

// The backend drops a land for a drone it believes is down, and that belief is wrong when
// a land toggle didn't take. Correct it, then land. On the floor this starts the motors,
// because land and takeoff are one toggle, so only a person may ask for it.
export async function forceLand(base: string): Promise<void> {
  const send = async (path: string, init?: RequestInit) => {
    const r = await fetch(`${base}${path}`, { method: "POST", signal: AbortSignal.timeout(3000), ...init });
    if (!r.ok) throw new Error(`${path}: HTTP ${r.status}`);
  };
  await send("/flight/assume", {
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ airborne: true }),
  });
  await send("/flight/land");
}

// The firmware climbs to its own hover height and reports no height back, so a low takeoff
// lets it climb, then holds the throttle down for a fixed time. It still goes high for a few
// seconds, and the 0.9 m is only true if the simulator's guessed rates match the drone.
const LOW_TAKEOFF_PLAN = "climb 0.3\ntakeoff\nmove 0 0 -0.9\n";

async function sendPlan(base: string, path: string, timeoutMs: number): Promise<void> {
  const r = await fetch(`${base}${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ text: LOW_TAKEOFF_PLAN }),
    signal: AbortSignal.timeout(timeoutMs),
  });
  if (!r.ok) {
    const data = await r.json().catch(() => ({}));
    throw new Error(data?.detail ?? `HTTP ${r.status}`);
  }
}

// Takes about 2 s on a Pi and sends nothing to the drone. Splitting it from the start lets
// several drones leave together, and lets a stop pressed meanwhile cancel the takeoff.
export async function prepareLowTakeoff(base: string): Promise<void> {
  if ((await presumedAirborne(base)) === true) throw new Error("already presumed in the air");
  await sendPlan(base, "/trajectory/plan", 15000);
}

export async function startLowTakeoff(base: string): Promise<void> {
  await sendPlan(base, "/trajectory/run", 3000);
}

// "Drone 2" when this tab points at a fleet port, otherwise the backend address.
export function backendLabel(base: string = API_BASE_URL): string {
  const port = Number(new URL(base).port);
  return FLEET.find((d) => d.port === port && new URL(base).hostname === "localhost")?.name ?? base;
}
