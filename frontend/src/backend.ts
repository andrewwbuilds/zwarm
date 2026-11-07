// Which backend this tab talks to.
//
// `?backend=8002` (a local port) or `?backend=http://host:port` picks one per tab, so
// several consoles can run side by side, one per drone. Without it, VITE_BACKEND_URL
// applies, then http://localhost:8000.

export interface FleetDrone {
  name: string;
  port: number;
}

// Local tunnel ports opened by tools/pi/fleet-tunnels.sh.
export const FLEET: FleetDrone[] = [
  { name: "Drone 1", port: 8001 },
  { name: "Drone 2", port: 8002 },
  { name: "Drone 3", port: 8003 },
];

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

// "Drone 2" when this tab points at a fleet port, otherwise the backend address.
export function backendLabel(base: string = API_BASE_URL): string {
  const port = Number(new URL(base).port);
  return FLEET.find((d) => d.port === port && new URL(base).hostname === "localhost")?.name ?? base;
}
