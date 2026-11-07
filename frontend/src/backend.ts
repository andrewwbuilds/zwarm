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

// "Drone 2" when this tab points at a fleet port, otherwise the backend address.
export function backendLabel(base: string = API_BASE_URL): string {
  const port = Number(new URL(base).port);
  return FLEET.find((d) => d.port === port && new URL(base).hostname === "localhost")?.name ?? base;
}
