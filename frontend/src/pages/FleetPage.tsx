import { useCallback, useEffect, useState } from "react";
import { FLEET, apiUrlForPort, type FleetDrone } from "../backend";

// Fleet page: video and status for every drone, plus fleet-wide takeoff, land and E-stop.
// Sticks stay with each drone's console (the "open console" links); this page never
// opens the control websocket.
//
// S2x drones use one toggle bit for takeoff and land and report nothing back, so the
// backend tracks a presumed state and drops a command that would reverse it. If that
// belief is wrong (the drone ignored a takeoff, or crashed), fix it with "mark landed" /
// "mark flying" before using the fleet buttons.

type Status = "checking" | "online" | "offline";

interface DroneState {
  status: Status;
  droneType: string;
  airborne: boolean | null;
}

const INITIAL: DroneState = { status: "checking", droneType: "", airborne: null };

async function post(port: number, path: string, body?: unknown): Promise<{ sent?: boolean; detail?: string }> {
  const r = await fetch(`${apiUrlForPort(port)}${path}`, {
    method: "POST",
    headers: body === undefined ? undefined : { "Content-Type": "application/json" },
    body: body === undefined ? undefined : JSON.stringify(body),
    signal: AbortSignal.timeout(3000),
  });
  const data = await r.json().catch(() => ({}));
  if (!r.ok) throw new Error(data?.detail ?? `HTTP ${r.status}`);
  return data;
}

function useFleetState() {
  const [states, setStates] = useState<Record<number, DroneState>>(
    () => Object.fromEntries(FLEET.map((d) => [d.port, INITIAL])),
  );

  const refresh = useCallback(async () => {
    const entries = await Promise.all(
      FLEET.map(async (d): Promise<[number, DroneState]> => {
        try {
          const r = await fetch(`${apiUrlForPort(d.port)}/capabilities`, { signal: AbortSignal.timeout(1500) });
          if (!r.ok) return [d.port, { ...INITIAL, status: "offline" }];
          const data = await r.json();
          const airborne = typeof data?.airborne === "boolean" ? data.airborne : null;
          return [d.port, { status: "online", droneType: data?.drone_type ?? "", airborne }];
        } catch {
          return [d.port, { ...INITIAL, status: "offline" }];
        }
      }),
    );
    setStates(Object.fromEntries(entries));
  }, []);

  useEffect(() => {
    refresh();
    const id = setInterval(refresh, 1000);
    return () => clearInterval(id);
  }, [refresh]);

  return { states, refresh };
}

function flightLabel(s: DroneState): string {
  if (s.status !== "online") return "";
  if (s.airborne === null) return "flight state not tracked";
  return s.airborne ? "presumed IN AIR" : "presumed on ground";
}

function DroneTile({ drone, state, onAssume }: {
  drone: FleetDrone;
  state: DroneState;
  onAssume: (drone: FleetDrone, airborne: boolean) => void;
}) {
  const base = apiUrlForPort(drone.port);
  // A failed MJPEG <img> never retries, so remount it each time the backend comes back.
  const [videoKey, setVideoKey] = useState(0);
  const [wasOnline, setWasOnline] = useState(false);
  if ((state.status === "online") !== wasOnline) {
    setWasOnline(state.status === "online");
    if (state.status === "online") setVideoKey((k) => k + 1);
  }

  const color = state.status === "online" ? "#3c3" : state.status === "offline" ? "#e44" : "#aaa";

  return (
    <div style={{ display: "flex", flexDirection: "column", minWidth: 0, minHeight: 0, border: "1px solid #333" }}>
      <div style={{ display: "flex", flexWrap: "wrap", alignItems: "center", gap: 8, padding: "6px 10px", background: "#111", color: "#ddd", fontFamily: "monospace", fontSize: 13 }}>
        <span style={{ width: 10, height: 10, borderRadius: 5, background: color }} />
        <strong>{drone.name}</strong>
        <span>:{drone.port}</span>
        <span>{state.status}{state.droneType ? ` · ${state.droneType}` : ""}</span>
        <span style={{ color: state.airborne ? "#fc6" : "#aaa" }}>{flightLabel(state)}</span>
        {state.status === "online" && state.airborne !== null && (
          <button
            title="Correct the presumed state. Sends nothing to the drone."
            onClick={() => onAssume(drone, !state.airborne)}
            style={{ fontSize: 11 }}
          >
            {state.airborne ? "mark landed" : "mark flying"}
          </button>
        )}
        <a
          href={`/?backend=${drone.port}`}
          target={`zwarm-console-${drone.port}`}
          style={{ marginLeft: "auto", color: "#8cf" }}
        >
          open console
        </a>
      </div>
      <div style={{ flex: 1, position: "relative", background: "#000", minHeight: 0 }}>
        {status === "online" ? (
          <img
            key={videoKey}
            src={`${base}/mjpeg`}
            alt={`${drone.name} video`}
            style={{ position: "absolute", inset: 0, width: "100%", height: "100%", objectFit: "contain" }}
          />
        ) : (
          <div style={{ color: "#777", fontFamily: "monospace", padding: 12 }}>
            {status === "offline" ? `No backend on localhost:${drone.port}. Is the tunnel open and the backend started?` : "Checking…"}
          </div>
        )}
      </div>
    </div>
  );
}

export default function FleetPage() {
  useEffect(() => {
    document.title = "Fleet monitor";
  }, []);

  return (
    <div
      style={{
        display: "grid",
        gridTemplateColumns: `repeat(${Math.min(FLEET.length, 3)}, 1fr)`,
        height: "100vh",
        background: "#000",
      }}
    >
      {FLEET.map((d) => (
        <DroneTile key={d.port} drone={d} />
      ))}
    </div>
  );
}
