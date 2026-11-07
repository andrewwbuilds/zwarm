import { useEffect, useState } from "react";
import { FLEET, apiUrlForPort, type FleetDrone } from "../backend";

// View-only fleet monitor: video and backend status for every drone. It never opens
// the control websocket, so it cannot send sticks or commands. Each tile links to that
// drone's console, which is where flying happens.

type Status = "checking" | "online" | "offline";

function DroneTile({ drone }: { drone: FleetDrone }) {
  const base = apiUrlForPort(drone.port);
  const [status, setStatus] = useState<Status>("checking");
  const [droneType, setDroneType] = useState<string>("");
  // A failed MJPEG <img> never retries, so remount it each time the backend comes back.
  const [videoKey, setVideoKey] = useState(0);

  useEffect(() => {
    let stop = false;
    let last: Status = "checking";
    const poll = async () => {
      let next: Status = "offline";
      try {
        const r = await fetch(`${base}/capabilities`, { signal: AbortSignal.timeout(1500) });
        if (r.ok) {
          const data = await r.json();
          next = "online";
          if (!stop) setDroneType(data?.drone_type ?? "");
        }
      } catch {
        next = "offline";
      }
      if (stop) return;
      if (next === "online" && last !== "online") setVideoKey((k) => k + 1);
      last = next;
      setStatus(next);
    };
    poll();
    const id = setInterval(poll, 2000);
    return () => {
      stop = true;
      clearInterval(id);
    };
  }, [base]);

  const color = status === "online" ? "#3c3" : status === "offline" ? "#e44" : "#aaa";

  return (
    <div style={{ display: "flex", flexDirection: "column", minWidth: 0, minHeight: 0, border: "1px solid #333" }}>
      <div style={{ display: "flex", alignItems: "center", gap: 8, padding: "6px 10px", background: "#111", color: "#ddd", fontFamily: "monospace", fontSize: 13 }}>
        <span style={{ width: 10, height: 10, borderRadius: 5, background: color }} />
        <strong>{drone.name}</strong>
        <span>:{drone.port}</span>
        <span>{status}{droneType ? ` · ${droneType}` : ""}</span>
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
