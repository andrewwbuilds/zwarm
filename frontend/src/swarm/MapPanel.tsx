import { useEffect, useState } from "react";
import PointCloud from "./PointCloud";
import type { Api, Cloud, MapStatus } from "./types";

// FleetCapture's states: idle, capturing, captured, building, done, failed.
const BUILDING = "building";
const DONE = "done";

function statusLine(status: MapStatus | null, points: number): string {
  if (!status) return "No build status yet.";
  const photos = `${status.photos} ${status.photos === 1 ? "photo" : "photos"}`;
  const drones = Object.keys(status.byDrone).length;
  const from = drones > 0 ? ` from ${drones} ${drones === 1 ? "drone" : "drones"}` : "";
  const placed = status.registered > 0 ? `, ${status.registered} placed` : "";
  const total = Math.max(status.points, points);
  const shown = total > 0 ? `, ${total.toLocaleString()} points` : "";
  const state = status.state.charAt(0).toUpperCase() + status.state.slice(1);
  const step = status.step ? ` (${status.step})` : "";
  return `${state}${step}: ${photos}${from}${placed}${shown}`;
}

export default function MapPanel({ api, status }: { api: Api; status: MapStatus | null }) {
  const [cloud, setCloud] = useState<Cloud | null>(null);
  const [problem, setProblem] = useState<string | null>(null);
  const [asked, setAsked] = useState(false);
  const [big, setBig] = useState(false);

  const stamp = status ? `${status.name}/${status.state}/${status.registered}` : "";
  const hasResult = status !== null && (status.registered > 0 || status.state === DONE);

  useEffect(() => {
    if (!hasResult) return;
    let live = true;
    api.points().then(
      (next) => {
        if (live && next.count > 0) setCloud(next);
      },
      (error: unknown) => {
        if (live) setProblem(`The points didn't load: ${error instanceof Error ? error.message : String(error)}.`);
      },
    );
    return () => {
      live = false;
    };
  }, [api, stamp, hasResult]);

  const building = asked || status?.state === BUILDING;
  const capturing = status?.state === "capturing";
  const built = status?.state === "done";

  const build = async () => {
    setProblem(null);
    setAsked(true);
    try {
      await api.buildMap();
    } catch (error) {
      setProblem(`The build didn't start: ${error instanceof Error ? error.message : String(error)}.`);
    } finally {
      setAsked(false);
    }
  };

  return (
    <section className={big ? "zw-3d zw-3d-big" : "zw-3d"} aria-label="3D map">
      <header>
        <h2>3D map</h2>
        <span className="zw-3d-actions">
          {cloud && (
            <button type="button" className="zw-btn zw-btn-small" onClick={() => setBig(!big)}>
              {big ? "Smaller" : "Enlarge"}
            </button>
          )}
          <button type="button" className="zw-btn zw-btn-small" disabled={building || capturing || built} onClick={() => void build()}>
            {building ? "Building" : capturing ? "Saving photos" : built ? "Built" : "Build map"}
          </button>
        </span>
      </header>
      <div className="zw-3d-view">
        {cloud
          ? <PointCloud cloud={cloud} />
          : <p className="zw-empty">No points yet. Fly a map mission, then press Build map.</p>}
      </div>
      <p className="zw-3d-status">{statusLine(status, cloud?.count ?? 0)}{cloud ? ". Drag to orbit." : ""}</p>
      {(status?.error || problem) && <p className="zw-fault">{problem ?? status?.error}</p>}
      {status?.warnings[0] && <p className="zw-3d-status">{status.warnings[0]}</p>}
    </section>
  );
}
