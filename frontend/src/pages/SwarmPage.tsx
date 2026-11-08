import { useEffect, useState } from "react";
import Conversation from "../swarm/Conversation";
import DronePanel from "../swarm/DronePanel";
import FindingsPanel from "../swarm/FindingsPanel";
import MapPanel from "../swarm/MapPanel";
import PanelBoundary from "../swarm/PanelBoundary";
import RoomMap from "../swarm/RoomMap";
import SafetyPanel from "../swarm/SafetyPanel";
import { missionLine, missionTitle } from "../swarm/format";
import { httpApi, resolveApiBase } from "../swarm/http";
import { mockApi } from "../swarm/mock";
import type { Api, Mission } from "../swarm/types";
import { useConsole, type Link } from "../swarm/useConsole";
import "../swarm/swarm.css";

// The operator console. Talks to the Zwarm API on http://localhost:8010, or wherever ?api= says
// (a port number or a full URL). ?mock=1 uses the pretend backend in swarm/mock.ts.

let sharedApi: Api | null = null;

// One client per page load. React's strict mode runs state initializers twice, and the mock
// keeps its pretend flight in the instance.
function getApi(): Api {
  if (sharedApi) return sharedApi;
  const params = new URLSearchParams(window.location.search);
  sharedApi = params.get("mock") ? mockApi() : httpApi(resolveApiBase(params.get("api")));
  return sharedApi;
}

function secondsSince(at: number | null): number | null {
  return at === null ? null : Math.max(0, (Date.now() - at) / 1000);
}

function Banner({ api, link }: { api: Api; link: Link }) {
  const age = secondsSince(link.lastOkAt);
  return (
    <div className="zw-banner" role="alert">
      <strong>No connection to the swarm backend</strong>
      <span>
        {api.label}: {link.error ?? "no answer"}.{" "}
        {age === null ? "No state has ever arrived." : `Everything below is ${Math.round(age)} s old.`}{" "}
        Start it with: cd backend, then uv run python -m zwarm sim. HOLD, LAND and E-STOP keep trying.
      </span>
    </div>
  );
}

function MissionHeader({ mission, down }: { mission: Mission | null; down: boolean }) {
  const active = mission?.kind ? mission : null;
  return (
    <section className="zw-mission" aria-label="Mission">
      <div>
        <h2>{missionTitle(mission)}</h2>
        <p className="zw-mission-sub">
          {active ? (
            <>
              {active.state === "running" ? "Running" : <b>{active.state}</b>} for {clock(active.elapsedS)}
              {down ? ". Not live" : ""}
            </>
          ) : "Say what you want done. Anything that takes off asks you first."}
        </p>
      </div>
      {active && (
        <div className="zw-percent">
          {Math.round(active.shareSeen * 100)}%
          <small>of this area seen</small>
        </div>
      )}
    </section>
  );
}

export default function SwarmPage() {
  const [api] = useState(getApi);
  const { state, link, events, findings, mapStatus, setFindingStatus } = useConsole(api);

  useEffect(() => {
    const before = document.title;
    document.title = "Zwarm console";
    return () => {
      document.title = before;
    };
  }, []);

  const staleS = link.down ? secondsSince(link.lastOkAt) : null;

  return (
    <div className="zw-root">
      {link.down && <Banner api={api} link={link} />}
      <div className="zw-grid">
        {/* Outside every error boundary, and first in tab order. */}
        <SafetyPanel api={api} estopped={state?.estopped ?? false} />

        <header className="zw-brand">
          <h1>Zwarm</h1>
          {state && (state.simulated
            ? <span className="zw-badge zw-badge-sim">SIMULATED</span>
            : <span className="zw-badge zw-badge-real">REAL DRONES</span>)}
        </header>

        <PanelBoundary name="mission" area="mission">
          <MissionHeader mission={state?.mission ?? null} down={link.down} />
        </PanelBoundary>

        <PanelBoundary name="conversation" area="talk">
          <Conversation api={api} events={events} agent={state?.agent ?? null} plan={state?.pendingPlan ?? null} />
        </PanelBoundary>

        <main className="zw-plan-area">
          <div className="zw-map-wrap">
            <PanelBoundary name="map">
              <RoomMap state={state} findings={findings} staleS={staleS} />
            </PanelBoundary>
          </div>
          <PanelBoundary name="findings" area="findings">
            <FindingsPanel api={api} findings={findings} onJudge={setFindingStatus} />
          </PanelBoundary>
          <PanelBoundary name="3D map" area="cloud">
            <MapPanel api={api} status={mapStatus} />
          </PanelBoundary>
        </main>

        <div className="zw-drones">
          <PanelBoundary name="drones">
            <DronePanel
              api={api}
              drones={state?.drones ?? []}
              mission={state?.mission ?? null}
              limitS={state?.flightLimitS ?? 270}
            />
          </PanelBoundary>
        </div>
      </div>
    </div>
  );
}
