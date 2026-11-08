import { useEffect, useRef, useState } from "react";
import { clock, droneColor, phaseLabel, trackingQuality, trackingText } from "./format";
import { VideoFeeds } from "./video";
import type { Api, Drone, Mission } from "./types";

declare module "react" {
  interface CSSProperties {
    "--zw-drone"?: string;
  }
}

const STALLED_S = 2;

function VideoTile({ feeds, id }: { feeds: VideoFeeds; id: string }) {
  const [note, setNote] = useState<string | null>("No video yet");
  const attach = useRef((canvas: HTMLCanvasElement | null) => feeds.attach(id, canvas));

  useEffect(() => {
    const timer = window.setInterval(() => {
      const age = feeds.age(id);
      if (age === null) setNote("No video yet");
      else setNote(age > STALLED_S ? `Video stalled ${Math.round(age)} s` : null);
    }, 500);
    return () => window.clearInterval(timer);
  }, [feeds, id]);

  return (
    <div className="zw-video">
      <canvas ref={attach.current} aria-label={`Live video from ${id}`} />
      {note && <span>{note}</span>}
    </div>
  );
}

function DroneRow({ drone, index, feeds, mission, limitS }: { drone: Drone; index: number; feeds: VideoFeeds; mission: Mission | null; limitS: number }) {
  const color = droneColor(index);
  const quality = trackingQuality(drone);
  const share = Math.min(1, drone.flightS / limitS);
  const timer = drone.flightS >= limitS ? "over" : share >= 0.8 ? "low" : "ok";
  const task = mission?.tasks[drone.id];
  const onGround = drone.phase === "ground";

  return (
    <article className="zw-drone" style={{ "--zw-drone": color }}>
      <VideoTile feeds={feeds} id={drone.id} />
      <div className="zw-drone-info">
        <header>
          <h3>{drone.id}</h3>
          <span>{phaseLabel(drone.phase)}</span>
        </header>
        <p className="zw-task">{task || (onGround ? "Waiting" : "No task")}</p>
        <div className={`zw-flight zw-flight-${timer}`} title="Flight time against the battery limit">
          <strong>{clock(drone.flightS)}</strong>
          <span>of {clock(limitS)}</span>
          <div className="zw-flight-bar" role="progressbar" aria-valuemin={0} aria-valuemax={limitS} aria-valuenow={Math.round(drone.flightS)} aria-label="Flight time">
            <i style={{ width: `${share * 100}%` }} />
          </div>
        </div>
      </div>
      <div className="zw-drone-foot">
        <p className={`zw-track zw-track-${quality}`}>
          <b>Tracking {quality}</b> {trackingText(drone)}
        </p>
        {!drone.connected && <p className="zw-fault">Link lost</p>}
        {drone.faults.map((fault) => <p className="zw-fault" key={fault}>{fault}</p>)}
      </div>
    </article>
  );
}

interface Props {
  api: Api;
  drones: Drone[];
  mission: Mission | null;
  limitS: number;
}

export default function DronePanel({ api, drones, mission, limitS }: Props) {
  const [feeds] = useState(() => new VideoFeeds(api));
  const ids = drones.map((d) => d.id).join("\n");

  useEffect(() => {
    feeds.setDrones(ids ? ids.split("\n") : []);
  }, [feeds, ids]);

  useEffect(() => {
    feeds.start();
    return () => feeds.stop();
  }, [feeds]);

  if (drones.length === 0) return <p className="zw-empty">No drones reported yet.</p>;
  return (
    <>
      {drones.map((drone, i) => (
        <DroneRow key={drone.id} drone={drone} index={i} feeds={feeds} mission={mission} limitS={limitS} />
      ))}
    </>
  );
}
