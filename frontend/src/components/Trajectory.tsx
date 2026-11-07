import { useEffect, useState } from "react";
import type { SimState } from "../hooks/useSim";
import { block, drop, num } from "../ui";
import { Calibration } from "./Calibration";
import { PROGRAMS } from "../programs";
import { API_BASE_URL } from "../backend";

const API = API_BASE_URL;
const KEY = "turbodrone.plan";
const EXAMPLE = `# x forward, y left, z up (m); headings deg, + is left
takeoff
speed 1
goto 2 0
turn 90
goto 2 2 1.6
yaw 0
goto 0 0 1.2
land
`;

type Point = number[]; // t, x, y, z, heading

interface Segment {
  t: number;
  dur: number;
  sticks: [number, number, number, number];
  action: string | null;
  text: string;
}

interface Planned {
  duration: number;
  segments: Segment[];
  path: Point[];
  warnings: string[];
  prediction: { track: Point[]; spread: Point[][]; events: [number, string][]; state: string } | null;
}

interface Status {
  running: boolean;
  t: number;
  duration: number;
  step: string | null;
  reason: string | null;
  origin: [number, number, number] | null;
  folder: string | null;
}

async function call(path: string, body?: object) {
  const r = await fetch(`${API}${path}`, body ? {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  } : undefined);
  const data = await r.json();
  if (!r.ok) throw new Error(data?.detail ?? r.statusText);
  return data;
}

function load(): string {
  try {
    return localStorage.getItem(KEY) ?? EXAMPLE;
  } catch {
    return EXAMPLE;
  }
}

function table(p: Planned): string {
  const rows = p.segments.map((s) => {
    const what = s.action ?? (s.sticks.some((v) => v) ? s.sticks.map((v) => num(v, 2, 6)).join("") : "");
    return `${s.t.toFixed(1).padStart(5)}${s.dur.toFixed(1).padStart(5)} ${what.padEnd(24)} ${s.text}`;
  });
  const lines = [`duration ${p.duration.toFixed(1)} s`, "    t  dur   thr   yaw   pit   rol", ...rows];
  const pr = p.prediction;
  if (pr) {
    const end = pr.track[pr.track.length - 1];
    lines.push(`end      x${num(end[1])} y${num(end[2])} z${num(end[3])} m  h${num(end[4], 0, 5)}  ${pr.state}`);
    lines.push(...pr.events.map(([t, m]) => `${t.toFixed(2).padStart(7)}  ${m}`));
  }
  lines.push(...p.warnings.map((w) => `warning: ${w}`));
  return lines.join("\n");
}

function Plot({ plan, flown }: { plan: Planned | null; flown: Point[] }) {
  const lines: { pts: Point[]; dash?: string; width?: number; opacity?: number }[] = [];
  if (plan?.prediction) {
    for (const t of plan.prediction.spread) lines.push({ pts: t, dash: "2 3", opacity: 0.35 });
    lines.push({ pts: plan.prediction.track, dash: "5 3" });
  }
  if (plan) lines.push({ pts: plan.path, width: 1.5 });
  if (flown.length) lines.push({ pts: flown, dash: "1 3", width: 2.5 });
  const all = lines.flatMap((l) => l.pts).concat([[0, 0, 0]]);
  const us = all.map((p) => -p[2]);
  const vs = all.map((p) => -p[1]);
  const u0 = Math.min(...us) - 0.5, u1 = Math.max(...us) + 0.5;
  const v0 = Math.min(...vs) - 0.5, v1 = Math.max(...vs) + 0.5;
  const span = Math.max(u1 - u0, v1 - v0, 2);
  const grid = [];
  for (let g = Math.ceil(u0); g <= u0 + span; g++) grid.push(<line key={`u${g}`} x1={g} x2={g} y1={v0} y2={v0 + span} vectorEffect="non-scaling-stroke" />);
  for (let g = Math.ceil(v0); g <= v0 + span; g++) grid.push(<line key={`v${g}`} y1={g} y2={g} x1={u0} x2={u0 + span} vectorEffect="non-scaling-stroke" />);
  return (
    <svg width={340} height={340} viewBox={`${u0} ${v0} ${span} ${span}`} style={{ display: "block" }}>
      <g stroke="currentColor" strokeOpacity={0.15} strokeWidth={1}>{grid}</g>
      <path d="M -0.08 0.08 L 0 -0.15 L 0.08 0.08" fill="none" stroke="currentColor" strokeWidth={1.5} vectorEffect="non-scaling-stroke" />
      {lines.map((l, i) => (
        <polyline key={i} fill="none" stroke="currentColor" strokeDasharray={l.dash} strokeWidth={l.width ?? 1}
          strokeOpacity={l.opacity ?? 1} vectorEffect="non-scaling-stroke"
          points={l.pts.map((p) => `${-p[2]},${-p[1]}`).join(" ")} />
      ))}
    </svg>
  );
}

export function Trajectory({ sim }: { sim: SimState | null }) {
  const [text, setText] = useState(load);
  const [plan, setPlan] = useState<Planned | null>(null);
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);
  const [status, setStatus] = useState<Status | null>(null);
  const [flown, setFlown] = useState<Point[]>([]);

  useEffect(() => {
    const id = setInterval(() => call("/trajectory").then(setStatus).catch(() => setStatus(null)), 250);
    return () => clearInterval(id);
  }, []);

  useEffect(() => {
    if (!status?.running || !status.origin || !sim) return;
    const [ox, oy, oyaw] = status.origin;
    const a = (-oyaw * Math.PI) / 180;
    const dx = sim.position[0] - ox, dy = sim.position[1] - oy;
    const point = [status.t, dx * Math.cos(a) - dy * Math.sin(a), dx * Math.sin(a) + dy * Math.cos(a), sim.height];
    setFlown((f) => [...f, point]);
  }, [sim, status?.running, status?.origin, status?.t]);

  const edit = (value: string) => {
    setText(value);
    setPlan(null);
    try {
      localStorage.setItem(KEY, value);
    } catch {
      /* storage can be unavailable */
    }
  };

  const act = (path: string, body?: object, then?: (data: never) => void) => async (e: { currentTarget: HTMLElement }) => {
    drop(e);
    setBusy(true);
    setError("");
    try {
      const data = await call(path, body ?? {});
      then?.(data as never);
    } catch (err) {
      setError(String((err as Error).message ?? err));
    } finally {
      setBusy(false);
    }
  };

  const run = status?.running
    ? `run     ${status.t.toFixed(1)} / ${status.duration.toFixed(1)} s  ${status.step ?? ""}`
    : status?.reason ? `run     ${status.reason}\nlog     runs/${status.folder?.split("/").pop() ?? ""}` : "run     idle";

  return (
    <>
      <p>
        trajectory{" "}
        <select value="" onKeyDown={block} onChange={(e) => { edit(e.target.value === "example" ? EXAMPLE : PROGRAMS[e.target.value]); drop(e); }}>
          <option value="">load...</option>
          <option>example</option>
          {Object.keys(PROGRAMS).map((name) => <option key={name}>{name}</option>)}
        </select>
      </p>
      <textarea rows={11} cols={42} spellCheck={false} value={text} onChange={(e) => edit(e.target.value)} />
      <p>
        <button disabled={busy} onClick={act("/trajectory/plan", { text }, (p: Planned) => setPlan(p))}>plan</button>{" "}
        <button disabled={busy || !plan || status?.running} onClick={act("/trajectory/run", { text }, () => setFlown([]))}>run</button>{" "}
        <button disabled={!status?.running} onClick={act("/trajectory/stop")}>stop</button>
      </p>
      <pre>{[error && `error   ${error}`, run, "plot    plan solid, predicted dashed,\n        flown dotted, grid 1 m"].filter(Boolean).join("\n")}</pre>
      <Plot plan={plan} flown={flown} />
      {plan && <pre>{table(plan)}</pre>}
      <Calibration folder={status && !status.running ? status.folder : null} />
    </>
  );
}
