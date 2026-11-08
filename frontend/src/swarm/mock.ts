// A pretend backend for ?mock=1, so the console can be seen working without one. Nothing here
// comes from the simulator or a drone. window.__zwarmMock.setDown(true) fakes a dead backend.

import { parsePoints } from "./parse";
import type {
  Api, ChatReply, Cloud, CoverageCell, Drone, Finding, FindingStatus, LogEvent, MapStatus, Plan, SafetyAction, Vec3,
  ZwarmState,
} from "./types";

const LO: Vec3 = [-3, -2.5, 0];
const HI: Vec3 = [3, 2.5, 2.4];
const CELL = 0.25;
const SPEED = 0.55;
const LIMIT_S = 270;
const HEIGHT = 1.1;

const PADS: Array<[number, number]> = [[-2.4, -1.5], [-2.4, 0], [-2.4, 1.5]];

const OBSTACLES = [
  { name: "table", lo: [0.6, -1.5, 0] as Vec3, hi: [1.6, -0.3, 0.75] as Vec3 },
  { name: "sofa", lo: [-1.6, 1.0, 0] as Vec3, hi: [-0.5, 2.3, 0.6] as Vec3 },
  { name: "shelf", lo: [2.4, 1.1, 0] as Vec3, hi: [3, 2.4, 1.8] as Vec3 },
];

const ANCHORS = [
  { id: "aruco:1", x: -1.5, y: -1.5 }, { id: "aruco:2", x: -1.5, y: 0 }, { id: "aruco:3", x: -1.5, y: 1.5 },
  { id: "aruco:4", x: 1.5, y: 1.5 }, { id: "aruco:5", x: 2, y: 0 }, { id: "aruco:6", x: 0, y: -2 },
];

const THINGS = [
  { id: "1", label: "chair", x: 1.9, y: -1.9, sure: true },
  { id: "2", label: "backpack", x: -0.4, y: -0.9, sure: true },
  { id: "3", label: "cable?", x: 0.4, y: 1.6, sure: false },
];

const STAGED_FINDINGS = [
  { x: 1.0, y: -0.2, summary: "Crack along the table's front leg, about 8 cm long.", severity: "high", confidence: 0.82, target: "table" },
  { x: 2.3, y: 0.9, summary: "Scorch mark on the floor beside the shelf.", severity: "medium", confidence: 0.66, target: "floor" },
  { x: -0.3, y: 1.1, summary: "Possible scuff on the sofa arm. Could be a shadow.", severity: "low", confidence: 0.41, target: "sofa" },
];

interface MockDrone {
  id: string;
  phase: Drone["phase"];
  p: [number, number, number];
  yaw: number;
  flightS: number;
  waypoints: Array<[number, number]>;
  next: number;
  phaseT: number;
  sigma: number;
  faults: string[];
}

const wait = (ms: number) => new Promise((resolve) => window.setTimeout(resolve, ms));

function regionBox(region: string): { x: [number, number]; y: [number, number] } {
  const x: [number, number] = [LO[0], HI[0]];
  const y: [number, number] = [LO[1], HI[1]];
  if (region.includes("north")) x[0] = 0;
  if (region.includes("south")) x[1] = 0;
  if (region.includes("east")) y[1] = 0;
  if (region.includes("west")) y[0] = 0;
  return { x, y };
}

// Back-and-forth lanes over one drone's share of the region.
function sweep(region: string, index: number, count: number): Array<[number, number]> {
  const box = regionBox(region);
  const width = (box.y[1] - box.y[0]) / count;
  const y0 = box.y[0] + width * index + 0.3;
  const y1 = box.y[0] + width * (index + 1) - 0.3;
  const out: Array<[number, number]> = [];
  let forward = true;
  for (let y = y0; y <= y1 + 1e-6; y += 0.6) {
    const ends: Array<[number, number]> = forward
      ? [[box.x[0] + 0.3, y], [box.x[1] - 0.3, y]]
      : [[box.x[1] - 0.3, y], [box.x[0] + 0.3, y]];
    out.push(...ends);
    forward = !forward;
  }
  return out;
}

class MockSwarm {
  down = false;
  private readonly drones: MockDrone[];
  private readonly coverage: Array<[number, number, number]> = [];
  private readonly log: LogEvent[] = [];
  private readonly found: Finding[] = [];
  private mission: { kind: string; region: string; target: string | null; t0: number; state: string } | null = null;
  private plan: (Plan & { kind: string; region: string }) | null = null;
  private map: MapStatus = { name: "room", state: "idle", photos: 0, byDrone: {}, registered: 0, error: null };
  private cloud: Cloud | null = null;
  private estopped = false;
  private busyUntil = 0;
  private last = performance.now();
  private staged = 0;
  private nextStage = 0;
  private buildDoneAt: number | null = null;

  constructor() {
    this.drones = PADS.map(([x, y], i) => ({
      id: `drone${i + 1}`,
      phase: "ground",
      p: [x, y, 0],
      yaw: 0,
      flightS: 0,
      waypoints: [],
      next: 0,
      phaseT: 0,
      sigma: 0.03,
      faults: [],
    }));
    for (let x = LO[0] + CELL / 2; x < HI[0]; x += CELL) {
      for (let y = LO[1] + CELL / 2; y < HI[1]; y += CELL) this.coverage.push([x, y, 0]);
    }
  }

  private say(kind: LogEvent["kind"], text: string, detail: string | null = null, refused = false): void {
    this.log.push({ n: this.log.length + 1, t: performance.now() / 1000, kind, text, detail, refused });
  }

  private assign(): void {
    const flying = this.drones.filter((d) => d.phase === "air" || d.phase === "takeoff");
    flying.forEach((d, i) => {
      d.waypoints = sweep(this.mission?.region ?? "all", i, flying.length);
      d.next = 0;
    });
  }

  private startMission(kind: string, region: string, target: string | null): void {
    this.mission = { kind, region, target, t0: performance.now(), state: "running" };
    this.nextStage = performance.now() + 5000;
    if (kind === "map") this.map = { ...this.map, state: "capturing" };
    this.drones.forEach((d, i) => {
      if (d.phase === "ground") {
        d.phase = "takeoff";
        d.phaseT = -i * 1.5;
      }
    });
    this.assign();
  }

  private advance(): void {
    const now = performance.now();
    const dt = Math.min(0.5, (now - this.last) / 1000);
    this.last = now;
    for (const d of this.drones) {
      d.phaseT += dt;
      if (d.phase === "takeoff") {
        if (d.phaseT > 0) d.p[2] = Math.min(HEIGHT, d.p[2] + dt * 0.7);
        if (d.p[2] >= HEIGHT) d.phase = "air";
      } else if (d.phase === "landing") {
        d.p[2] = Math.max(0, d.p[2] - dt * 0.6);
        if (d.p[2] <= 0) d.phase = "ground";
      }
      if (d.phase === "air" || d.phase === "takeoff") d.flightS += dt;
      d.sigma = 0.05 + 0.04 * (1 + Math.sin(now / 1700 + d.p[0]));
      if (d.phase === "air" && this.mission?.state === "running" && d.waypoints.length) {
        const [gx, gy] = d.waypoints[d.next % d.waypoints.length];
        const dx = gx - d.p[0];
        const dy = gy - d.p[1];
        const distance = Math.hypot(dx, dy);
        if (distance < 0.08) d.next += 1;
        else {
          d.p[0] += (dx / distance) * SPEED * dt;
          d.p[1] += (dy / distance) * SPEED * dt;
          const want = Math.atan2(dy, dx);
          const turn = Math.atan2(Math.sin(want - d.yaw), Math.cos(want - d.yaw));
          d.yaw += turn * Math.min(1, dt * 3);
        }
        this.markSeen(d);
      }
    }
    this.stageFindings(now);
    this.stageMap(now);
  }

  private markSeen(d: MockDrone): void {
    const cos = Math.cos(d.yaw);
    const sin = Math.sin(d.yaw);
    for (const cell of this.coverage) {
      const dx = cell[0] - d.p[0];
      const dy = cell[1] - d.p[1];
      const ahead = dx * cos + dy * sin;
      if (ahead > 0.2 && ahead < 1.6 && Math.abs(-dx * sin + dy * cos) < 0.6 + ahead * 0.6) cell[2] = Math.min(1, cell[2] + 0.04);
    }
  }

  private stageFindings(now: number): void {
    if (this.mission?.kind !== "inspect" || this.mission.state !== "running") return;
    if (now < this.nextStage || this.staged >= STAGED_FINDINGS.length) return;
    this.nextStage = now + 7000;
    const staged = STAGED_FINDINGS[this.staged];
    const drone = this.drones[this.staged % this.drones.length];
    this.staged += 1;
    this.found.push({ ...staged, id: String(this.staged), t: now / 1000, drone: drone.id, z: 0.2, status: "pending" });
    this.say("finding", `Possible damage for review: ${staged.summary}`);
  }

  private stageMap(now: number): void {
    if (this.map.state === "capturing" && Math.random() < 0.3) {
      const drone = this.drones[Math.floor(Math.random() * this.drones.length)];
      this.map = {
        ...this.map,
        photos: this.map.photos + 1,
        byDrone: { ...this.map.byDrone, [drone.id]: (this.map.byDrone[drone.id] ?? 0) + 1 },
      };
    }
    if (this.buildDoneAt !== null && now > this.buildDoneAt) {
      this.buildDoneAt = null;
      this.cloud = makeCloud();
      this.map = { ...this.map, state: "done", registered: Math.round(Math.max(this.map.photos, 60) * 0.85) };
      this.say("mission", `3D map built: ${this.cloud.count.toLocaleString()} points.`);
    }
  }

  private share(): number {
    return this.coverage.filter((c) => c[2] >= 0.3).length / this.coverage.length;
  }

  private sharesByRegion(): Array<{ name: string; share: number }> {
    return ["north", "south", "east", "west"].map((name) => {
      const box = regionBox(name);
      const cells = this.coverage.filter((c) => c[0] >= box.x[0] && c[0] <= box.x[1] && c[1] >= box.y[0] && c[1] <= box.y[1]);
      return { name, share: cells.filter((c) => c[2] >= 0.3).length / cells.length };
    }).sort((a, b) => a.share - b.share);
  }

  state(): ZwarmState {
    this.advance();
    const running = this.mission;
    const tasks: Record<string, string> = {};
    for (const d of this.drones) {
      if (d.phase === "stopped") tasks[d.id] = "Motors cut";
      else if (running?.state === "running") tasks[d.id] = `${running.kind === "inspect" ? "Inspect" : "Sweep"} ${running.region === "all" ? "lane" : running.region + " lane"} ${d.next + 1}`;
    }
    return {
      simulated: true,
      estopped: this.estopped,
      flightLimitS: LIMIT_S,
      mission: running && {
        kind: running.kind,
        state: running.state,
        region: running.region,
        target: running.target,
        shareSeen: this.share(),
        elapsedS: (performance.now() - running.t0) / 1000,
        tasks,
        flightS: Object.fromEntries(this.drones.map((d) => [d.id, d.flightS])),
      },
      drones: this.drones.map((d) => {
        const lane = d.waypoints[d.next % Math.max(1, d.waypoints.length)];
        const inMission = d.phase === "air" && running?.state === "running" && lane !== undefined;
        return {
          id: d.id,
          phase: d.phase,
          p: [d.p[0], d.p[1], d.p[2]],
          yawDeg: (d.yaw * 180) / Math.PI,
          sigma: d.sigma,
          fixAge: 0.1 + Math.abs(Math.sin(performance.now() / 900 + d.p[1])) * 0.5,
          flightS: d.flightS,
          faults: d.faults,
          connected: true,
          path: inMission ? [[lane[0], lane[1], HEIGHT]] : [],
          goal: null,
        };
      }),
      room: { lo: LO, hi: HI, obstacles: OBSTACLES, anchors: ANCHORS, northYawDeg: 0 },
      coverage: { cell: CELL, cells: this.coverage.map((c): CoverageCell => [c[0], c[1], c[2]]), shareSeen: this.share() },
      things: this.share() > 0.08 ? THINGS : [],
      agent: { model: "nvidia/nemotron-3-ultra-550b-a55b", busy: performance.now() < this.busyUntil, online: true },
      pendingPlan: this.plan && { id: this.plan.id, summary: this.plan.summary, steps: this.plan.steps },
    };
  }

  events(after: number): LogEvent[] {
    return this.log.filter((e) => e.n > after);
  }

  private flying(): boolean {
    return this.drones.some((d) => d.phase === "air" || d.phase === "takeoff");
  }

  private proposeMission(kind: string, region: string): string {
    if (this.flying()) {
      this.say("tool", `start_mission kind=${kind} region=${region}`, "ok, drones redistributed");
      this.startMission(kind, region, null);
      return `Switched to ${kind} over ${region}. Drones redistributed.`;
    }
    this.say("tool", `get_status`, "3 drones on the ground, 0:00 flown");
    this.say("tool", `start_mission kind=${kind} region=${region}`, "needs approval: nothing is flying");
    this.plan = {
      id: `p${this.log.length}`,
      kind,
      region,
      summary: `Take off all three drones and ${kind} ${region === "all" ? "the whole room" : `the ${region} side`}.`,
      steps: [
        "drone1 takes the west third of the room",
        "drone2 takes the middle third",
        "drone3 takes the east third",
        "All land before the 4:30 battery limit",
      ],
    };
    return "Here's the plan. It needs your approval because the drones would take off.";
  }

  async chat(text: string): Promise<ChatReply> {
    this.say("said", text);
    const lower = text.toLowerCase();
    if (/\bland\b/.test(lower)) {
      await this.safety("land");
      this.say("reply", "Landing now. That went straight to the drones without the model.");
      return { reply: "Landing now. That went straight to the drones without the model.", toolCalls: [] };
    }
    this.busyUntil = performance.now() + 1200;
    await wait(1200);
    let reply: string;
    if (/haven't|have not|gap|explored/.test(lower)) {
      const gaps = this.sharesByRegion();
      this.say("tool", "coverage_gaps", gaps.map((g) => `${g.name} ${Math.round(g.share * 100)}%`).join(", "));
      reply = `The ${gaps[0].name} side is least covered, ${Math.round(gaps[0].share * 100)}% seen. Next I'd look there, low and slow near the middle of that wall.`;
    } else if (/focus|north|south|east|west/.test(lower)) {
      const region = /north/.test(lower) ? "north" : /south/.test(lower) ? "south" : /east/.test(lower) ? "east" : "west";
      this.say("tool", `focus_region region=${region}`, "ok, 3 drones reassigned");
      if (this.mission) this.mission.region = region;
      else this.say("tool", `start_mission kind=explore region=${region}`, "refused: nothing to refocus", true);
      this.assign();
      reply = `Focusing on the ${region} side. All three drones have new lanes there.`;
    } else if (/inspect|damage/.test(lower)) {
      reply = this.proposeMission("inspect", "all");
    } else if (/stop/.test(lower)) {
      this.say("tool", "stop_mission", "ok, drones holding");
      if (this.mission) this.mission.state = "stopped";
      reply = "Mission stopped. The drones are holding.";
    } else if (/build/.test(lower)) {
      this.say("tool", "build_map", "started");
      await this.buildMap();
      reply = "Building the 3D map from every drone's photos.";
    } else if (/map|explore/.test(lower)) {
      reply = this.proposeMission(/explore/.test(lower) ? "explore" : "map", "all");
    } else {
      this.say("tool", "get_status", `${this.drones.length} drones, ${Math.round(this.share() * 100)}% covered`);
      reply = `${Math.round(this.share() * 100)}% of the room is covered so far.`;
    }
    this.say("reply", reply);
    return { reply, toolCalls: [] };
  }

  approve(id: string): void {
    if (this.plan?.id !== id) throw new Error("that plan is no longer waiting");
    const { kind, region } = this.plan;
    this.plan = null;
    this.estopped = false;
    this.say("mission", `Plan approved. Taking off to ${kind} ${region}.`);
    this.startMission(kind, region, null);
  }

  reject(id: string): void {
    if (this.plan?.id !== id) throw new Error("that plan is no longer waiting");
    this.plan = null;
    this.say("mission", "Plan rejected. Nothing took off.");
  }

  findings(): Finding[] {
    return this.found;
  }

  setFinding(id: string, status: FindingStatus): void {
    const finding = this.found.find((f) => f.id === id);
    if (finding) finding.status = status;
  }

  async photo(id: string): Promise<Blob> {
    const finding = this.found.find((f) => f.id === id);
    return drawFrame(`Finding ${id}`, finding?.x ?? 0, finding?.y ?? 0, true);
  }

  async video(id: string): Promise<Blob | null> {
    const d = this.drones.find((candidate) => candidate.id === id);
    return d ? drawFrame(id, d.p[0], d.p[1] + d.yaw, false) : null;
  }

  mapStatus(): MapStatus {
    return this.map;
  }

  async buildMap(): Promise<void> {
    this.map = { ...this.map, state: "building" };
    this.buildDoneAt = performance.now() + 5000;
  }

  points(): Cloud {
    return this.cloud ?? { count: 0, xyz: new Float32Array(0), rgb: new Uint8ClampedArray(0) };
  }

  async safety(action: SafetyAction): Promise<void> {
    if (action === "estop") {
      this.estopped = true;
      for (const d of this.drones) {
        d.phase = "stopped";
        d.p[2] = 0;
      }
      if (this.mission) this.mission.state = "stopped";
      this.say("fault", "E-STOP pressed. Every motor cut.");
      return;
    }
    if (action === "hold") {
      if (this.mission) this.mission.state = "holding";
      this.say("mission", "HOLD pressed. Drones are holding where they are.");
      return;
    }
    for (const d of this.drones) {
      if (d.phase === "air" || d.phase === "takeoff") d.phase = "landing";
    }
    if (this.mission) this.mission.state = "stopped";
    this.say("mission", action === "land" ? "LAND pressed. Drones are landing." : "Drones are landing where they are.");
  }
}

// A rough room: floor, two walls and three boxes, enough to check the orbit control.
function makeCloud(): Cloud {
  const rows: number[][] = [];
  const shade = (base: number) => base + Math.random() * 30 - 15;
  for (let i = 0; i < 26000; i += 1) {
    const x = LO[0] + Math.random() * (HI[0] - LO[0]);
    const y = LO[1] + Math.random() * (HI[1] - LO[1]);
    rows.push([x, y, 0, shade(150), shade(125), shade(95)]);
  }
  for (let i = 0; i < 12000; i += 1) {
    const x = LO[0] + Math.random() * (HI[0] - LO[0]);
    rows.push([x, HI[1], Math.random() * HI[2], shade(195), shade(195), shade(210)]);
    rows.push([HI[0], LO[1] + Math.random() * (HI[1] - LO[1]), Math.random() * HI[2], shade(170), shade(200), shade(215)]);
  }
  for (const o of OBSTACLES) {
    for (let i = 0; i < 4000; i += 1) {
      const x = o.lo[0] + Math.random() * (o.hi[0] - o.lo[0]);
      const y = o.lo[1] + Math.random() * (o.hi[1] - o.lo[1]);
      const z = Math.random() < 0.5 ? o.hi[2] : Math.random() * o.hi[2];
      rows.push([x, y, z, shade(120), shade(70), shade(50)]);
    }
  }
  return parsePoints(rows);
}

async function drawFrame(label: string, a: number, b: number, damaged: boolean): Promise<Blob> {
  const canvas = new OffscreenCanvas(320, 240);
  const ctx = canvas.getContext("2d");
  if (!ctx) throw new Error("no 2D canvas");
  const floor = ctx.createLinearGradient(0, 0, 0, 240);
  floor.addColorStop(0, "#6d6256");
  floor.addColorStop(1, "#b5a690");
  ctx.fillStyle = floor;
  ctx.fillRect(0, 0, 320, 240);
  ctx.strokeStyle = "rgba(40,30,20,0.5)";
  ctx.lineWidth = 2;
  ctx.beginPath();
  for (let i = -6; i <= 6; i += 1) {
    ctx.moveTo(160 + (i + (b % 1)) * 16, 0);
    ctx.lineTo(160 + (i + (b % 1)) * 90, 240);
  }
  for (let j = 0; j < 7; j += 1) {
    const y = ((j + (a % 1)) / 7) ** 2 * 240;
    ctx.moveTo(0, y);
    ctx.lineTo(320, y);
  }
  ctx.stroke();
  if (damaged) {
    ctx.strokeStyle = "#1a1208";
    ctx.lineWidth = 4;
    ctx.beginPath();
    ctx.moveTo(90, 60);
    ctx.lineTo(130, 110);
    ctx.lineTo(120, 140);
    ctx.lineTo(180, 190);
    ctx.stroke();
  }
  ctx.fillStyle = "#fff";
  ctx.font = "600 18px sans-serif";
  ctx.fillText(`${label} (mock)`, 10, 24);
  return canvas.convertToBlob({ type: "image/jpeg", quality: 0.7 });
}

declare global {
  interface Window {
    __zwarmMock?: { setDown: (down: boolean) => void };
  }
}

export function mockApi(): Api {
  const swarm = new MockSwarm();
  window.__zwarmMock = {
    setDown: (down) => {
      swarm.down = down;
    },
  };
  // Failing polls fail fast here. E-STOP goes through the same gate, as it would for real.
  const up = async <T,>(task: () => T | Promise<T>): Promise<T> => {
    if (swarm.down) throw new Error("can't connect (mock backend is down)");
    return task();
  };
  return {
    label: "the mock backend",
    state: () => up(() => swarm.state()),
    events: (after) => up(() => swarm.events(after)),
    chat: (text) => up(() => swarm.chat(text)),
    approve: (id) => up(() => swarm.approve(id)),
    reject: (id) => up(() => swarm.reject(id)),
    findings: () => up(() => swarm.findings()),
    setFinding: (id, status) => up(() => swarm.setFinding(id, status)),
    photo: (id) => up(() => swarm.photo(id)),
    video: (drone) => up(() => swarm.video(drone)),
    mapStatus: () => up(() => swarm.mapStatus()),
    buildMap: () => up(() => swarm.buildMap()),
    points: () => up(() => swarm.points()),
    safety: (action) => up(() => swarm.safety(action)),
  };
}
