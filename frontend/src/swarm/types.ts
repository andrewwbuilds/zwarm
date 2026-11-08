// What the console knows about the swarm, after parse.ts has checked the JSON.
// World frame: x points north, y points west, z up, metres. Yaw is degrees, 0 along +x,
// positive turning toward +y.

export type Vec3 = readonly [number, number, number];

export interface Drone {
  id: string;
  phase: string;
  p: Vec3 | null;
  yawDeg: number;
  sigma: number | null;
  fixAge: number | null;
  flightS: number;
  faults: string[];
  connected: boolean;
  path: Vec3[];
  goal: Vec3 | null;
}

export interface Mission {
  kind: string | null;
  state: string;
  region: string;
  shareSeen: number;
  target: string | null;
  reason: string | null;
  elapsedS: number;
  tasks: Record<string, string>;
  flightS: Record<string, number>;
}

export interface Obstacle {
  name: string;
  lo: Vec3;
  hi: Vec3;
}

export interface Anchor {
  id: string;
  x: number;
  y: number;
}

export interface Room {
  lo: Vec3;
  hi: Vec3;
  obstacles: Obstacle[];
  anchors: Anchor[];
  northYawDeg: number;
}

export type CoverageCell = readonly [x: number, y: number, seen: number];

export interface Coverage {
  cell: number;
  cells: CoverageCell[];
  shareSeen: number;
}

export interface Thing {
  id: string;
  label: string;
  x: number;
  y: number;
  sure: boolean;
}

export interface AgentInfo {
  model: string;
  busy: boolean;
  online: boolean;
}

export interface Plan {
  id: string;
  summary: string;
  steps: string[];
}

export interface ZwarmState {
  simulated: boolean;
  estopped: boolean;
  flightLimitS: number;
  mission: Mission | null;
  drones: Drone[];
  room: Room | null;
  coverage: Coverage | null;
  things: Thing[];
  agent: AgentInfo;
  pendingPlan: Plan | null;
}

export const EVENT_KINDS = ["said", "reply", "tool", "report", "mission", "finding", "fault"] as const;
export type EventKind = (typeof EVENT_KINDS)[number];

export interface LogEvent {
  n: number;
  t: number;
  kind: EventKind;
  text: string;
  detail: string | null;
  refused: boolean;
}

export interface ToolCall {
  text: string;
  detail: string | null;
  refused: boolean;
}

export interface ChatReply {
  reply: string;
  toolCalls: ToolCall[];
}

export interface Decision {
  ok: boolean;
  reply: string;
}

export type FindingStatus = "pending" | "confirmed" | "dismissed";

export interface Finding {
  id: string;
  t: number;
  drone: string;
  x: number | null;
  y: number | null;
  summary: string;
  severity: string;
  confidence: number | null;
  status: FindingStatus;
  target: string | null;
  views: number;
}

export interface MapStatus {
  name: string;
  state: string;
  photos: number;
  byDrone: Record<string, number>;
  registered: number;
  points: number;
  step: string | null;
  warnings: string[];
  error: string | null;
}

export interface Cloud {
  count: number;
  xyz: Float32Array;
  rgb: Uint8ClampedArray;
}

export type SafetyAction = "hold" | "land" | "land-here" | "estop";

// Everything the console asks of the backend. http.ts and mock.ts both implement it.
export interface Api {
  label: string;
  state(): Promise<ZwarmState>;
  events(after: number): Promise<LogEvent[]>;
  chat(text: string): Promise<ChatReply>;
  approve(planId: string): Promise<Decision>;
  reject(planId: string): Promise<Decision>;
  findings(): Promise<Finding[]>;
  setFinding(id: string, status: FindingStatus): Promise<void>;
  photo(id: string): Promise<Blob>;
  video(drone: string): Promise<Blob | null>;
  mapStatus(): Promise<MapStatus>;
  buildMap(): Promise<void>;
  points(): Promise<Cloud>;
  safety(action: SafetyAction): Promise<void>;
}
