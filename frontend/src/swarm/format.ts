// Words and colours shared by the panels.

import type { Drone, Mission } from "./types";

export const DRONE_COLORS = ["#ff8fcf", "#6fe9ff", "#c9f26b", "#ffb36b", "#c3a6ff"] as const;

export const droneColor = (index: number): string => DRONE_COLORS[index % DRONE_COLORS.length];

export function clock(seconds: number): string {
  const whole = Math.max(0, Math.floor(seconds));
  return `${Math.floor(whole / 60)}:${String(whole % 60).padStart(2, "0")}`;
}

const PHASES: Record<string, string> = {
  ground: "On the ground",
  takeoff: "Taking off",
  check: "Testing its sticks",
  air: "Flying",
  landing: "Landing",
  stopped: "Motors cut",
};

export const phaseLabel = (phase: string): string => PHASES[phase] ?? phase;

export const isFlying = (phase: string): boolean => phase === "air" || phase === "check" || phase === "takeoff";

const REGIONS: Record<string, string> = {
  all: "the whole room",
  center: "the center",
  north: "the north side",
  south: "the south side",
  east: "the east side",
  west: "the west side",
};

export function regionLabel(region: string): string {
  return REGIONS[region] ?? `the ${region.replace(/_/g, "-")} corner`;
}

export function missionTitle(mission: Mission | null): string {
  if (!mission || !mission.kind) return "No mission";
  const where = regionLabel(mission.region);
  switch (mission.kind) {
    case "explore":
      return `Exploring ${where}`;
    case "map":
      return `Mapping ${where}`;
    case "inspect":
      return mission.target ? `Inspecting ${mission.target}` : `Inspecting ${where}`;
    case "search":
      return mission.target ? `Looking for ${mission.target}` : `Searching ${where}`;
    default:
      return `${mission.kind} over ${where}`;
  }
}

// One line under the mission title. MissionManager states: starting, running, analyzing, done,
// stopped, failed.
export function missionLine(mission: Mission): string {
  const time = clock(mission.elapsedS);
  const reason = mission.reason ? `. ${mission.reason}` : "";
  switch (mission.state) {
    case "starting":
      return "Taking off";
    case "running":
      return `Running for ${time}`;
    case "analyzing":
      return `Flying done after ${time}. Checking the photos`;
    case "done":
      return `Done after ${time}${reason}`;
    case "stopped":
      return `Stopped after ${time}${reason}`;
    case "failed":
      return `Failed after ${time}${reason}`;
    default:
      return `${mission.state}, ${time}${reason}`;
  }
}

export type Quality = "good" | "fair" | "poor" | "lost";

// How far to trust a drone's position, from its uncertainty and the age of its last anchor fix.
export function trackingQuality(drone: Pick<Drone, "sigma" | "fixAge">): Quality {
  const { sigma, fixAge } = drone;
  if (fixAge !== null && fixAge > 4) return "lost";
  if (sigma === null) return fixAge === null ? "lost" : "fair";
  if (sigma > 0.6) return "lost";
  if (sigma > 0.3 || (fixAge !== null && fixAge > 2)) return "poor";
  if (sigma > 0.12) return "fair";
  return "good";
}

export function trackingText(drone: Pick<Drone, "sigma" | "fixAge">): string {
  const parts: string[] = [];
  if (drone.sigma !== null) parts.push(`±${drone.sigma.toFixed(2)} m`);
  if (drone.fixAge !== null) parts.push(`fix ${drone.fixAge.toFixed(1)} s ago`);
  return parts.length ? parts.join(", ") : "no anchor fix yet";
}
