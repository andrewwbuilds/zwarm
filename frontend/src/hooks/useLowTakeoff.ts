import { useState } from "react";

const KEY = "zwarm.lowTakeoff";

// Off until asked for, then remembered, so a reload can't quietly bring back a full-height takeoff.
export function useLowTakeoff() {
  const [low, setLow] = useState(() => localStorage.getItem(KEY) === "1");
  const change = (value: boolean) => {
    localStorage.setItem(KEY, value ? "1" : "0");
    setLow(value);
  };
  return [low, change] as const;
}
