import { useRef, useState } from "react";
import type { Api, SafetyAction } from "./types";

const ESTOP_RETRY_MS = 250;
const ESTOP_GIVE_UP_MS = 30000;

type Outcome =
  | { kind: "idle" }
  | { kind: "sending"; action: SafetyAction; tries: number }
  | { kind: "delivered"; action: SafetyAction }
  | { kind: "failed"; action: SafetyAction; message: string };

const NAMES: Record<SafetyAction, string> = {
  hold: "HOLD",
  land: "LAND",
  "land-here": "Land where they are",
  estop: "E-STOP",
};

const describe = (outcome: Outcome): string => {
  switch (outcome.kind) {
    case "idle":
      return "";
    case "sending":
      return outcome.tries > 1
        ? `${NAMES[outcome.action]} not delivered yet. Sending again, try ${outcome.tries}.`
        : `Sending ${NAMES[outcome.action]}.`;
    case "delivered":
      return `${NAMES[outcome.action]} delivered.`;
    case "failed":
      return `${NAMES[outcome.action]} not delivered: ${outcome.message}.`;
    default: {
      const exhaustive: never = outcome;
      return exhaustive;
    }
  }
};

export default function SafetyPanel({ api, estopped }: { api: Api; estopped: boolean }) {
  const [outcome, setOutcome] = useState<Outcome>({ kind: "idle" });
  // True while E-STOP is still retrying. Pressing again doesn't start a second loop.
  const retrying = useRef(false);

  const once = (action: SafetyAction) => {
    setOutcome({ kind: "sending", action, tries: 1 });
    api.safety(action).then(
      () => setOutcome({ kind: "delivered", action }),
      (error: unknown) => setOutcome({ kind: "failed", action, message: error instanceof Error ? error.message : String(error) }),
    );
  };

  // Asks nothing. Sends again until the backend says yes. One attempt at a time, so a hung
  // backend can't fill the browser's connections with stuck E-STOPs.
  const estop = () => {
    if (retrying.current) return;
    retrying.current = true;
    const started = performance.now();
    const attempt = async (tries: number) => {
      setOutcome({ kind: "sending", action: "estop", tries });
      try {
        await api.safety("estop");
        retrying.current = false;
        setOutcome({ kind: "delivered", action: "estop" });
      } catch (error) {
        if (performance.now() - started < ESTOP_GIVE_UP_MS) {
          window.setTimeout(() => void attempt(tries + 1), ESTOP_RETRY_MS);
          return;
        }
        retrying.current = false;
        const reason = error instanceof Error ? error.message : String(error);
        setOutcome({ kind: "failed", action: "estop", message: `${reason}. Use E-STOP all on the /fleet page, or pull the batteries` });
      }
    };
    void attempt(1);
  };

  const failed = outcome.kind === "failed";

  return (
    <section className="zw-safety" aria-label="Flight controls">
      <button type="button" className="zw-big" onClick={() => once("hold")}>HOLD</button>
      <button type="button" className="zw-big" onClick={() => once("land")}>LAND</button>
      <button
        type="button"
        className={`zw-estop${estopped ? " zw-estop-on" : ""}`}
        onPointerDown={estop}
        onClick={estop}
        title="Cuts every motor now. Drones in the air fall."
      >
        <span>{estopped ? "STOPPED" : "E-STOP"}</span>
      </button>
      <div className={`zw-safety-note${failed ? " zw-safety-failed" : ""}`} role="status">
        <span>{describe(outcome)}</span>
        <button type="button" className="zw-link" onClick={() => once("land-here")}>Land where they are</button>
      </div>
    </section>
  );
}
