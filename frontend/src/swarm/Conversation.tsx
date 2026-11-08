import { useEffect, useMemo, useRef, useState, type FormEvent } from "react";
import { listen, speechSupported, type Listener } from "./speech";
import type { AgentInfo, Api, EventKind, LogEvent, Plan, ToolCall } from "./types";

// The operator sees the command at once, and the server's events replace it when they arrive.
// Lines the backend never echoes as events stay in the log, placed where they happened.
type LocalLine =
  | { key: number; after: number; kind: "said" | "reply"; text: string }
  | { key: number; after: number; kind: "tool"; call: ToolCall }
  | { key: number; after: number; kind: "error"; text: string };

interface Line {
  key: string;
  kind: EventKind | "error";
  text: string;
  detail: string | null;
  refused: boolean;
}

const MAX_RUNNING = 2;

// The backend acts on these with no model, so a busy model is no reason to hold one back. It decides
// what really counts, so sending one extra sentence by mistake costs nothing.
const STOP_WORDS = /\b(stop|hold|land|halt|abort|freeze|e-?stop|emergency|down|home|come back|bring)\b/i;
const STOP_MAX_WORDS = 6;

function isStopPhrase(text: string): boolean {
  return STOP_WORDS.test(text) && text.trim().split(/\s+/).length <= STOP_MAX_WORDS;
}

const EXAMPLES = [
  "Map this entire area in 3D",
  "Inspect these things for damage",
  "Which areas haven't been explored yet?",
  "Stop mapping and focus on the north side",
];

function mergeLines(events: LogEvent[], local: LocalLine[]): Line[] {
  const visible = local.filter((line) => {
    if (line.kind === "error") return true;
    if (line.kind === "tool") return !events.some((e) => e.kind === "tool" && e.n > line.after);
    return !events.some((e) => e.kind === line.kind && e.n > line.after && e.text === line.text);
  });
  const rows: Array<{ order: number; tie: number; line: Line }> = events.map((e) => ({
    order: e.n,
    tie: 0,
    line: { key: `e${e.n}`, kind: e.kind, text: e.text, detail: e.detail, refused: e.refused },
  }));
  for (const line of visible) {
    rows.push({
      order: line.after,
      tie: line.key + 1,
      line: {
        key: `l${line.key}`,
        kind: line.kind,
        text: line.kind === "tool" ? line.call.text : line.text,
        detail: line.kind === "tool" ? line.call.detail : null,
        refused: line.kind === "tool" && line.call.refused,
      },
    });
  }
  rows.sort((a, b) => a.order - b.order || a.tie - b.tie);
  return rows.map((row) => row.line);
}

function LogLine({ line }: { line: Line }) {
  switch (line.kind) {
    case "said":
      return <p className="zw-said">{line.text}</p>;
    case "reply":
      return <p className="zw-reply">{line.text}</p>;
    case "tool":
      return (
        <p className={`zw-tool${line.refused ? " zw-tool-refused" : ""}`}>
          <code>{line.text}</code>
          {line.detail && <span>{line.refused ? `Refused: ${line.detail}` : line.detail}</span>}
        </p>
      );
    case "report":
      return <p className="zw-report">{line.text}</p>;
    case "mission":
      return <p className="zw-note">{line.text}</p>;
    case "finding":
      return <p className="zw-note zw-note-finding">{line.text}</p>;
    case "fault":
    case "error":
      return <p className="zw-note zw-note-fault">{line.text}</p>;
    default: {
      const exhaustive: never = line.kind;
      return exhaustive;
    }
  }
}

function PlanCard({ api, plan, onError }: { api: Api; plan: Plan; onError: (text: string) => void }) {
  const [busy, setBusy] = useState(false);

  const decide = async (approve: boolean) => {
    setBusy(true);
    try {
      const decision = await (approve ? api.approve(plan.id) : api.reject(plan.id));
      if (!decision.ok) onError(decision.reply || `The plan wasn't ${approve ? "approved" : "rejected"}.`);
    } catch (error) {
      onError(`The plan wasn't ${approve ? "approved" : "rejected"}: ${error instanceof Error ? error.message : String(error)}.`);
    } finally {
      setBusy(false);
    }
  };

  return (
    <section className="zw-plan" aria-label="Plan waiting for approval">
      <h2>Approve this plan to take off</h2>
      <p>{plan.summary}</p>
      {plan.steps.length > 0 && (
        <ol>
          {plan.steps.map((step, i) => <li key={`${i}-${step}`}>{step}</li>)}
        </ol>
      )}
      <div className="zw-plan-buttons">
        <button type="button" className="zw-btn zw-btn-go" disabled={busy} onClick={() => void decide(true)}>Approve</button>
        <button type="button" className="zw-btn" disabled={busy} onClick={() => void decide(false)}>Reject</button>
      </div>
    </section>
  );
}

function MicIcon() {
  return (
    <svg viewBox="0 0 24 24" width="28" height="28" aria-hidden="true">
      <rect x="9" y="3" width="6" height="11" rx="3" fill="currentColor" />
      <path d="M5.5 11.5a6.5 6.5 0 0 0 13 0M12 18v3M8.5 21h7" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" />
    </svg>
  );
}

interface Props {
  api: Api;
  events: LogEvent[];
  agent: AgentInfo | null;
  plan: Plan | null;
}

export default function Conversation({ api, events, agent, plan }: Props) {
  const [local, setLocal] = useState<LocalLine[]>([]);
  const [draft, setDraft] = useState("");
  const [running, setRunning] = useState(0);
  // The mic's callback outlives the render it was made in, so it reads the count from here.
  const runningNow = useRef(0);
  const [listening, setListening] = useState(false);
  const [micError, setMicError] = useState<string | null>(null);
  const listener = useRef<Listener | null>(null);
  const logRef = useRef<HTMLDivElement>(null);
  const pinned = useRef(true);
  const nextKey = useRef(0);
  const hasSpeech = useMemo(speechSupported, []);

  const lines = useMemo(() => mergeLines(events, local), [events, local]);

  useEffect(() => {
    const log = logRef.current;
    if (log && pinned.current) log.scrollTop = log.scrollHeight;
  }, [lines, running, plan]);

  useEffect(() => () => listener.current?.stop(), []);

  const lastN = events.length ? events[events.length - 1].n : 0;

  const addError = (text: string) => {
    nextKey.current += 1;
    const key = nextKey.current;
    setLocal((old) => [...old, { key, after: lastN, kind: "error", text }]);
  };

  const send = async (raw: string) => {
    const text = raw.trim();
    if (!text) return;
    if (runningNow.current >= MAX_RUNNING && !isStopPhrase(text)) {
      setDraft(text);
      addError("Two commands are still running. HOLD, LAND and E-STOP work now, or send this again in a moment.");
      return;
    }
    setDraft("");
    pinned.current = true;
    const after = lastN;
    const key = () => {
      nextKey.current += 1;
      return nextKey.current;
    };
    setLocal((old) => [...old, { key: key(), after, kind: "said", text }]);
    runningNow.current += 1;
    setRunning(runningNow.current);
    try {
      const answer = await api.chat(text);
      const added: LocalLine[] = answer.toolCalls.map((call) => ({ key: key(), after, kind: "tool", call }));
      if (answer.reply) added.push({ key: key(), after, kind: "reply", text: answer.reply });
      setLocal((old) => [...old, ...added]);
    } catch (error) {
      const reason = error instanceof Error ? error.message : String(error);
      setLocal((old) => [...old, { key: key(), after, kind: "error", text: `"${text}" didn't get through: ${reason}.` }]);
    } finally {
      runningNow.current -= 1;
      setRunning(runningNow.current);
    }
  };

  const submit = (e: FormEvent) => {
    e.preventDefault();
    void send(draft);
  };

  const toggleMic = () => {
    if (listening) {
      listener.current?.stop();
      return;
    }
    setMicError(null);
    listener.current = listen({
      onInterim: setDraft,
      onFinal: (text) => void send(text),
      onError: setMicError,
      onEnd: () => {
        listener.current = null;
        setListening(false);
      },
    });
    if (listener.current) setListening(true);
  };

  const onScroll = () => {
    const log = logRef.current;
    if (log) pinned.current = log.scrollHeight - log.scrollTop - log.clientHeight < 40;
  };

  const thinking = running > 0 || agent?.busy === true;
  const modelName = agent?.model ? agent.model.split("/").pop() : "Nemotron";

  return (
    <section className="zw-talk" aria-label="Conversation with Nemotron">
      <div className="zw-log" ref={logRef} onScroll={onScroll} aria-live="polite">
        {lines.length === 0 && (
          <div className="zw-examples">
            <p>Tell the drones what you want done.</p>
            {EXAMPLES.map((example) => (
              <button type="button" key={example} className="zw-example" onClick={() => void send(example)}>{example}</button>
            ))}
          </div>
        )}
        {lines.map((line) => <LogLine key={line.key} line={line} />)}
      </div>

      {plan && <PlanCard key={plan.id} api={api} plan={plan} onError={addError} />}

      <div className={`zw-agent${agent && !agent.online ? " zw-agent-off" : ""}`}>
        {agent && !agent.online
          ? "Nemotron can't be reached. Fixed phrases still work: land, hold, stop, take off."
          : thinking ? `${modelName} is working` : `${modelName} is ready`}
        {thinking && <span className="zw-dots" aria-hidden="true" />}
      </div>

      <form className="zw-command" onSubmit={submit}>
        <input
          value={draft}
          onChange={(e) => setDraft(e.target.value)}
          placeholder={listening ? "Listening" : "Say or type a command"}
          aria-label="Command for the swarm"
          autoComplete="off"
          autoFocus
        />
        <button
          type="button"
          className={`zw-mic${listening ? " zw-mic-on" : ""}`}
          onClick={toggleMic}
          disabled={!hasSpeech}
          aria-pressed={listening}
          aria-label={listening ? "Stop listening" : "Speak a command"}
          title={hasSpeech ? "Speak a command" : "This browser has no speech recognition. Use Chrome, Edge or Safari."}
        >
          <MicIcon />
        </button>
      </form>
      {micError && <p className="zw-mic-error" role="alert">{micError}</p>}
    </section>
  );
}
