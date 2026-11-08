import { useEffect, useRef, useState } from "react";
import type { Api, Finding, FindingStatus } from "./types";

const PHOTO_RETRIES = 3;
const PHOTO_RETRY_MS = 3000;

// Fetches one finding's photo and keeps it as a blob URL for as long as the row is on screen.
function usePhoto(api: Api, id: string): string | null {
  const [url, setUrl] = useState<string | null>(null);

  useEffect(() => {
    let live = true;
    let made: string | null = null;
    let timer: number | undefined;
    const load = async (left: number) => {
      try {
        const blob = await api.photo(id);
        if (!live) return;
        made = URL.createObjectURL(blob);
        setUrl(made);
      } catch {
        if (live && left > 0) timer = window.setTimeout(() => void load(left - 1), PHOTO_RETRY_MS);
      }
    };
    void load(PHOTO_RETRIES);
    return () => {
      live = false;
      window.clearTimeout(timer);
      if (made) URL.revokeObjectURL(made);
      setUrl(null);
    };
  }, [api, id]);

  return url;
}

function severityClass(severity: string): string {
  const word = severity.toLowerCase();
  if (word.includes("high") || word.includes("severe") || word.includes("critical")) return "zw-sev-high";
  if (word.includes("med") || word.includes("moderate")) return "zw-sev-med";
  return "zw-sev-low";
}

interface RowProps {
  api: Api;
  finding: Finding;
  onJudge: (id: string, status: FindingStatus) => void;
  onEnlarge: (finding: Finding, url: string) => void;
}

function FindingRow({ api, finding, onJudge, onEnlarge }: RowProps) {
  const url = usePhoto(api, finding.id);
  const sure = finding.confidence === null ? null : Math.round(finding.confidence * 100);

  return (
    <li className={`zw-finding zw-finding-${finding.status}`}>
      <button
        type="button"
        className="zw-photo"
        disabled={!url}
        onClick={() => url && onEnlarge(finding, url)}
        aria-label={`Enlarge the photo from ${finding.drone}`}
      >
        {url ? <img src={url} alt="" /> : <span>No photo yet</span>}
      </button>
      <div className="zw-finding-body">
        <p className="zw-finding-summary">
          <b className={severityClass(finding.severity)}>{finding.severity}</b> {finding.summary}
        </p>
        <p className="zw-finding-meta">
          {finding.drone}{finding.target ? `, ${finding.target}` : ""}{sure === null ? "" : `, ${sure}% sure`}
          {finding.views > 1 ? `, ${finding.views} looks` : ""}
        </p>
        {finding.status === "pending" ? (
          <div className="zw-finding-buttons">
            <button type="button" className="zw-btn zw-btn-small zw-btn-go" onClick={() => onJudge(finding.id, "confirmed")}>Confirm</button>
            <button type="button" className="zw-btn zw-btn-small" onClick={() => onJudge(finding.id, "dismissed")}>Dismiss</button>
          </div>
        ) : (
          <p className="zw-finding-verdict">
            {finding.status === "confirmed" ? "Confirmed" : "Dismissed"}
            <button type="button" className="zw-link" onClick={() => onJudge(finding.id, "pending")}>Undo</button>
          </p>
        )}
      </div>
    </li>
  );
}

interface Props {
  api: Api;
  findings: Finding[];
  onJudge: (id: string, status: FindingStatus) => Promise<void>;
}

export default function FindingsPanel({ api, findings, onJudge }: Props) {
  const [problem, setProblem] = useState<string | null>(null);
  const [large, setLarge] = useState<{ finding: Finding; url: string } | null>(null);
  const dialog = useRef<HTMLDialogElement>(null);

  useEffect(() => {
    if (large) dialog.current?.showModal();
    else dialog.current?.close();
  }, [large]);

  const judge = (id: string, status: FindingStatus) => {
    setProblem(null);
    onJudge(id, status).catch((error: unknown) => {
      setProblem(`That wasn't saved: ${error instanceof Error ? error.message : String(error)}.`);
    });
  };

  const order: Record<FindingStatus, number> = { pending: 0, confirmed: 1, dismissed: 2 };
  const sorted = [...findings].sort((a, b) => order[a.status] - order[b.status] || b.t - a.t);
  const waiting = findings.filter((f) => f.status === "pending").length;

  return (
    <section className="zw-findings" aria-label="Findings for review">
      <h2>{waiting > 0 ? `${waiting} ${waiting === 1 ? "finding needs" : "findings need"} your review` : "Findings"}</h2>
      {problem && <p className="zw-fault">{problem}</p>}
      {sorted.length === 0 ? (
        <p className="zw-empty">Nothing yet. Start an inspection and possible damage shows up here for you to check.</p>
      ) : (
        <ul>
          {sorted.map((finding) => (
            <FindingRow key={finding.id} api={api} finding={finding} onJudge={judge} onEnlarge={(f, url) => setLarge({ finding: f, url })} />
          ))}
        </ul>
      )}
      <dialog ref={dialog} className="zw-dialog" onClose={() => setLarge(null)}>
        {large && (
          <>
            <img src={large.url} alt={`Photo from ${large.finding.drone}`} />
            <p><b className={severityClass(large.finding.severity)}>{large.finding.severity}</b> {large.finding.summary}</p>
            <form method="dialog"><button className="zw-btn" type="submit">Close</button></form>
          </>
        )}
      </dialog>
    </section>
  );
}
