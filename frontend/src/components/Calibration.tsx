import { useState } from "react";
import { drop } from "../ui";
import { API_BASE_URL } from "../backend";

const API = API_BASE_URL;

const FIELDS = [
  ["hover_height", "hover height m"],
  ["final_height", "final height m"],
  ["landed_x", "landed x m"],
  ["landed_y", "landed y m"],
  ["yaw_deg", "turned deg"],
] as const;

async function post(path: string, body: object) {
  const r = await fetch(`${API}${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  const data = await r.json();
  if (!r.ok) throw new Error(data?.detail ?? r.statusText);
  return data;
}

export function Calibration({ folder }: { folder: string | null }) {
  const [values, setValues] = useState<Record<string, string>>({});
  const [minutes, setMinutes] = useState("");
  const [note, setNote] = useState("");
  const [busy, setBusy] = useState(false);

  const save = async (e: { currentTarget: HTMLElement }) => {
    drop(e);
    try {
      const r = await post("/trajectory/measure", values);
      setNote(`saved to runs/${r.folder}/measured.json`);
      setValues({});
    } catch (err) {
      setNote(`error   ${(err as Error).message}`);
    }
  };

  const fit = async (e: { currentTarget: HTMLElement }) => {
    drop(e);
    setBusy(true);
    setNote("fitting, this can take a minute");
    try {
      setNote((await post("/calibrate", { flight_min: minutes })).report);
    } catch (err) {
      setNote(`error   ${(err as Error).message}`);
    } finally {
      setBusy(false);
    }
  };

  return (
    <>
      <p>calibration</p>
      {folder && (
        <p>
          measured on runs/{folder.split("/").pop()}<br />
          {FIELDS.map(([key, label]) => (
            <label key={key}>
              {label}{" "}
              <input type="number" step="0.01" size={6} value={values[key] ?? ""}
                onChange={(e) => setValues({ ...values, [key]: e.target.value })} />
              <br />
            </label>
          ))}
          <button onClick={save}>save measurements</button>
        </p>
      )}
      <p>
        hover minutes on a full battery{" "}
        <input type="number" step="0.1" size={5} value={minutes} onChange={(e) => setMinutes(e.target.value)} />
        <br />
        <button disabled={busy} onClick={fit}>fit simulator to all runs</button>
      </p>
      {note && <pre>{note}</pre>}
    </>
  );
}
