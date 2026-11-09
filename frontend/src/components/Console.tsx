import { useEffect, useRef, useState, type ReactNode } from "react";
import type { CameraTiltDirection, SpeedTier, useControls } from "../hooks/useControls";
import type { SimState, useSim } from "../hooks/useSim";
import { usePlugins } from "../hooks/usePlugins";
import { useLowTakeoff } from "../hooks/useLowTakeoff";
import { VideoFeed } from "./VideoFeed";
import { DrawingOverlay } from "./DrawingOverlay";
import { block, drop, num } from "../ui";
import { API_BASE_URL, forceLand, prepareLowTakeoff, presumedAirborne, startLowTakeoff } from "../backend";

type Controls = ReturnType<typeof useControls>;
type Sim = ReturnType<typeof useSim>;

const MODES = { inc: "keyboard", abs: "gamepad", mouse: "trackpoint" } as const;
const WINDS = [0, 0.5, 1, 1.5, 2, 2.5, 3, 4, 5, 6];


function pilot(c: Controls): string {
  const a = c.axes;
  return [
    `drone   ${c.droneType}`,
    `sticks  t${num(a.throttle, 2, 5)} y${num(a.yaw, 2, 5)} p${num(a.pitch, 2, 5)} r${num(a.roll, 2, 5)}`,
    `input   ${MODES[c.mode]}  speed ${c.speedTier}`,
  ].join("\n");
}

function telemetry(s: SimState): string {
  const [x, y, z] = s.position;
  const [vx, vy, vz] = s.velocity;
  const [roll, pitch, yaw] = s.attitude;
  return [
    `sim t   ${s.t.toFixed(2)} s`,
    `state   ${s.state}`,
    `leds    ${s.leds}`,
    `pos     ${num(x)} ${num(y)} ${num(z)} m`,
    `vel     ${num(vx)} ${num(vy)} ${num(vz)} m/s`,
    `att     ${num(roll, 1)} ${num(pitch, 1)} ${num(yaw, 1)} deg`,
    `agl     ${s.height.toFixed(2)} m`,
    `batt    ${(s.battery * 100).toFixed(1)} %  ${s.voltage.toFixed(2)} V  ${s.current.toFixed(2)} A`,
    `link    ${s.link ? "up" : "down"}  ${s.rc_rate.toFixed(1)} Hz`,
    `rx      ${s.accepted} ok  ${s.rejected} rejected`,
    `radio   ${s.rssi.toFixed(1)} dBm  ${s.distance.toFixed(1)} m`,
    `flow    ${s.flow ? "valid" : "none"}`,
    `video   ${s.video ? "streaming" : "off"}`,
    `wind    ${s.wind.speed.toFixed(1)} m/s  ${s.wind.direction.toFixed(0)} deg  gust ${s.wind.gust.toFixed(1)}`,
  ].join("\n");
}

function Plugins() {
  const { pluginsEnabled, availablePlugins, runningPlugins, togglePlugin } = usePlugins();
  if (!pluginsEnabled) return null;
  return (
    <p>
      plugins{" "}
      {availablePlugins.map((name) => (
        <button key={name} onClick={async (e) => {
          drop(e);
          const running = runningPlugins.has(name);
          await togglePlugin(name);
          window.dispatchEvent(new CustomEvent(running ? "plugin:stopped" : "plugin:running"));
        }}>
          {name} {runningPlugins.has(name) ? "on" : "off"}
        </button>
      ))}
    </p>
  );
}

interface StabilizerAxis { speed: number; error: number; trim: number; stick: number }
interface StabilizerStatus {
  enabled: boolean;
  state: string;
  fault: string | null;
  gain: number;
  frame_ms: number;
  inliers: number;
  forward: StabilizerAxis;
  lateral: StabilizerAxis;
}

/* Corrects drift from the video. Off until asked for: see docs/decisions/0001-visual-stabilizer.md. */
function Stabilizer() {
  const [status, setStatus] = useState<StabilizerStatus | null>(null);

  useEffect(() => {
    let cancelled = false;
    const load = async () => {
      try {
        const response = await fetch(`${API_BASE_URL}/stabilizer`);
        if (response.ok && !cancelled) setStatus(await response.json());
      } catch {
        // The backend may be restarting. Keep the last status until it answers.
      }
    };
    load();
    const interval = setInterval(load, 500);
    return () => { cancelled = true; clearInterval(interval); };
  }, []);

  if (!status) return null;

  const toggle = async (e: { currentTarget: HTMLElement }) => {
    drop(e);
    const response = await fetch(`${API_BASE_URL}/stabilizer`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ enabled: !status.enabled }),
    });
    if (response.ok) setStatus(await response.json());
  };

  const { forward: f, lateral: l } = status;
  return (
    <>
      <p><button onClick={toggle}>stabilizer {status.enabled ? "on" : "off"}</button></p>
      {status.enabled && (
        <pre>{[
          `stab    ${status.state}  gain ${status.gain.toFixed(2)}  ${status.inliers} pts  ${status.frame_ms.toFixed(1)} ms`,
          `  fwd   speed ${num(f.speed, 2, 5)} stick ${num(f.stick, 2, 5)} trim ${num(f.trim, 2, 5)}`,
          `  side  speed ${num(l.speed, 2, 5)} stick ${num(l.stick, 2, 5)} trim ${num(l.trim, 2, 5)}`,
          ...(status.fault ? [`  ${status.fault}`] : []),
        ].join("\n")}</pre>
      )}
    </>
  );
}

function SimControls({ sim }: { sim: Sim }) {
  const s = sim.state;
  if (!s) return <pre>simulator offline</pre>;
  return (
    <>
      <p>
        scene{" "}
        <select value={s.scene} onKeyDown={block} onChange={(e) => { sim.reset({ scene: e.target.value }); drop(e); }}>
          {s.scenes.map((name) => <option key={name}>{name}</option>)}
        </select>{" "}
        firmware{" "}
        <select value={s.firmware} title={s.firmwares[s.firmware]} onKeyDown={block}
          onChange={(e) => { sim.reset({ firmware: e.target.value }); drop(e); }}>
          {Object.keys(s.firmwares).map((name) => <option key={name}>{name}</option>)}
        </select>
      </p>
      <p>
        wind{" "}
        <select value={WINDS.includes(s.wind.speed) ? s.wind.speed : ""} onKeyDown={block}
          onChange={(e) => { sim.wind(Number(e.target.value)); drop(e); }}>
          {!WINDS.includes(s.wind.speed) && <option value="">{s.wind.speed}</option>}
          {WINDS.map((w) => <option key={w} value={w}>{w}</option>)}
        </select>{" "}
        m/s{" "}
        <button onClick={(e) => { sim.reset({}); drop(e); }}>power cycle</button>
      </p>
      <pre>{s.events.slice().reverse().map(([t, m]) => `${t.toFixed(2).padStart(7)}  ${m}`).join("\n")}</pre>
    </>
  );
}

export function Console({ controls, sim, children }: { controls: Controls; sim: Sim; children?: ReactNode }) {
  const c = controls;
  const caps = c.commandCapabilities;
  // Set when a land press would be dropped, which also shows the force land button.
  const [landNote, setLandNote] = useState<string | null>(null);
  const [lowTakeoff, setLowTakeoff] = useLowTakeoff();
  const [takeoffNote, setTakeoffNote] = useState<string | null>(null);
  const [preparing, setPreparing] = useState(false);
  // A stop pressed while a low takeoff is being prepared must also cancel the takeoff.
  const stopped = useRef(false);

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => { if (e.key === "Escape") stopped.current = true; };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, []);

  const pressTakeoff = async () => {
    setLandNote(null);
    setTakeoffNote(null);
    if (!lowTakeoff) {
      c.takeOff();
      return;
    }
    stopped.current = false;
    setPreparing(true);
    try {
      await prepareLowTakeoff(API_BASE_URL);
      if (stopped.current) setTakeoffNote("Low takeoff cancelled by the stop.");
      else await startLowTakeoff(API_BASE_URL);
    } catch (e) {
      setTakeoffNote(`Low takeoff NOT sent: ${(e as Error).message}`);
    } finally {
      setPreparing(false);
    }
  };

  const pressLand = async () => {
    stopped.current = true;
    if ((await presumedAirborne(API_BASE_URL)) === false) {
      setLandNote("Land NOT sent: the backend thinks this drone is on the ground.");
      return;
    }
    setLandNote(null);
    c.land();
  };

  const pressForceLand = async () => {
    try {
      await forceLand(API_BASE_URL);
      setLandNote(null);
    } catch (e) {
      setLandNote(`Force land failed: ${(e as Error).message}`);
    }
  };

  const scheme = (mode: Controls["mode"]) => (e: { currentTarget: HTMLElement }) => {
    if (mode === "mouse") document.body.requestPointerLock();
    c.setMode(mode);
    drop(e);
  };

  const tilt = (direction: CameraTiltDirection) => ({
    onPointerDown: () => c.setCameraTiltDirection(direction),
    onPointerUp: () => c.setCameraTiltDirection(0),
    onPointerLeave: () => c.setCameraTiltDirection(0),
  });

  return (
    <div style={{ display: "flex", height: "100vh" }}>
      <div style={{ flex: 1, position: "relative", background: "#000" }}>
        <VideoFeed src={`${API_BASE_URL}/mjpeg`} />
        <DrawingOverlay />
      </div>
      <div className="raw" style={{ width: 400, padding: 8, overflowY: "auto" }}>
        <pre>{pilot(c)}{sim.state ? "\n" + telemetry(sim.state) : ""}</pre>
        <p>
          <button disabled={!caps.takeoff || preparing} onClick={(e) => { pressTakeoff(); drop(e); }}>takeoff</button>{" "}
          <button disabled={!caps.land} onClick={(e) => { pressLand(); drop(e); }}>land</button>{" "}
          <button disabled={!caps.estop} onClick={(e) => { stopped.current = true; setLandNote(null); c.emergencyStop(); drop(e); }}>e-stop (esc)</button>
        </p>
        <p>
          <label title="Climbs to its normal height first, then comes down. Not measured on a real drone yet.">
            <input type="checkbox" checked={lowTakeoff} onKeyDown={block}
              onChange={(e) => { setLowTakeoff(e.target.checked); drop(e); }} />{" "}
            low takeoff (1 ft)
          </label>
          {preparing && " preparing..."}
        </p>
        {takeoffNote && <p style={{ color: "#f96" }}>{takeoffNote}</p>}
        {landNote && (
          <p style={{ color: "#f96" }}>
            {landNote} Still in the air?{" "}
            <button onClick={(e) => { pressForceLand(); drop(e); }}>force land</button>
            <br />
            On the floor, force land starts the motors.
          </p>
        )}
        <p>
          <button onClick={scheme("inc")}>keyboard</button>{" "}
          <button onClick={scheme("abs")} disabled={!c.gamepadConnected}>gamepad</button>{" "}
          <button onClick={scheme("mouse")}>trackpoint</button>
        </p>
        {caps.speed_control && (
          <p>
            speed{" "}
            <select value={c.speedTier} onKeyDown={block}
              onChange={(e) => { c.setSpeedTier(Number(e.target.value) as SpeedTier); drop(e); }}>
              <option value={0}>0</option>
              <option value={1}>1</option>
              <option value={2}>2</option>
            </select>
          </p>
        )}
        {caps.camera_tilt && (
          <p>
            tilt (pgup/pgdn) <button {...tilt(1)}>up</button> <button {...tilt(-1)}>down</button>
          </p>
        )}
        <Stabilizer />
        <Plugins />
        {children}
        {sim.active && <SimControls sim={sim} />}
      </div>
    </div>
  );
}
