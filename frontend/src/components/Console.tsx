import type { ReactNode } from "react";
import type { CameraTiltDirection, SpeedTier, useControls } from "../hooks/useControls";
import type { SimState, useSim } from "../hooks/useSim";
import { usePlugins } from "../hooks/usePlugins";
import { VideoFeed } from "./VideoFeed";
import { DrawingOverlay } from "./DrawingOverlay";
import { block, drop, num } from "../ui";
import { API_BASE_URL } from "../backend";

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
        <VideoFeed src="http://localhost:8000/mjpeg" />
        <DrawingOverlay />
      </div>
      <div className="raw" style={{ width: 400, padding: 8, overflowY: "auto" }}>
        <pre>{pilot(c)}{sim.state ? "\n" + telemetry(sim.state) : ""}</pre>
        <p>
          <button disabled={!caps.takeoff} onClick={(e) => { c.takeOff(); drop(e); }}>takeoff</button>{" "}
          <button disabled={!caps.land} onClick={(e) => { c.land(); drop(e); }}>land</button>{" "}
          <button disabled={!caps.estop} onClick={(e) => { c.emergencyStop(); drop(e); }}>e-stop (esc)</button>
        </p>
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
        <Plugins />
        {children}
        {sim.active && <SimControls sim={sim} />}
      </div>
    </div>
  );
}
