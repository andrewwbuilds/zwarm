// Pulls the newest frame of each drone in turn and paints it straight onto its canvas, with no
// React render per frame. One request at a time over all the drones.

import type { Api } from "./types";

const PAUSE_MS = 60;
const RETRY_NO_VIDEO_MS = 1000;

export class VideoFeeds {
  private readonly api: Api;
  private ids: string[] = [];
  private readonly canvases = new Map<string, HTMLCanvasElement>();
  private readonly lastFrame = new Map<string, number>();
  private readonly skipUntil = new Map<string, number>();
  private next = 0;
  private running = false;
  private timer: number | undefined;

  constructor(api: Api) {
    this.api = api;
  }

  setDrones(ids: string[]): void {
    this.ids = ids;
  }

  attach(id: string, canvas: HTMLCanvasElement | null): void {
    if (canvas) this.canvases.set(id, canvas);
    else this.canvases.delete(id);
  }

  // Seconds since the last frame, or null if this drone has never sent one.
  age(id: string): number | null {
    const at = this.lastFrame.get(id);
    return at === undefined ? null : (performance.now() - at) / 1000;
  }

  start(): void {
    if (this.running) return;
    this.running = true;
    void this.step();
  }

  stop(): void {
    this.running = false;
    window.clearTimeout(this.timer);
  }

  private async step(): Promise<void> {
    const id = this.nextDrone();
    if (id !== null) {
      try {
        await this.fetchFrame(id);
      } catch {
        this.skipUntil.set(id, performance.now() + RETRY_NO_VIDEO_MS);
      }
    }
    if (this.running) this.timer = window.setTimeout(() => void this.step(), PAUSE_MS);
  }

  private nextDrone(): string | null {
    const now = performance.now();
    for (let i = 0; i < this.ids.length; i += 1) {
      const id = this.ids[(this.next + i) % this.ids.length];
      if ((this.skipUntil.get(id) ?? 0) <= now) {
        this.next = (this.next + i + 1) % this.ids.length;
        return id;
      }
    }
    return null;
  }

  private async fetchFrame(id: string): Promise<void> {
    const blob = await this.api.video(id);
    if (!blob) {
      this.skipUntil.set(id, performance.now() + RETRY_NO_VIDEO_MS);
      return;
    }
    const bitmap = await createImageBitmap(blob);
    try {
      const canvas = this.canvases.get(id);
      const ctx = canvas?.getContext("2d");
      if (canvas && ctx) {
        if (canvas.width !== bitmap.width) canvas.width = bitmap.width;
        if (canvas.height !== bitmap.height) canvas.height = bitmap.height;
        ctx.drawImage(bitmap, 0, 0);
        this.lastFrame.set(id, performance.now());
      }
    } finally {
      bitmap.close();
    }
  }
}
