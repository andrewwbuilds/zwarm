import type { KeyboardEvent } from "react";

export const num = (v: number, digits = 2, width = 7) => ((v >= 0 ? "+" : "") + v.toFixed(digits)).padStart(width);
export const block = (e: KeyboardEvent) => e.preventDefault();
export const drop = (e: { currentTarget: HTMLElement }) => e.currentTarget.blur();
