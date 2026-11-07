export const PROGRAMS: Record<string, string> = {
  "calibrate hover": `# calibration: hover height, over a flat floor. Fly it two or three times.
# measure: floor to drone while it holds
takeoff
hold 6
land
`,
  "calibrate climb": `# calibration: climb rate
# measure: floor to drone during the last hold
takeoff
hold 2
climb 0.5
move 0 0 1
hold 4
land
`,
  "calibrate speed": `# calibration: speed. Needs 4 m of clear, flat, textured floor ahead.
# measure: where it lands, x forward and y left of the takeoff spot
takeoff
speed 1
move 2.5 0
hold 1
land
`,
  "calibrate spin": `# calibration: turn rate and camera field of view, from the video.
# start in the middle of a room, 1.5 m or more from walls, with things
# in view all around; nothing to measure by hand
takeoff
hold 2
rate 90
turn 450
hold 2
land
`,
};
