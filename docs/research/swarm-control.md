# Swarm control for ZWARM

Research date: 2026-11-07. Scope: a small indoor fleet of S20/S2x drones first.
Other protocol families need their own hardware validation. This is an architecture
recommendation, not a demonstrated swarm capability or measured fleet-size limit.

Update: this repo went a different way on positioning. There are no room cameras and no markers on the drones. See `docs/decisions/0004-floor-tag-tracking.md`. The rest stands as research.

## Recommendation

Use a central coordinator, external tracking in a shared room coordinate system,
and one independently supervised control worker per drone. Keep the stock flight
controller responsible for stabilization. Begin with two stock S20s, each connected
through its own Wi-Fi adapter on a Linux host. Isolate the adapters and workers in
network namespaces. Keep the Mac as the operator console if desired.

Prove single-drone closed-loop position hold before attempting formation flight.
For an expanded S20 fleet, evaluate an ESP32-S3 UART radio bridge on one airframe.
If reliable swarm operation matters more than reusing these airframes, compare
Crazyflie plus Crazyswarm2 and a supported positioning system before buying a fleet.

## What this repository establishes

- `protocols/s2x_rc_protocol_adapter.py` sends 20-byte stick-control packets to UDP
  8080. The flight controller service defaults to 80 Hz; this is a packet send rate,
  not evidence of an 80 Hz physical control bandwidth.
- `protocols/s2x_video_protocol.py` defaults to 172.16.10.1 and binds the fixed video
  port 8888 on all interfaces. Simply constructing multiple adapters in the same
  network namespace will not provide reliable per-drone routing/video isolation.
- `web_server.py` owns a single active flight controller and receiver. It needs a
  fleet layer or separate worker processes to support multiple drones.
- `control/trajectory.py` predicts paths and runs timed stick commands. It does not
  close a position feedback loop.
- Independent monocular SLAM instances do not establish a shared metric map.
  Coordinate alignment, scale, yaw and tracking failures remain problems.
- `sim/server.py` wraps one simulated vehicle. Multiple server processes alone do
  not simulate inter-drone collisions or aerodynamic interactions.
- `docs/research/S2x.md` reports hardware captures showing that the stock Wi-Fi
  board forwards control packets unchanged over 3.3 V UART, 115200 8N1. It also
  emits optical-flow reports at approximately 22 Hz. These are existing research
  records; this review did not independently repeat the hardware measurements.

## Connectivity

Stock route: Linux host -> dedicated Wi-Fi adapter per drone -> drone access point.
Each worker and adapter live in an isolated network namespace, so repeated drone
IP addresses and video ports can coexist. A veth management link connects each
worker to the coordinator; the worker terminates/proxies drone traffic. Do not
route ambiguous drone IP addresses directly from the coordinator. Linux wireless
PHY namespace support must be checked on the selected adapters/drivers.

Separate small Linux gateways, each with drone Wi-Fi and wired Ethernet backhaul,
are an alternative when USB adapter/driver isolation is troublesome. Avoid Wi-Fi
network switching to time-share one radio. Virtual interfaces are not an assumed
substitute for independent radios. Verify each airframe's SSID/BSSID, gateway,
channel and behavior with multiple drones powered on. Do not assume these stock
boards support joining a common access point or changing their channels.

Network namespaces isolate devices, routes and ports [1]; they do not create
extra radio airtime. Several stock access points can still contend on one channel.
Disable unnecessary onboard video during control experiments and measure command
age, loss and jitter with the entire fleet active.

Modified route: PC -> USB ESP32 gateway -> ESP-NOW -> ESP32 bridge -> stock FC UART.
This is a proposed design, not firmware already present here. Use individual IDs,
sequence numbers, command expiration, application acknowledgments for discrete
actions, and a local watchdog. Keep newest stick setpoints; never replay a backlog
of old movements. Start near the stock app's documented 20 Hz command cadence and
increase only when latency measurements and hardware behavior justify it.

ESP-NOW removes per-drone Wi-Fi associations, but still shares the 2.4 GHz medium.
Espressif documents a 20-peer paired-device limit, at most 17 encrypted peers
(default 7), shared-channel requirements, and no guarantee of application receipt
from a successful MAC callback [2]. These are protocol/configuration limits, not a
validated fleet capacity. Prefer encrypted unicast initially; multicast encryption
is unsupported. This control bridge does not automatically retain camera video.

The stock S20 board also supplies optical flow to the flight controller. Replacing
it with a bridge that sends zero flow removes that stabilization input. A faithful
replacement must acquire and forward real flow, or an inline proxy must preserve
the stock reports. Use a designed UART multiplexer/proxy; never directly connect
two actively driven TX outputs. Decode FC return telemetry and characterize actual
FC link-loss behavior before relying on a bridge watchdog's fallback action.

## Shared positioning and control

For a low-cost indoor prototype, mount distinct ArUco markers on the airframes and
use calibrated external cameras. OpenCV supports marker ID detection and pose
estimation from known marker geometry and camera calibration [3]. Start with an
overhead view for detection/identity, then add an oblique view for improved depth
and occlusion coverage. A floor-plane pixel conversion is wrong for drones at
varying heights. Single-marker monocular depth can be noisy; measure 3D accuracy
throughout the flight volume before closing the loop. Motion capture is the more
established alternative when available. Do not assume adding markers guarantees
the accuracy or latency needed for formation flight.

Each worker consumes a timestamped pose, estimated velocity and confidence. A
position/velocity controller computes pitch, roll, throttle and yaw stick commands
in the drone's body frame. Identify each airframe's deadbands, yaw response, delay
and speed response first; account for the stock position/altitude hold loops.
Use simulator tuning as a starting hypothesis, then fit measured hardware data.
Proposed initial targets are 30-60 Hz tracking and 20-50 Hz outer-loop updates;
these are engineering targets, not proven minimums or achieved performance.

Central coordination is simpler for the initial fleet: assign formation slots,
schedule takeoff/landing, produce smooth bounded trajectories, reserve separation
volumes, and supervise worker health. Use a common clock/start time for coordinated
segments, but let each worker track its own measured state. Choose separation
using vehicle size, pose uncertainty, delay, stopping distance and downwash. Avoid
vertical stacking. Begin with separated lanes and planned trajectories; advanced
reactive avoidance comes after tracking and control validation.

Independent local failure handling is essential: expire stale setpoints and poses,
reject new mission motion after loss of tracking, and execute a tested recovery.
Neutral sticks are not guaranteed position hold. S20 takeoff and landing share a
toggle, so blind retries can reverse the intended action. Require observed flight
state and deduplicate action requests; a network acknowledgment proves receipt,
not that the aircraft landed. Motor-cut is a separate emergency action and causes
a fall. Coordinator failure must not leave workers replaying stale commands.

## Framework choice

Crazyswarm2 is a strong architecture reference: per-robot identities, fleet
commands, tracking integration, simulation and trajectory execution [4,5]. It is
built for Bitcraze flight computers, so S20 support needs custom transport and
closed-loop control integration. A ROS 2 wrapper alone does not supply those.
Its documented Buffered Voronoi Cell collision avoidance executes in Crazyflie
firmware and requires motion capture; it cannot be enabled on an S20 by setting
the same parameter [6].

Keep the first S20 implementation in the existing Python backend, with explicit
transport/tracker/controller interfaces. Add ROS 2 integration when its tracking,
visualization or planning ecosystem is useful. If changing hardware, Crazyflie
with Crazyswarm2 avoids much of the reverse engineering. Lighthouse needs a
positioning deck on each compatible vehicle and installed base stations [7]; it
is not a software-only positioning upgrade for the current stock drones.

## Staged validation

1. Two powered stock drones, props removed: verify identity, packet routing and
   video isolation; measure delay/loss with both connections active.
2. External tracking: validate metric position, yaw, identity persistence,
   occlusion handling and end-to-end pose age over the intended volume.
3. One drone: identify response, close the position loop, demonstrate repeatable
   hover/path tracking and tested command/pose-loss recovery.
4. Two drones: separated hovers, then coordinated movement in distinct lanes;
   stagger takeoff/landing and verify separation under injected delays/dropouts.
5. Expand only after measured error and recovery behavior support it. Test shared
   world collisions in simulation and hardware downwash separately. Evaluate an
   ESP32 bridge on one drone before replicating modifications across the fleet.

No fleet-size, cost or schedule guarantee follows from the current evidence.
Tracking coverage and tail latency under full load determine the practical limit.

## Sources

1. Linux network namespaces: https://www.man7.org/linux/man-pages/man7/network_namespaces.7.html
2. Espressif ESP-NOW (ESP32-S3): https://docs.espressif.com/projects/esp-idf/en/stable/esp32s3/api-reference/network/esp_now.html
3. OpenCV ArUco detection and pose: https://docs.opencv.org/4.13.0/d5/dae/tutorial_aruco_detection.html
4. Crazyswarm2 scope: https://imrclab.github.io/crazyswarm2/
5. Crazyswarm2 configuration, positioning and simulation: https://imrclab.github.io/crazyswarm2/usage.html
6. Crazyswarm2 collision avoidance and trajectories: https://imrclab.github.io/crazyswarm2/howto.html
7. Bitcraze Lighthouse setup: https://www.bitcraze.io/documentation/tutorials/getting-started-with-lighthouse/
