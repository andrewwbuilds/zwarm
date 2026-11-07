"""The 2.4 GHz WiFi link between the drone and the Mac."""

from __future__ import annotations

import math
import random

# measured: the Mac saw -38 and -47 dBm with the drone close by.
RSSI_AT_1M_DBM = -42.0  # estimate, fitted to those readings
PATH_LOSS_EXPONENT = 2.7  # estimate: typical indoors or near the ground
# spec: 50 m of video range and 100 m of control range. These midpoints reproduce both.
VIDEO_MIDPOINT_DBM = -88.0
CONTROL_MIDPOINT_DBM = -96.0
LOSS_SOFTNESS_DB = 2.0  # estimate


class RadioLink:
    def __init__(self, rng: random.Random):
        self.rng = rng
        self.pilot_distance_m = 1.5  # estimate: where the Mac sits relative to the takeoff spot
        self.offset_m = 0.0  # how far the drone has flown from the takeoff spot
        self.enabled = True
        self.extra_loss = 0.0

    def distance_m(self) -> float:
        return math.hypot(self.pilot_distance_m, self.offset_m)

    def rssi_dbm(self) -> float:
        return RSSI_AT_1M_DBM - 10.0 * PATH_LOSS_EXPONENT * math.log10(max(self.distance_m(), 0.1))

    def loss_probability(self, kind: str) -> float:
        if not self.enabled:
            return 1.0
        midpoint = VIDEO_MIDPOINT_DBM if kind == "video" else CONTROL_MIDPOINT_DBM
        margin = self.rssi_dbm() - midpoint
        loss = 1.0 / (1.0 + math.exp(margin / LOSS_SOFTNESS_DB))
        return min(1.0, loss + self.extra_loss)

    def one_way_delay_s(self) -> float:
        """measured: ping to the drone took 3.2 to 27 ms, 11.7 ms on average."""
        return 0.0015 + self.rng.lognormvariate(math.log(0.004), 0.7)

    def transmit(self, kind: str) -> float | None:
        """How long a packet takes to arrive, or None if it is lost."""
        if self.rng.random() < self.loss_probability(kind):
            return None
        return self.one_way_delay_s()
