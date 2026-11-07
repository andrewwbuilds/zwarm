# S20 & S29 Controller model

from models.base_rc import BaseRCModel
from models.control_profile import ControlProfile
from control.strategies import IncrementalStrategy
from models.stick_range import StickRange

class S2xDroneModel(BaseRCModel):
    """Model for S2x protocol drones (S20, S29)"""
    
    STICK_RANGE = StickRange(60, 128, 200)   # ← tailorable per drone

    # Control packets that carry each E-stop press (0.25 s at 80 Hz), so one lost
    # UDP packet can't silently cancel a stop.
    ESTOP_PACKETS = 20

    PRESETS = {
        "normal":     ControlProfile("normal",     2.08, 4.86, 0.5, 0.02),
        "precise":    ControlProfile("precise",    1.39, 5.56, 0.3, 0.01),
        "aggressive": ControlProfile("aggressive", 4.17, 3.89, 1.5, 0.11),
    }

    def __init__(self, profile: str | ControlProfile = "normal"):
        # BaseRCModel handles STICK_RANGE, presets and profile application
        super().__init__(stick_range=self.STICK_RANGE, profile=profile)

        self.strategy = IncrementalStrategy()   # default

        # one-shot flags
        self.takeoff_flag = False
        self.land_flag = False
        self.stop_flag = False
        self.stop_packets_left = 0
        self.headless_flag = False
        self.calibration_flag = False

        # Takeoff and land share one toggle bit and the drone reports nothing back,
        # so this tracks what was last sent. It is a belief, not a measurement: a
        # takeoff the drone ignored, or a crash, leaves it wrong until corrected.
        self.airborne = False

        # misc
        self.speed = 20    # matches 0x14 from dumps
        self.speed_index = 2

        # Track last direction for each axis
        self.last_throttle_dir = 0
        self.last_yaw_dir = 0
        self.last_pitch_dir = 0
        self.last_roll_dir = 0
    
    def update(self, dt, axes):
        self.strategy.update(self, dt, axes)
    
    def takeoff(self) -> bool:
        """Send takeoff unless the drone is already presumed airborne.

        The bit is a toggle, so a second takeoff would land the drone. Returns
        whether the command went out.
        """
        if self.airborne:
            return False
        self.takeoff_flag = True
        self.airborne = True
        return True

    def land(self) -> bool:
        """
        Request the stock app's land action, unless the drone is presumed grounded.

        In the HiTurbo app this shares the same one-shot flight bit as takeoff,
        so the protocol adapter maps the semantic intent rather than a distinct
        dedicated land bit. Sent to a grounded drone, it would take off.
        Returns whether the command went out.
        """
        if not self.airborne:
            return False
        self.land_flag = True
        self.airborne = False
        return True

    def emergency_stop(self):
        """Hold the emergency stop flag for the next ESTOP_PACKETS packets."""
        self.stop_flag = True
        self.stop_packets_left = self.ESTOP_PACKETS
        self.airborne = False

    def set_speed_index(self, speed_index: int) -> None:
        """Set the Macrochip app speed tier: 0=low, 1=medium, 2=full."""
        self.speed_index = max(0, min(2, int(speed_index)))
    
    def get_control_state(self):
        """Get current control state as a dict"""
        return {
            "throttle":  self.throttle,
            "yaw":       self.yaw,
            "pitch":     self.pitch,
            "roll":      self.roll,
            "speed_index": self.speed_index,
        }

    def set_strategy(self, strategy) -> None:
        self.strategy = strategy
