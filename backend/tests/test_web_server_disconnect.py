import asyncio
import unittest
from unittest import mock

from fastapi import WebSocketDisconnect

import web_server


class FakeWebSocket:
    """Delivers the given messages, then disconnects like a closed browser tab."""

    def __init__(self, messages):
        self._messages = list(messages)

    async def accept(self):
        pass

    async def receive_json(self):
        if self._messages:
            return self._messages.pop(0)
        raise WebSocketDisconnect()


class WebSocketDisconnectTests(unittest.TestCase):
    HELD_STICKS = {"type": "axes", "mode": "inc", "throttle": 1, "yaw": 0, "pitch": 1, "roll": 0}

    def _disconnect_with_sticks_held(self, plugin_manager=None):
        controller = mock.Mock()
        with mock.patch.object(web_server, "flight_controller", controller), \
                mock.patch.object(web_server, "plugin_manager", plugin_manager):
            asyncio.run(web_server.ws_endpoint(FakeWebSocket([self.HELD_STICKS])))
        return controller

    def test_closing_the_client_centres_the_sticks(self):
        controller = self._disconnect_with_sticks_held()

        controller.set_axes_from.assert_called_with("disconnect", 0, 0, 0, 0)

    def test_running_plugin_keeps_control_after_disconnect(self):
        plugins = mock.Mock()
        plugins.running.return_value = ["follow"]

        controller = self._disconnect_with_sticks_held(plugins)

        controller.set_axes_from.assert_not_called()


class SwarmCommandTests(unittest.TestCase):
    SWARM = {"type": "axes", "mode": "linear", "source": "swarm", "ttl": 0.3,
             "throttle": 0.1, "yaw": 0.0, "pitch": 0.2, "roll": -0.2}
    CONSOLE = {"type": "axes", "mode": "abs", "throttle": 1, "yaw": 0, "pitch": 1, "roll": 0}

    def setUp(self):
        web_server.swarm_claim_until = 0.0
        self.addCleanup(setattr, web_server, "swarm_claim_until", 0.0)

    def _run(self, messages):
        controller = mock.Mock()
        with mock.patch.object(web_server, "flight_controller", controller), \
                mock.patch.object(web_server, "plugin_manager", None), \
                mock.patch.object(web_server, "runner", None):
            asyncio.run(web_server.ws_endpoint(FakeWebSocket(messages)))
        return controller

    def test_swarm_axes_are_linear_and_expire(self):
        controller = self._run([self.SWARM])
        controller.set_axes_from.assert_any_call("swarm", 0.1, 0.0, 0.2, -0.2)
        controller.expire_axes_after.assert_called_once_with(0.3)
        strategy = controller.model.set_strategy.call_args[0][0]
        self.assertIsInstance(strategy, web_server.LinearStrategy)

    def test_swarm_disconnect_centres_the_sticks(self):
        controller = self._run([self.SWARM])
        controller.set_axes_from.assert_called_with("disconnect", 0, 0, 0, 0)

    def test_console_cannot_fly_while_the_swarm_holds_the_drone(self):
        web_server.swarm_claim_until = web_server.time.monotonic() + 30
        controller = self._run([self.CONSOLE, {"type": "takeoff"}])
        controller.model.takeoff.assert_not_called()
        for call in controller.set_axes_from.call_args_list:
            self.assertNotEqual(call.args[0], "frontend")

    def test_console_can_still_stop_while_the_swarm_holds_the_drone(self):
        web_server.swarm_claim_until = web_server.time.monotonic() + 30
        controller = self._run([{"type": "estop"}, {"type": "land"}])
        controller.model.emergency_stop.assert_called_once()
        controller.model.land.assert_called_once()


class AxesExpiryTests(unittest.TestCase):
    def test_sticks_centre_when_the_deadline_passes(self):
        from services.flight_controller import FlightController
        fc = FlightController(mock.Mock(), mock.Mock())
        fc.set_axes_from("swarm", 0.5, 0.5, 0.5, 0.5)
        fc.expire_axes_after(-1.0)
        fc._expire_axes()
        self.assertEqual((fc.throttle_dir, fc.yaw_dir, fc.pitch_dir, fc.roll_dir), (0, 0, 0, 0))

    def test_a_fresh_command_cancels_the_deadline(self):
        from services.flight_controller import FlightController
        fc = FlightController(mock.Mock(), mock.Mock())
        fc.set_axes_from("swarm", 0.5, 0.5, 0.5, 0.5)
        fc.expire_axes_after(-1.0)
        fc.set_axes_from("swarm", 0.4, 0.0, 0.0, 0.0)
        fc._expire_axes()
        self.assertEqual(fc.throttle_dir, 0.4)


if __name__ == "__main__":
    unittest.main()
