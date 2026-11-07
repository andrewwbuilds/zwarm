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


if __name__ == "__main__":
    unittest.main()
