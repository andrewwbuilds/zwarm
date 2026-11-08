"""Turns what an operator says into one action, or into nothing.

Fixed phrases, on purpose: this drives real aircraft, so a sentence it doesn't recognise must do
nothing rather than a best guess. "stop" holds position. Only an explicit phrase cuts the motors.
Speech recognition, when added, feeds text in here and gets the same rules.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

ALIASES = {
    "bag": "backpack", "rucksack": "backpack", "sofa": "couch", "human": "person", "someone": "person",
    "somebody": "person", "anyone": "person", "anybody": "person", "man": "person", "woman": "person", "people": "person",
    "plant": "potted plant", "phone": "cell phone", "table": "dining table", "television": "tv", "ball": "sports ball",
    "teddy": "teddy bear", "backpacks": "backpack", "chairs": "chair", "bottles": "bottle", "cups": "cup",
}
EXPLANATION = ('Try: "take off", "find the backpack", "where is the chair", "what have you seen", "go to 1 2", '
               '"drone a go to the chair", "line up", "status", "forget the box", "stop", "land", "stop and land", "land here", '
               'or "emergency stop".')

_ESTOP = re.compile(r"(emergency stop|e stop|estop|kill (the )?motors|cut (the )?motors)")
_TAKEOFF = re.compile(r"(take ?off|lift off|launch)")
# Said before a landing: "stop and land", "stop mapping and land", "wrap it up and bring them home".
_WORK = r"(?:(?:the )?(?:mission|mapping|map|exploring|exploration|inspecting|inspection|searching|search|scanning|work)|everything|that|this|it)"
_THEN = rf"(?:(?:stop|end|wrap up|wrap it up)(?: {_WORK})?(?: and| then)? )?"  # not "finish": that may mean do the work first
_THEM = r"(?:them|them all|all of them|it|everyone|everybody|everything|all|the drones|all the drones|all drones|the fleet|the swarm|the team)"
_LAND_HERE = re.compile(rf"{_THEN}(?:land (?:here|now|where you are|right here|right now)|put {_THEM} down (?:here|right here|where they are|right where they are))")
_LAND = re.compile(rf"{_THEN}(?:land|come back|come home|go home|return|return home|return to base|come down|get down|go down|touch down"
                   rf"|(?:come|go|fly|head|get) (?:back|home)(?: and| then)? land|return(?: and| then)? land"
                   rf"|(?:bring|take|put|get|set) {_THEM}(?: back)? (?:down|home)|bring {_THEM} (?:back|in)|bring {_THEM} back(?: and| then)? land)")
_HOLD = re.compile(r"(stop|stop searching|stop looking|hold|hold position|wait|stay|stay there|freeze|pause|cancel|abort)")
_STATUS = re.compile(r"(status|report|how are you|what are you doing|where are you|who is flying)")
_LIST = re.compile(r"(what have you (seen|found)|what did you (see|find)|what do you (see|know|remember)|"
                   r"what is in the room|whats in the room|list everything)")
_HELP = re.compile(r"(help|what can you do|commands)")
_FIND = re.compile(r"(?:go )?(?:find|search for|look for|locate|hunt for)\s+(.+)")
_WHERE = re.compile(r"(?:where is|wheres|where are|have you seen|did you see|do you see|is there|are there)\s+(.+)")
_FORGET_ALL = re.compile(r"forget (everything|it all|all of it)")
_FORGET = re.compile(r"forget\s+(.+)")
_FORM = re.compile(r"(?:form|make|get in|get into)\s+(?:an? )?(line|ring|circle|grid)")
_NUMBER = r"(-?\d+(?:\.\d+)?)"
_GO_POINT = re.compile(rf"(?:go|fly|move) to\s+{_NUMBER}\s+{_NUMBER}(?:\s+{_NUMBER})?")
_GO_THING = re.compile(r"(?:go|fly|move) to\s+(.+)")
_ADDRESS = re.compile(r"(?:drone\s+)?([a-z0-9]+)\s*,\s*(.+)|drone\s+([a-z0-9]+)\s+(.+)")


@dataclass(frozen=True)
class Command:
    verb: str
    label: str | None = None
    drone: str | None = None
    point: tuple[float, ...] | None = None
    shape: str | None = None
    problem: str | None = None


def clean_label(words: str) -> str:
    words = re.sub(r"^(?:(?:the|a|an|my|our|your|any|some|that|this)\s+)+", "", words.strip())
    words = re.sub(r"\s+(?:please|now|for me|in the room|around here)$", "", words).strip()
    return ALIASES.get(words, words)


def _normal(text: str) -> str:
    text = text.lower().strip()
    text = text.replace("'", "")
    # Keep minus signs and decimal points on numbers; everything else that isn't a word goes.
    text = re.sub(r"(?<![\d\s])-|-(?!\d)", " ", text)
    text = re.sub(r"\.(?!\d)", " ", text)
    text = re.sub(r"[^a-z0-9,.\- ]+", " ", text)
    text = re.sub(r"\s*,\s*", ", ", text)
    text = re.sub(r"\s+", " ", text).strip(" ,")
    text = re.sub(r"^(please|ok|okay|hey|now),?\s+", "", text)
    text = re.sub(r"\s+please$", "", text)
    return text


def parse(text: str, drones=()) -> Command:
    """One command from one sentence. An unrecognised sentence gives the verb 'unknown'."""
    said = _normal(text)
    if not said:
        return Command("unknown")
    # "a, go to 1, 2" names a drone, but "go to 1, 2" is only a list of coordinates.
    addressed = _ADDRESS.fullmatch(said)
    drone = None
    if addressed:
        name, rest = (addressed.group(1), addressed.group(2)) if addressed.group(1) else (addressed.group(3), addressed.group(4))
        if name in drones or said.startswith("drone "):
            drone, said = name, rest
    flat = said.replace(",", " ")
    flat = re.sub(r"\s+", " ", flat).strip()

    if _ESTOP.fullmatch(flat):
        return _fleet_only(Command("estop"), drone)
    if _LAND_HERE.fullmatch(flat):
        return _fleet_only(Command("land_here"), drone)
    if _TAKEOFF.fullmatch(flat):
        return _fleet_only(Command("takeoff"), drone)
    if _LAND.fullmatch(flat):
        return _fleet_only(Command("land"), drone)
    if _HOLD.fullmatch(flat):
        return _fleet_only(Command("hold"), drone)
    if _STATUS.fullmatch(flat):
        return Command("status")
    if _LIST.fullmatch(flat):
        return Command("list")
    if _HELP.fullmatch(flat):
        return Command("help")
    if _FORGET_ALL.fullmatch(flat):
        return Command("forget")
    if match := _GO_POINT.fullmatch(flat):
        return Command("goto", drone=drone, point=tuple(float(v) for v in match.groups() if v is not None))
    if match := _GO_THING.fullmatch(flat):
        return Command("goto", drone=drone, label=clean_label(match.group(1)))
    if match := _FORM.fullmatch(flat):
        return _fleet_only(Command("form", shape="ring" if match.group(1) == "circle" else match.group(1)), drone)
    if flat in ("line up", "spread out"):
        return _fleet_only(Command("form", shape="line" if flat == "line up" else "grid"), drone)
    if match := _FIND.fullmatch(flat):
        return _fleet_only(Command("find", label=clean_label(match.group(1))), drone)
    if match := _WHERE.fullmatch(flat):
        return Command("where", label=clean_label(match.group(1)))
    if match := _FORGET.fullmatch(flat):
        return Command("forget", label=clean_label(match.group(1)))
    return Command("unknown")


def _fleet_only(command: Command, drone: str | None) -> Command:
    """These act on every drone. Naming one must not quietly land the whole fleet."""
    if drone is None:
        return command
    return Command("refused", problem=f'"{command.verb}" applies to the whole fleet, so I did nothing. Say it without naming a drone.')
