"""Teleoperation controller factories.

Controllers expose the same small interface as the physical leader arm so
the recorder can save identical LeRobot v3 frames regardless of input source.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from threading import Lock
from typing import Literal, Protocol

from so101_nexus_core.teleop.leader import LeaderProtocol, get_leader

ControllerKind = Literal["leader", "keyboard", "gamepad"]

KEYBOARD_HELP = (
    "Keyboard controls: q/a shoulder_pan, w/s shoulder_lift, e/d elbow_flex, "
    "r/f wrist_flex, t/g wrist_roll, y/h gripper, space resets."
)

DEFAULT_KEY_BINDINGS: dict[str, tuple[str, float]] = {
    "q": ("shoulder_pan", 1.0),
    "a": ("shoulder_pan", -1.0),
    "w": ("shoulder_lift", 1.0),
    "s": ("shoulder_lift", -1.0),
    "e": ("elbow_flex", 1.0),
    "d": ("elbow_flex", -1.0),
    "r": ("wrist_flex", 1.0),
    "f": ("wrist_flex", -1.0),
    "t": ("wrist_roll", 1.0),
    "g": ("wrist_roll", -1.0),
    "y": ("gripper", 1.0),
    "h": ("gripper", -1.0),
}


class ControllerProtocol(Protocol):
    """Minimal interface required by the teleop recorder."""

    def connect(self) -> None:
        """Prepare the controller for use."""
        ...

    def disconnect(self) -> None:
        """Release controller resources."""
        ...

    def get_action(self) -> dict:
        """Return joint readings in degrees using ``<joint>.pos`` keys."""
        ...


@dataclass
class KeyboardController:
    """Keyboard-driven joint-position controller.

    The controller keeps an absolute joint target in degrees. Key presses
    nudge one joint by ``step_deg`` so downstream conversion and dataset
    writing can stay identical to the leader-arm path.
    """

    joint_names: tuple[str, ...]
    step_deg: float = 2.0
    min_deg: float = -180.0
    max_deg: float = 180.0
    key_bindings: dict[str, tuple[str, float]] = field(default_factory=lambda: dict(DEFAULT_KEY_BINDINGS))

    def __post_init__(self) -> None:
        self._positions = {name: 0.0 for name in self.joint_names}
        self._lock = Lock()
        self._listener = None

    def connect(self) -> None:
        """Start listening for keyboard events."""
        try:
            from pynput import keyboard
        except ImportError as exc:  # pragma: no cover - depends on optional teleop extra
            raise RuntimeError("Keyboard control requires the optional 'pynput' dependency.") from exc

        self._listener = keyboard.Listener(on_press=self._on_press)
        self._listener.start()
        print(KEYBOARD_HELP)

    def disconnect(self) -> None:
        """Stop listening for keyboard events."""
        if self._listener is not None:
            self._listener.stop()
            self._listener = None

    def get_action(self) -> dict:
        """Return the current joint-position target as leader-style readings."""
        with self._lock:
            return {f"{name}.pos": float(self._positions[name]) for name in self.joint_names}

    def reset_positions(self) -> None:
        """Reset all keyboard joint targets to their home position."""
        with self._lock:
            for name in self.joint_names:
                self._positions[name] = 0.0

    def apply_key(self, key_name: str) -> None:
        """Apply one logical key press; useful for tests and non-pynput callers."""
        key_name = key_name.lower()
        if key_name == "space":
            self.reset_positions()
            return

        binding = self.key_bindings.get(key_name)
        if binding is None:
            return

        joint_name, direction = binding
        if joint_name not in self._positions:
            return

        with self._lock:
            value = self._positions[joint_name] + direction * self.step_deg
            self._positions[joint_name] = min(max(value, self.min_deg), self.max_deg)

    def _on_press(self, key) -> None:
        key_name = getattr(key, "char", None)
        if key_name is None:
            key_name = getattr(key, "name", "")
        self.apply_key(str(key_name))


def get_controller(
    controller: ControllerKind,
    robot_type: str,
    joint_names: tuple[str, ...],
    leader_port: str,
    leader_id: str,
) -> ControllerProtocol | LeaderProtocol:
    """Create a controller without connecting it."""
    if controller == "leader":
        return get_leader(robot_type, leader_port, leader_id)
    if controller == "keyboard":
        return KeyboardController(joint_names)
    if controller == "gamepad":
        raise NotImplementedError("Gamepad control is not implemented yet.")
    raise ValueError(f"Unknown controller: {controller}")
