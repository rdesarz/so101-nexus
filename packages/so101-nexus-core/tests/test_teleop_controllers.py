"""Tests for teleop controller abstractions."""

from __future__ import annotations

import numpy as np
import pytest

from so101_nexus_core.config import SO101_JOINT_NAMES
from so101_nexus_core.teleop.controllers import KeyboardController, get_controller
from so101_nexus_core.teleop.recorder import convert_controller_action


def test_keyboard_controller_emits_leader_style_joint_readings() -> None:
    controller = KeyboardController(SO101_JOINT_NAMES, step_deg=5.0)

    controller.apply_key("q")
    controller.apply_key("w")
    controller.apply_key("s")

    action = controller.get_action()
    assert set(action) == {f"{name}.pos" for name in SO101_JOINT_NAMES}
    assert action["shoulder_pan.pos"] == 5.0
    assert action["shoulder_lift.pos"] == 0.0


def test_keyboard_controller_reset_key_zeros_all_joints() -> None:
    controller = KeyboardController(SO101_JOINT_NAMES, step_deg=5.0)

    controller.apply_key("q")
    controller.apply_key("y")
    controller.apply_key("space")

    assert all(value == 0.0 for value in controller.get_action().values())


def test_keyboard_controller_action_converts_with_existing_recorder_path() -> None:
    controller = KeyboardController(SO101_JOINT_NAMES, step_deg=90.0)
    controller.apply_key("q")

    converted = convert_controller_action(
        controller.get_action(),
        SO101_JOINT_NAMES,
        wrist_roll_offset_deg=0.0,
    )

    assert converted.shape == (len(SO101_JOINT_NAMES),)
    np.testing.assert_allclose(converted[0], np.pi / 2)


def test_get_controller_returns_keyboard_controller() -> None:
    controller = get_controller(
        "keyboard",
        robot_type="so101",
        joint_names=SO101_JOINT_NAMES,
        leader_port="/dev/null",
        leader_id="unused",
    )

    assert isinstance(controller, KeyboardController)


def test_get_controller_gamepad_placeholder() -> None:
    with pytest.raises(NotImplementedError, match="Gamepad control"):
        get_controller(
            "gamepad",
            robot_type="so101",
            joint_names=SO101_JOINT_NAMES,
            leader_port="/dev/null",
            leader_id="unused",
        )
