"""Unit tests for pure helpers in so101_nexus_core.teleop.app.

The ``app`` module is designed to be importable on a base install; gradio,
lerobot, and cv2 are imported lazily inside ``main()`` and individual
callbacks. These tests exercise the pure-logic helpers that need no
gradio runtime.
"""

from __future__ import annotations

import sys
import types

import pytest

from so101_nexus_core.teleop.app import (
    _build_field_selection,
    _connect_controller,
    _connect_leader,
    _create_dataset,
    _progress_text,
    _run_init_worker,
)
from so101_nexus_core.teleop.dataset import OVERHEAD_KEY, WRIST_KEY


def test_progress_text_formats_episode_count() -> None:
    assert _progress_text(2, 5) == "**Episode 2 / 5**"


def test_progress_text_formats_zero_completed() -> None:
    assert _progress_text(0, 10) == "**Episode 0 / 10**"


def test_build_field_selection_all_keys() -> None:
    selection = _build_field_selection([WRIST_KEY, OVERHEAD_KEY, "task"])

    assert selection.wrist_image is True
    assert selection.overhead_image is True
    assert selection.task is True


def test_build_field_selection_empty() -> None:
    selection = _build_field_selection([])

    assert selection.wrist_image is False
    assert selection.overhead_image is False
    assert selection.task is False


def test_build_field_selection_only_wrist() -> None:
    selection = _build_field_selection([WRIST_KEY])

    assert selection.wrist_image is True
    assert selection.overhead_image is False
    assert selection.task is False


def test_connect_leader_wraps_connect_failure_in_runtime_error(monkeypatch) -> None:
    """If leader.connect() raises, _connect_leader wraps it with a hint message."""

    class _FailingLeader:
        def connect(self) -> None:
            raise OSError("permission denied")

    def _fake_get_controller(_controller_type, _robot_type, _joint_names, _port, _leader_id):
        return _FailingLeader()

    monkeypatch.setattr("so101_nexus_core.teleop.app.get_controller", _fake_get_controller)

    with pytest.raises(RuntimeError, match="Failed to connect on /dev/ttyACM0") as excinfo:
        _connect_leader("so101", "/dev/ttyACM0", "leader_a")

    assert "lerobot-find-port" in str(excinfo.value)
    assert isinstance(excinfo.value.__cause__, OSError)


def test_connect_leader_returns_connected_leader_on_success(monkeypatch) -> None:
    """Happy path: _connect_leader returns the leader after a successful connect."""
    state = {"connected": False}

    class _OkLeader:
        def connect(self) -> None:
            state["connected"] = True

    monkeypatch.setattr(
        "so101_nexus_core.teleop.app.get_controller",
        lambda *_a, **_kw: _OkLeader(),
    )

    leader = _connect_leader("so101", "/dev/ttyACM0", "leader_a")

    assert state["connected"] is True
    assert isinstance(leader, _OkLeader)


def test_connect_controller_keyboard_success(monkeypatch) -> None:
    """The generalized controller path starts the selected controller."""
    state = {"connected": False}

    class _Keyboard:
        def connect(self) -> None:
            state["connected"] = True

    monkeypatch.setattr(
        "so101_nexus_core.teleop.app.get_controller",
        lambda *_a, **_kw: _Keyboard(),
    )

    controller = _connect_controller(
        "keyboard",
        "so101",
        ("shoulder_pan",),
        "/dev/ttyACM0",
        "leader_a",
    )

    assert state["connected"] is True
    assert isinstance(controller, _Keyboard)


def test_create_dataset_disconnects_leader_on_failure(monkeypatch) -> None:
    """If LeRobotDataset.create raises, the leader is disconnected and a RuntimeError is raised."""
    disconnect_calls = {"n": 0}

    class _StubLeader:
        def disconnect(self) -> None:
            disconnect_calls["n"] += 1

    class _RaisingDataset:
        @classmethod
        def create(cls, **_kwargs):
            raise ValueError("schema mismatch")

    fake_module = types.ModuleType("lerobot.datasets.lerobot_dataset")
    fake_module.LeRobotDataset = _RaisingDataset  # type: ignore[attr-defined]

    for name, mod in [
        ("lerobot", types.ModuleType("lerobot")),
        ("lerobot.datasets", types.ModuleType("lerobot.datasets")),
        ("lerobot.datasets.lerobot_dataset", fake_module),
    ]:
        monkeypatch.setitem(sys.modules, name, mod)

    leader = _StubLeader()
    with pytest.raises(RuntimeError, match="Failed to create dataset"):
        _create_dataset("local/test", 30, "so101", {}, leader)

    assert disconnect_calls["n"] == 1


def test_create_dataset_returns_dataset_on_success(monkeypatch) -> None:
    """Happy path: _create_dataset returns the LeRobotDataset instance."""
    seen = {}

    class _OkDataset:
        @classmethod
        def create(cls, **kwargs):
            seen.update(kwargs)
            return cls()

    fake_module = types.ModuleType("lerobot.datasets.lerobot_dataset")
    fake_module.LeRobotDataset = _OkDataset  # type: ignore[attr-defined]

    for name, mod in [
        ("lerobot", types.ModuleType("lerobot")),
        ("lerobot.datasets", types.ModuleType("lerobot.datasets")),
        ("lerobot.datasets.lerobot_dataset", fake_module),
    ]:
        monkeypatch.setitem(sys.modules, name, mod)

    class _StubLeader:
        def disconnect(self) -> None:
            pass

    ds = _create_dataset("local/test", 30, "so101", {"action": {}}, _StubLeader())

    assert isinstance(ds, _OkDataset)
    assert seen["repo_id"] == "local/test"
    assert seen["fps"] == 30
    assert seen["robot_type"] == "so101"
    assert seen["features"] == {"action": {}}


def test_run_init_worker_keyboard_initializes_session_without_hardware(monkeypatch) -> None:
    """Smoke-test init wiring for keyboard control without touching real devices."""

    class _Controller:
        def connect(self) -> None:
            pass

        def disconnect(self) -> None:
            pass

    class _Dataset:
        pass

    monkeypatch.setattr("so101_nexus_core.teleop.app.import_backend_for_env_id", lambda _env_id: None)
    monkeypatch.setattr(
        "so101_nexus_core.teleop.app._connect_controller",
        lambda *_args, **_kwargs: _Controller(),
    )
    monkeypatch.setattr(
        "so101_nexus_core.teleop.app._create_dataset",
        lambda *_args, **_kwargs: _Dataset(),
    )

    session: dict = {}
    init_state: dict = {}
    _run_init_worker(
        session,
        init_state,
        leader_port="/dev/null",
        controller_type="keyboard",
        env_id="MuJoCoReach-v1",
        robot_type="so101",
        leader_id="unused",
        fps=30,
        wrist_wh=(64, 64),
        overhead_wh=(64, 64),
        repo_id="local/test",
        num_episodes=1,
        action_space="joint_pos",
        max_steps=2,
        countdown=0,
        wrist_roll_offset_deg=0.0,
        field_selection=_build_field_selection([]),
    )

    assert init_state["done"] is True
    assert init_state.get("error") is None
    assert session["controller_type"] == "keyboard"
    assert session["env_id"] == "MuJoCoReach-v1"
    assert isinstance(session["dataset"], _Dataset)
