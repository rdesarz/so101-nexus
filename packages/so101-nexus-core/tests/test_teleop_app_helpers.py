"""Unit tests for pure helpers in so101_nexus_core.teleop.app.

The ``app`` module is designed to be importable on a base install; gradio,
lerobot, and cv2 are imported lazily inside ``main()`` and individual
callbacks. These tests exercise the pure-logic helpers that need no
gradio runtime.
"""

from __future__ import annotations

import sys
import types

import numpy as np
import pytest

from so101_nexus_core.teleop.app import (
    _build_field_selection,
    _cb_approve_episode,
    _cb_discard_episode,
    _cb_start_init,
    _connect_controller,
    _connect_leader,
    _create_dataset,
    _progress_text,
    _run_init_worker,
)
from so101_nexus_core.teleop.dataset import OVERHEAD_KEY, WRIST_KEY
from so101_nexus_core.teleop.recorder import RecordingState


class _FakeGradio(types.ModuleType):
    class Error(Exception):
        pass

    @staticmethod
    def Walkthrough(**kwargs):
        return {"Walkthrough": kwargs}

    @staticmethod
    def update(**kwargs):
        return {"update": kwargs}


class _Dataset:
    def __init__(self) -> None:
        self.frames: list[dict] = []
        self.clear_calls = 0
        self.save_calls = 0

    def add_frame(self, frame: dict) -> None:
        self.frames.append(frame)

    def save_episode(self) -> None:
        self.save_calls += 1

    def clear_episode_buffer(self) -> None:
        self.clear_calls += 1


def test_progress_text_formats_episode_count() -> None:
    assert _progress_text(2, 5) == "**Episode 2 / 5**"


def test_progress_text_formats_zero_completed() -> None:
    assert _progress_text(0, 10) == "**Episode 0 / 10**"


def test_discard_episode_shows_start_button_for_rerecord(monkeypatch) -> None:
    monkeypatch.setitem(sys.modules, "gradio", _FakeGradio("gradio"))

    state = RecordingState(num_episodes=2)
    state.episode_actions.append(np.array([1.0], dtype=np.float32))
    dataset = _Dataset()

    result = _cb_discard_episode({"state": state, "dataset": dataset})

    assert result[0] == {"Walkthrough": {"selected": 2}}
    assert result[1] == {
        "update": {"value": "Episode discarded. Ready to re-record. Click the button below."}
    }
    assert result[2] == {"update": {"visible": True}}
    assert result[3] == {"update": {"value": "**Episode 0 / 2**"}}
    assert dataset.clear_calls == 1
    assert state.episode_actions == []


def test_approve_episode_shows_start_button_for_next_recording(monkeypatch) -> None:
    monkeypatch.setitem(sys.modules, "gradio", _FakeGradio("gradio"))

    state = RecordingState(num_episodes=2)
    state.episode_actions.append(np.array([1.0], dtype=np.float32))
    state.episode_states.append(np.array([1.0], dtype=np.float32))
    dataset = _Dataset()
    session = {
        "state": state,
        "dataset": dataset,
        "action_space": "joint_pos",
        "field_selection": _build_field_selection([]),
    }

    result = _cb_approve_episode(session)

    assert result[0] == {"Walkthrough": {"selected": 2}}
    assert result[1] == {
        "update": {"value": "Episode saved! Ready to record the next one. Click the button below."}
    }
    assert result[2] == {"update": {"visible": True}}
    assert result[3] == {"update": {"value": "**Episode 1 / 2**"}}
    assert result[4] == {"update": {}}
    assert dataset.save_calls == 1
    assert len(dataset.frames) == 1


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


def test_start_init_callback_matches_gradio_input_order(monkeypatch) -> None:
    """The Gradio input list passes env_id before controller_type."""

    class _FakeGradio(types.ModuleType):
        class Error(Exception):
            pass

        @staticmethod
        def Walkthrough(**kwargs):
            return kwargs

    class _ImmediateThread:
        def __init__(self, target, args, daemon=False):
            self._target = target
            self._args = args
            self.daemon = daemon

        def start(self) -> None:
            self._target(*self._args)

    seen = {}

    def _fake_run_init_worker(
        _session,
        _init_state,
        _leader_port,
        controller_type,
        env_id,
        *_args,
    ) -> None:
        seen["controller_type"] = controller_type
        seen["env_id"] = env_id

    monkeypatch.setitem(sys.modules, "gradio", _FakeGradio("gradio"))
    monkeypatch.setattr("so101_nexus_core.teleop.app.threading.Thread", _ImmediateThread)
    monkeypatch.setattr(
        "so101_nexus_core.teleop.app._run_init_worker",
        _fake_run_init_worker,
    )

    result = _cb_start_init(
        {},
        {},
        "/dev/null",
        "default_leader",
        "MuJoCoReach-v1",
        "keyboard",
        "so101",
        "",
        30,
        64,
        64,
        64,
        64,
        "",
        1,
        "joint_pos",
        2,
        0,
        0.0,
        [],
    )

    assert result == {"selected": 1}
    assert seen == {"controller_type": "keyboard", "env_id": "MuJoCoReach-v1"}
