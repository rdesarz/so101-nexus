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
    _connect_leader,
    _create_dataset,
    _progress_text,
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

    def _fake_get_leader(_robot_type, _port, _leader_id):
        return _FailingLeader()

    monkeypatch.setattr("so101_nexus_core.teleop.app.get_leader", _fake_get_leader)

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
        "so101_nexus_core.teleop.app.get_leader",
        lambda *_a, **_kw: _OkLeader(),
    )

    leader = _connect_leader("so101", "/dev/ttyACM0", "leader_a")

    assert state["connected"] is True
    assert isinstance(leader, _OkLeader)


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
