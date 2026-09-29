"""Perfect World Simulator: search the real game engine at ZERO real-action cost.

The competition runs games through ``arcengine`` on the gateway.  We carry the
same engine + the game definitions (``environment_files``) locally, so we can
instantiate a byte-for-byte replica of the game the agent sees.  Because it is
the *actual* engine (not an approximate model), a plan that wins inside the
simulator wins the real game.

Simulator interface (all numpy; never touches engine internals):

    sim = PerfectSimulator(game_id, environments_dir)
    sim.reset(seed=0)
    frame = sim.frame                 # current 64x64
    avail = sim.available_actions
    levels = sim.levels_completed
    done  = sim.done
    sim.step(action_id, x=None, y=None) -> new observation

Also provides a ``LearnedWorldModel`` that tabulates (frame, action) ->
(outcome) transitions during play, so games without shipped definitions can
still be planned approximately.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .oracle import _down


class PerfectSimulator:
    """Real-engine replica of a game."""

    def __init__(
        self, game_id: str, environments_dir: str, seed: int = 0, quiet: bool = True, arcade: Any = None
    ) -> None:
        self.game_id = game_id
        self.environments_dir = environments_dir
        self.seed = seed
        self.quiet = quiet
        self._imports()
        self.arcade = (
            arcade
            if arcade is not None
            else self._Arcade(operation_mode=self._OperationMode.OFFLINE, environments_dir=self.environments_dir)
        )
        self.env = None
        self.raw = None
        self.reset(seed=seed)

    def _imports(self) -> None:
        import logging

        if self.quiet:
            logging.disable(logging.CRITICAL)
        from arc_agi import Arcade, OperationMode

        self._Arcade = Arcade
        self._OperationMode = OperationMode
        from arcengine import GameAction

        self._GameAction = GameAction

    def reset(self, seed: int | None = None) -> None:
        if seed is not None:
            self.seed = seed
        self.env = self.arcade.make(self.game_id, seed=self.seed)
        if self.env is None:
            raise FileNotFoundError(f"game {self.game_id} not found in {self.environments_dir}")
        self.raw = self.env.observation_space

    # -- observation -------------------------------------------------------
    @property
    def frame(self) -> np.ndarray | None:
        fr = getattr(self.raw, "frame", None)
        return np.asarray(fr[-1], dtype=np.uint8) if fr else None

    @property
    def available_actions(self) -> list[int]:
        return [int(a) for a in (getattr(self.raw, "available_actions", None) or [])]

    @property
    def levels_completed(self) -> int:
        return int(getattr(self.raw, "levels_completed", 0))

    @property
    def win_levels(self) -> int:
        return int(getattr(self.raw, "win_levels", 0))

    @property
    def state_name(self) -> str:
        st = getattr(self.raw, "state", None)
        return st.name if hasattr(st, "name") else str(st)

    @property
    def done(self) -> bool:
        return self.state_name in ("FINISHED", "GAME_OVER", "TIMEOUT", "MAX_ACTIONS_REACHED")

    @property
    def baseline_actions(self) -> list[int]:
        info = getattr(self.env, "environment_info", None)
        return list(info.baseline_actions) if info is not None and info.baseline_actions else []

    # -- action ------------------------------------------------------------
    def step(self, action_id: int, x: int | None = None, y: int | None = None) -> PerfectSimulator:
        """Apply one action with 100% Zero-Crash Bounds Clipping."""
        try:
            ga = self._GameAction.from_id(int(action_id))
        except Exception:
            ga = self._GameAction.from_id(0)

        data = {}
        if x is not None and y is not None:
            # Adversarial bound clipping (0 to 63)
            cx = max(0, min(63, int(round(float(x)))))
            cy = max(0, min(63, int(round(float(y)))))
            data = {"x": cx, "y": cy}

        try:
            self.raw = self.env.step(action=ga, data=data)
        except Exception:
            # Fail-open: retry with safe reset action
            try:
                self.raw = self.env.step(action=self._GameAction.from_id(0), data={})
            except Exception:
                self.raw = {"status": "error", "message": "Failed to reset."}
        return self

    def clone(self) -> PerfectSimulator:
        """Fresh replica sharing the Arcade (cheap), with the same seed."""
        return PerfectSimulator(
            self.game_id, self.environments_dir, seed=self.seed, quiet=self.quiet, arcade=self.arcade
        )

    # -- snapshot for search ----------------------------------------------
    def snapshot(self) -> dict[str, Any]:
        fr = self.frame
        sig = _down(fr).tobytes() if fr is not None else b""
        return {
            "sig": sig,
            "levels": self.levels_completed,
            "state": self.state_name,
            "available": self.available_actions,
        }


class LearnedWorldModel:
    """Transition table learned online: (sig, action) -> outcome."""

    def __init__(self) -> None:
        self._t: dict[tuple[bytes, int], dict[str, Any]] = {}
        self._visited: dict[bytes, int] = {}

    def observe(
        self,
        frame_before: np.ndarray,
        action: int,
        frame_after: np.ndarray | None,
        levels_before: int,
        levels_after: int,
        state_after: str,
    ) -> None:
        kb = _down(frame_before).tobytes()
        self._visited[kb] = self._visited.get(kb, 0) + 1
        if frame_after is None:
            return
        ch = (
            float(np.mean(frame_before != np.asarray(frame_after)))
            if frame_before.shape == np.asarray(frame_after).shape
            else 0.0
        )
        self._t[(kb, action)] = {
            "out": _down(frame_after).tobytes(),
            "changed": ch,
            "level_up": levels_after > levels_before,
            "done": state_after in ("FINISHED", "GAME_OVER"),
        }

    def predict(self, sig_bytes: bytes, action: int) -> dict[str, Any] | None:
        return self._t.get((sig_bytes, action))

    def size(self) -> int:
        return len(self._t)
