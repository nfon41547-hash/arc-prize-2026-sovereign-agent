"""Validated game/level abstraction registry.

This is a compact prior compiled from the team's real 25-game rune manifest.
It is advisory only: observed transitions and the live environment retain veto
power, so a manifest label can never directly fire an action.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class AbstractSkill:
    game_id: str
    archetype: str
    level_count: int
    grid_size: tuple[int, int]
    tags: tuple[str, ...]
    data_keys: tuple[str, ...]
    evidence_source: str


class AbstractionSkillRegistry:
    """Read-only, provenance-carrying abstraction lookup."""

    def __init__(self, skills: dict[str, AbstractSkill], *, source: str) -> None:
        self._skills = dict(skills)
        self.source = source

    @classmethod
    def from_manifest(cls, path: str | Path) -> AbstractionSkillRegistry:
        manifest_path = Path(path)
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        games = payload.get("games")
        if not isinstance(games, list):
            raise ValueError("rune manifest must contain a games list")
        skills: dict[str, AbstractSkill] = {}
        for game in games:
            game_id = str(game["id"])
            if game_id in skills:
                raise ValueError(f"duplicate game id: {game_id}")
            size = tuple(int(v) for v in game.get("grid_size", (0, 0)))
            if len(size) != 2 or min(size) <= 0:
                raise ValueError(f"invalid grid_size for {game_id}")
            skills[game_id] = AbstractSkill(
                game_id=game_id,
                archetype=str(game.get("archetype", "unknown")),
                level_count=int(game["n_levels"]),
                grid_size=(size[0], size[1]),
                tags=tuple(str(x) for x in game.get("tags", ())),
                data_keys=tuple(str(x) for x in game.get("data_keys", ())),
                evidence_source=str(manifest_path),
            )
        return cls(skills, source=str(manifest_path))

    def get(self, game_id: str) -> AbstractSkill | None:
        return self._skills.get(str(game_id))

    def summary(self) -> dict[str, Any]:
        archetypes: dict[str, int] = {}
        levels = 0
        for skill in self._skills.values():
            archetypes[skill.archetype] = archetypes.get(skill.archetype, 0) + 1
            levels += skill.level_count
        return {
            "games": len(self._skills),
            "levels": levels,
            "archetypes": dict(sorted(archetypes.items())),
            "source": self.source,
            "advisory_only": True,
        }

    def as_skill(self, game_id: str, level: int) -> dict[str, Any] | None:
        """Return a bounded prompt/tool skill; never returns an action."""
        item = self.get(game_id)
        if item is None or not 0 <= int(level) < item.level_count:
            return None
        return {
            "game_id": item.game_id,
            "archetype": item.archetype,
            "level": int(level),
            "grid_size": list(item.grid_size),
            "tags": list(item.tags),
            "advisory_only": True,
        }
