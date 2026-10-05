"""V6 score-aware priority scheduler overlay.

Preserves the Daniel/TAAF public interface and adds an opt-in D' composition:
    P = A*M*C + B*phi
where A is immediate RHAE-like level value, M is measured pace multiplier,
C is completion-likelihood proxy from action/token spend, B is continuation
value, and phi fades future value near the shared run deadline.
"""
from __future__ import annotations
import math
import os
from dataclasses import dataclass

TAIL_LOOKUP_80K: dict[int, tuple[float, ...]] = {
    6: (3.9082328014, 5.0756939323, 5.3532912917, 5.6080111343, 4.0898867400, 0.0),
    7: (3.9262240771, 5.1669653335, 5.6943514417, 6.6500041602, 6.6562876217, 4.7715345300, 0.0),
    8: (3.9291260268, 5.1855629875, 5.7851444897, 7.0053702980, 7.8471367942, 7.7045641090, 5.4531823200, 0.0),
    9: (3.9295002558, 5.1885168225, 5.8035897108, 7.0958580947, 8.2469236991, 9.0442694281, 8.7528405964, 6.1348301100, 0.0),
    10: (3.9295397677, 5.1888936715, 5.8065473797, 7.1136691211, 8.3474656955, 9.4884771003, 10.2414020621, 9.8011170838, 6.8164779000, 0.0),
}
TAIL_LOOKUP_REMAINING: dict[int, tuple[float, ...]] = {
    n: (8.0,) * (n - 3) + (7.0, 5.0, 0.0) for n in range(6, 11)
}

def parse_total_levels(value: object) -> int | None:
    if value is None or isinstance(value, bool): return None
    try: number = float(value)
    except (TypeError, ValueError, OverflowError): return None
    if not math.isfinite(number) or number <= 0 or not number.is_integer(): return None
    return min(10, max(6, int(number)))

@dataclass
class ProgressPace:
    completed: int = 0
    log_cost_ratio: float = 0.0
    def record_completion(self, tokens_used: float, reference_tokens: float) -> None:
        observation = math.log(max(1.0, tokens_used) / max(1.0, reference_tokens))
        self.log_cost_ratio = observation if self.completed == 0 else 0.6*self.log_cost_ratio + 0.4*observation
        self.completed += 1
    def cost_multiplier(self) -> float:
        confidence = self.completed / (self.completed + 3.0)
        return math.exp(min(math.log(2.0), max(math.log(0.5), confidence*self.log_cost_ratio)))

@dataclass(frozen=True)
class PrioritySnapshot:
    level: int
    actions: int
    tokens: float
    cost_multiplier: float = 1.0
    total_levels: int | None = None

def _components(state: PrioritySnapshot, *, tail_fraction, human_actions, action_scale,
                token_scale, action_weight, tail_base, tail_cap, normalize_score,
                tail_lookup, tail_efficiency):
    level = max(1, state.level)
    total_levels = parse_total_levels(getattr(state, 'total_levels', None)) or 10
    actions, tokens = max(0, state.actions), max(0.0, state.tokens)
    pace = min(2.0, max(0.5, state.cost_multiplier))
    efficiency = (human_actions/(human_actions+actions))**2
    current_value = level*efficiency
    if normalize_score:
        current_value *= 55.0/(total_levels*(total_levels+1)/2)
    if tail_lookup == 'remaining':
        tail = TAIL_LOOKUP_REMAINING[total_levels][min(total_levels, level)-1]
    elif tail_lookup:
        fe = min(1.0, max(0.0, tail_efficiency)) if math.isfinite(tail_efficiency) else 0.8
        tail = TAIL_LOOKUP_80K[total_levels][min(total_levels, level)-1] * fe
    else:
        tail = min(tail_cap, tail_base + level - 1)
    phi = 1.0 if tail_fraction is None else min(1.0, max(0.0, tail_fraction))
    chance = action_weight*0.5**((actions/action_scale)**2) + (1-action_weight)*0.5**((tokens/(token_scale*pace))**2)
    return level, pace, efficiency, current_value, tail, phi, chance

def priority_value(state: PrioritySnapshot, *, endgame: bool=False, tail_fraction: float|None=None,
                   human_actions: float=25.0, action_scale: float=115.0, token_scale: float=62000.0,
                   action_weight: float=0.25, tail_base: float=6.0, tail_cap: float=6.0,
                   normalize_score: bool=False, tail_lookup: bool|str=False,
                   tail_efficiency: float=0.8) -> int:
    level, pace, efficiency, A, B, phi, C = _components(
        state, tail_fraction=tail_fraction, human_actions=human_actions,
        action_scale=action_scale, token_scale=token_scale, action_weight=action_weight,
        tail_base=tail_base, tail_cap=tail_cap, normalize_score=normalize_score,
        tail_lookup=tail_lookup, tail_efficiency=tail_efficiency)
    if endgame and tail_fraction is None:
        value = ((level*0.25 + A) if normalize_score else level*(0.25+efficiency))/pace*100
        return max(1, int(value))
    formula = os.environ.get('ARC3_PRIORITY_FORMULA','legacy').strip().lower()
    if formula in {'dprime','d-prime','affectify'}:
        # D': immediate bankable value is hazard/pace aware; continuation is
        # separately faded rather than being crushed by current-level hazard.
        M = 1.0/pace
        value = (A*M*C + B*phi)*100
    else:
        value = (A + B*phi)*C/pace*100
    return max(1, int(value))
