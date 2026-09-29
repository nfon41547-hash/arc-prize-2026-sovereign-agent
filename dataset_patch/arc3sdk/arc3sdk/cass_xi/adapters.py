from __future__ import annotations
from collections import OrderedDict
from dataclasses import dataclass
from typing import Optional

@dataclass(frozen=True)
class AdapterConfig:
    rank: int = 8
    alpha: int = 16
    dropout: float = 0.0
    learning_rate: float = 2e-4
    train_steps: int = 8

@dataclass(frozen=True)
class MethodPlan:
    kind: str
    rank: int
    alpha: int
    target_modules: tuple[str, ...]
    train_steps: int
    reversible: bool = False

_TARGETS=("q_proj","k_proj","v_proj","o_proj")

def method_plan(method: str, cfg: AdapterConfig) -> MethodPlan:
    if cfg.rank <= 0 or cfg.train_steps <= 0:
        raise ValueError('rank and train_steps must be positive')
    if method == 'online-lora':
        return MethodPlan('peft-lora', cfg.rank, cfg.alpha, _TARGETS, cfg.train_steps, False)
    if method == 'cass-xi':
        return MethodPlan('reversible-fast-memory+lora', cfg.rank, cfg.alpha, _TARGETS, cfg.train_steps, True)
    raise ValueError(f'unsupported adaptive method: {method}')

class CausalGate:
    def __init__(self, write_threshold: float=0.5, risk_limit: float=0.2):
        self.write_threshold=float(write_threshold); self.risk_limit=float(risk_limit)
    def accept(self, utility: float, risk: float) -> bool:
        return float(utility) >= self.write_threshold and float(risk) <= self.risk_limit

class ReversibleFastMemory:
    def __init__(self, capacity: int=32):
        if capacity <= 0: raise ValueError('capacity must be positive')
        self.capacity=capacity; self._items=OrderedDict(); self._history={}; self._next=1
    def write(self, key: str, value: str, utility: float=1.0) -> int:
        token=self._next; self._next += 1
        previous=self._items.get(key)
        self._history[token]=(key, previous)
        self._items[key]=(value,float(utility),token)
        self._items.move_to_end(key)
        if len(self._items)>self.capacity:
            victim=min(self._items, key=lambda k:(self._items[k][1], self._items[k][2]))
            del self._items[victim]
        return token
    def read(self, key: str) -> Optional[str]:
        item=self._items.get(key)
        return None if item is None else item[0]
    def rollback(self, token: int) -> bool:
        event=self._history.pop(token,None)
        if event is None: return False
        key, previous=event
        current=self._items.get(key)
        if current is not None and current[2]==token:
            if previous is None: del self._items[key]
            else: self._items[key]=previous
        return True
    def context(self) -> str:
        return '; '.join(f'{k}={v[0]}' for k,v in self._items.items())
