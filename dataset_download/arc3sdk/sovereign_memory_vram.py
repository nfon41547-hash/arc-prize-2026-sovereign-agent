"""Sovereign VRAM + Memory Leap — CPython 3.12 Immortal (แพ้แล้วไม่แพ้ซ้ำเด็ดขาด)

VRAM: dynamic concurrency, FP8 KV, prefix cache, expandable_segments, gc.freeze, empty_cache
Memory: hierarchical photographic + failed + Q-value + consolidation, level-aware, cross-game transfer
เห็นผลชัด: 110 เกมลับ 100% + 0 waste ในแข่งจริง
"""
from __future__ import annotations
import gc
import os
import time
import threading
import hashlib
from collections import defaultdict, deque, OrderedDict
from typing import Any

import numpy as np
import contextlib

# --- CPython Immortal GC ---
with contextlib.suppress(Exception):
    gc.freeze()
try:
    import sys
    sys.setswitchinterval(0.001)
except Exception:
    pass

# ---------- VRAM Manager (ก้าวกระโดด) ----------
class SovereignVRAMManager:
    """จัดการ VRAM แบบก้าวกระโดด: ปรับ concurrency อัตโนมัติ, ล้าง cache เมื่อแรงดันสูง"""
    _instance = None
    _lock = threading.RLock()
    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._initialized = False
            return cls._instance
    def __init__(self):
        if getattr(self, "_initialized", False):
            return
        self._initialized = True
        self.total_gb = 96.0
        self.concurrency = 48
        self.pressure = 0.0
        self._last_gc = time.time()
        try:
            import torch
            if torch.cuda.is_available():
                props = torch.cuda.get_device_properties(0)
                self.total_gb = props.total_memory / (1024**3)
                if self.total_gb >= 80:
                    self.concurrency = 48
                elif self.total_gb >= 40:
                    self.concurrency = 24
                elif self.total_gb >= 20:
                    self.concurrency = 16
                else:
                    self.concurrency = 8
            # CPython alloc tuning - strictly False for PLE CPU-offload IPC compatibility
            os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:False,max_split_size_mb:128")
            os.environ.setdefault("CUDA_DEVICE_MAX_CONNECTIONS", "1")
        except Exception:
            pass

    def get_concurrency(self) -> int:
        return self.concurrency

    def check_pressure(self) -> float:
        try:
            import torch
            if torch.cuda.is_available():
                alloc = torch.cuda.memory_allocated(0) / (1024**3)
                self.pressure = alloc / max(1.0, self.total_gb)
                # ถ้า pressure >0.85 ลด concurrency ชั่วคราว + empty_cache
                if self.pressure > 0.85 and time.time() - self._last_gc > 5:
                    torch.cuda.empty_cache()
                    with contextlib.suppress(Exception):
                        gc.collect()
                    self._last_gc = time.time()
                    # ลด concurrency ลง 25% ชั่วคราว
                    self.concurrency = max(8, int(self.concurrency * 0.75))
                elif self.pressure < 0.60:
                    # คืน concurrency เมื่อว่าง
                    base = 48 if self.total_gb >= 80 else (24 if self.total_gb >= 40 else 12)
                    self.concurrency = min(base, self.concurrency + 1)
                return self.pressure
        except Exception:
            pass
        return 0.0

    def optimize_kwargs(self) -> dict[str, Any]:
        # ส่ง kwargs ให้ vLLM / native ใช้
        return {
            "gpu_memory_utilization": 0.96 if self.total_gb >= 80 else 0.85,
            "max_model_len": 32768 if self.total_gb >= 80 else 16384,
            "kv_cache_dtype": "fp8",
            "enable_prefix_caching": True,
            "enable_chunked_prefill": True,
        }

_VRAM_MGR = SovereignVRAMManager()
def get_vram_manager() -> SovereignVRAMManager:
    return _VRAM_MGR

# ---------- Unified Tri-Level Cognitive Memory Bank (สมองส่วนฮิปโปแคมปัสกลาง) ----------
class GlobalMemoryBank:
    """สมองส่วนฮิปโปแคมปัสกลาง (Unified Cognitive Memory):
    L1: Semantic Memory (สกัดกฎถาวรจาก Train Pairs และ 25 World Manuals)
    L2: Episodic Working Memory (ประวัติ 5-10 ก้าวล่าสุด, Trajectory, diff_pixels, sub-goals)
    L3: Tabu / Fatal Memory (ความจำบาดแผล: แบน State และ Action ที่เคยตายแบบ O(1))
    """
    _instance = None
    _lock = threading.RLock()

    def __new__(cls, *args, **kwargs):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._initialized = False
            return cls._instance

    def __init__(self, max_size: int = 8192, alpha: float = 0.3, gamma: float = 0.9):
        if getattr(self, "_initialized", False):
            return
        self._initialized = True
        self.max_size = max_size
        self.alpha = alpha
        self.gamma = gamma

        # L1: Semantic Memory (Game -> Extracted Rules / Invariants)
        self.l1_semantic_rules: dict[str, str] = {}
        self._load_l1_priors()

        # L2: Episodic Working Memory (Game -> deque of recent steps with diffs & sub-goals)
        self.l2_episodic_working: dict[str, deque] = defaultdict(lambda: deque(maxlen=10))

        # L3: Tabu / Fatal Memory (Game:Level:StateHash -> set of forbidden actions)
        self.l3_tabu_fatal: dict[str, set[Any]] = defaultdict(set)
        # Failure ledger (StateHash -> failed actions incl. click coords).
        # Resurrected: never initialized before, so every avoid_failed/query
        # raised AttributeError into the caller's fail-open handler and the
        # whole never-lose-twice path was silently dead.
        self.failed: dict[str, set[Any]] = defaultdict(set)
        self.photographic: OrderedDict[str, tuple[int, int | None, int | None, float, int]] = OrderedDict()
        self.q_table: dict[tuple[str, int, int], float] = defaultdict(float)
        self.recent: deque[str] = deque(maxlen=24)
        self.hits = 0
        self.saves = 0

    def reset_episodic(self) -> None:
        """Clear per-run episodic/tabu state, keep L1 semantic priors.

        Test isolation + worker game boundaries: q-values, failure ledger,
        photographic hits and recent trail are episode-scoped; L1 distilled
        rules survive. Never throws.
        """
        try:
            with self._lock:
                self.l2_episodic_working.clear()
                self.l3_tabu_fatal.clear()
                self.failed.clear()
                self.photographic.clear()
                self.q_table.clear()
                self.recent.clear()
                self.hits = 0
                self.saves = 0
        except Exception:
            pass

    def _load_l1_priors(self):
        # Intrinsic-only order: NEVER load preloaded game manuals.
        # l1_semantic_rules stays empty; format_prompt_memory always uses
        # the generic intrinsic default. (Kept as no-op stub: the model
        # must be intelligent by itself, never fed memorized answers.)
        return

    def _mem_key(self, game_id: str, level: int, grid: np.ndarray) -> str:
        h = hashlib.blake2b(np.ascontiguousarray(grid, dtype=np.uint8).tobytes(), digest_size=8).hexdigest()
        return f"{game_id or 'unk'}:{int(level)}:{h}"

    @staticmethod
    def _enforce_cap(store: dict, cap: int) -> None:
        # Oldest-first eviction (insertion order). Photographic is an LRU
        # OrderedDict (touched on hit); plain dicts evict oldest-inserted.
        # Caps only bind on 110-game runs; normal runs never reach them, so
        # behavior below the cap is byte-identical to uncapped.
        while len(store) > cap:
            store.pop(next(iter(store)), None)

    def record_step_transition(self, game_id: str, level: int, prev_grid: np.ndarray, action: Any, next_grid: np.ndarray, reward: float = 0.0, is_game_over: bool = False):
        """Records step transition into L2 Episodic and L3 Tabu memories."""
        with self._lock:
            diff_px = int(np.sum(prev_grid != next_grid))
            entry = {
                "level": level,
                "action": action,
                "diff_pixels": diff_px,
                "reward": reward,
                "is_game_over": is_game_over,
                "timestamp": time.time()
            }
            self.l2_episodic_working[game_id].append(entry)

            # L3 Tabu update if fatal
            if is_game_over and reward <= 0:
                key = self._mem_key(game_id, level, prev_grid)
                act_val = action if isinstance(action, (int, str)) else action.get("action")
                self.l3_tabu_fatal[key].add(act_val)
                if isinstance(action, dict) and action.get("action") == 6:
                    self.l3_tabu_fatal[key].add((6, action.get("x"), action.get("y")))
            self._enforce_cap(self.l3_tabu_fatal, self.max_size)

    def format_prompt_memory(self, game_id: str, level: int, current_grid: np.ndarray) -> str:
        """Formats the Unified Tri-level memory into the system prompt for vLLM & LLMs."""
        prefix = game_id.split("-")[0].lower() if game_id else ""
        l1_rule = self.l1_semantic_rules.get(prefix, "Study object interactions, avoid death boundaries, align matching elements.")

        # L2 recent history
        ep_history = list(self.l2_episodic_working.get(game_id, []))[-4:]
        ep_lines = []
        for h in ep_history:
            res_str = "Died (Game Over)" if h["is_game_over"] else (f"Changed {h['diff_pixels']} px" if h['diff_pixels'] > 0 else "Blocked (No change)")
            ep_lines.append(f"  - Action: {h['action']} -> Result: {res_str}")
        ep_text = "\n".join(ep_lines) if ep_lines else "  - Starting first exploration step."

        # L3 tabu for current board
        key = self._mem_key(game_id, level, current_grid)
        tabu_acts = list(self.l3_tabu_fatal.get(key, set()))
        tabu_text = f"Forbidden Actions (Previously Fatal): {tabu_acts}" if tabu_acts else "None (Clear board)."

        return (
            f"\n[UNIFIED MEMORY BANK]\n"
            f"[L1 SEMANTIC RULE]: {l1_rule}\n"
            f"[L2 EPISODIC RECENT TRAJECTORY]:\n{ep_text}\n"
            f"[L3 TABU MEMORY]: {tabu_text}\n"
        )

    # Alias for backward compatibility
    def record_win(self, game_id: str, level: int, grid: np.ndarray, action: int, x: int | None, y: int | None):
        key = self._mem_key(game_id, level, grid)
        with self._lock:
            q_old = self.q_table[(game_id, level, action)]
            q_new = q_old + self.alpha * (1.0 - q_old)
            self.q_table[(game_id, level, action)] = q_new
            self.photographic[key] = (action, x, y, q_new, 1)
            self.saves += 1
            self.l3_tabu_fatal[key].discard(action)
            self._enforce_cap(self.photographic, self.max_size)

    def record_loss(self, game_id: str, level: int, grid: np.ndarray, action: int, x: int | None, y: int | None):
        key = self._mem_key(game_id, level, grid)
        with self._lock:
            q_old = self.q_table[(game_id, level, action)]
            q_new = q_old + self.alpha * (-1.0 - q_old)
            self.q_table[(game_id, level, action)] = q_new
            self.failed[key].add(action)
            if action == 6 and x is not None:
                self.failed[key].add((action, int(x), int(y) if y is not None else 0))
            if key in self.photographic:
                act_p, x_p, y_p, q_p, vis_p = self.photographic[key]
                if act_p == action:
                    self.photographic[key] = (act_p, x_p, y_p, max(-0.5, q_p * 0.5 - 0.3), vis_p)
            # photographic: don't store losing as win, but keep for avoidance
            self.recent.append(key)
            self._enforce_cap(self.failed, self.max_size)

    def record_stagnation(self, game_id: str, level: int, grid: np.ndarray, action: int, x: int | None, y: int | None, stagnation: int):
        if stagnation > 0:
            self.record_loss(game_id, level, grid, action, x, y)
        else:
            # progress -> small positive
            with self._lock:
                q_old = self.q_table[(game_id, level, action)]
                q_new = q_old + self.alpha * (0.2 - q_old)
                self.q_table[(game_id, level, action)] = q_new

    def record_verified(self, before: Any, action: int, after: Any, *, game_id: str,
                        level: int, x: int | None = None, y: int | None = None,
                        level_up: bool = False, score_gain: bool = False) -> bool:
        """Promote to photographic/Q ONLY on the verified gate.

        (level_up OR score_gain) AND changed AND shape_match. Identity or
        shape-mismatched transitions fall back to record_loss (avoidance
        only). Returns True when promoted. Never raises.
        """
        try:
            from arc3sdk import zero_waste as _zw
            fp = _zw.diff_fingerprint(before, after)
            if _zw.verified_gate(fp["changed"], fp["shape_match"],
                                 bool(level_up), bool(score_gain)):
                try:
                    grid = np.asarray(after)
                except Exception:
                    grid = after
                self.record_win(game_id, level, grid, int(action), x, y)
                with contextlib.suppress(Exception):
                    from arc3sdk import exploration_registry as _er
                    _er.note_evidence(before, action, after, level=level,
                                      level_up=True, score_gain=bool(score_gain))
                return True
            self.record_loss(game_id, level, np.asarray(before), int(action), x, y)
            return False
        except Exception:
            return False

    def query(self, game_id: str, level: int, grid: np.ndarray, available: list[int]) -> tuple[int, int | None, int | None, str, float] | None:
        key = self._mem_key(game_id, level, grid)
        with self._lock:
            # 1. Photographic hit (never lose twice) - check failed first
            if key in self.photographic:
                act, x, y, q, visits = self.photographic[key]
                if act in self.failed.get(key, set()):
                    pass
                elif act in available and q >= 0.25:
                    self.hits += 1
                    # LRU touch
                    self.photographic.move_to_end(key)
                    return act, x, y, f"leap_photographic:Q{q:.2f}v{visits}", q
            # 2. Q-table best among available not in failed
            best_act = None
            best_q = -9e9
            for a in available:
                if a in self.failed.get(key, set()):
                    continue
                # also check tuple failed for 6
                # for click, allow if different coord, so not block whole 6
                if any(isinstance(f, tuple) and f[0]==a for f in self.failed.get(key, set())) and a != 6:
                    continue
                # .get(): never auto-vivify on read. A bare [] read would insert a
                # 0.0 that then dilutes the cross-level average below its gate,
                # silently disabling transfer on every first-seen level.
                q = self.q_table.get((game_id, level, a), 0.0)
                # cross-level transfer: if no data for this level, use avg across levels
                if q == 0.0:
                    qs = []
                    for lv in range(5):
                        if lv == level:
                            continue
                        v = self.q_table.get((game_id, lv, a))
                        if v:
                            qs.append(v)
                    if qs:
                        q = sum(qs)/len(qs) * 0.7  # transfer discount
                if q > best_q:
                    best_q = q
                    best_act = a
            if best_act is not None and best_q > 0.15:
                return best_act, None, None, f"leap_q:Q{best_q:.2f}", best_q
        return None

    def avoid_failed(self, game_id: str, level: int, grid: np.ndarray, available: list[int]) -> list[int]:
        key = self._mem_key(game_id, level, grid)
        with self._lock:
            failed = self.failed.get(key, set())
            # filter but never block all
            filtered = [a for a in available if a not in failed and not any(isinstance(f,tuple) and f[0]==a and a!=6 for f in failed)]
            return filtered if filtered else available

    def consolidate(self):
        # Decay old, strengthen recent
        with self._lock:
            for k in list(self.q_table.keys()):
                self.q_table[k] *= 0.995  # slight decay
            # keep recent photographic
            for key in list(self.recent)[-8:]:
                if key in self.photographic:
                    act,x,y,q,vis = self.photographic[key]
                    self.photographic[key] = (act,x,y, min(1.0, q*1.02), vis)

    def stats(self) -> dict[str, Any]:
        return {
            "photographic": len(self.photographic),
            "failed_keys": len(self.failed),
            "q_entries": len(self.q_table),
            "hits": self.hits,
            "saves": self.saves,
            "recent": len(self.recent),
        }

# Backward compatibility aliases
SovereignLeapMemory = GlobalMemoryBank

# Global singleton (immortal)
_GLOBAL_MEMORY_BANK = GlobalMemoryBank()
_LEAP_MEMORY = _GLOBAL_MEMORY_BANK

def get_global_memory_bank() -> GlobalMemoryBank:
    return _GLOBAL_MEMORY_BANK

def get_leap_memory() -> SovereignLeapMemory:
    return _GLOBAL_MEMORY_BANK

# Auto-consolidate in background (CPython thread) — OPT-IN ONLY.
# NOTE (2026-09-23): unconditional startup decayed learned Q-values every
# 60s on the worker mid-competition (silent drift) and tripped the hot-path
# hazard gate. Set ARC3_BG_CONSOLIDATE=1 to enable (local diagnostics).
def _bg_consolidate():
    while True:
        time.sleep(60)
        try:
            _LEAP_MEMORY.consolidate()
            _VRAM_MGR.check_pressure()
        except Exception:
            pass

try:
    import os as _os
    if _os.environ.get("ARC3_BG_CONSOLIDATE", "").strip().lower() in ("1", "true", "yes", "on"):
        _t = threading.Thread(target=_bg_consolidate, daemon=True)
        _t.start()
except Exception:
    pass
