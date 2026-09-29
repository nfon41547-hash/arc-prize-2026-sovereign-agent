from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from math import sqrt

from .intelligence import normalize_adaptation, forgetting, ood_gain, Observation, IntelligenceModel

CONFIG_KEYS = ('rank','active_layers','precision','fusion')
FIELDS = (*CONFIG_KEYS, 'seed','task','adapt','forget','ood','measured_latency_us','peak_vram_mb','source')

@dataclass(frozen=True)
class Outcome:
    adapt: float
    forget: float
    ood: float


def derive_outcome(*, frozen, after, oracle, retain_before, retain_after, ood_frozen, ood_after):
    return Outcome(normalize_adaptation(after, frozen, oracle),
                   forgetting(retain_before, retain_after),
                   ood_gain(ood_after, ood_frozen))


def config_key(c):
    return tuple(c[k] for k in CONFIG_KEYS)


class ResultStore:
    def __init__(self, path): self.path=Path(path)
    def append(self, config, outcome, *, seed, task, measured_latency_us='', peak_vram_mb='', source='measured'):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        exists=self.path.exists() and self.path.stat().st_size>0
        if hasattr(outcome, '__dict__'): outcome=outcome.__dict__
        row={**{k:config[k] for k in CONFIG_KEYS}, 'seed':seed, 'task':task,
             'adapt':outcome['adapt'], 'forget':outcome['forget'], 'ood':outcome['ood'],
             'measured_latency_us':measured_latency_us, 'peak_vram_mb':peak_vram_mb, 'source':source}
        with self.path.open('a', newline='') as f:
            w=csv.DictWriter(f, fieldnames=FIELDS)
            if not exists: w.writeheader()
            w.writerow(row)
    def read(self):
        if not self.path.exists(): return []
        with self.path.open() as f: return list(csv.DictReader(f))


def _typed(r):
    return {'rank':int(r['rank']), 'active_layers':int(r['active_layers']),
            'precision':r['precision'], 'fusion':r['fusion'],
            'adapt':float(r['adapt']), 'forget':float(r['forget']), 'ood':float(r['ood'])}


def select_next(candidates, measured, *, exploration=1.0):
    if not measured: raise ValueError('anchor measurements are required before active selection')
    typed=[_typed(r) if isinstance(r.get('rank'), str) else r for r in measured]
    seen={config_key(r) for r in typed}
    unseen=[c for c in candidates if config_key(c) not in seen]
    if not unseen: return None
    obs=[Observation({k:r[k] for k in CONFIG_KEYS}, r['adapt'],r['forget'],r['ood']) for r in typed]
    model=IntelligenceModel(k=min(5,len(obs))).fit(obs)
    # Cost-aware uncertainty acquisition. Quality is learned only from observations;
    # distance/uncertainty drives exploration, while predicted quality drives exploitation.
    best=None
    for c in unseen:
        p=model.predict(c)
        utility=(p.adapt_mean + p.ood_mean - p.forget_mean)
        uncertainty=p.adapt_sigma+p.ood_sigma+p.forget_sigma
        cost=max(1.0, c['rank']*c['active_layers'])
        score=(utility + exploration*uncertainty)/sqrt(cost)
        item=(score, c)
        if best is None or item[0]>best[0]: best=item
    return best[1]
