from __future__ import annotations
from dataclasses import dataclass
import random

@dataclass(frozen=True)
class ExactMatchTask:
    task_id: str
    prompt: str
    answer: str
    split: str  # adapt | retain | ood

def score_exact(prediction: str, answer: str) -> float:
    norm=lambda x: ' '.join(str(x).strip().lower().split())
    return float(norm(prediction)==norm(answer))

def balanced_split(tasks, seed=0):
    items=list(tasks); random.Random(seed).shuffle(items)
    mid=(len(items)+1)//2
    return items[:mid],items[mid:]

def aggregate_scores(tasks, predictions):
    missing=[t.task_id for t in tasks if t.task_id not in predictions]
    if missing: raise ValueError(f'missing observed predictions: {missing[:5]}')
    groups={}
    for t in tasks:
        groups.setdefault(t.split,[]).append(score_exact(predictions[t.task_id],t.answer))
    return {k:sum(v)/len(v) for k,v in groups.items() if v}
