from dataclasses import dataclass
from math import sqrt


def normalize_adaptation(after, frozen, oracle):
    d=oracle-frozen
    if d <= 0: raise ValueError('oracle must exceed frozen baseline')
    return (after-frozen)/d

def forgetting(retain_before, retain_after):
    return max(0.0, retain_before-retain_after)

def ood_gain(after, frozen):
    return after-frozen

@dataclass(frozen=True)
class Observation:
    config: dict
    adapt: float
    forget: float
    ood: float

@dataclass(frozen=True)
class Prediction:
    adapt_mean: float; adapt_sigma: float
    forget_mean: float; forget_sigma: float
    ood_mean: float; ood_sigma: float

class IntelligenceModel:
    '''Small dependency-free kNN surrogate. It never invents a prior: fit observations first.'''
    def __init__(self, k=5): self.k=k; self._obs=[]
    def fit(self, observations): self._obs=list(observations); return self
    @staticmethod
    def _distance(a,b):
        rank=abs(a['rank']-b['rank'])/28.0
        layers=abs(a['active_layers']-b['active_layers'])/6.0
        precision=0.0 if a['precision']==b['precision'] else 1.0
        fusion_order={'unfused':0,'grouped2':1,'single-fused':2,'persistent':3}
        fusion=abs(fusion_order[a['fusion']]-fusion_order[b['fusion']])/3.0
        return sqrt(rank*rank+layers*layers+precision*precision+fusion*fusion)
    @staticmethod
    def _weighted(values, distances):
        ws=[1.0/(d+1e-6) for d in distances]
        sw=sum(ws); mean=sum(w*v for w,v in zip(ws,values))/sw
        var=sum(w*(v-mean)**2 for w,v in zip(ws,values))/sw
        # uncertainty includes neighbor disagreement and distance-to-evidence
        sigma=sqrt(var) + min(distances)*0.05
        return mean,sigma
    def predict(self, config):
        if not self._obs: raise ValueError('intelligence predictions require observed benchmark outcomes')
        near=sorted(((self._distance(config,o.config),o) for o in self._obs), key=lambda x:x[0])[:max(1,min(self.k,len(self._obs)))]
        ds=[x[0] for x in near]; os=[x[1] for x in near]
        am,asig=self._weighted([o.adapt for o in os],ds)
        fm,fsig=self._weighted([o.forget for o in os],ds)
        om,osig=self._weighted([o.ood for o in os],ds)
        return Prediction(am,asig,fm,fsig,om,osig)
