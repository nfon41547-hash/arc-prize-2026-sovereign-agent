"""SOBU-Ω + Cognitive Utility V6.
Evidence-gated action arbitration for ARC-AGI-3. Stdlib-only except callers may
pass numpy-like grids. Novel actions fail open; only observed evidence can make
a hard negative verdict.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from collections import defaultdict, deque
import hashlib, json, math, os, threading, time
from pathlib import Path

_EPS=1e-12

def _b(v, d=False):
    s=str(v if v is not None else '').strip().lower()
    return s in {'1','true','yes','on'} if s else d

def _f(name, default):
    try: return float(os.environ.get(name, default))
    except Exception: return float(default)

def _i(name, default):
    try: return int(os.environ.get(name, default))
    except Exception: return int(default)

def _entropy(p):
    p=min(1-_EPS,max(_EPS,float(p)))
    return -(p*math.log(p)+(1-p)*math.log(1-p))

def _eig(a,b):
    # Expected one-step entropy reduction of Beta-Bernoulli predictive belief.
    p=a/(a+b); h=_entropy(p)
    p1=(a+1)/(a+b+1); p0=a/(a+b+1)
    return max(0.0, h-(p*_entropy(p1)+(1-p)*_entropy(p0)))

def _beta_lcb(a,b,z=1.0):
    n=a+b; m=a/n; var=(a*b)/(n*n*(n+1))
    return max(0.0,min(1.0,m-z*math.sqrt(max(0.0,var))))

def _fp(grid):
    try:
        import numpy as np
        a=np.asarray(grid)
        if a.ndim>2:
            while a.ndim>2 and a.shape[0]==1: a=a[0]
        if a.ndim==2 and min(a.shape)>2: a=a[1:-1,1:-1]
        raw=a.tobytes()
    except Exception:
        raw=repr(grid).encode('utf-8','replace')
    return hashlib.blake2b(raw,digest_size=12).hexdigest()

def _sig(action_id,x=None,y=None):
    aid=str(action_id or '').strip().upper()
    return f'{aid}({x},{y})' if x is not None or y is not None else aid

@dataclass
class Stat:
    n:int=0; useful:int=0; changed:int=0; noop:int=0; fatal:int=0
    reward_sum:float=0.0; latency_sum:float=0.0
    def mean_reward(self): return self.reward_sum/max(1,self.n)
    def latency(self): return self.latency_sum/max(1,self.n)

@dataclass
class Game:
    level:int=0
    last_action:str|None=None
    recent_states:deque=field(default_factory=lambda:deque(maxlen=24))
    exact:dict=field(default_factory=lambda:defaultdict(Stat))
    action:dict=field(default_factory=lambda:defaultdict(Stat))
    reason:dict=field(default_factory=lambda:defaultdict(Stat))
    pair:dict=field(default_factory=lambda:defaultdict(Stat))
    events:int=0

class Controller:
    def __init__(self):
        self.games=defaultdict(Game); self.lock=threading.RLock()
    @property
    def mode(self): return os.environ.get('ARC3_V6_MODE','off').strip().lower()
    @property
    def enabled(self): return self.mode in {'shadow','control'}
    def _stat(self,g,level,state,action): return g.exact[(int(level),state,action)]
    def assess(self, game_id, level, grid, action_id, x=None, y=None, reason='analyzer', raw_confidence=None):
        with self.lock:
            g=self.games[str(game_id)]; state=_fp(grid); action=_sig(action_id,x,y)
            st=self._stat(g,level,state,action); ag=g.action[(int(level),action)]; rs=g.reason[str(reason or 'unknown')]
            # Conservative empirical posteriors. Novel state-actions retain a broad prior.
            pa,pb=1+st.useful,1+max(0,st.n-st.useful)
            ca,cb=1+st.changed,1+max(0,st.n-st.changed)
            fa,fb=1+st.fatal,3+max(0,st.n-st.fatal)
            na,nb=1+st.noop,1+max(0,st.n-st.noop)
            p_use=pa/(pa+pb); p_chg=ca/(ca+cb); p_fatal=fa/(fa+fb); p_noop=na/(na+nb)
            # Goal-value proxy is learned only from realized transitions; board change has low weight.
            dphi=0.75*p_use + 0.15*p_chg - 0.85*p_fatal - 0.25*p_noop
            info=_eig(pa,pb) * (1.0/(1.0+0.5*st.n))
            synergy=0.0
            if g.last_action:
                ps=g.pair[(int(level),g.last_action,action)]
                if ps.n>=2:
                    synergy=max(0.0, ps.mean_reward()-ag.mean_reward())
            repeat=min(1.0,st.n/max(1.0,_f('ARC3_V6_REPEAT_SCALE',4)))
            cycle=1.0 if state in list(g.recent_states)[-4:] else 0.0
            fatal_raw=min(1.0,p_fatal+0.5*p_noop)
            evidence=st.n/(st.n+2.0)
            fatal_risk=fatal_raw*evidence
            latency=max(0.0,ag.latency()-_f('ARC3_V6_LATENCY_TARGET_S',0.50))/_f('ARC3_V6_LATENCY_TARGET_S',0.50)
            latency*=ag.n/(ag.n+2.0)
            utility=(dphi + _f('ARC3_V6_BETA_INFO',0.35)*info + _f('ARC3_V6_ETA_SYNERGY',0.20)*synergy
                     -_f('ARC3_V6_LAMBDA_R',0.20)*repeat -_f('ARC3_V6_LAMBDA_K',0.20)*cycle
                     -_f('ARC3_V6_LAMBDA_F',0.65)*fatal_risk -_f('ARC3_V6_LAMBDA_L',0.05)*latency)
            rr_a,rr_b=1+rs.useful,1+max(0,rs.n-rs.useful)
            lcb=_beta_lcb(rr_a,rr_b,_f('ARC3_V6_LCB_Z',1.0))
            if raw_confidence is not None:
                # raw confidence is only a bounded tie-breaker; never treated as calibrated probability.
                try: utility += 0.03*(min(1.0,max(0.0,float(raw_confidence)))-0.5)
                except Exception: pass
            minobs=_i('ARC3_V6_MIN_OBSERVATIONS',2)
            hard_negative = st.n>=minobs and (p_fatal>=_f('ARC3_V6_FATAL_VETO',0.60) or p_noop>=_f('ARC3_V6_NOOP_VETO',0.70))
            return {'state':state,'action':action,'n':st.n,'delta_phi':dphi,'info':info,'synergy':synergy,
                    'R':repeat,'K':cycle,'F':fatal_risk,'F_raw':fatal_raw,'L':latency,'utility':utility,'reason_lcb':lcb,
                    'hard_negative':hard_negative}
    def compare(self, game_id, level, grid, analyzer, tier, tier_reason, tier_conf):
        aa,ax,ay=analyzer; ta,tx,ty=tier
        a=self.assess(game_id,level,grid,aa,ax,ay,'analyzer',None)
        t=self.assess(game_id,level,grid,ta,tx,ty,tier_reason,tier_conf)
        allow=True; why='fail-open'
        if t['hard_negative']:
            allow=False; why='tier-hard-negative'
        elif self.mode=='control':
            # Only arbitrate after enough local evidence. Before that, preserve the proven V5 gate.
            minobs=_i('ARC3_V6_MIN_OBSERVATIONS',2); margin=_f('ARC3_V6_OVERRIDE_MARGIN',0.08)
            if t['n']>=minobs and a['n']>=minobs:
                allow=t['utility'] > a['utility'] + margin
                why='utility-margin' if allow else 'analyzer-utility-higher'
            elif t['n']>=minobs and t['utility'] <= _f('ARC3_V6_TAU',0.0):
                allow=False; why='tier-nonpositive-utility'
        return {'allow_tier':allow,'why':why,'analyzer':a,'tier':t}
    def observe(self, game_id, level, before, after, action_id, x=None, y=None, reason='analyzer',
                latency_s=0.0, score_delta=0.0, level_delta=0, fatal=False, run_complete=False):
        with self.lock:
            g=self.games[str(game_id)]; state=_fp(before); nxt=_fp(after); action=_sig(action_id,x,y)
            changed=state!=nxt; noop=not changed and not level_delta and not score_delta
            strong=bool(run_complete or level_delta>0 or score_delta>0)
            useful=bool(strong or (changed and not fatal))
            reward=(1.5 if run_complete else 1.0 if level_delta>0 else 0.8 if score_delta>0 else 0.15 if changed else -0.25)
            if fatal: reward-=1.25
            st=self._stat(g,level,state,action); ag=g.action[(int(level),action)]; rs=g.reason[str(reason or 'unknown')]
            for s in (st,ag,rs):
                s.n+=1; s.useful+=int(useful); s.changed+=int(changed); s.noop+=int(noop); s.fatal+=int(bool(fatal)); s.reward_sum+=reward; s.latency_sum+=max(0.0,float(latency_s or 0.0))
            if g.last_action:
                ps=g.pair[(int(level),g.last_action,action)]; ps.n+=1; ps.useful+=int(useful); ps.changed+=int(changed); ps.noop+=int(noop); ps.fatal+=int(bool(fatal)); ps.reward_sum+=reward
            g.last_action=action; g.recent_states.append(nxt); g.level=int(level)+max(0,int(level_delta)); g.events+=1
            payload={'game':str(game_id),'level':int(level),'state':state,'next_state':nxt,'action':action,'reason':str(reason),
                     'changed':changed,'noop':noop,'fatal':bool(fatal),'score_delta':float(score_delta or 0.0),
                     'level_delta':int(level_delta or 0),'reward':reward,'latency_s':float(latency_s or 0.0)}
            self._audit(payload); return payload
    def _audit(self,payload):
        if not _b(os.environ.get('ARC3_V6_AUDIT','1'),True): return
        try:
            root=Path(os.environ.get('ARC3_V6_EVIDENCE_DIR','/kaggle/working/v6_evidence')); root.mkdir(parents=True,exist_ok=True)
            with (root/'sobu_events.jsonl').open('a',encoding='utf-8') as f: f.write(json.dumps(payload,sort_keys=True,allow_nan=False)+'\n')
        except Exception: pass

_SHARED=Controller()
def shared_controller(): return _SHARED
