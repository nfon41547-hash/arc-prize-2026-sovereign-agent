def pareto_front(rows, *, maximize=(), minimize=()):
    def dominates(a,b):
        no_worse=all(a[k]>=b[k] for k in maximize) and all(a[k]<=b[k] for k in minimize)
        better=any(a[k]>b[k] for k in maximize) or any(a[k]<b[k] for k in minimize)
        return no_worse and better
    return [r for i,r in enumerate(rows) if not any(i!=j and dominates(o,r) for j,o in enumerate(rows))]

def uncertainty_aware_pareto(rows, *, maximize=(), minimize=(), deterministic_minimize=(), z=1.0):
    '''Conservative dominance: A dominates B only when confidence bounds separate.'''
    def dominates(a,b):
        checks=[]; strict=[]
        for mean,sigma in maximize:
            a_lo=a[mean]-z*a[sigma]; b_hi=b[mean]+z*b[sigma]
            checks.append(a_lo >= b_hi); strict.append(a_lo > b_hi)
        for mean,sigma in minimize:
            a_hi=a[mean]+z*a[sigma]; b_lo=b[mean]-z*b[sigma]
            checks.append(a_hi <= b_lo); strict.append(a_hi < b_lo)
        for key in deterministic_minimize:
            checks.append(a[key] <= b[key]); strict.append(a[key] < b[key])
        return all(checks) and any(strict)
    return [r for i,r in enumerate(rows) if not any(i!=j and dominates(o,r) for j,o in enumerate(rows))]
