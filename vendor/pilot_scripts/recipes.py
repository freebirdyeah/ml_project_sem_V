"""Candidate recipe library: ordered sequences of the 7 ABC AIG operations."""
import random, csv
OPS = ['rw','rwz','rf','rfz','rs','rsz','b']
CMD = {'rw':'rewrite','rwz':'rewrite -z','rf':'refactor','rfz':'refactor -z',
       'rs':'resub','rsz':'resub -z','b':'balance'}
KNOWN = {
 'resyn':  'b rw rwz b rwz b',
 'resyn2': 'b rw rf b rw rwz b rfz rwz b',
}
def sample(rng, kind, n):
    w = {'area':   dict(rw=3,rwz=3,rf=3,rfz=3,rs=3,rsz=3,b=1),
         'delay':  dict(rw=1,rwz=1,rf=1,rfz=1,rs=0.5,rsz=0.5,b=6),
         'mixed':  dict(rw=2,rwz=2,rf=2,rfz=2,rs=2,rsz=2,b=3),
         'random': dict(rw=1,rwz=1,rf=1,rfz=1,rs=1,rsz=1,b=1)}[kind]
    return rng.choices(OPS, weights=[w[o] for o in OPS], k=n)
def build(n_total=60, seed=7):
    rng = random.Random(seed)
    lib = {}
    for name, s in KNOWN.items(): lib[s] = ('known', name)
    quota = {'area':12,'delay':12,'mixed':12,'random': n_total-len(KNOWN)-36}
    for kind,q in quota.items():
        while sum(1 for v in lib.values() if v[0]==kind) < q:
            seq = sample(rng, kind, rng.randint(6,12))
            # reject immediate repeats of balance (b b is a no-op) to keep recipes meaningful
            if any(a=='b' and b=='b' for a,b in zip(seq,seq[1:])): continue
            s = ' '.join(seq)
            if s not in lib: lib[s] = (kind, '')
    rows=[]
    for i,(s,(kind,name)) in enumerate(lib.items()):
        rows.append(dict(recipe_id=f'R{i:03d}', kind=kind, name=name, ops=s,
                         abc=';'.join(CMD[o] for o in s.split())))
    return rows
if __name__=='__main__':
    rows = build()
    with open('out/recipes.csv','w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    print(len(rows),'recipes'); [print(r['recipe_id'],r['kind'],r['ops']) for r in rows[:8]]
