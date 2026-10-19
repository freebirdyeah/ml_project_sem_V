"""Explicit ABC operation vocabulary and deterministic recipe extension."""
import random
import pandas as pd

OPS = ['rw', 'rwz', 'rf', 'rfz', 'rs', 'rsz', 'b']
COMMANDS = dict(zip(OPS, ['rewrite', 'rewrite -z', 'refactor', 'refactor -z', 'resub', 'resub -z', 'balance']))

def extend(pilot, total=120, seed=17):
    rows = pd.read_csv(pilot).to_dict('records')
    if total < len(rows):
        return pd.DataFrame(rows[:total])
    if total != 120:
        raise ValueError('The extension is defined for 120 recipes; use the pilot for 60.')
    rng = random.Random(seed)
    seen = {r['ops'] for r in rows}
    weights = {'area': [3,3,3,3,3,3,1], 'delay': [1,1,1,1,.5,.5,6],
               'mixed': [2,2,2,2,2,2,3], 'random': [1]*7}
    for kind, count in [('area',12),('delay',12),('mixed',12),('random',24)]:
        for _ in range(count):
            while True:
                ops = rng.choices(OPS, weights=weights[kind], k=rng.randint(6,12))
                text = ' '.join(ops)
                if text not in seen and not any(a == b == 'b' for a,b in zip(ops,ops[1:])):
                    break
            seen.add(text)
            rows.append(dict(recipe_id=f'R{len(rows):03d}', kind=kind, name='', ops=text,
                             abc=';'.join(COMMANDS[o] for o in ops)))
    return pd.DataFrame(rows)

def encode_recipe(text, mode='full'):
    ops = text.split()
    if not 6 <= len(ops) <= 12 or any(o not in OPS for o in ops):
        raise ValueError(f'Invalid recipe: {text}')
    f = {'r_len':len(ops), 'r_nz':sum(o.endswith('z') for o in ops)}
    f.update({f'r_cnt_{o}':ops.count(o) for o in OPS})
    if mode in ('positions','full'):
        f.update({f'r_pos{p}_{o}':int(p<len(ops) and ops[p]==o) for p in range(12) for o in OPS})
    if mode == 'full':
        f.update({f'r_tr_{a}_{b}':sum(x==a and y==b for x,y in zip(ops,ops[1:])) for a in OPS for b in OPS})
        f.update(r_first_b=int(ops[0]=='b'),r_last_b=int(ops[-1]=='b'))
    return f
