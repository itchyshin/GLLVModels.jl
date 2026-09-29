#!/usr/bin/env python3
"""Build the NB re-run list and a cost-balanced task map (not a submission).

Usage: python3 nb_rerun_tasks.py <slowdown_ratio> [target_hours_per_task]
Cost per dataset = old-kernel mean seconds for its cell (from harvest/) x slowdown_ratio.
Writes nb_rerun_list.csv (4,800 rows) and nb_rerun_tasks.csv (task,first,last,est_secs).
"""
import csv, glob, sys, collections
ratio = float(sys.argv[1]); target = float(sys.argv[2]) * 3600 if len(sys.argv) > 2 else 3 * 3600
t = collections.defaultdict(float); c = collections.defaultdict(set)
for f in glob.glob('harvest/*.csv'):
    for r in csv.DictReader(open(f)):
        if r.get('family') != 'nb':
            continue
        k = (int(r['n']), int(r['p']), int(r['K_true'])); t[k] += float(r['secs'] or 0); c[k].add(r['rep'])
cost = {k: ratio * t[k] / len(c[k]) for k in t}
rows = [(n, p, K, rep) for (n, p, K) in sorted(cost, key=lambda k: -cost[k]) for rep in range(1, 201)]
with open('nb_rerun_list.csv', 'w') as io:
    io.write('family,n,p,K_true,rep\n')
    for n, p, K, rep in rows:
        io.write(f'nb,{n},{p},{K},{rep}\n')
tasks, first, acc = [], 1, 0.0
for i, (n, p, K, rep) in enumerate(rows, start=1):
    acc += cost[(n, p, K)]
    if acc >= target:
        tasks.append((first, i, acc)); first, acc = i + 1, 0.0
if first <= len(rows):
    tasks.append((first, len(rows), acc))
with open('nb_rerun_tasks.csv', 'w') as io:
    io.write('task,first,last,est_secs\n')
    for j, (a, b, s) in enumerate(tasks, start=1):
        io.write(f'{j},{a},{b},{s:.0f}\n')
tot = sum(s for _, _, s in tasks)
print(f'{len(rows)} datasets, {len(tasks)} tasks, est {tot/3600:.0f} core-h, '
      f'max task {max(s for _,_,s in tasks)/3600:.2f} h, max single dataset {max(cost.values())/3600:.2f} h')
