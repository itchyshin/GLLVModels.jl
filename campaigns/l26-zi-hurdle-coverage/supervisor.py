#!/usr/bin/env python3
"""L26 supervisor: dispatches (cell, rep) tasks to N Julia workers in cell order,
enforces a hard per-task timeout (kills the worker; logs status=timeout), restarts
dead workers (logs status=crash), and appends one receipt row per cell x rep.
Resumable: (cell, rep) pairs already in the receipts CSV are skipped."""
import argparse, csv, os, selectors, signal, subprocess, sys, time

FIELDS = ["cell", "family", "pi", "n", "p", "K", "rep", "seed", "status", "converged", "iterations",
          "logLik", "pd_hessian", "n_nan_se", "n_boundary_terms", "grad_maxabs", "r_hat",
          "prop_zero", "t_fit_s", "t_wald_s", "t_total_s", "error", "pkg_sha", "julia"]
FAMILIES = ["ZIP", "ZINB", "HurdlePoisson"]; PIS = [0.05, 0.2, 0.5]; NS = [100, 300, 1000]

def spec(cell):
    c = cell - 1
    return FAMILIES[c // 9], PIS[(c % 9) // 3], NS[c % 3]

def seed_of(cell, rep):
    return 20261016 + 10000 * cell + rep

ap = argparse.ArgumentParser()
ap.add_argument("--plan", required=True, help="ordered cell:reps list, e.g. 4:30,5:15,6:5")
ap.add_argument("--workers", type=int, default=4)
ap.add_argument("--timeout", default="100:600,300:900,1000:1800", help="n:seconds hard timeouts")
ap.add_argument("--out", default="/home/user/l26/out")
args = ap.parse_args()

def parse_cells(s):
    out = []
    for part in s.split(","):
        if "-" in part:
            a, b = part.split("-"); out += list(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return out

PLAN = [(int(a), int(b)) for a, b in (x.split(":") for x in args.plan.split(","))]
SHA = subprocess.run(["git", "-C", "/home/user/GLLVModels.jl", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
JV = subprocess.run(["julia", "--version"], capture_output=True, text=True).stdout.strip()
TIMEOUT = {int(k): float(v) for k, v in (x.split(":") for x in args.timeout.split(","))}
RECEIPTS = os.path.join(args.out, "receipts.csv")
LOG = open(os.path.join(args.out, "supervisor.log"), "a", buffering=1)

def log(msg):
    LOG.write(time.strftime("%Y-%m-%dT%H:%M:%S ") + msg + "\n")

done = set()
if os.path.exists(RECEIPTS):
    with open(RECEIPTS) as f:
        for r in csv.DictReader(f):
            done.add((int(r["cell"]), int(r["rep"])))
new_file = not os.path.exists(RECEIPTS)
rf = open(RECEIPTS, "a", newline="", buffering=1)
rw = csv.writer(rf)
if new_file:
    rw.writerow(FIELDS)

queue = [(c, r) for c, R in PLAN for r in range(1, R + 1) if (c, r) not in done]
PLAN_FILE = os.path.join(args.out, "plan.txt")
plan_mtime = os.path.getmtime(PLAN_FILE) if os.path.exists(PLAN_FILE) else None

def maybe_reload_plan():
    """Re-read plan.txt when it changes: re-queue (cell, rep) pairs that are neither done nor in flight."""
    global PLAN, queue, plan_mtime
    if not os.path.exists(PLAN_FILE):
        return
    m = os.path.getmtime(PLAN_FILE)
    if m == plan_mtime:
        return
    plan_mtime = m
    try:
        PLAN = [(int(a), int(b)) for a, b in (x.split(":") for x in open(PLAN_FILE).read().strip().split(","))]
    except Exception as e:
        log(f"plan reload failed: {e}"); return
    inflight = {w["task"] for w in workers.values() if w["task"] is not None}
    queue = [(c, r) for c, R in PLAN for r in range(1, R + 1) if (c, r) not in done and (c, r) not in inflight]
    log(f"plan reloaded: {len(queue)} tasks pending")

log(f"start: {len(queue)} tasks pending, {len(done)} already done, workers={args.workers}, timeouts={TIMEOUT}")

ENV = dict(os.environ, JULIA_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
sel = selectors.DefaultSelector()
workers = {}   # wid -> dict(proc, task, started, buf, ready)
wcount = 0

def spawn():
    global wcount
    wcount += 1
    wid = wcount
    pf = os.path.join(args.out, f"params_w{wid:03d}.csv")
    errf = open(os.path.join(args.out, f"worker_w{wid:03d}.stderr"), "w")
    p = subprocess.Popen(["julia", "--project=/home/user/GLLVModels.jl", "/home/user/l26/worker.jl", pf],
                         stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=errf, env=ENV,
                         text=True, bufsize=1)
    os.set_blocking(p.stdout.fileno(), False)
    workers[wid] = dict(proc=p, task=None, started=None, buf="", ready=False, errf=errf)
    sel.register(p.stdout, selectors.EVENT_READ, wid)
    log(f"spawned worker {wid} pid {p.pid}")

def write_row(cell, rep, status, err="", t=float("nan")):
    fam, pi, n = spec(cell)
    row = dict(cell=cell, family=fam, pi=pi, n=n, p=10, K=2, rep=rep, seed=seed_of(cell, rep),
               status=status, converged="false", iterations=-1, logLik="NaN", pd_hessian="false",
               n_nan_se=-1, n_boundary_terms=-1, grad_maxabs="NaN", r_hat="NaN", prop_zero="NaN",
               t_fit_s="NaN", t_wald_s="NaN", t_total_s=t, error=err, pkg_sha=SHA, julia=JV)
    rw.writerow([row[k] for k in FIELDS])
    done.add((cell, rep))

def kill(wid):
    w = workers.pop(wid)
    try: sel.unregister(w["proc"].stdout)
    except Exception: pass
    try: w["proc"].kill()
    except Exception: pass
    try: w["proc"].wait(timeout=10)
    except Exception: pass
    w["errf"].close()

def dispatch(wid):
    w = workers[wid]
    if queue:
        cell, rep = queue.pop(0)
        w["task"] = (cell, rep); w["started"] = None; w["phase_t"] = time.time()
        w["proc"].stdin.write(f"{cell} {rep}\n"); w["proc"].stdin.flush()
    else:
        w["task"] = None

stop = False
def handle_sig(*_):
    global stop
    stop = True
signal.signal(signal.SIGTERM, handle_sig)

for _ in range(args.workers):
    spawn()

ndone = 0
while not stop:
    maybe_reload_plan()
    if not queue and all(w["task"] is None and w["ready"] for w in workers.values()):
        break
    for key, _ in sel.select(timeout=1.0):
        wid = key.data
        if wid not in workers:
            continue
        w = workers[wid]
        try:
            chunk = w["proc"].stdout.read()
        except Exception:
            chunk = None
        if chunk:
            w["buf"] += chunk
        while "\n" in w["buf"]:
            line, w["buf"] = w["buf"].split("\n", 1)
            if line == "READY":
                w["ready"] = True; dispatch(wid)
            elif line.startswith("START "):
                w["started"] = time.time()
            elif line.startswith("RESULT\t"):
                vals = line.split("\t")[1:]
                if len(vals) == len(FIELDS) - 2:
                    rw.writerow(vals + [SHA, JV])
                    done.add((int(vals[0]), int(vals[6])))
                else:
                    c, r = w["task"]; write_row(c, r, "badresult", err=line[:200])
                ndone += 1
                dispatch(wid)
    now = time.time()
    for wid in list(workers):
        w = workers[wid]
        p = w["proc"]
        if p.poll() is not None:
            if w["task"] is not None:
                c, r = w["task"]
                write_row(c, r, "crash", err=f"worker exited rc={p.returncode}")
                log(f"worker {wid} crashed on cell {c} rep {r} rc={p.returncode}")
            else:
                log(f"worker {wid} exited rc={p.returncode}")
            kill(wid)
            if queue:
                spawn()
            continue
        if w["task"] is not None and w["started"] is not None:
            c, r = w["task"]
            lim = TIMEOUT[spec(c)[2]]
            if now - w["started"] > lim:
                write_row(c, r, "timeout", err=f"hard timeout {lim:.0f}s (fit+Wald)", t=round(now - w["started"], 1))
                log(f"TIMEOUT worker {wid} cell {c} rep {r} after {now - w['started']:.0f}s; killing")
                w["task"] = None
                kill(wid)
                if queue:
                    spawn()
    if not workers and queue:
        spawn()

for wid in list(workers):
    try:
        workers[wid]["proc"].stdin.close()
    except Exception:
        pass
for wid in list(workers):
    try: workers[wid]["proc"].wait(timeout=30)
    except Exception: pass
    kill(wid)
log(f"finished: {ndone} results this run; remaining queue {len(queue)}")
