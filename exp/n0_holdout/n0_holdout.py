#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Out-of-grid holdout for the two-path cost model -- corrected instrument.

Two defects in exp/n0_robust invalidated the table it produced, so the holdout
is run on a corrected instrument rather than on top of the published one.

  defect 1 (origin)   the three JVM rows scanned v[0..n) while every Python
                      row scanned v[LOFF..LOFF+n).  v is value-sorted, v[0]~0
                      and v[LOFF] is the median, so the JVM predicate was
                      false on every row and no count check existed for those
                      rows.  Fixed by N0Bench3, which takes LOFF.
  defect 2 (band)     candidates were np.sort(choice(N, 8192)), so the first
                      4096 were the 4096 smallest ids in the sample, all below
                      the threshold: path B timed rows that a band could never
                      return, and the pass rate rose with k (0% at k<=4096,
                      50% at k=8192).  Fixed here: the table is stored in
                      insertion order and candidates are drawn from the rows
                      that actually satisfy the threshold, so a band selects
                      scattered rows, which is the access path B must pay for.

The holdout question, unchanged: the coefficients are fitted on
n in [512, 65536], k in [8, 4096], and the corpus decides at (394, 182).  n=394
is below the fitted scan range and the corpus band is at k/|s| = 0.46, far
above the fitted k/|s| = 0.06.  This script fits nothing on the holdout: it
applies the fitted coefficients to
  * scan lengths below the fitted minimum, including n = 394,
  * band sizes above the fitted maximum, and
  * three instances that differ from the fitted one (replicate data, an L2
    sized group, a group four times larger with a different selectivity),
and reports prediction error, winner agreement, and the cost of misrouting.

This supersedes exp/n0_robust, whose three defects are documented in README.md
("Corrections"): a JVM scan that started at row 0 of a value-sorted table and so
passed the predicate zero times, a band drawn as the k smallest identifiers, and
a predicate expression 3.6x slower than its equivalent, which compressed kappa
toward 1. Set N0_JDK if javac is not on PATH.

Outputs: n0_holdout.json
"""
import json
import os
import shutil
import subprocess
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.abspath(os.path.join(HERE, "..", "n0_robust"))
OUT = os.path.join(HERE, "n0_holdout.json")
def _find_jdk():
    """$N0_JDK, else whatever javac the shell can see, else empty."""
    p = os.environ.get("N0_JDK")
    if p and os.path.exists(os.path.join(p, "bin", "javac")) or (
            p and os.path.exists(os.path.join(p, "bin", "javac.exe"))):
        return p
    jc = shutil.which("javac")
    return os.path.dirname(os.path.dirname(jc)) if jc else ""


JDK = _find_jdk()

THETA_DEFAULT = 500000
OMEGA = 1
N_ROWS = 1 << 22
CAND_POOL = 32768

FIT_NS = [512, 1024, 2048, 4096, 8192, 16384, 32768, 65536]
FIT_KS = [8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]

HOLD_NS = [64, 128, 256, 394, 1024, 4096, 16384, 65536]
HOLD_KS = [182, 512, 2048, 6144, 12288, 24576, 30720]
REPS = 25

CORPUS_N, CORPUS_K = 394, 182

BACKENDS = ["numpy", "pyloop", "sqlite", "java c2", "java c1", "java int"]

INSTANCES = [
    dict(tag="F fit", N=N_ROWS, G=65536, theta=500000, seed=20261105,
         note="the fitted configuration, re-measured on the corrected instrument"),
    dict(tag="H1 replicate", N=N_ROWS, G=65536, theta=500000, seed=111,
         note="same shape, different data"),
    dict(tag="H2 L2 group", N=N_ROWS, G=16384, theta=500000, seed=222,
         note="scope's streaming region fits in L2"),
    dict(tag="H3 large group", N=N_ROWS, G=262144, theta=600000, seed=333,
         note="4x the fitted group, 40% pass rate"),
]


# ------------------------------------------------------------------ instance
def build(N, G, theta, seed, pool=CAND_POOL):
    """Rows in insertion order; the band is the set that satisfies theta."""
    rng = np.random.default_rng(seed)
    v = rng.integers(0, 10 ** 6, N).astype(np.int64)
    mod = (rng.random(N) < 0.6).astype(np.int8)
    band = np.flatnonzero(v >= theta)
    if band.size < pool:
        raise RuntimeError("band too small: %d" % band.size)
    cand = rng.choice(band, pool, replace=False)     # unsorted: no id bias
    loff = (N // G // 2) * G
    gsorted = np.sort(v[loff:loff + G])
    return dict(v=v, mod=mod, cand=cand, loff=loff, G=G, theta=theta,
                gsorted=gsorted, N=N)


# ------------------------------------------------------------------ backends
class NumpyB:
    name = "numpy"

    def __init__(self, inst):
        self.i = inst
        self.v, self.m = inst["v"], inst["mod"]

    def R(self, n):
        lo = self.i["loff"]
        v, m = self.v[lo:lo + n], self.m[lo:lo + n]
        return int(np.count_nonzero((v >= self.i["theta"]) & ~(m == 1)))

    def B(self, k):
        c = self.i["cand"][:k]
        v, m = self.v.take(c), self.m.take(c)
        return int(np.count_nonzero((v >= self.i["theta"]) & ~(m == 1)))

    def rank(self):
        s = self.i["gsorted"]
        t = self.i["theta"]
        return int(np.searchsorted(s, t + 1000) - np.searchsorted(s, t - 1000))


class PyLoopB:
    name = "pyloop"

    def __init__(self, inst):
        self.i = inst
        self.v = inst["v"].tolist()
        self.m = inst["mod"].tolist()
        self.c = inst["cand"].tolist()
        self.s = inst["gsorted"].tolist()

    def R(self, n):
        v, m, lo, th = self.v, self.m, self.i["loff"], self.i["theta"]
        c = 0
        for i in range(lo, lo + n):
            if v[i] >= th and m[i] != 1:
                c += 1
        return c

    def B(self, k):
        v, m, th, cand = self.v, self.m, self.i["theta"], self.c
        c = 0
        for j in range(k):
            i = cand[j]
            if v[i] >= th and m[i] != 1:
                c += 1
        return c

    def rank(self):
        import bisect
        t = self.i["theta"]
        return bisect.bisect_right(self.s, t + 1000) - bisect.bisect_left(self.s, t - 1000)


SQL_R = ("SELECT SUM(CASE WHEN v>=? AND mod<>1 THEN 1 ELSE 0 END) "
         "FROM claims WHERE id>? AND id<=?")
SQL_B = ("SELECT SUM(CASE WHEN c.v>=? AND c.mod<>1 THEN 1 ELSE 0 END) "
         "FROM cand x JOIN claims c ON c.id=x.id WHERE x.ord<?")


class SqliteB:
    name = "sqlite"

    def __init__(self, inst):
        import sqlite3
        self.i = inst
        p = os.path.join(HERE, "claims_%d_%d_%d_%d.db"
                         % (inst["N"], inst["G"], inst["theta"],
                            inst.get("seed", 0)))
        self.conn = sqlite3.connect(p)
        for pr in ("journal_mode=OFF", "synchronous=OFF",
                   "cache_size=-524288", "temp_store=MEMORY"):
            self.conn.execute("PRAGMA " + pr)
        self.conn.execute("DROP TABLE IF EXISTS claims")
        self.conn.execute("DROP TABLE IF EXISTS cand")
        self.conn.execute("CREATE TABLE claims (id INTEGER PRIMARY KEY, "
                          "v INTEGER, mod INTEGER)")
        self.conn.executemany("INSERT INTO claims VALUES (?,?,?)",
                              zip(range(1, inst["N"] + 1), inst["v"].tolist(),
                                  inst["mod"].tolist()))
        self.conn.execute("CREATE TABLE cand (ord INTEGER PRIMARY KEY, id INTEGER)")
        self.conn.executemany("INSERT INTO cand VALUES (?,?)",
                              zip(range(len(inst["cand"])),
                                  (inst["cand"] + 1).tolist()))
        self.conn.commit()

    def R(self, n):
        lo = self.i["loff"]
        r = self.conn.execute(SQL_R, (self.i["theta"], lo, lo + n)).fetchone()[0]
        return int(r or 0)

    def B(self, k):
        r = self.conn.execute(SQL_B, (self.i["theta"], k)).fetchone()[0]
        return int(r or 0)


class JavaB:
    def __init__(self, inst, label, flags, workdir):
        self.i, self.label, self.flags, self.workdir = inst, label, flags, workdir

    def run(self, ns, ks, reps):
        d = self.workdir
        np.save(os.path.join(d, "_v.npy"), self.i["v"])
        np.save(os.path.join(d, "_m.npy"), self.i["mod"])
        np.save(os.path.join(d, "_c.npy"), self.i["cand"].astype(np.int64))
        np.save(os.path.join(d, "_g.npy"), self.i["gsorted"])
        cmd = [os.path.join(JDK, "bin", "java.exe")] + self.flags + [
            "-cp", d, "N0Bench3",
            os.path.join(d, "_v.npy"), os.path.join(d, "_m.npy"),
            os.path.join(d, "_c.npy"), os.path.join(d, "_g.npy"),
            str(self.i["loff"]), str(self.i["theta"]), str(reps),
            ",".join(map(str, ns)), ",".join(map(str, ks))]
        out = subprocess.run(cmd, capture_output=True, text=True)
        if out.returncode != 0:
            raise RuntimeError(out.stderr[-800:])
        return json.loads(out.stdout)


# ---------------------------------------------------------------- measuring
def timed(fn, reps):
    fn()
    vals = []
    acc = 0
    for _ in range(reps):
        t0 = time.perf_counter_ns()
        acc += fn()
        vals.append(float(time.perf_counter_ns() - t0))
    return vals, acc


def fixed_cost(fn, calls):
    best = float("inf")
    for _ in range(5):
        acc = 0
        t0 = time.perf_counter_ns()
        for _ in range(calls):
            acc += fn()
        best = min(best, (time.perf_counter_ns() - t0) / calls)
    return best


def fitline(xs, ys):
    a, b = np.polyfit(np.asarray(xs, float), np.asarray(ys, float), 1)
    return float(a), float(b)


def r2of(xs, ys):
    xs, ys = np.asarray(xs, float), np.asarray(ys, float)
    a, b = np.polyfit(xs, ys, 1)
    ss = float(np.sum((ys - ys.mean()) ** 2))
    return float(1 - np.sum((ys - (a * xs + b)) ** 2) / ss)


def measure_python(inst, ns, ks, ref):
    out = {}
    for cls in (NumpyB, PyLoopB, SqliteB):
        t0 = time.time()
        b = cls(inst)
        reps = 9 if cls is PyLoopB else REPS
        tR, tB = {}, {}
        for n in ns:
            s, acc = timed(lambda n=n: b.R(n), reps)
            if acc != ref["R"][n] * reps:
                raise RuntimeError("%s R count %d vs %d at n=%d"
                                   % (cls.name, acc, ref["R"][n] * reps, n))
            tR[n] = min(s)
        for k in ks:
            s, acc = timed(lambda k=k: b.B(k), reps)
            if acc != ref["B"][k] * reps:
                raise RuntimeError("%s B count at k=%d" % (cls.name, k))
            tB[k] = min(s)
        FR = fixed_cost(lambda: b.R(0), 5000 if cls is PyLoopB else 20000)
        FB = fixed_cost(lambda: b.B(0), 5000 if cls is PyLoopB else 20000)
        RK = fixed_cost(lambda: b.rank(), 2000) if hasattr(b, "rank") else 0.0
        out[cls.name] = dict(tR=tR, tB=tB, FR=FR, FB=FB, RK=RK,
                             setup_s=round(time.time() - t0, 1))
    return out


def measure_java(inst, ns, ks, ref):
    compile_once()
    out = {}
    for label, flags in (("java c2", []),
                         ("java c1", ["-XX:TieredStopAtLevel=1"]),
                         ("java int", ["-Xint"])):
        t0 = time.time()
        b = JavaB(inst, label, flags, HERE)
        r = b.run(ns, ks, REPS)
        for i, n in enumerate(ns):
            if r["counts_R"][i] != ref["R"][n]:
                raise RuntimeError("%s R count mismatch at n=%d: %d vs %d"
                                   % (label, n, r["counts_R"][i], ref["R"][n]))
        for i, k in enumerate(ks):
            if r["counts_B"][i] != ref["B"][k]:
                raise RuntimeError("%s B count mismatch at k=%d" % (label, k))
        out[label] = dict(
            tR={p["n"]: float(p["ns"]) for p in r["points_R"]},
            tB={p["k"]: float(p["ns"]) for p in r["points_B"]},
            FR=r["fixed_R_ns"], FB=r["fixed_B_ns"], RK=r["rank_ns"],
            setup_s=round(time.time() - t0, 1))
    return out


def compile_once():
    cls = os.path.join(HERE, "N0Bench3.class")
    if os.path.exists(cls):
        return
    r = subprocess.run([os.path.join(JDK, "bin", "javac.exe"), "-d", HERE,
                        os.path.join(HERE, "N0Bench3.java")],
                       capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError("javac: " + r.stderr[-600:])


def coefficients(row, ns, ks):
    # points under ~100 ns sit at the timer's quantisation floor and would drag
    # the intercept; the fitted ranges start at 512 rows / 8 candidates, where
    # every backend is well above it.
    ns_f = [n for n in ns if row["tR"][n] >= 100]
    ks_f = [k for k in ks if row["tB"][k] >= 100]
    aS, _ = fitline(ns_f, [row["tR"][n] for n in ns_f])
    aP, _ = fitline(ks_f, [row["tB"][k] for k in ks_f])
    kappa = aS / aP
    n0 = (row["FB"] - row["FR"]) / aS
    return dict(aS=aS, aP=aP, FR=row["FR"], FB=row["FB"],
                kappa=kappa, n0=n0, dp=(row["FB"] - row["FR"]) / aP,
                r2R=r2of(ns_f, [row["tR"][n] for n in ns_f]),
                r2B=r2of(ks_f, [row["tB"][k] for k in ks_f]),
                nfit=len(ns_f), kfit=len(ks_f))


def evaluate(coef, ns, ks, tR, tB):
    pR = {n: coef["FR"] + coef["aS"] * n for n in ns}
    pB = {k: coef["FB"] + coef["aP"] * k for k in ks}
    # timers below ~50 ns carry quantisation noise; exclude rather than divide
    rs = [n for n in ns if tR[n] >= 50]
    bs = [k for k in ks if tB[k] >= 50]
    mR = 100 * float(np.mean([abs(pR[n] - tR[n]) / tR[n] for n in rs])) if rs else float("nan")
    mB = 100 * float(np.mean([abs(pB[k] - tB[k]) / tB[k] for k in bs])) if bs else float("nan")

    buckets, cp, co = {}, 0.0, 0.0
    wrong = 0
    for n in ns:
        for k in ks:
            meas = tB[k] < tR[n]
            pred = pB[k] < pR[n]
            key = ("in" if 512 <= n <= 65536 else "out") + "/" + \
                  ("in" if 8 <= k <= 4096 else "out")
            b = buckets.setdefault(key, [0, 0])
            b[0] += int(meas == pred)
            b[1] += 1
            if meas != pred:
                wrong += 1
            cp += tB[k] if pred else tR[n]
            co += min(tB[k], tR[n])
    return dict(mape_R=round(mR, 2), mape_B=round(mB, 2),
                agreements={k: dict(ok=v[0], n=v[1], pct=round(100 * v[0] / v[1], 1))
                            for k, v in buckets.items()},
                wrong=wrong, pairs=len(ns) * len(ks),
                regret=round(cp / co, 4),
                corpus=dict(
                    measured_R=round(tR[CORPUS_N], 1),
                    measured_B=round(tB[CORPUS_K], 1),
                    measured_band=bool(tB[CORPUS_K] < tR[CORPUS_N])))


def main():
    compile_once()
    out = dict(instrument="corrected (insertion-order rows, band-true candidates, "
                          "JVM scans from LOFF)",
               fit_ns=FIT_NS, fit_ks=FIT_KS, hold_ns=HOLD_NS, hold_ks=HOLD_KS,
               reps=REPS, instances=[])

    coefs = {}
    for inst in INSTANCES:
        G = inst["G"]
        theta = inst["theta"]
        print("\n=== %-14s N=%d G=%d theta=%d seed=%d  (%s)"
              % (inst["tag"], inst["N"], G, theta, inst["seed"], inst["note"]),
              flush=True)
        dat = build(inst["N"], G, theta, inst["seed"])
        ns_all = sorted(set([x for x in FIT_NS if x <= G] +
                            [x for x in HOLD_NS if x <= G]))
        ks_all = sorted(set([x for x in FIT_KS if x <= G] +
                            [x for x in HOLD_KS if x <= G]))

        ref = {"R": {}, "B": {}}
        nb = NumpyB(dat)
        ref["R"] = {n: nb.R(n) for n in ns_all}
        ref["B"] = {k: nb.B(k) for k in ks_all}
        print("   measured n=%s" % ns_all)
        print("   measured k=%s" % ks_all, flush=True)

        rows = measure_python(dat, ns_all, ks_all, ref)
        rows.update(measure_java(dat, ns_all, ks_all, ref))

        ns_fit = [x for x in FIT_NS if x <= G]
        ks_fit = [x for x in FIT_KS if x <= G]
        rec = dict(tag=inst["tag"], N=inst["N"], G=G, theta=theta,
                   seed=inst["seed"], note=inst["note"], backends={})

        for name in BACKENDS:
            r = rows[name]
            coef = coefficients(r, ns_fit, ks_fit)
            coefs.setdefault(name, coef)
            ev_fit = evaluate(coef, ns_fit, ks_fit, r["tR"], r["tB"])
            ns_h = [x for x in HOLD_NS if x <= G]
            ks_h = [x for x in HOLD_KS if x <= G]
            ev_h = evaluate(coef, ns_h, ks_h, r["tR"], r["tB"])
            full = evaluate(coef, ns_all, ks_all, r["tR"], r["tB"])
            rec["backends"][name] = dict(
                coef=dict(aS=round(coef["aS"], 4), aP=round(coef["aP"], 4),
                          FR=round(coef["FR"], 1), FB=round(coef["FB"], 1),
                          kappa=round(coef["kappa"], 4), n0=round(coef["n0"], 1),
                          dp=round(coef["dp"], 1), r2R=round(coef["r2R"], 4),
                          r2B=round(coef["r2B"], 4)),
                rank_ns=round(r["RK"], 1), setup_s=r["setup_s"],
                tR={str(n): round(v, 1) for n, v in sorted(r["tR"].items())},
                tB={str(k): round(v, 1) for k, v in sorted(r["tB"].items())},
                fit=ev_fit, holdout=ev_h, allpairs=full)
            print("   %-9s kappa=%.3f n0=%7.1f dp=%6.1f  fit: agr %5.1f%%  "
                  "holdout: MAPE %.1f/%.1f%% agr %5.1f%% regret %.3f  corpus %s"
                  % (name, coef["kappa"], coef["n0"], coef["dp"],
                     ev_fit["agreements"].get("in/in", {}).get("pct", float("nan")),
                     ev_h["mape_R"], ev_h["mape_B"],
                     ev_h["agreements"].get("out/out", ev_h["agreements"].get("out/in", {"pct": float("nan")}))["pct"],
                     ev_h["regret"],
                     "band" if ev_h["corpus"]["measured_band"] else "scan"),
                  flush=True)
        out["instances"].append(rec)

    out["fit_coefficients"] = {k: dict(aS=round(v["aS"], 4), aP=round(v["aP"], 4),
                                       FR=round(v["FR"], 1), FB=round(v["FB"], 1),
                                       kappa=round(v["kappa"], 4),
                                       n0=round(v["n0"], 1), dp=round(v["dp"], 1))
                               for k, v in coefs.items()}
    json.dump(out, open(OUT, "w", encoding="utf-8"), indent=1)
    print("\nwrote", OUT)


if __name__ == "__main__":
    main()
