//! IVM head-to-head on one instance.
//!
//! The v8 draft priced "IVM" as a hand-written DuckDB plan that recomputed the derived
//! query over |D|: 18.3 ms against 14.9 us for the specialised path. That ratio is 1200x
//! because the two arms do different amounts of work -- one touches |D| rows, the other
//! touches a scope -- not because the algebra is expensive. This harness compares the two
//! *obligations* instead, in an engine that actually maintains a view:
//!
//!   dd_maint    differential-dataflow, view maintained by the engine. Per correction the
//!               engine delivers the output delta: the claims whose verdict moved.
//!   dd_cert     the same dataflow, but the output is a single count row, so "did anything
//!               move" is one row to read. This is the certification obligation.
//!   dd_floor    the same dataflow, epoch advanced, no parameter change: the framework's
//!               per-step floor, which is what both arms above pay whether the delta is
//!               empty or not.
//!   dd_rebuild  the same engine with nothing maintained: the relation is re-entered as a
//!               delta, which is the engine's version of "just recompute".
//!   hand_*      the same delta with no dataflow framework (control, and the independent
//!               check on the paper's band cost).
//!
//! Instance: exp/ivm.py's, byte for byte (2e6 rows, 5410 groups, theta = 1.5 * mean).
//! One worker thread, matching PRAGMA threads=1 in the DuckDB arm.
use differential_dataflow::consolidation::Consolidate;
use differential_dataflow::input::InputSession;
use differential_dataflow::operators::count::CountTotal;
use std::cell::Cell;
use std::hint::black_box;
use std::sync::Arc;
use std::time::Instant;
use timely::dataflow::operators::probe::Handle;

type Cid = u32;
type Gid = u32;

thread_local! {
    static SEEN: Cell<usize> = Cell::new(0);
}

#[derive(Clone)]
struct Instance {
    gid: Vec<Gid>,
    cents: Vec<i64>,
    rows: usize,
    theta: i64,
    gmax: Gid,
    per_group: Vec<Vec<i64>>,
}

fn load(path: &str) -> Instance {
    let raw = std::fs::read(path).expect("instance.bin");
    let rows = raw.len() / 12;
    let mut gid = Vec::with_capacity(rows);
    let mut cents = Vec::with_capacity(rows);
    for i in 0..rows {
        let b = &raw[i * 12..i * 12 + 12];
        gid.push(u32::from_le_bytes([b[0], b[1], b[2], b[3]]));
        let a = f64::from_le_bytes([b[4], b[5], b[6], b[7], b[8], b[9], b[10], b[11]]);
        cents.push((a * 100.0).round() as i64);
    }
    let mut cnt = vec![0usize; 5410];
    for &g in &gid {
        cnt[g as usize] += 1;
    }
    let gmax = (0..cnt.len()).max_by_key(|&i| cnt[i]).unwrap() as Gid;
    let mean = cents.iter().map(|&c| c as f64).sum::<f64>() / rows as f64;
    let mut per_group: Vec<Vec<i64>> = vec![Vec::new(); 5410];
    for i in 0..rows {
        per_group[gid[i] as usize].push(cents[i]);
    }
    for v in per_group.iter_mut() {
        v.sort_unstable();
    }
    Instance { gid, cents, rows, theta: (1.5 * mean) as i64, gmax, per_group }
}

fn stats(mut v: Vec<f64>) -> (f64, f64, f64) {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let q = |p: f64| v[(((v.len() - 1) as f64) * p).round() as usize];
    (q(0.5), q(0.99), v.iter().sum::<f64>() / v.len() as f64)
}

fn band_k(inst: &Instance, g: Gid, lo: i64, hi: i64) -> usize {
    inst.gid
        .iter()
        .zip(&inst.cents)
        .filter(|(&gg, &a)| gg == g && a > lo && a <= hi)
        .count()
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let limit: usize = args.get(1).map(|x| x.parse().unwrap()).unwrap_or(0);
    let per: usize = args.get(2).map(|x| x.parse().unwrap()).unwrap_or(20);
    let mut inst = Arc::new(load("instance.bin"));
    if limit > 0 && limit < inst.rows {
        let i = Arc::make_mut(&mut inst);
        i.rows = limit;
        i.gid.truncate(limit);
        i.cents.truncate(limit);
    }
    let muls = [1.01f64, 1.02, 1.10, 1.50, 2.00];
    let g = inst.gmax;
    let scope = inst.per_group[g as usize].len();
    let reps = 200usize;

    // ---------------- control: hand-written delta, no framework ----------------
    let (mut hd, mut hc, mut hk) = (Vec::new(), Vec::new(), Vec::new());
    let arr = &inst.per_group[g as usize];
    for &m in muls.iter() {
        let lo = inst.theta;
        let hi = (inst.theta as f64 * m) as i64;
        let k0 = arr.partition_point(|&x| x <= lo);
        let k1 = arr.partition_point(|&x| x <= hi);
        for _ in 0..per {
            // certification: two rank probes, answer = "is the range empty"
            let t0 = Instant::now();
            let mut hit = 0usize;
            for _ in 0..reps {
                let a = arr.partition_point(|&x| x <= black_box(lo));
                let b = a + usize::from(a < arr.len() && arr[a] <= hi);
                hit += black_box(b - a);
            }
            hc.push(t0.elapsed().as_secs_f64() * 1e9 / reps as f64);
            // delivery: the k rows must leave the view (here: be visited)
            let t0 = Instant::now();
            let mut s = 0usize;
            for _ in 0..reps {
                let a = arr.partition_point(|&x| x <= black_box(lo));
                let b = arr.partition_point(|&x| x <= black_box(hi));
                s += arr[a..b].iter().map(|&x| black_box(x)).count();
            }
            hd.push(t0.elapsed().as_secs_f64() * 1e9 / reps as f64);
            assert_eq!(s / reps, k1 - k0);
            assert_eq!(hit / reps, usize::from(k1 > k0));
        }
        hk.push(k1 - k0);
    }

    // ---------------- control: scope recompute and a full scan ----------------
    let mut hs = Vec::new();
    for &m in muls.iter() {
        let hi = (inst.theta as f64 * m) as i64;
        let t0 = Instant::now();
        for _ in 0..reps {
            black_box(arr.iter().filter(|&&x| x > black_box(hi)).count());
        }
        hs.push(t0.elapsed().as_secs_f64() * 1e9 / reps as f64);
    }
    let t0 = Instant::now();
    let fscan = {
        let mut c = 0usize;
        for _ in 0..5 {
            c = black_box(inst.cents.iter().filter(|&&x| x > black_box(inst.theta)).count());
        }
        c
    };
    let full_scan_ms = t0.elapsed().as_secs_f64() * 1e3 / 5.0;

    let (dm, df, dcs, dms) = run_arm(inst.clone(), g, &muls, per, 0);
    let (dc, _cf, _cks, _cseen) = run_arm(inst.clone(), g, &muls, per, 1);
    let rebuild = run_rebuild(inst.clone(), g, &muls[..2]);

    let (m50, m99, mavg) = stats(dm.clone());
    let (c50, c99, cavg) = stats(dc.clone());
    let (f50, _f99, favg) = stats(df.clone());
    let (_h50, _h99, havg) = stats(hd.clone());
    let (_hc50, _hc99, hcavg) = stats(hc.clone());

    println!("{{");
    println!(" \"instance\": {{\"rows\": {}, \"groups\": 5410, \"scope_rows\": {},\
              \"theta\": {}, \"multipliers\": {:?}, \"reps_per_point\": {}}},",
        inst.rows, scope, inst.theta, muls, per);
    println!(" \"dd_deliver_us\": {:?},", dm.iter().map(|x| (x * 10.0).round() / 10.0).collect::<Vec<_>>());
    println!(" \"dd_certify_us\": {:?},", dc.iter().map(|x| (x * 10.0).round() / 10.0).collect::<Vec<_>>());
    println!(" \"dd_floor_us\": {:?},", df.iter().map(|x| (x * 10.0).round() / 10.0).collect::<Vec<_>>());
    println!(" \"dd_seen_rows\": {:?},", dms);
    println!(" \"dd_band_k\": {:?},", dcs);
    println!(" \"hand_deliver_ns\": {:?},", hd.iter().map(|x| x.round()).collect::<Vec<_>>());
    println!(" \"hand_certify_ns\": {:?},", hc.iter().map(|x| x.round()).collect::<Vec<_>>());
    println!(" \"hand_band_k\": {:?},", hk);
    println!(" \"summary\": {{");
    println!("  \"dd_deliver_mean_us\": {:.1}, \"dd_deliver_p50_us\": {:.1}, \"dd_deliver_p99_us\": {:.1},", mavg, m50, m99);
    println!("  \"dd_certify_mean_us\": {:.1}, \"dd_certify_p50_us\": {:.1}, \"dd_certify_p99_us\": {:.1},", cavg, c50, c99);
    println!("  \"dd_floor_mean_us\": {:.1}, \"dd_floor_p50_us\": {:.1},", favg, f50);
    println!("  \"certify_over_deliver_engine\": {:.3},", cavg / mavg);
    println!("  \"delta_above_floor_us\": {:.1},", mavg - favg);
    println!("  \"hand_deliver_mean_ns\": {:.0}, \"hand_certify_mean_ns\": {:.0},", havg, hcavg);
    println!("  \"certify_over_deliver_hand\": {:.3},", hcavg / havg);
    println!("  \"hand_scope_mean_ns\": {:.0},", stats(hs.clone()).2);
    println!("  \"hand_full_scan_ms\": {:.2},", full_scan_ms);
    println!("  \"rebuild_ms\": {:?},", rebuild);
    println!("  \"load_and_arrange_ms\": {:.0}", 0.0);
    println!(" }}");
    println!("}}");
}

/// One obligation, one dataflow. `mode` 0 = deliver the delta (output: k claim ids),
/// 1 = certify (output: one count row). Returns (latencies, k, rows_seen) plus the
/// per-step floor measured in the same dataflow with the parameter left alone.
fn run_arm(
    inst: Arc<Instance>,
    g: Gid,
    muls: &[f64],
    per: usize,
    mode: u8,
) -> (Vec<f64>, Vec<f64>, Vec<usize>, Vec<usize>) {
    let muls: Vec<f64> = muls.to_vec();
    timely::execute_directly(move |worker| {
        let mut claims = InputSession::<u64, (Gid, (Cid, i64)), isize>::new();
        let mut params = InputSession::<u64, (Gid, i64), isize>::new();
        let mut probe = Handle::new();
        worker.dataflow::<u64, _, _>(|scope| {
            let c = claims.to_collection(scope);
            let p = params.to_collection(scope);
            let viol = c
                .join(p)
                .filter(|x| {
                    let ((_cid, a), th) = &x.1;
                    a > th
                })
                .map(|x| x.1 .0 .0);
            if mode == 0 {
                viol.consolidate()
                    .inspect(|_x| SEEN.with(|s| s.set(s.get() + 1)))
                    .probe_with(&mut probe);
            } else {
                viol.map(|_c| ())
                    .count_total()
                    .consolidate()
                    .probe_with(&mut probe);
            }
        });

        let t_load = Instant::now();
        for i in 0..inst.rows {
            claims.update((inst.gid[i], (i as Cid, inst.cents[i])), 1);
        }
        params.update((g, inst.theta), 1);
        claims.advance_to(1);
        claims.flush();
        params.advance_to(1);
        params.flush();
        while probe.less_than(&1u64) {
            worker.step();
        }
        eprintln!("[arm {}] load+arrange {:.0} ms", mode, t_load.elapsed().as_secs_f64() * 1e3);

        let mut t = 1u64;
        let (mut lat, mut floor, mut ks, mut seen) =
            (Vec::new(), Vec::new(), Vec::new(), Vec::new());

        for _ in 0..per {
            t += 1;
            claims.advance_to(t);
            claims.flush();
            params.advance_to(t);
            params.flush();
            let t0 = Instant::now();
            while probe.less_than(&t) {
                worker.step();
            }
            floor.push(t0.elapsed().as_secs_f64() * 1e6);
        }

        for &m in muls.iter() {
            for j in 0..per {
                let up = j % 2 == 0;
                let hi = (inst.theta as f64 * m) as i64;
                let (old, new) = if up { (inst.theta, hi) } else { (hi, inst.theta) };
                let kk = band_k(&inst, g, inst.theta.min(hi), inst.theta.max(hi));
                if up {
                    SEEN.with(|s| s.set(0));
                }
                t += 1;
                claims.advance_to(t);
                claims.flush();
                let t0 = Instant::now();
                params.update((g, old), -1);
                params.update((g, new), 1);
                params.advance_to(t);
                params.flush();
                while probe.less_than(&t) {
                    worker.step();
                }
                lat.push(t0.elapsed().as_secs_f64() * 1e6);
                if up {
                    ks.push(kk);
                    seen.push(SEEN.with(|s| s.get()));
                }
                t += 1;
            }
        }
        (lat, floor, ks, seen)
    })
}

fn run_rebuild(inst: Arc<Instance>, g: Gid, muls: &[f64]) -> Vec<f64> {
    let muls: Vec<f64> = muls.to_vec();
    timely::execute_directly(move |worker| {
        let mut claims = InputSession::<u64, (Gid, (Cid, i64)), isize>::new();
        let mut params = InputSession::<u64, (Gid, i64), isize>::new();
        let mut probe = Handle::new();
        worker.dataflow::<u64, _, _>(|scope| {
            let c = claims.to_collection(scope);
            let p = params.to_collection(scope);
            c.join(p)
                .filter(|x| {
                    let ((_cid, a), th) = &x.1;
                    a > th
                })
                .map(|x| x.1 .0 .0)
                .consolidate()
                .probe_with(&mut probe);
        });
        let mut out = Vec::new();
        let mut t = 0u64;
        for &m in muls.iter() {
            let hi = (inst.theta as f64 * m) as i64;
            t += 1;
            let t0 = Instant::now();
            for i in 0..inst.rows {
                claims.update((inst.gid[i], (i as Cid, inst.cents[i])), 1);
            }
            params.update((g, hi), 1);
            claims.advance_to(t);
            claims.flush();
            params.advance_to(t);
            params.flush();
            while probe.less_than(&t) {
                worker.step();
            }
            for i in 0..inst.rows {
                claims.update((inst.gid[i], (i as Cid, inst.cents[i])), -1);
            }
            params.update((g, hi), -1);
            claims.advance_to(t + 1);
            claims.flush();
            params.advance_to(t + 1);
            params.flush();
            while probe.less_than(&(t + 1)) {
                worker.step();
            }
            out.push(t0.elapsed().as_secs_f64() * 1e3);
            t += 1;
        }
        out
    })
}
