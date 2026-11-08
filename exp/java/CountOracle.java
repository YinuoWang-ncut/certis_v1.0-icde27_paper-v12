import java.util.SplittableRandom;

/**
 * Count-aware oracle: what does it actually cost to learn k = |B|?
 *
 * The v8 paper charges the selector a rank probe -- two binary searches in each
 * affected group's sorted array -- and measured 3,851 ns per group under numpy.
 * That number is suspicious: two binary searches over 1,024 ints in L2 should be
 * two orders of magnitude cheaper than two numpy calls.  So this benchmark
 * re-prices every candidate structure inside one JVM, on one memory layout, and
 * reports ns/query and ns/update for each:
 *
 *   A SORTED    sorted int[] per group + binary search   (the probe, C-speed)
 *   B PREFIX    dense prefix sums over atoms             O(1) query / O(P) update
 *   C FENWICK   binary indexed tree over atoms           O(log P) both
 *   D SAMPLE    hash-sample of s claims per group        O(s) query / O(1) update
 *
 * A and B/C are exact.  D is approximate, and that is the point: routing errors
 * cost performance, never correctness, because both paths return exact answers.
 * So the right question is not "can we compute k for free" but "how much
 * maintenance does a k we can trust enough to route on cost, and when does that
 * beat paying a probe per correction".
 *
 * Usage: java CountOracle <atoms-per-group> <values-per-group>
 */
public class CountOracle {

    static final int G = 5410;          // groups, as in exp/rank_probe.py
    static final int S = 256;           // sample size, structure D
    static final int Q = 2000;          // queries per measurement round
    static final int U = 2000;          // updates per measurement round
    static volatile long sink;          // defeats dead-code elimination

    static int NPG;                     // values per group
    static int P;                       // atoms per group

    static int[][] sorted;              // A: NPG sorted atom indices
    static int[][] prefix;              // B: P+1 prefix sums
    static int[][] bit;                 // C: P+1 fenwick tree
    static int[][] sample;              // D: S sampled atom indices, sorted

    static int[] qLo = new int[Q], qHi = new int[Q];

    // ------------------------------------------------------------------ build

    static void build(long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        sorted = new int[G][NPG];
        prefix = new int[G][P + 1];
        bit = new int[G][P + 1];
        sample = new int[G][S];

        for (int g = 0; g < G; g++) {
            // atoms drawn without the batch structure of the real corpus: each
            // value lands on an atom uniformly, which is the worst case for a
            // count structure (P = NPG means almost no atom is reused)
            int[] v = sorted[g];
            for (int i = 0; i < NPG; i++) v[i] = r.nextInt(P);
            java.util.Arrays.sort(v);

            // B and C are built from the same multiset
            int[] cnt = new int[P];
            for (int i = 0; i < NPG; i++) cnt[v[i]]++;
            int acc = 0;
            for (int a = 0; a < P; a++) {
                prefix[g][a] = acc;
                acc += cnt[a];
            }
            prefix[g][P] = acc;
            for (int a = 0; a < P; a++) {
                for (int i = a + 1; i <= P; i += i & -i) bit[g][i] += cnt[a];
            }

            // D: S independent uniform positions, kept sorted
            for (int i = 0; i < S; i++) sample[g][i] = v[r.nextInt(NPG)];
            java.util.Arrays.sort(sample[g]);
        }
    }

    // ---------------------------------------------------------------- queries

    static int lowerBound(int[] a, int n, int key) {
        int lo = 0, hi = n;
        while (lo < hi) {
            int mid = (lo + hi) >>> 1;
            if (a[mid] < key) lo = mid + 1; else hi = mid;
        }
        return lo;
    }

    static int prefixSum(int[] t, int i) {         // sum of [0, i)
        int s = 0;
        for (; i > 0; i -= i & -i) s += t[i];
        return s;
    }

    // ---------------------------------------------------------------- timing

    interface Body { long run(); }

    static long best(Body b, int reps) {
        long m = Long.MAX_VALUE;
        for (int i = 0; i < reps; i++) {
            long t0 = System.nanoTime();
            sink += b.run();
            long dt = System.nanoTime() - t0;
            if (dt < m) m = dt;
        }
        return m;
    }

    static void drawQueries(long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        for (int i = 0; i < Q; i++) {
            int a = r.nextInt(P);
            int b = Math.min(P, a + Math.max(1, P / 8));
            qLo[i] = a; qHi[i] = b;
        }
    }

    // ------------------------------------------------------------------- main

    public static void main(String[] args) {
        P = Integer.parseInt(args[0]);
        NPG = Integer.parseInt(args[1]);
        long seed = args.length > 2 ? Long.parseLong(args[2]) : 7L;

        build(seed);
        drawQueries(seed + 1);

        // ---- queries, per group per query
        long qa = best(() -> {
            long s = 0;
            for (int i = 0; i < Q; i++)
                for (int g = 0; g < G; g++)
                    s += lowerBound(sorted[g], NPG, qHi[i])
                       - lowerBound(sorted[g], NPG, qLo[i]);
            return s;
        }, 7);

        long qb = best(() -> {
            long s = 0;
            for (int i = 0; i < Q; i++)
                for (int g = 0; g < G; g++)
                    s += prefix[g][qHi[i]] - prefix[g][qLo[i]];
            return s;
        }, 7);

        long qc = best(() -> {
            long s = 0;
            for (int i = 0; i < Q; i++)
                for (int g = 0; g < G; g++)
                    s += prefixSum(bit[g], qHi[i]) - prefixSum(bit[g], qLo[i]);
            return s;
        }, 7);

        long qd = best(() -> {
            long s = 0;
            for (int i = 0; i < Q; i++)
                for (int g = 0; g < G; g++)
                    s += lowerBound(sample[g], S, qHi[i])
                       - lowerBound(sample[g], S, qLo[i]);
            return s;
        }, 7);

        double perGroupFactor = (double) Q * G;

        // ---- a scan, so that query costs can be expressed in rows
        long sc = best(() -> {
            long s = 0;
            int lo = qLo[0];
            for (int g = 0; g < G; g++) {
                int[] v = sorted[g];
                for (int i = 0; i < NPG; i++) if (v[i] >= lo) s++;
            }
            return s;
        }, 7);
        double nsPerRowScan = (double) sc / ((double) G * NPG);

        // ---- updates: one value correction in one group
        int[] ug = new int[U], uOld = new int[U], uNew = new int[U];
        SplittableRandom ru = new SplittableRandom(seed + 2);
        for (int i = 0; i < U; i++) {
            ug[i] = ru.nextInt(G);
            uOld[i] = ru.nextInt(P);
            uNew[i] = ru.nextInt(P);
        }

        // A: delete old, insert new -- two memmoves in the sorted array
        long ua = best(() -> {
            long s = 0;
            for (int i = 0; i < U; i++) {
                int[] v = sorted[ug[i]];
                int pos = lowerBound(v, NPG, uOld[i]);
                if (pos < NPG) {                       // the value may not be there
                    System.arraycopy(v, pos + 1, v, pos, NPG - pos - 1);
                    int ins = lowerBound(v, NPG - 1, uNew[i]);
                    System.arraycopy(v, ins, v, ins + 1, NPG - 1 - ins);
                    v[ins] = uNew[i];
                }
                s += v[0];
            }
            return s;
        }, 7);

        // B: O(P) prefix update
        long ub = best(() -> {
            long s = 0;
            for (int i = 0; i < U; i++) {
                int[] pf = prefix[ug[i]];
                int d = uNew[i] >= uOld[i] ? 1 : -1;   // move one value up or down
                for (int a = Math.min(uOld[i], uNew[i]) + 1; a <= P; a++) pf[a] += d;
                s += pf[P];
            }
            return s;
        }, 7);

        // C: O(log P) fenwick update, twice
        long uc = best(() -> {
            long s = 0;
            for (int i = 0; i < U; i++) {
                int[] t = bit[ug[i]];
                for (int a = uOld[i] + 1; a <= P; a += a & -a) t[a] -= 1;
                for (int a = uNew[i] + 1; a <= P; a += a & -a) t[a] += 1;
                s += t[1];
            }
            return s;
        }, 7);

        // D: O(1) amortised -- one sampled position replaced and re-sorted in place
        long ud = best(() -> {
            long s = 0;
            for (int i = 0; i < U; i++) {
                int[] sp = sample[ug[i]];
                int pos = lowerBound(sp, S, uOld[i]);
                if (pos < S && sp[pos] == uOld[i]) {
                    System.arraycopy(sp, pos + 1, sp, pos, S - pos - 1);
                    int ins = lowerBound(sp, S - 1, uNew[i]);
                    System.arraycopy(sp, ins, sp, ins + 1, S - 1 - ins);
                    sp[ins] = uNew[i];
                }
                s += sp[0];
            }
            return s;
        }, 7);

        // ---- accuracy of the sampled oracle, and how often its verdict differs
        // The decision is k < kappa*(n - n0), i.e. at |D| = 1e6 essentially the
        // ratio r = k/n against kappa = 0.44.  A sample of S claims estimates r
        // with sd sqrt(r(1-r)/S); the margin in the corpus is 0.463 - 0.44 = 0.023,
        // so S has to be large enough to resolve a 5% gap.  That is the design
        // question, and it is answered by sweeping S rather than guessing.
        int trials = 100_000;
        SplittableRandom re = new SplittableRandom(seed + 3);
        double kappa = 0.44, n0 = 451, n = 1_000_000;
        double thr = kappa * (1.0 - n0 / n);
        int[] sizes = {256, 1024, 4096, 16384};
        System.out.println("sample-size sweep (r drawn around the corpus median "
                + "0.463, +-0.10)");
        for (int si = 0; si < sizes.length; si++) {
            int s = sizes[si];
            double[] err = new double[trials];
            int flips = 0;
            for (int t = 0; t < trials; t++) {
                double r = 0.363 + 0.20 * re.nextDouble();
                int hits = 0;
                for (int i = 0; i < s; i++) if (re.nextDouble() < r) hits++;
                double rHat = (double) hits / s;
                err[t] = Math.abs(rHat - r) / r;
                if ((r < thr) != (rHat < thr)) flips++;
            }
            java.util.Arrays.sort(err);
            System.out.printf("  S=%-6d relerr p50=%.3f p90=%.3f   routing flips=%.1f%%%n",
                    s, err[trials / 2], err[(int) (0.9 * trials)],
                    100.0 * flips / trials);
        }
        int flips = 0, decidedB = 0, decidedBexact = 0;
        double[] relErr = new double[trials];

        java.util.Arrays.sort(relErr);

        double nsPerQueryA = qa / perGroupFactor;
        double nsPerQueryB = qb / perGroupFactor;
        double nsPerQueryC = qc / perGroupFactor;
        double nsPerQueryD = qd / perGroupFactor;
        double nsPerUpdA = (double) ua / U;
        double nsPerUpdB = (double) ub / U;
        double nsPerUpdC = (double) uc / U;
        double nsPerUpdD = (double) ud / U;

        // The exact oracle replaces q probes by q queries plus build plus u
        // updates.  Ignoring the one-off build, it pays for itself while
        //      u/q  <  (probe - query) / update
        // which is the only number a loader needs to decide whether to keep one.
        double beB = (nsPerQueryA - nsPerQueryB) / nsPerUpdB;
        double beC = (nsPerQueryA - nsPerQueryC) / nsPerUpdC;
        double beD = (nsPerQueryA - nsPerQueryD) / nsPerUpdD;

        System.out.printf("atoms=%d values_per_group=%d groups=%d%n", P, NPG, G);
        System.out.printf("scan ns/row       : %.3f%n", nsPerRowScan);
        System.out.printf("probe in rows     : %.0f%n", nsPerQueryA / nsPerRowScan);
        System.out.printf("per-group query ns : A=%.1f B=%.1f C=%.1f D=%.1f%n",
                nsPerQueryA, nsPerQueryB, nsPerQueryC, nsPerQueryD);
        System.out.printf("per-group update ns: A=%.1f B=%.1f C=%.1f D=%.1f%n",
                nsPerUpdA, nsPerUpdB, nsPerUpdC, nsPerUpdD);
        System.out.printf("max updates/query : B=%.2f C=%.2f D=%.2f%n", beB, beC, beD);
        System.out.printf("probe at g=5410   : %.2f ms  (scan of |D| is 20 ms)%n",
                nsPerQueryA * G / 1e6);
        System.out.println("sink=" + sink);
    }
}
