// n0 holdout -- Java backend, corrected scope origin.
//
// Defects this file fixes, both present in N0Bench2:
//
//  1. N0Bench2 scanned v[0..n) while every Python backend scanned
//     v[LOFF..LOFF+n).  The table is value-sorted, so v[0] ~ 0 and v[LOFF] is
//     the median: the Java path evaluated a predicate that was false on every
//     row.  Counts were never checked for the JVM rows, so the mismatch was
//     silent.  The scan now starts at LOFF.
//  2. N0Bench2 binary-searched [0, perGroup) for the rank probe, i.e. the
//     smallest values in the table rather than the scope's values.  The probe
//     now searches a sorted copy of the scope's values, which is what a
//     per-group ordered index actually holds.
//
// The instance is stored in insertion order (see n0_holdout.py): a value band
// then selects scattered rows, which is the access pattern path B exists to
// pay for.  The probe reads a prebuilt ordered array, so no sort is charged to
// either path.

import java.io.*;
import java.nio.*;
import java.nio.file.*;
import java.util.*;

public class N0Bench3 {

    static long[] v;
    static byte[] mod;
    static int[] cand;
    static long[] gsorted;
    static long sink = 0;

    static long[] readNpyLong(String path) throws IOException {
        byte[] all = Files.readAllBytes(Paths.get(path));
        int hdr = 10 + ((all[8] & 0xff) | ((all[9] & 0xff) << 8));
        ByteBuffer bb = ByteBuffer.wrap(all, hdr, all.length - hdr)
                                 .order(ByteOrder.LITTLE_ENDIAN);
        long[] a = new long[bb.remaining() / 8];
        bb.asLongBuffer().get(a);
        return a;
    }

    static byte[] readNpyByte(String path) throws IOException {
        byte[] all = Files.readAllBytes(Paths.get(path));
        int hdr = 10 + ((all[8] & 0xff) | ((all[9] & 0xff) << 8));
        return Arrays.copyOfRange(all, hdr, all.length);
    }

    static int[] readNpyInt(String path) throws IOException {
        long[] a = readNpyLong(path);
        int[] b = new int[a.length];
        for (int i = 0; i < a.length; i++) b[i] = (int) a[i];
        return b;
    }

    static int pathR(int lo, int n, int theta) {
        int c = 0;
        for (int i = lo; i < lo + n; i++) {
            if (v[i] >= theta && !((mod[i] & 1) == 1)) c++;
        }
        return c;
    }

    static int pathB(int k, int theta) {
        int c = 0;
        for (int j = 0; j < k; j++) {
            int i = cand[j];
            if (v[i] >= theta && !((mod[i] & 1) == 1)) c++;
        }
        return c;
    }

    static int pathRank(int theta) {
        int lo = lowerBound(0, gsorted.length, (long) theta - 1000);
        int hi = upperBound(0, gsorted.length, (long) theta + 1000);
        return hi - lo;
    }

    static int lowerBound(int lo, int hi, long t) {
        while (lo < hi) {
            int m = (lo + hi) >>> 1;
            if (gsorted[m] < t) lo = m + 1; else hi = m;
        }
        return lo;
    }

    static int upperBound(int lo, int hi, long t) {
        while (lo < hi) {
            int m = (lo + hi) >>> 1;
            if (gsorted[m] <= t) lo = m + 1; else hi = m;
        }
        return lo;
    }

    interface IntFn { int apply(); }

    static double[] repsSeries(IntFn f, int reps) {
        double[] out = new double[reps];
        for (int i = 0; i < reps; i++) {
            long t0 = System.nanoTime();
            sink += f.apply();
            out[i] = System.nanoTime() - t0;
        }
        return out;
    }

    static double timeAvg(IntFn f, int calls) {
        long t0 = System.nanoTime();
        for (int i = 0; i < calls; i++) sink += f.apply();
        return (System.nanoTime() - t0) / (double) calls;
    }

    static String arr(double[] a) {
        StringBuilder sb = new StringBuilder("[");
        for (int i = 0; i < a.length; i++) {
            if (i > 0) sb.append(",");
            sb.append(String.format(java.util.Locale.ROOT, "%.1f", a[i]));
        }
        return sb.append("]").toString();
    }

    public static void main(String[] args) throws Exception {
        v = readNpyLong(args[0]);
        mod = readNpyByte(args[1]);
        cand = readNpyInt(args[2]);
        gsorted = readNpyLong(args[3]);
        int loff = Integer.parseInt(args[4]);
        int theta = Integer.parseInt(args[5]);
        int reps = Integer.parseInt(args[6]);
        int[] ns = Arrays.stream(args[7].split(",")).mapToInt(Integer::parseInt).toArray();
        int[] ks = Arrays.stream(args[8].split(",")).mapToInt(Integer::parseInt).toArray();

        for (int i = 0; i < 2000; i++) {
            sink += pathR(loff, Math.min(65536, v.length - loff), theta);
            sink += pathB(Math.min(1024, cand.length), theta);
            sink += pathRank(theta);
        }

        StringBuilder sb = new StringBuilder();
        sb.append("{\"counts_R\":[");
        for (int i = 0; i < ns.length; i++) {
            if (i > 0) sb.append(",");
            sb.append(pathR(loff, ns[i], theta));
        }
        sb.append("],\"counts_B\":[");
        for (int i = 0; i < ks.length; i++) {
            if (i > 0) sb.append(",");
            sb.append(pathB(ks[i], theta));
        }
        sb.append("],\"points_R\":[");
        for (int i = 0; i < ns.length; i++) {
            final int n = ns[i];
            sink += pathR(loff, n, theta);
            double[] s = repsSeries(() -> pathR(loff, n, theta), reps);
            if (i > 0) sb.append(",");
            sb.append("{\"n\":").append(n).append(",\"ns\":")
              .append((long) Arrays.stream(s).min().getAsDouble())
              .append(",\"reps\":").append(arr(s)).append("}");
        }
        sb.append("],\"points_B\":[");
        for (int i = 0; i < ks.length; i++) {
            final int k = ks[i];
            sink += pathB(k, theta);
            double[] s = repsSeries(() -> pathB(k, theta), reps);
            if (i > 0) sb.append(",");
            sb.append("{\"k\":").append(k).append(",\"ns\":")
              .append((long) Arrays.stream(s).min().getAsDouble())
              .append(",\"reps\":").append(arr(s)).append("}");
        }
        sb.append("],");
        double fr = timeAvg(() -> pathR(loff, 0, theta), 20000);
        double fb = timeAvg(() -> pathB(0, theta), 20000);
        double rk = timeAvg(() -> pathRank(theta), 20000);
        sb.append("\"fixed_R_ns\":").append(String.format(java.util.Locale.ROOT, "%.2f", fr));
        sb.append(",\"fixed_B_ns\":").append(String.format(java.util.Locale.ROOT, "%.2f", fb));
        sb.append(",\"rank_ns\":").append(String.format(java.util.Locale.ROOT, "%.2f", rk));
        sb.append(",\"rank_count\":").append(pathRank(theta));
        sb.append(",\"sink\":").append(sink).append("}");
        System.out.println(sb.toString());
    }
}
