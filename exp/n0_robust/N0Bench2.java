// n0 robustness -- Java backend, repeated-timing edition.
//
// Same semantics as N0Bench: rows (v: long, mod: byte), verdict
// v >= theta AND NOT (mod AND omega); path R visits the scope's n rows,
// path B visits the k candidate ids the rank probe already located.
//
// Differences from N0Bench, both needed for the robustness study:
//   1. every repetition is emitted, not just the minimum, so the interval on
//      n0 can be bootstrapped from measured spread instead of assumed noise;
//   2. the rank probe is restricted to the group (PER_GROUP entries), which is
//      the scope it actually bounds -- N0Bench searched the whole array.

import java.io.*;
import java.nio.*;
import java.nio.file.*;
import java.util.*;

public class N0Bench2 {

    static long[] v;
    static byte[] mod;
    static int[] cand;
    static long sink = 0;
    static int perGroup;

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

    static int pathR(int lo, int n, int theta, int omega) {
        int c = 0;
        for (int i = lo; i < lo + n; i++) {
            if (v[i] >= theta && !((mod[i] & 1) == 1 && omega == 1)) c++;
        }
        return c;
    }

    static int pathB(int k, int theta, int omega) {
        int c = 0;
        for (int j = 0; j < k; j++) {
            int i = cand[j];
            if (v[i] >= theta && !((mod[i] & 1) == 1 && omega == 1)) c++;
        }
        return c;
    }

    static int pathRank(int theta) {
        int lo = 0, hi = perGroup;
        return upperBound(lo, hi, theta + 1000) - lowerBound(lo, hi, theta - 1000);
    }

    static int lowerBound(int lo, int hi, long t) {
        while (lo < hi) {
            int m = (lo + hi) >>> 1;
            if (v[m] < t) lo = m + 1; else hi = m;
        }
        return lo;
    }

    static int upperBound(int lo, int hi, long t) {
        while (lo < hi) {
            int m = (lo + hi) >>> 1;
            if (v[m] <= t) lo = m + 1; else hi = m;
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
        perGroup = Integer.parseInt(args[3]);
        int theta = Integer.parseInt(args[4]);
        int omega = Integer.parseInt(args[5]);
        int reps = Integer.parseInt(args[6]);
        int[] ns = Arrays.stream(args[7].split(",")).mapToInt(Integer::parseInt).toArray();
        int[] ks = Arrays.stream(args[8].split(",")).mapToInt(Integer::parseInt).toArray();

        for (int i = 0; i < 2000; i++) {
            sink += pathR(0, perGroup, theta, omega);
            sink += pathB(1024, theta, omega);
            sink += pathRank(theta);
        }

        StringBuilder sb = new StringBuilder();
        sb.append("{\"points_R\":[");
        for (int i = 0; i < ns.length; i++) {
            final int n = ns[i];
            sink += pathR(0, n, theta, omega);
            double[] s = repsSeries(() -> pathR(0, n, theta, omega), reps);
            if (i > 0) sb.append(",");
            sb.append("{\"n\":").append(n).append(",\"ns\":")
              .append((long) Arrays.stream(s).min().getAsDouble())
              .append(",\"reps\":").append(arr(s)).append("}");
        }
        sb.append("],\"points_B\":[");
        for (int i = 0; i < ks.length; i++) {
            final int k = ks[i];
            sink += pathB(k, theta, omega);
            double[] s = repsSeries(() -> pathB(k, theta, omega), reps);
            if (i > 0) sb.append(",");
            sb.append("{\"k\":").append(k).append(",\"ns\":")
              .append((long) Arrays.stream(s).min().getAsDouble())
              .append(",\"reps\":").append(arr(s)).append("}");
        }
        sb.append("],");
        double fr = timeAvg(() -> pathR(0, 0, theta, omega), 20000);
        double fb = timeAvg(() -> pathB(0, theta, omega), 20000);
        double rk = timeAvg(() -> pathRank(theta), 20000);
        sb.append("\"fixed_R_ns\":").append(String.format(java.util.Locale.ROOT, "%.2f", fr));
        sb.append(",\"fixed_B_ns\":").append(String.format(java.util.Locale.ROOT, "%.2f", fb));
        sb.append(",\"rank_ns\":").append(String.format(java.util.Locale.ROOT, "%.2f", rk));
        sb.append(",\"rank_count\":").append(pathRank(theta));
        sb.append(",\"sink\":").append(sink).append("}");
        System.out.println(sb.toString());
    }
}
