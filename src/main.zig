const std = @import("std");

const solar_mass = 4.0 * std.math.pi * std.math.pi;
const days_per_year = 365.24;

const N = 5;
const PAIRS = N * (N - 1) / 2; // 10

const V4 = @Vector(4, f64);
const V4f = @Vector(4, f32);

// Approximate 1/sqrt(x) using vrsqrtps + Goldschmidt refinement (matches C version)
inline fn rsqrt_v4(s: V4) V4 {
    const q: V4f = @floatCast(s);
    const r: V4f = asm ("vrsqrtps %[src], %[dst]"
        : [dst] "=x" (-> V4f),
        : [src] "x" (q),
    );
    const x_init: V4 = @floatCast(r);
    const y = s * x_init * x_init;
    const a = y * @as(V4, @splat(0.375)) * y;
    const b = y * @as(V4, @splat(1.25)) - @as(V4, @splat(1.875));
    return x_init * (a - b);
}

// Horizontal sum of squared components for 4 V4 vectors -> V4 of distance²
// Each V4 is (_, x, y, z); we want x²+y²+z² for each
inline fn hsum_sq4(r0: V4, r1: V4, r2: V4, r3: V4) V4 {
    const x0 = r0 * r0;
    const x1 = r1 * r1;
    const x2 = r2 * r2;
    const x3 = r3 * r3;

    // hadd pairs: t0 = hadd(x0, x1), t1 = hadd(x2, x3)
    const t0: V4 = asm ("vhaddpd %[b], %[a], %[dst]"
        : [dst] "=x" (-> V4),
        : [a] "x" (x0),
          [b] "x" (x1),
    );
    const t1: V4 = asm ("vhaddpd %[b], %[a], %[dst]"
        : [dst] "=x" (-> V4),
        : [a] "x" (x2),
          [b] "x" (x3),
    );

    // permute + blend to finish horizontal reduction
    const y0: V4 = asm ("vperm2f128 $0x21, %[b], %[a], %[dst]"
        : [dst] "=x" (-> V4),
        : [a] "x" (t0),
          [b] "x" (t1),
    );
    const y1: V4 = asm ("vblendpd $0b1100, %[b], %[a], %[dst]"
        : [dst] "=x" (-> V4),
        : [a] "x" (t0),
          [b] "x" (t1),
    );

    return y0 + y1;
}

// Body data: each body stored as V4 = (_, x, y, z)
var mass: [N]f64 = undefined;
var pos: [N]V4 = undefined;
var vel: [N]V4 = undefined;

fn initBodies() void {
    // sun
    mass[0] = solar_mass;
    pos[0] = @splat(0.0);
    vel[0] = @splat(0.0);

    // jupiter
    mass[1] = 9.54791938424326609e-04 * solar_mass;
    pos[1] = V4{ 0, 4.84143144246472090e+00, -1.16032004402742839e+00, -1.03622044471123109e-01 };
    vel[1] = V4{ 0, 1.66007664274403694e-03 * days_per_year, 7.69901118419740425e-03 * days_per_year, -6.90460016972063023e-05 * days_per_year };

    // saturn
    mass[2] = 2.85885980666130812e-04 * solar_mass;
    pos[2] = V4{ 0, 8.34336671824457987e+00, 4.12479856412430479e+00, -4.03523417114321381e-01 };
    vel[2] = V4{ 0, -2.76742510726862411e-03 * days_per_year, 4.99852801234917238e-03 * days_per_year, 2.30417297573763929e-05 * days_per_year };

    // uranus
    mass[3] = 4.36624404335156298e-05 * solar_mass;
    pos[3] = V4{ 0, 1.28943695621391310e+01, -1.51111514016986312e+01, -2.23307578892655734e-01 };
    vel[3] = V4{ 0, 2.96460137564761618e-03 * days_per_year, 2.37847173959480950e-03 * days_per_year, -2.96589568540237556e-05 * days_per_year };

    // neptune
    mass[4] = 5.15138902046611451e-05 * solar_mass;
    pos[4] = V4{ 0, 1.53796971148509165e+01, -2.59193146099879641e+01, 1.79258772950371181e-01 };
    vel[4] = V4{ 0, 2.68067772490389322e-03 * days_per_year, 1.62824170038242295e-03 * days_per_year, -9.51592254519715870e-05 * days_per_year };
}

fn offsetMomentum() void {
    var o: V4 = @splat(0.0);
    for (0..N) |i| {
        o += @as(V4, @splat(mass[i])) * vel[i];
    }
    vel[0] = o * @as(V4, @splat(-1.0 / solar_mass));
}

// Compute pair deltas and rsqrt of distances
inline fn kernel(r: *[PAIRS + 3]V4, w: *align(32) [PAIRS + 3]f64) void {
    var k: usize = 0;
    for (1..N) |i| {
        for (0..i) |j| {
            r[k] = pos[i] - pos[j];
            k += 1;
        }
    }

    // Process 4 pairs at a time
    comptime var ck: usize = 0;
    inline while (ck < PAIRS) : (ck += 4) {
        const dsq = hsum_sq4(r[ck], r[ck + 1], r[ck + 2], r[ck + 3]);
        const inv = rsqrt_v4(dsq);
        @as(*align(32) [4]f64, @ptrCast(&w[ck])).* = @bitCast(inv);
    }
}

fn advance(n_steps: usize, dt: f64) void {
    var r: [PAIRS + 3]V4 = undefined;
    var w: [PAIRS + 3]f64 align(32) = undefined;

    // Pad with 1.0 to avoid division by zero in rsqrt
    r[PAIRS] = @splat(1.0);
    r[PAIRS + 1] = @splat(1.0);
    r[PAIRS + 2] = @splat(1.0);

    const rt: V4 = @splat(dt);

    var rm: [N]V4 = undefined;
    for (0..N) |i| {
        rm[i] = @splat(mass[i]);
    }

    for (0..n_steps) |_| {
        kernel(&r, &w);

        // mag = rsqrt^2 * rsqrt * dt = dt / (dsq * sqrt(dsq))
        comptime var ck: usize = 0;
        inline while (ck < PAIRS) : (ck += 4) {
            const wv: V4 = @as(*align(32) const [4]f64, @ptrCast(&w[ck])).*;
            const mag = wv * wv * wv * rt;
            @as(*align(32) [4]f64, @ptrCast(&w[ck])).* = @bitCast(mag);
        }

        // Update velocities
        var k: usize = 0;
        for (1..N) |i| {
            for (0..i) |j| {
                const t: V4 = r[k] * @as(V4, @splat(w[k]));
                vel[i] -= t * rm[j];
                vel[j] += t * rm[i];
                k += 1;
            }
        }

        // Update positions
        for (0..N) |i| {
            pos[i] += vel[i] * rt;
        }
    }
}

fn energy() f64 {
    var e: f64 = 0.0;
    for (0..N) |i| {
        const v = vel[i];
        e += 0.5 * mass[i] * (v[1] * v[1] + v[2] * v[2] + v[3] * v[3]);
        for (i + 1..N) |j| {
            const d = pos[i] - pos[j];
            const dsq = d[1] * d[1] + d[2] * d[2] + d[3] * d[3];
            e -= (mass[i] * mass[j]) / @sqrt(dsq);
        }
    }
    return e;
}

pub fn main() !void {
    var buf: [4096]u8 = undefined;
    var w = std.fs.File.stdout().writer(&buf);
    const stdout = &w.interface;
    const args = try std.process.argsAlloc(std.heap.page_allocator);
    defer std.process.argsFree(std.heap.page_allocator, args);

    const n: usize = if (args.len > 1)
        try std.fmt.parseInt(usize, args[1], 10)
    else
        1000;

    initBodies();
    offsetMomentum();
    try stdout.print("{d:.9}\n", .{energy()});

    advance(n, 0.01);

    try stdout.print("{d:.9}\n", .{energy()});
    try stdout.flush();
}
