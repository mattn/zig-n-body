const std = @import("std");

const solar_mass = 4.0 * std.math.pi * std.math.pi;
const days_per_year = 365.24;

const n_bodies = 5;
const n_pairs = n_bodies * (n_bodies - 1) / 2; // 10
const simd_width = 4;
const n_padded = ((n_pairs + simd_width - 1) / simd_width) * simd_width; // 12
const n_chunks = n_padded / simd_width; // 3

const V4 = @Vector(simd_width, f64);

// SoA layout
var x: [n_bodies]f64 = undefined;
var y: [n_bodies]f64 = undefined;
var z: [n_bodies]f64 = undefined;
var vx: [n_bodies]f64 = undefined;
var vy: [n_bodies]f64 = undefined;
var vz: [n_bodies]f64 = undefined;
var mass: [n_bodies]f64 = undefined;

// Comptime pair index tables
const pair_i = blk: {
    var arr: [n_padded]usize = undefined;
    var k: usize = 0;
    for (0..n_bodies) |i| {
        for (i + 1..n_bodies) |_| {
            arr[k] = i;
            k += 1;
        }
    }
    while (k < n_padded) : (k += 1) arr[k] = 0;
    break :blk arr;
};

const pair_j = blk: {
    var arr: [n_padded]usize = undefined;
    var k: usize = 0;
    for (0..n_bodies) |i| {
        for (i + 1..n_bodies) |j| {
            arr[k] = j;
            k += 1;
        }
    }
    while (k < n_padded) : (k += 1) arr[k] = 0;
    break :blk arr;
};

fn initBodies() void {
    const init_x = [n_bodies]f64{ 0, 4.84143144246472090e+00, 8.34336671824457987e+00, 1.28943695621391310e+01, 1.53796971148509165e+01 };
    const init_y = [n_bodies]f64{ 0, -1.16032004402742839e+00, 4.12479856412430479e+00, -1.51111514016986312e+01, -2.59193146099879641e+01 };
    const init_z = [n_bodies]f64{ 0, -1.03622044471123109e-01, -4.03523417114321381e-01, -2.23307578892655734e-01, 1.79258772950371181e-01 };
    const init_vx = [n_bodies]f64{ 0, 1.66007664274403694e-03 * days_per_year, -2.76742510726862411e-03 * days_per_year, 2.96460137564761618e-03 * days_per_year, 2.68067772490389322e-03 * days_per_year };
    const init_vy = [n_bodies]f64{ 0, 7.69901118419740425e-03 * days_per_year, 4.99852801234917238e-03 * days_per_year, 2.37847173959480950e-03 * days_per_year, 1.62824170038242295e-03 * days_per_year };
    const init_vz = [n_bodies]f64{ 0, -6.90460016972063023e-05 * days_per_year, 2.30417297573763929e-05 * days_per_year, -2.96589568540237556e-05 * days_per_year, -9.51592254519715870e-05 * days_per_year };
    const init_mass = [n_bodies]f64{ solar_mass, 9.54791938424326609e-04 * solar_mass, 2.85885980666130812e-04 * solar_mass, 4.36624404335156298e-05 * solar_mass, 5.15138902046611451e-05 * solar_mass };

    x = init_x;
    y = init_y;
    z = init_z;
    vx = init_vx;
    vy = init_vy;
    vz = init_vz;
    mass = init_mass;
}

fn advance(dt: f64) void {
    // SoA pair deltas stored flat for SIMD
    var dx: [n_padded]f64 align(32) = undefined;
    var dy: [n_padded]f64 align(32) = undefined;
    var dz: [n_padded]f64 align(32) = undefined;
    var mag_arr: [n_padded]f64 align(32) = undefined;

    // Compute deltas (unrolled at comptime)
    inline for (0..n_pairs) |k| {
        dx[k] = x[pair_i[k]] - x[pair_j[k]];
        dy[k] = y[pair_i[k]] - y[pair_j[k]];
        dz[k] = z[pair_i[k]] - z[pair_j[k]];
    }
    // Pad
    inline for (n_pairs..n_padded) |k| {
        dx[k] = 0;
        dy[k] = 0;
        dz[k] = 0;
    }

    // SIMD: compute mag = dt / (dsq * sqrt(dsq))
    const dt_v: V4 = @splat(dt);
    inline for (0..n_chunks) |c| {
        const base = c * simd_width;
        const vdx: V4 = @as(*align(32) const [simd_width]f64, @ptrCast(dx[base..][0..simd_width])).*;
        const vdy: V4 = @as(*align(32) const [simd_width]f64, @ptrCast(dy[base..][0..simd_width])).*;
        const vdz: V4 = @as(*align(32) const [simd_width]f64, @ptrCast(dz[base..][0..simd_width])).*;
        const dsq = vdx * vdx + vdy * vdy + vdz * vdz;
        const dist = @sqrt(dsq);
        const mag_v = dt_v / (dsq * dist);
        @as(*align(32) [simd_width]f64, @ptrCast(mag_arr[base..][0..simd_width])).* = mag_v;
    }

    // Update velocities (unrolled at comptime for zero-overhead indexing)
    inline for (0..n_pairs) |k| {
        const i = pair_i[k];
        const j = pair_j[k];
        const m = mag_arr[k];
        const mmj = mass[j] * m;
        const mmi = mass[i] * m;

        vx[i] -= dx[k] * mmj;
        vy[i] -= dy[k] * mmj;
        vz[i] -= dz[k] * mmj;
        vx[j] += dx[k] * mmi;
        vy[j] += dy[k] * mmi;
        vz[j] += dz[k] * mmi;
    }

    // Update positions
    inline for (0..n_bodies) |i| {
        x[i] += dt * vx[i];
        y[i] += dt * vy[i];
        z[i] += dt * vz[i];
    }
}

fn energy() f64 {
    var e: f64 = 0.0;
    for (0..n_bodies) |i| {
        e += 0.5 * mass[i] * (vx[i] * vx[i] + vy[i] * vy[i] + vz[i] * vz[i]);
        for (i + 1..n_bodies) |j| {
            const ddx = x[i] - x[j];
            const ddy = y[i] - y[j];
            const ddz = z[i] - z[j];
            const dsq = ddx * ddx + ddy * ddy + ddz * ddz;
            e -= (mass[i] * mass[j]) / @sqrt(dsq);
        }
    }
    return e;
}

fn offsetMomentum() void {
    var px: f64 = 0;
    var py: f64 = 0;
    var pz: f64 = 0;
    for (0..n_bodies) |i| {
        px += vx[i] * mass[i];
        py += vy[i] * mass[i];
        pz += vz[i] * mass[i];
    }
    vx[0] = -px / solar_mass;
    vy[0] = -py / solar_mass;
    vz[0] = -pz / solar_mass;
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

    for (0..n) |_| {
        advance(0.01);
    }

    try stdout.print("{d:.9}\n", .{energy()});
    try stdout.flush();
}
