const std = @import("std");

const solar_mass = 4.0 * std.math.pi * std.math.pi;
const days_per_year = 365.24;

const Body = struct {
    x: V3,
    v: V3,
    mass: f64,
};

const V3 = @Vector(3, f64);

const n_bodies = 5;

var bodies = [n_bodies]Body{
    // Sun
    .{
        .x = V3{ 0, 0, 0 },
        .v = V3{ 0, 0, 0 },
        .mass = solar_mass,
    },
    // Jupiter
    .{
        .x = V3{
            4.84143144246472090e+00,
            -1.16032004402742839e+00,
            -1.03622044471123109e-01,
        },
        .v = V3{
            1.66007664274403694e-03 * days_per_year,
            7.69901118419740425e-03 * days_per_year,
            -6.90460016972063023e-05 * days_per_year,
        },
        .mass = 9.54791938424326609e-04 * solar_mass,
    },
    // Saturn
    .{
        .x = V3{
            8.34336671824457987e+00,
            4.12479856412430479e+00,
            -4.03523417114321381e-01,
        },
        .v = V3{
            -2.76742510726862411e-03 * days_per_year,
            4.99852801234917238e-03 * days_per_year,
            2.30417297573763929e-05 * days_per_year,
        },
        .mass = 2.85885980666130812e-04 * solar_mass,
    },
    // Uranus
    .{
        .x = V3{
            1.28943695621391310e+01,
            -1.51111514016986312e+01,
            -2.23307578892655734e-01,
        },
        .v = V3{
            2.96460137564761618e-03 * days_per_year,
            2.37847173959480950e-03 * days_per_year,
            -2.96589568540237556e-05 * days_per_year,
        },
        .mass = 4.36624404335156298e-05 * solar_mass,
    },
    // Neptune
    .{
        .x = V3{
            1.53796971148509165e+01,
            -2.59193146099879641e+01,
            1.79258772950371181e-01,
        },
        .v = V3{
            2.68067772490389322e-03 * days_per_year,
            1.62824170038242295e-03 * days_per_year,
            -9.51592254519715870e-05 * days_per_year,
        },
        .mass = 5.15138902046611451e-05 * solar_mass,
    },
};

const n_pairs = n_bodies * (n_bodies - 1) / 2;

fn advance(bs: *[n_bodies]Body, dt: f64) void {
    const dt_v: V3 = @splat(dt);

    var dx: [n_pairs]V3 = undefined;
    {
        var k: usize = 0;
        for (0..n_bodies) |i| {
            for (i + 1..n_bodies) |j| {
                dx[k] = bs[i].x - bs[j].x;
                k += 1;
            }
        }
    }

    var mag: [n_pairs]f64 = undefined;
    for (0..n_pairs) |k| {
        const d = dx[k];
        const dsq = @reduce(.Add, d * d);
        const dist = @sqrt(dsq);
        mag[k] = dt / (dsq * dist);
    }

    {
        var k: usize = 0;
        for (0..n_bodies) |i| {
            for (i + 1..n_bodies) |j| {
                const m_j: V3 = @splat(bs[j].mass * mag[k]);
                const m_i: V3 = @splat(bs[i].mass * mag[k]);
                bs[i].v -= dx[k] * m_j;
                bs[j].v += dx[k] * m_i;
                k += 1;
            }
        }
    }

    for (bs) |*b| {
        b.x += dt_v * b.v;
    }
}

fn energy(bs: []const Body) f64 {
    var e: f64 = 0.0;
    for (bs, 0..) |b, i| {
        e += 0.5 * b.mass * @reduce(.Add, b.v * b.v);
        for (bs[i + 1 ..]) |b2| {
            const dx = b.x - b2.x;
            const dsq = @reduce(.Add, dx * dx);
            e -= (b.mass * b2.mass) / @sqrt(dsq);
        }
    }
    return e;
}

fn offsetMomentum(bs: *[n_bodies]Body) void {
    var p = V3{ 0, 0, 0 };
    for (bs[1..]) |b| {
        p += @as(V3, @splat(b.mass)) * b.v;
    }
    bs[0].v = -p / @as(V3, @splat(solar_mass));
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

    offsetMomentum(&bodies);
    try stdout.print("{d:.9}\n", .{energy(&bodies)});

    for (0..n) |_| {
        advance(&bodies, 0.01);
    }

    try stdout.print("{d:.9}\n", .{energy(&bodies)});
    try stdout.flush();
}
