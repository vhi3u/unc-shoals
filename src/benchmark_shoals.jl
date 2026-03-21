#!/usr/bin/env julia
#
# benchmark_shoals.jl — Ablation-style performance benchmark for flow_over_shoals.jl
#
# Usage:
#   julia +1.11.2 --project=. src/benchmark_shoals.jl              # CPU, small grid
#   julia +1.11.2 --project=. src/benchmark_shoals.jl --gpu        # GPU, small grid
#   julia +1.11.2 --project=. src/benchmark_shoals.jl --gpu --full # GPU, production grid
#   julia +1.11.2 --project=. src/benchmark_shoals.jl --steps 50   # More steps for accuracy
#
# Each level adds one component on top of the previous. The marginal cost
# of each addition reveals what is expensive.
#

using Oceananigans
using Oceananigans.Grids: Periodic, Bounded, minimum_zspacing
using Oceananigans.Units
using Oceananigans.BoundaryConditions: OpenBoundaryCondition, FieldBoundaryConditions
using Oceananigans.TurbulenceClosures
using Oceananigans.Solvers: ConjugateGradientPoissonSolver, FFTBasedPoissonSolver
using Oceananigans.Models: buoyancy_operation
using Oceananigans.OutputWriters
using Oceananigans.Forcings
using Statistics: mean
using Oceanostics: KineticEnergy, RossbyNumber
using SeawaterPolynomials.TEOS10
using Printf
using NCDatasets
using CUDA
using CUDA: has_cuda_gpu

# ═══════════════════════════════════════════════════════════════════════════
#  CLI argument parsing
# ═══════════════════════════════════════════════════════════════════════════

use_gpu  = "--gpu"  in ARGS
full_res = "--full" in ARGS

N_steps = 20                             # default number of timed steps
for (i, a) in enumerate(ARGS)
    if a == "--steps" && i < length(ARGS)
        N_steps = parse(Int, ARGS[i+1])
    end
end

if use_gpu && has_cuda_gpu()
    arch = GPU()
elseif use_gpu
    @warn "Requested GPU but no CUDA GPU found — falling back to CPU"
    arch = CPU()
else
    arch = CPU()
end

@info "Benchmark config" arch N_steps full_res

# ═══════════════════════════════════════════════════════════════════════════
#  Shared physics (copied from flow_over_shoals.jl)
# ═══════════════════════════════════════════════════════════════════════════

include("dshoal_vn_param.jl")   # parameterised shoal bathymetry

# Grid dimensions
if full_res
    Nx, Ny, Nz = (arch isa GPU ? (200, 600, 50) : (50, 150, 10))
else
    Nx, Ny, Nz = (30, 30, 10)  # small grid for fast profiling
end
Lx, Ly, Lz = 100e3, 300e3, 50.0

# ── T/S profiles (identical to flow_over_shoals.jl) ──────────────────────
const δ_smooth = 2.5
@inline _smooth_step_z(z, z0) = 0.5 * (1.0 - tanh((z - z0) / δ_smooth))

@inline function T_north_pwl(z)
    z1, z2, z3 = -5.0, -15.0, -35.0
    v1, v2, v3 = 20.5389, 17.8875, 14.3323
    m12 = (v2 - v1) / (z2 - z1); m23 = (v3 - v2) / (z3 - z2)
    val1 = v1; val2 = v1 + m12 * (z - z1); val3 = v2 + m23 * (z - z2); val4 = v3
    w1 = _smooth_step_z(z, z1); w2 = _smooth_step_z(z, z2); w3 = _smooth_step_z(z, z3)
    return val1 * (1 - w1) + val2 * (w1 - w2) + val3 * (w2 - w3) + val4 * w3
end
@inline function T_south_pwl(z)
    z1, z2, z3 = -5.0, -15.0, -30.0
    v1, v2, v3 = 24.5378, 24.3073, 23.4116
    m12 = (v2 - v1) / (z2 - z1); m23 = (v3 - v2) / (z3 - z2)
    val1 = v1; val2 = v1 + m12 * (z - z1); val3 = v2 + m23 * (z - z2); val4 = v3
    w1 = _smooth_step_z(z, z1); w2 = _smooth_step_z(z, z2); w3 = _smooth_step_z(z, z3)
    return val1 * (1 - w1) + val2 * (w1 - w2) + val3 * (w2 - w3) + val4 * w3
end
@inline function S_north_pwl(z)
    z1, z2, z3 = -5.0, -15.0, -35.0
    v1, v2, v3 = 32.6264, 33.7062, 33.2648
    m12 = (v2 - v1) / (z2 - z1); m23 = (v3 - v2) / (z3 - z2)
    val1 = v1; val2 = v1 + m12 * (z - z1); val3 = v2 + m23 * (z - z2); val4 = v3
    w1 = _smooth_step_z(z, z1); w2 = _smooth_step_z(z, z2); w3 = _smooth_step_z(z, z3)
    return val1 * (1 - w1) + val2 * (w1 - w2) + val3 * (w2 - w3) + val4 * w3
end
@inline function S_south_pwl(z)
    z1, z2, z3 = -5.0, -15.0, -30.0
    v1, v2, v3 = 35.5830, 35.9986, 36.1776
    m12 = (v2 - v1) / (z2 - z1); m23 = (v3 - v2) / (z3 - z2)
    val1 = v1; val2 = v1 + m12 * (z - z1); val3 = v2 + m23 * (z - z2); val4 = v3
    w1 = _smooth_step_z(z, z1); w2 = _smooth_step_z(z, z2); w3 = _smooth_step_z(z, z3)
    return val1 * (1 - w1) + val2 * (w1 - w2) + val3 * (w2 - w3) + val4 * w3
end

# ── Velocity profile ─────────────────────────────────────────────────────
const v₀ = 0.10
@inline function v∞(x, z, t, p)
    xC = 3e3; xS = 60e3; Lw = p.Lx
    k1 = 80 / Lw; k2 = 40 / Lw
    s1 = 1 / (1 + exp(-k1 * (x - xC)))
    s2 = 1 / (1 + exp(k2 * (x - xS)))
    s = (s1 - 1) + s2
    return p.v₀ * clamp(s, 0.0, 1.0)
end

# ── Sponge masks ─────────────────────────────────────────────────────────
@inline function south_mask(x, y, z, p)
    y1 = p.Ls
    return (0 <= y <= y1) ? (1 - y / y1) : 0.0
end
@inline function north_mask(x, y, z, p)
    y0 = p.Ly - p.Ls; y1 = p.Ly
    return (y0 <= y <= y1) ? (y - y0) / (y1 - y0) : 0.0
end
@inline function east_mask(x, y, z, p)
    x0 = p.Lx - p.Le; x1 = p.Lx
    return (x0 <= x <= x1) ? (x - x0) / (x1 - x0) : 0.0
end

# ── Sponge forcing functions (periodic_y version with min) ───────────────
@inline sponge_u(x, y, z, t, u, p) = -min(
    south_mask(x, y, z, p) * u / p.τₛ,
    north_mask(x, y, z, p) * u / p.τₙ,
    east_mask(x, y, z, p) * u / p.τₑ)
@inline sponge_v(x, y, z, t, v, p) = -min(
    south_mask(x, y, z, p) * (v - v∞(x, z, t, p)) / p.τₛ,
    north_mask(x, y, z, p) * (v - v∞(x, z, t, p)) / p.τₙ,
    east_mask(x, y, z, p) * v / p.τₑ)
@inline sponge_w(x, y, z, t, w, p) = -min(
    south_mask(x, y, z, p) * w / p.τₛ,
    north_mask(x, y, z, p) * w / p.τₙ,
    east_mask(x, y, z, p) * w / p.τₑ)
@inline sponge_T(x, y, z, t, T, p) = -min(
    south_mask(x, y, z, p) * (T - T_south_pwl(z)) / p.τ_ts,
    north_mask(x, y, z, p) * (T - T_north_pwl(z)) / p.τ_ts,
    east_mask(x, y, z, p) * (T - p.Tₑ) / p.τ_ts)
@inline sponge_S(x, y, z, t, S, p) = -min(
    south_mask(x, y, z, p) * (S - S_south_pwl(z)) / p.τ_ts,
    north_mask(x, y, z, p) * (S - S_north_pwl(z)) / p.τ_ts,
    east_mask(x, y, z, p) * (S - p.Sₑ) / p.τ_ts)

# shared parameter NamedTuple
const sponge_params = (
    Lx=Lx, Ly=Ly, Lz=Lz, v₀=v₀,
    Ls=50e3, Le=60e3,
    τₙ=6*3600.0, τₛ=6*3600.0, τₑ=24*3600.0, τ_ts=6*3600.0,
    Tₑ=23.11, Sₑ=35.5)

# ═══════════════════════════════════════════════════════════════════════════
#  Benchmark harness
# ═══════════════════════════════════════════════════════════════════════════

"""
    benchmark_timesteps(model, Δt, N_steps; warmup=3)

Run `warmup` + `N_steps` time steps and return the average walltime per
timed step (in seconds). Synchronises GPU after each step.
"""
function benchmark_timesteps(model, Δt, N_steps; warmup=3)
    sim = Simulation(model, Δt=Δt, stop_iteration=warmup + N_steps)

    # Warm-up phase (allow JIT + GPU kernel compilation)
    for _ in 1:warmup
        time_step!(model, Δt)
    end
    if arch isa GPU
        CUDA.@sync nothing   # ensure GPU finishes warm-up
    end

    # Timed phase
    t0 = time_ns()
    for _ in 1:N_steps
        time_step!(model, Δt)
    end
    if arch isa GPU
        CUDA.@sync nothing
    end
    elapsed = (time_ns() - t0) / 1e9

    # Try to read CG iterations if applicable
    cg_iters = nothing
    solver = model.pressure_solver
    if hasproperty(solver, :conjugate_gradient_solver)
        cg_iters = solver.conjugate_gradient_solver.iteration
    end

    return elapsed / N_steps, cg_iters
end

"""
    init_model!(model)

Set a simple initial condition so the model has non-trivial fields.
"""
function init_model!(model)
    u, v, w = model.velocities
    uᵢ = 0.005 * rand(size(u)...); uᵢ .-= mean(uᵢ)
    vᵢ = 0.005 * rand(size(v)...); vᵢ .-= mean(vᵢ); vᵢ .+= v₀
    wᵢ = 0.005 * rand(size(w)...); wᵢ .-= mean(wᵢ)
    set!(model, u=uᵢ, v=vᵢ, w=wᵢ,
         T=(x,y,z) -> T_south_pwl(z),
         S=(x,y,z) -> S_south_pwl(z))
end

# ═══════════════════════════════════════════════════════════════════════════
#  Build each configuration level
# ═══════════════════════════════════════════════════════════════════════════

const Δt_bench = 60.0   # 1-minute fixed timestep for all benchmarks

results = Vector{@NamedTuple{name::String, time_per_step::Float64, cg_iters::Any}}()

function run_level(name, model)
    @info "────── Benchmarking: $name ──────"
    init_model!(model)
    t, cg = benchmark_timesteps(model, Δt_bench, N_steps)
    push!(results, (; name, time_per_step=t, cg_iters=cg))
    @info @sprintf("  → %.4f s/step%s", t, isnothing(cg) ? "" : @sprintf(" (%d CG iters)", cg))
    GC.gc()   # free previous model memory
end

# ── Grids ────────────────────────────────────────────────────────────────
flat_grid = RectilinearGrid(arch;
    size=(Nx, Ny, Nz), halo=(4, 4, 4),
    x=(0, Lx), y=(0, Ly), z=(-Lz, 0),
    topology=(Bounded, Periodic, Bounded))

# Immersed boundary grid (shoal)
Hs = 15.0; sigma = 8e3; shoal_length = 20e3
slope_bottom = dshoal_param_bottom(Ly; Hs, sigma, shoal_length)
GFB = GridFittedBottom(slope_bottom)
ib_grid = ImmersedBoundaryGrid(flat_grid, GFB)

# Drag coefficient (same as flow_over_shoals.jl)
const κᵛᵏ = 0.4
const Rz = 2.5e-4
z_0 = Rz * Lz
z₁ = minimum_zspacing(flat_grid, Center(), Center(), Center()) / 2
c_dz = (κᵛᵏ / log(z₁ / z_0))^2
drag = BulkDrag(coefficient=c_dz)

# Forcings
FT = Forcing(sponge_T, field_dependencies=:T, parameters=sponge_params)
FS = Forcing(sponge_S, field_dependencies=:S, parameters=sponge_params)
Fᵤ = Forcing(sponge_u, field_dependencies=:u, parameters=sponge_params)
Fᵥ = Forcing(sponge_v, field_dependencies=:v, parameters=sponge_params)
F_w = Forcing(sponge_w, field_dependencies=:w, parameters=sponge_params)
coriolis = FPlane(latitude=35.2480)

# ═══════════════════════════════════════════════════════════════════════════
#  Level 0: Bare flat grid (FFT solver, no closures)
# ═══════════════════════════════════════════════════════════════════════════
@info "Building Level 0..."
model0 = NonhydrostaticModel(flat_grid;
    timestepper=:RungeKutta3,
    tracers=(:T, :S),
    buoyancy=SeawaterBuoyancy())
run_level("0: Bare (flat, FFT solver)", model0)

# ═══════════════════════════════════════════════════════════════════════════
#  Level 1: + Immersed boundary + CG pressure solver
# ═══════════════════════════════════════════════════════════════════════════
@info "Building Level 1..."
model1 = NonhydrostaticModel(ib_grid;
    timestepper=:RungeKutta3,
    pressure_solver=ConjugateGradientPoissonSolver(ib_grid),
    tracers=(:T, :S),
    buoyancy=SeawaterBuoyancy())
run_level("1: + IB + CG solver", model1)

# ═══════════════════════════════════════════════════════════════════════════
#  Level 2: + WENO(order=5) advection
# ═══════════════════════════════════════════════════════════════════════════
@info "Building Level 2..."
model2 = NonhydrostaticModel(ib_grid;
    timestepper=:RungeKutta3,
    advection=WENO(order=5),
    pressure_solver=ConjugateGradientPoissonSolver(ib_grid),
    tracers=(:T, :S),
    buoyancy=SeawaterBuoyancy())
run_level("2: + WENO(5)", model2)

# ═══════════════════════════════════════════════════════════════════════════
#  Level 3: + DynamicSmagorinsky closure
# ═══════════════════════════════════════════════════════════════════════════
@info "Building Level 3..."
model3 = NonhydrostaticModel(ib_grid;
    timestepper=:RungeKutta3,
    advection=WENO(order=5),
    closure=DynamicSmagorinsky(),
    pressure_solver=ConjugateGradientPoissonSolver(ib_grid),
    tracers=(:T, :S),
    buoyancy=SeawaterBuoyancy())
run_level("3: + DynamicSmagorinsky", model3)

# ═══════════════════════════════════════════════════════════════════════════
#  Level 4: + Coriolis
# ═══════════════════════════════════════════════════════════════════════════
@info "Building Level 4..."
model4 = NonhydrostaticModel(ib_grid;
    timestepper=:RungeKutta3,
    advection=WENO(order=5),
    closure=DynamicSmagorinsky(),
    pressure_solver=ConjugateGradientPoissonSolver(ib_grid),
    tracers=(:T, :S),
    buoyancy=SeawaterBuoyancy(),
    coriolis=coriolis)
run_level("4: + Coriolis (FPlane)", model4)

# ═══════════════════════════════════════════════════════════════════════════
#  Level 5: + Sponge forcings
# ═══════════════════════════════════════════════════════════════════════════
@info "Building Level 5..."
model5 = NonhydrostaticModel(ib_grid;
    timestepper=:RungeKutta3,
    advection=WENO(order=5),
    closure=DynamicSmagorinsky(),
    pressure_solver=ConjugateGradientPoissonSolver(ib_grid),
    tracers=(:T, :S),
    buoyancy=SeawaterBuoyancy(),
    coriolis=coriolis,
    forcing=(u=Fᵤ, v=Fᵥ, w=F_w, T=FT, S=FS))
run_level("5: + Sponge forcings", model5)

# ═══════════════════════════════════════════════════════════════════════════
#  Level 6: + BulkDrag on immersed boundary
# ═══════════════════════════════════════════════════════════════════════════
@info "Building Level 6..."
drag_bcs = (
    u=FieldBoundaryConditions(immersed=drag),
    v=FieldBoundaryConditions(immersed=drag),
    w=FieldBoundaryConditions(immersed=drag))
model6 = NonhydrostaticModel(ib_grid;
    timestepper=:RungeKutta3,
    advection=WENO(order=5),
    closure=DynamicSmagorinsky(),
    pressure_solver=ConjugateGradientPoissonSolver(ib_grid),
    tracers=(:T, :S),
    buoyancy=SeawaterBuoyancy(),
    coriolis=coriolis,
    forcing=(u=Fᵤ, v=Fᵥ, w=F_w, T=FT, S=FS),
    boundary_conditions=drag_bcs)
run_level("6: + BulkDrag (IB)", model6)

# ═══════════════════════════════════════════════════════════════════════════
#  Level 7: + Output writers (2D slices)
# ═══════════════════════════════════════════════════════════════════════════
@info "Building Level 7..."
model7 = NonhydrostaticModel(ib_grid;
    timestepper=:RungeKutta3,
    advection=WENO(order=5),
    closure=DynamicSmagorinsky(),
    pressure_solver=ConjugateGradientPoissonSolver(ib_grid),
    tracers=(:T, :S),
    buoyancy=SeawaterBuoyancy(),
    coriolis=coriolis,
    forcing=(u=Fᵤ, v=Fᵥ, w=F_w, T=FT, S=FS),
    boundary_conditions=drag_bcs)

u7, v7, w7 = model7.velocities
T7 = model7.tracers.T; S7 = model7.tracers.S

u7c = @at (Center, Center, Center) u7
v7c = @at (Center, Center, Center) v7
w7c = @at (Center, Center, Center) w7
KE7 = KineticEnergy(model7)
slice7 = (; u_c=u7c, v_c=v7c, w_c=w7c, T=T7, S=S7, KE=KE7)

sim7 = Simulation(model7, Δt=Δt_bench, stop_iteration=1)
sim7.output_writers[:surface] = NetCDFWriter(model7, slice7,
    filename=joinpath(tempdir(), "bench_top.nc"),
    schedule=TimeInterval(Δt_bench),
    indices=(:, :, Nz),
    overwrite_existing=true)
sim7.output_writers[:midy] = NetCDFWriter(model7, slice7,
    filename=joinpath(tempdir(), "bench_midy.nc"),
    schedule=TimeInterval(Δt_bench),
    indices=(:, round(Int, Ny/2), :),
    overwrite_existing=true)
sim7.output_writers[:midx] = NetCDFWriter(model7, slice7,
    filename=joinpath(tempdir(), "bench_midx.nc"),
    schedule=TimeInterval(Δt_bench),
    indices=(round(Int, Nx/2), :, :),
    overwrite_existing=true)

# Use simulation-based timing for this level
init_model!(model7)
for _ in 1:3; time_step!(model7, Δt_bench); end  # warmup
if arch isa GPU; CUDA.@sync nothing; end
t0 = time_ns()
for _ in 1:N_steps; time_step!(model7, Δt_bench); end
if arch isa GPU; CUDA.@sync nothing; end
t7 = (time_ns() - t0) / 1e9 / N_steps
cg7 = nothing
if hasproperty(model7.pressure_solver, :conjugate_gradient_solver)
    cg7 = model7.pressure_solver.conjugate_gradient_solver.iteration
end
push!(results, (; name="7: + Output writers (slices)", time_per_step=t7, cg_iters=cg7))
@info @sprintf("  → %.4f s/step%s", t7, isnothing(cg7) ? "" : @sprintf(" (%d CG iters)", cg7))
GC.gc()

# ═══════════════════════════════════════════════════════════════════════════
#  Level 8: + Cross-correlation diagnostics + time avg writer
# ═══════════════════════════════════════════════════════════════════════════
@info "Building Level 8..."
model8 = NonhydrostaticModel(ib_grid;
    timestepper=:RungeKutta3,
    advection=WENO(order=5),
    closure=DynamicSmagorinsky(),
    pressure_solver=ConjugateGradientPoissonSolver(ib_grid),
    tracers=(:T, :S),
    buoyancy=SeawaterBuoyancy(),
    coriolis=coriolis,
    forcing=(u=Fᵤ, v=Fᵥ, w=F_w, T=FT, S=FS),
    boundary_conditions=drag_bcs)

u8, v8, w8 = model8.velocities
T8 = model8.tracers.T; S8 = model8.tracers.S

u8c = @at (Center, Center, Center) u8
v8c = @at (Center, Center, Center) v8
w8c = @at (Center, Center, Center) w8
KE8 = KineticEnergy(model8)
Ro8 = RossbyNumber(model8)
slice8 = (; u_c=u8c, v_c=v8c, w_c=w8c, T=T8, S=S8, KE=KE8, Ro=Ro8)

# Cross-correlations (same as flow_over_shoals.jl)
uu8 = Field((@at (Center, Center, Center) u8 * u8))
vv8 = Field((@at (Center, Center, Center) v8 * v8))
ww8 = Field((@at (Center, Center, Center) w8 * w8))
uT8 = Field((@at (Center, Center, Center) u8 * T8))
uS8 = Field((@at (Center, Center, Center) u8 * S8))

sim8 = Simulation(model8, Δt=Δt_bench, stop_iteration=1)
sim8.output_writers[:surface] = NetCDFWriter(model8, slice8,
    filename=joinpath(tempdir(), "bench8_top.nc"),
    schedule=TimeInterval(Δt_bench),
    indices=(:, :, Nz),
    overwrite_existing=true)
sim8.output_writers[:midy] = NetCDFWriter(model8, slice8,
    filename=joinpath(tempdir(), "bench8_midy.nc"),
    schedule=TimeInterval(Δt_bench),
    indices=(:, round(Int, Ny/2), :),
    overwrite_existing=true)
sim8.output_writers[:midx] = NetCDFWriter(model8, slice8,
    filename=joinpath(tempdir(), "bench8_midx.nc"),
    schedule=TimeInterval(Δt_bench),
    indices=(round(Int, Nx/2), :, :),
    overwrite_existing=true)
sim8.output_writers[:avg] = NetCDFWriter(model8,
    (; u=u8, v=v8, w=w8, T=T8, S=S8, uu=uu8, vv=vv8, ww=ww8, uT=uT8, uS=uS8),
    schedule=AveragedTimeInterval(10days, window=10days),
    filename=joinpath(tempdir(), "bench8_avg.nc"),
    overwrite_existing=true)
sim8.output_writers[:full3d] = NetCDFWriter(model8,
    (; u_c=u8c, v_c=v8c, w_c=w8c, T=T8, S=S8, KE=KE8, Ro=Ro8),
    filename=joinpath(tempdir(), "bench8_full3d.nc"),
    schedule=TimeInterval(20days),
    overwrite_existing=true)

init_model!(model8)
for _ in 1:3; time_step!(model8, Δt_bench); end
if arch isa GPU; CUDA.@sync nothing; end
t0 = time_ns()
for _ in 1:N_steps; time_step!(model8, Δt_bench); end
if arch isa GPU; CUDA.@sync nothing; end
t8 = (time_ns() - t0) / 1e9 / N_steps
cg8 = nothing
if hasproperty(model8.pressure_solver, :conjugate_gradient_solver)
    cg8 = model8.pressure_solver.conjugate_gradient_solver.iteration
end
push!(results, (; name="8: + Cross-corr + time avg", time_per_step=t8, cg_iters=cg8))
@info @sprintf("  → %.4f s/step%s", t8, isnothing(cg8) ? "" : @sprintf(" (%d CG iters)", cg8))

# ═══════════════════════════════════════════════════════════════════════════
#  Print results table
# ═══════════════════════════════════════════════════════════════════════════

println()
println("═" ^ 82)
println("  BENCHMARK RESULTS  ($(N_steps) timed steps per level, grid=$(Nx)×$(Ny)×$(Nz), arch=$(arch))")
println("═" ^ 82)

baseline = results[1].time_per_step

# Header
@printf("  %-35s  %9s  %9s  %8s  %6s  %s\n",
    "Configuration", "Total s", "Margin s", "Margin%", "Ratio", "CG iters")
println("  " * "─"^80)

let prev_time = 0.0
    for (i, r) in enumerate(results)
        margin = r.time_per_step - prev_time
        margin_pct = baseline > 0 ? (margin / results[end].time_per_step) * 100 : 0.0
        ratio = baseline > 0 ? r.time_per_step / baseline : 0.0
        cg_str = isnothing(r.cg_iters) ? "—" : string(r.cg_iters)

        if i == 1
            @printf("  %-35s  %9.4f  %9s  %8s  %6.1fx  %s\n",
                r.name, r.time_per_step, "—", "—", 1.0, cg_str)
        else
            @printf("  %-35s  %9.4f  %+9.4f  %7.1f%%  %6.1fx  %s\n",
                r.name, r.time_per_step, margin, margin_pct, ratio, cg_str)
        end
        prev_time = r.time_per_step
    end
end

println("═" ^ 82)
println()

# Cleanup temp files
for f in filter(n -> startswith(n, "bench"), readdir(tempdir()))
    try rm(joinpath(tempdir(), f)) catch end
end

@info "Benchmark complete."
