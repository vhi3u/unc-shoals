# ═══════════════════════════════════════════════════════════════════════════
# Nonhydrostatic LES of the REAL Diamond Shoals, scaled down by dynamic
# similarity -- version 2 of src/les_shoal.jl, after Chor's rough-seamounts
# (github.com/tomchor/rough-seamounts): real bathymetry, smoothed and rescaled,
# open inflow/outflow instead of a periodic channel.
#
#   julia --project src/les_diamond_real.jl <config> [f1|f2|f4|f8|f16] [smoothing_km]
#
#   julia --project src/les_diamond_real.jl DS_summer f2        # GEBCO as is
#   julia --project src/les_diamond_real.jl DS_summer f2 2      # smoothed 2 km
#
# ── BATHYMETRY ─────────────────────────────────────────────────────────────
# data/diamond_shoals_frame.nc (tools/make_les_bathymetry.jl): GEBCO 2026
# (2024 to the south) in the shoal's frame at real scale, 80 x 160 km, ocean
# only (Pamlico Sound cut off), depth clamped to [3, 50] m.
#   X  along the shoal axis (132.5 deg), the cape at X = 25 km
#   Y  across it, the NE side (upstream in southward flow) at small Y, the
#      cape at Y = 60 km
# Optional smoothing (third argument, km at real scale): a Gaussian of that
# standard deviation over the water only, as Chor's L (there a fraction of
# the seamount's FWHM). GEBCO's 15" grid (~0.4 km) is the finest roughness
# available; smoothing removes more, to test how much the roughness matters.
#
# ── SCALING ────────────────────────────────────────────────────────────────
# As in les_shoal.jl: the prototype is the real shoal (L = 10 km, half its
# reach; H = 19 m, shelf -24 m to crest -5 m), matched in Ro_h = V/(fL) and
# Fr_h = V/(NH). The model steepens it to alpha = H/L = 0.1: H_m = 40 m,
# L_m = 400 m, so depths scale by mu = 40/19 and lengths by lambda = 0.04.
# The domain is 3.2 x 6.4 km x 105 m.
#
# ── BOUNDARIES (the change from les_shoal.jl) ──────────────────────────────
# y is Open at both ends, as in Chor's seamount runs:
#   south (inflow, the NE side of the real shoal): v = V_inf imposed, u = w = 0,
#     b = N^2 z
#   north (outflow): PerturbationAdvection radiation (inflow timescale 2 min,
#     outflow 30 min, Chor's values) for v, u, w and b
# The flow is carried in by the inflow condition, so neither the sponge nor
# the drag-balancing body force of les_shoal.jl is needed. Kept: the
# geostrophic cross-shore force -f V_inf, and log-law drag (natural
# roughness, z0 = 2.5e-3 H_m) on the immersed bottom; the flat floor offshore
# of the break is free slip, as in Chor's runs.
# x: the domain wall at x = 0 (land along it) and a free-slip wall offshore.
#
# ── NUMERICS ───────────────────────────────────────────────────────────────
# NonhydrostaticModel, RK3, WENO(order=5), DynamicSmagorinsky (Lagrangian,
# every 5 steps, as Chor uses), b = N^2 z. Pressure: CG preconditioned by the
# FFT solver, tolerance 1e-4 and at most 50 iterations (Zhao's fjord LES):
# the land is real and spans the full depth, so the FFT solver alone would
# leak through it. 10 advective periods of spin-up + 20 of statistics.
# ═══════════════════════════════════════════════════════════════════════════

using Oceananigans
using Oceananigans.Units
using Oceananigans.TurbulenceClosures: DynamicSmagorinsky
using Oceananigans.Solvers: ConjugateGradientPoissonSolver
using Oceananigans.Architectures: on_architecture
using NCDatasets
using Printf
using Random
using CUDA: has_cuda_gpu

include(joinpath(@__DIR__, "ProgressMessengers.jl"))
using .ProgressMessengers: TimedMessenger

# ── command line ─────────────────────────────────────────────────────────
config_name = length(ARGS) >= 1 ? ARGS[1] : "DS_summer"
factor = length(ARGS) >= 2 ? parse(Int, replace(ARGS[2], "f" => "")) : 1
smooth_km = length(ARGS) >= 3 ? parse(Float64, ARGS[3]) : 0.0
factor in (1, 2, 4, 8, 16) || error("resolution factor must be f1, f2, f4, f8 or f16")
simname = @sprintf("LES2_%s_S%g_f%d", config_name, smooth_km, factor)
arch = has_cuda_gpu() ? GPU() : CPU()
@info "LES v2 (real bathymetry) $simname on $arch"

# ── prototype and configurations (as les_shoal.jl) ──────────────────────
const prototype_ = (; L = 10e3, H = 19.0, f = 8.42e-5, Zsh = -24.0)
const configs = Dict(
    "DS_base"        => (; V = 0.10, N = 0.0174),
    "DS_summer"      => (; V = 0.10, N = 0.0338),
    "DS_summer_slow" => (; V = 0.04, N = 0.0338),
)
haskey(configs, config_name) || error("unknown config $config_name; one of $(keys(configs))")
cfg = configs[config_name]
Ro_h = cfg.V / (prototype_.f * prototype_.L)
Fr_h = cfg.V / (cfg.N * prototype_.H)

const α_m = 0.1
const H_m = 40.0
const L_m = H_m / α_m
const V_m = 0.1
μ = H_m / prototype_.H
λ = L_m / prototype_.L
f_m = V_m / (Ro_h * L_m)
N_m = V_m / (Fr_h * H_m)
const N²_m = N_m^2
T_adv = L_m / V_m

# ── real bathymetry, smoothed and rescaled ──────────────────────────────
bathy_file = joinpath(dirname(@__DIR__), "data", "diamond_shoals_frame.nc")
Xp, Yp, Dp, h_max_p = NCDataset(bathy_file) do ds
    Float64.(ds["X"][:]), Float64.(ds["Y"][:]), Float64.(ds["depth"][:, :]), Float64(ds.attrib["h_max"])
end
dxp = Xp[2] - Xp[1]                                    # km
waterp = Dp .< 0

# Gaussian smoothing of the depth over water only (land stays land)
function smooth_water(D, water, σ_cells)
    σ_cells <= 0 && return copy(D)
    r = ceil(Int, 3σ_cells)
    k = [exp(-i^2 / (2σ_cells^2)) for i in -r:r]
    function pass(A, W, dim)
        B = zeros(size(A)); Wn = zeros(size(A))
        n = size(A, dim)
        for idx in CartesianIndices(A)
            i = idx[dim]
            for (o, kk) in zip(-r:r, k)
                j = i + o
                (1 <= j <= n) || continue
                src = dim == 1 ? CartesianIndex(j, idx[2]) : CartesianIndex(idx[1], j)
                B[idx] += kk * A[src] * W[src]
                Wn[idx] += kk * W[src]
            end
        end
        return B, Wn
    end
    W = Float64.(water)
    B1, W1 = pass(D, W, 1)
    A1 = [W1[i] > 0 ? B1[i] / W1[i] : 0.0 for i in eachindex(B1)]
    A1 = reshape(A1, size(D))
    B2, W2 = pass(A1, Float64.(W1 .> 0), 2)
    S = [water[i] ? (W2[i] > 0 ? B2[i] / W2[i] : D[i]) : 0.0 for i in eachindex(B2)]
    return reshape(S, size(D))
end
Dp_s = smooth_water(Dp, waterp, smooth_km / dxp)

# model domain and grid (as les_shoal.jl)
Lx_m, Ly_m = λ * 1e3 * (Xp[end] + dxp / 2), λ * 1e3 * (Yp[end] + dxp / 2)
Lz = μ * h_max_p
y_cape = λ * 1e3 * 60.0
x_cape = λ * 1e3 * 25.0
Δh = 4.0 * factor
Δz0 = 1.0 * factor
z_fine = μ * abs(prototype_.Zsh) + 5.0
function z_faces(Lz, Δz0, z_fine)
    zf = collect(0.0:-Δz0:-z_fine)
    Δ = Δz0
    while zf[end] > -Lz
        Δ = min(1.12 * Δ, 5.0 * Δz0)
        push!(zf, zf[end] - Δ)
    end
    zf[end] = -Lz
    if length(zf) > 2 && (zf[end-1] - zf[end]) < 0.5 * (zf[end-2] - zf[end-1])
        deleteat!(zf, length(zf) - 1)
    end
    return reverse(zf)
end
zf = z_faces(Lz, Δz0, z_fine)
Nx, Ny, Nz = round(Int, Lx_m / Δh), round(Int, Ly_m / Δh), length(zf) - 1

@info @sprintf("Ro_h = %.3f, Fr_h = %.3f, S_h = %.2f, Gamma = %.3f; smoothing %.1f km (real scale)", Ro_h, Fr_h, Ro_h / Fr_h, α_m * Fr_h, smooth_km)
@info @sprintf("model: %.0f x %.0f x %.0f m, f = %.2e, N = %.2e, T_adv = %.0f s; grid %d x %d x %d = %.1f M cells",
               Lx_m, Ly_m, Lz, f_m, N_m, T_adv, Nx, Ny, Nz, Nx * Ny * Nz / 1e6)

# bathymetry on the model's cell centres: land where the nearest prototype
# cell is land, otherwise bilinear over the water points around it
xc = [(i - 0.5) * Δh for i in 1:Nx]; yc = [(j - 0.5) * Δh for j in 1:Ny]
function sample(D, water, Xp, Yp, xm, ym)
    X = xm / (λ * 1e3); Y = ym / (λ * 1e3)            # back to real km
    fx = clamp((X - Xp[1]) / (Xp[2] - Xp[1]) + 1, 1, length(Xp)); fy = clamp((Y - Yp[1]) / (Yp[2] - Yp[1]) + 1, 1, length(Yp))
    i0 = min(floor(Int, fx), length(Xp) - 1); j0 = min(floor(Int, fy), length(Yp) - 1)
    a, b = fx - i0, fy - j0
    water[round(Int, fx), round(Int, fy)] || return 0.0
    num = 0.0; den = 0.0
    for (di, dj, w) in ((0, 0, (1-a)*(1-b)), (1, 0, a*(1-b)), (0, 1, (1-a)*b), (1, 1, a*b))
        if water[i0+di, j0+dj]
            num += w * D[i0+di, j0+dj]; den += w
        end
    end
    return den > 0 ? num / den : 0.0
end
bottom_cpu = [μ * sample(Dp_s, waterp, Xp, Yp, xc[i], yc[j]) for i in 1:Nx, j in 1:Ny]   # m, negative; 0 = land
@info @sprintf("bathymetry: %.1f%% water, depth %.1f to %.1f m", 100 * count(<(0), bottom_cpu) / length(bottom_cpu),
               -maximum(bottom_cpu[bottom_cpu .< 0]), -minimum(bottom_cpu))

underlying = RectilinearGrid(arch; size = (Nx, Ny, Nz), halo = (4, 4, 4),
                             x = (0, Lx_m), y = (0, Ly_m), z = zf,
                             topology = (Bounded, Open, Bounded))
grid = ImmersedBoundaryGrid(underlying, GridFittedBottom(on_architecture(arch, bottom_cpu)))

# ── boundary conditions ─────────────────────────────────────────────────
z₀ = 2.5e-3 * H_m
z₁ = Δz0 / 2
C_d = (0.4 / log(z₁ / z₀))^2
drag = BulkDrag(coefficient = C_d)
@info @sprintf("bottom drag on the immersed bottom: z0 = %.3f m, C_d = %.4f", z₀, C_d)

outflow = PerturbationAdvection(inflow_timescale = 2minutes, outflow_timescale = 30minutes)
@inline b_ext(x, z, t, N²) = N² * z
u_bcs = FieldBoundaryConditions(south = ValueBoundaryCondition(0.0),
                                north = ValueBoundaryCondition(0.0; scheme = outflow),
                                immersed = ImmersedBoundaryCondition(bottom = drag))
v_bcs = FieldBoundaryConditions(south = NormalFlowBoundaryCondition(V_m),
                                north = NormalFlowBoundaryCondition(V_m; scheme = outflow),
                                immersed = ImmersedBoundaryCondition(bottom = drag))
w_bcs = FieldBoundaryConditions(south = ValueBoundaryCondition(0.0),
                                north = ValueBoundaryCondition(0.0; scheme = outflow))
b_bcs = FieldBoundaryConditions(south = ValueBoundaryCondition(b_ext; parameters = N²_m),
                                north = ValueBoundaryCondition(b_ext; parameters = N²_m, scheme = outflow))

# geostrophic cross-shore pressure gradient for the mean flow V_inf
@inline Fu_(x, y, z, t, p) = -p.f * p.V
forcing = (; u = Forcing(Fu_, parameters = (; f = f_m, V = V_m)))

# ── model ────────────────────────────────────────────────────────────────
closure = DynamicSmagorinsky(schedule = IterationInterval(5))
pressure_solver = ConjugateGradientPoissonSolver(grid; reltol = 1e-4, maxiter = 50)
model = NonhydrostaticModel(grid;
                            pressure_solver,
                            advection = WENO(order = 5),
                            timestepper = :RungeKutta3,
                            buoyancy = BuoyancyTracer(),
                            tracers = :b,
                            coriolis = FPlane(f = f_m),
                            closure,
                            forcing,
                            boundary_conditions = (u = u_bcs, v = v_bcs, w = w_bcs, b = b_bcs))

# ── output, checkpoints, run ─────────────────────────────────────────────
outroot = isdir("/glade/derecho/scratch/nguyen") ? "/glade/derecho/scratch/nguyen/unc-shoals/les" :
          joinpath(dirname(@__DIR__), "les_output")
outdir = joinpath(outroot, simname)
mkpath(outdir)
@info "Output directory: $outdir"
pickup = any(startswith("checkpoint_iteration"), readdir(outdir))

if !pickup
    Random.seed!(1234)
    ε(x, y, z) = 1e-3 * V_m * (2rand() - 1)
    set!(model, u = ε, v = (x, y, z) -> V_m + ε(x, y, z), w = ε, b = (x, y, z) -> N²_m * z)
end

const T_RUN = 30                                   # 10 spin-up + 20 statistics
wall_limit = factor == 1 ? 23hours : 11.5hours
simulation = Simulation(model; Δt = 0.2 * Δz0 / V_m, stop_time = T_RUN * T_adv, wall_time_limit = wall_limit)
# diffusive_cfl: DynamicSmagorinsky is explicit, so a spike in eddy viscosity
# can outrun an advective-only step (the first f2 runs reached a diffusive
# CFL of 0.69 uncapped). Zhao's fjord LES caps it at 0.5; 0.8 here.
wizard = TimeStepWizard(cfl = 0.8, diffusive_cfl = 0.8, max_change = 1.05, min_change = 0.2, max_Δt = 1 / N_m)
simulation.callbacks[:wizard] = Callback(wizard, IterationInterval(4))
simulation.callbacks[:progress] = Callback(TimedMessenger(), IterationInterval(100))

u, v, w = model.velocities
b = model.tracers.b
ωz = Field(∂x(v) - ∂y(u))
fields2d = (; u, v, w, b, ωz)

zc = (zf[1:end-1] .+ zf[2:end]) ./ 2
k_mid = searchsortedfirst(zc, μ * prototype_.Zsh / 2)
j_axis = clamp(round(Int, y_cape / Δh), 1, Ny)
i_crest = clamp(round(Int, (x_cape + λ * 8.5e3) / Δh), 1, Nx)          # middle of the 17 km crest
Δt_slice = 0.5 * T_adv
ow = !pickup
for (name, idx) in ((:top, (:, :, Nz)), (:midshelf, (:, :, k_mid)), (:axis_xz, (:, j_axis, :)), (:crest_yz, (i_crest, :, :)))
    simulation.output_writers[name] = NetCDFWriter(model, fields2d; filename = joinpath(outdir, "$(name).nc"),
        indices = idx, schedule = TimeInterval(Δt_slice), overwrite_files = ow)
end
averages = (; u, v, w, b, uu = Field(u * u), vv = Field(v * v), ww = Field(w * w),
            uv = Field(u * v), ub = Field(u * b), vb = Field(v * b), wb = Field(w * b))
simulation.output_writers[:avg] = NetCDFWriter(model, averages; filename = joinpath(outdir, "time_avg_3d.nc"),
    schedule = AveragedTimeInterval(10T_adv, window = 10T_adv), overwrite_files = ow)
simulation.output_writers[:checkpointer] = Checkpointer(model; dir = outdir, prefix = "checkpoint",
    schedule = TimeInterval(5T_adv), cleanup = true)

NCDataset(joinpath(outdir, "metadata.nc"), "c") do ds
    for (k, v) in pairs((; config = config_name, factor, smooth_km, Ro_h, Fr_h, S_h = Ro_h / Fr_h, alpha = α_m,
                          H_m, L_m, V_m, f = f_m, N = N_m, T_adv, mu = μ, lambda = λ, C_d, z0 = z₀,
                          Lx = Lx_m, Ly = Ly_m, Lz, x_cape, y_cape, Nx, Ny, Nz, dh = Δh, dz_top = Δz0,
                          proto_L = prototype_.L, proto_H = prototype_.H, proto_f = prototype_.f,
                          proto_V = cfg.V, proto_N = cfg.N, bathymetry = basename(bathy_file)))
        ds.attrib[String(k)] = v
    end
end

@info "Starting: $(T_RUN) advective periods ($(round(T_RUN * T_adv / 86400, digits = 2)) days), pickup = $pickup"
run!(simulation; pickup)
if simulation.model.clock.time < T_RUN * T_adv
    Oceananigans.OutputWriters.checkpoint(simulation)
    @info "Stopped at the walltime limit after $(round(simulation.model.clock.time / T_adv, digits = 2)) advective periods; checkpoint written. Resubmit to continue."
else
    @info "Done: $(simulation.model.clock.time / T_adv) advective periods"
end
