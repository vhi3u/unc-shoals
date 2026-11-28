# ═══════════════════════════════════════════════════════════════════════════
# Nonhydrostatic LES of a Diamond Shoals-shaped cape shoal, scaled down by
# dynamic similarity (after Chor & Wenegrat 2025, github.com/tomchor/
# submesoscale-headland).
#
#   julia --project src/les_shoal.jl <config> [f1|f2|f4|f8|f16]
#
#   julia --project src/les_shoal.jl DS_base f2     # half resolution
#   julia --project src/les_shoal.jl DS_summer      # full resolution
#
# A separate model from the hydrostatic sweep (flow_over_shoals_sweep.jl). It
# shares only the bathymetry (dshoal_diamond.jl) and the progress messenger.
#
# ── THE SCALING ────────────────────────────────────────────────────────────
# The real shoal cannot be resolved down to its turbulence: it is ~20 km
# across. As Chor & Wenegrat do, the model is a small box whose
# dimensionless numbers match the real flow:
#   Ro_h = V / (f L)      headland Rossby number
#   Fr_h = V / (N H)      headland Froude number
#   S_h  = Ro_h / Fr_h    slope Burger number (sets the regime in their paper)
# with the geometry fixed in metres and f and N chosen to hit Ro_h and Fr_h.
# The prototype (Diamond Shoals, prototype_ below): L = 10 km (half the ~20 km
# the shoal reaches out from the cape), H = 19 m (shelf -24 m, crest -5 m),
# V = 0.1 m/s at the shoal (the open v0 = 0.2 runs), f = 8.42e-5.
#
# The real shoal is very gentle (alpha = H/L ~ 0.002): shrinking it at that
# shape saves nothing. So the model is DISTORTED: steepened to alpha = 0.1
# (H_m = 40 m of relief, L_m = 400 m). Everything vertical is scaled by
# mu = H_m / H (depths, the coast, the crest), everything horizontal by
# lambda = L_m / L. That is legitimate while the flow stays nearly hydrostatic
# at the shoal scale, Gamma = alpha Fr_h << 1 (0.03 here; Chor & Wenegrat go
# to 0.25).
#
# FRICTION. Bottom roughness is "natural", as in Chor & Wenegrat:
# z0 = 2.5e-3 H_m, with a log-law drag coefficient at the first grid point.
# The distortion cannot keep both of the real shelf's friction numbers:
#   spin-down / advection   alpha / C_d : real 0.002/0.0028 = 0.7; here
#                           0.1/0.06 ~ 1.6 (at full resolution) -- close
#   boundary layer / depth  ~ 0.4 sqrt(C_d) Ro_h / alpha : real ~1.3 (the
#                           bottom layer fills the shelf); here ~0.1 -- a thin
#                           bottom layer under a frictionless interior, as in
#                           Chor & Wenegrat's self-similar assumption.
# So this LES is the Chor-like counterpart of our shoal: friction acts at
# about the right rate, but the bottom layer no longer fills the column.
#
# ── THE MODEL ──────────────────────────────────────────────────────────────
#   NonhydrostaticModel, rigid lid, RK3, WENO(order=5), DynamicSmagorinsky
#   (Lagrangian-averaged dynamic coefficient), linear buoyancy b = N^2 z.
#   x: coast wall at 0 (Bounded), y: Periodic, z: Bounded, from -Lz to 0.
#   Bathymetry: dshoal_diamond.jl with a 45-degree land cape, all lengths
#   scaled; land is the immersed bottom at z = 0.
#   Flow: uniform V_inf, held by three body forces --
#     u: -f V_inf, the cross-shore pressure gradient in geostrophic balance
#        with V_inf (Chor & Wenegrat's "geostrophy" term)
#     v: C_d V_inf^2 / h(x, y), so that a uniform V_inf balances the bottom
#        drag everywhere. Chor & Wenegrat need none: their flat bottom is
#        free slip. Here every bottom has drag, and without it a periodic
#        channel loses its transport (the hydrostatic periodic runs kept only
#        a third of it).
#     sponge: the first 10% of y relaxes u -> 0, v -> V_inf, w -> 0,
#        b -> N^2 z at 40 V_inf / L_sponge (~20 e-folds per crossing), so the
#        wake does not come back round.
#   Run: 20 advective periods (T = L_m / V) of spin-up, 60 of statistics.
#
# Output: /glade/derecho/scratch/nguyen/unc-shoals/les/<simname>/ on Casper
# (les_output/<simname>/ elsewhere). Checkpoints let a run continue in a
# second job: resubmitting the same job picks up where it stopped.
# ═══════════════════════════════════════════════════════════════════════════

using Oceananigans
using Oceananigans.Units
using Oceananigans.TurbulenceClosures: DynamicSmagorinsky
using Oceananigans.Solvers: ConjugateGradientPoissonSolver
using NCDatasets
using Printf
using Random
using CUDA: has_cuda_gpu

include(joinpath(@__DIR__, "dshoal_diamond.jl"))
include(joinpath(@__DIR__, "ProgressMessengers.jl"))
using .ProgressMessengers: TimedMessenger

# ── command line ─────────────────────────────────────────────────────────
config_name = length(ARGS) >= 1 ? ARGS[1] : "DS_base"
factor = length(ARGS) >= 2 ? parse(Int, replace(ARGS[2], "f" => "")) : 1
factor in (1, 2, 4, 8, 16) || error("resolution factor must be f1, f2, f4, f8 or f16")
simname = "LES_$(config_name)_f$(factor)"
arch = has_cuda_gpu() ? GPU() : CPU()
@info "LES $simname on $arch"

# ── prototype (Diamond Shoals) and configurations ────────────────────────
const prototype_ = (; L = 10e3, H = 19.0, f = 8.42e-5, Zs = -5.0, Zsh = -24.0,
                     tip_depth = -13.0, h_deep = -50.0, h_coast = -5.0,
                     crest_length = 17e3, taper_length = 3e3, sigma = 4.6e3,
                     shoreface_width = 10e3, break_distance = 22e3, rise_width = 2e3,
                     cape_length = 25e3, Lx = 80e3, Ly = 160e3, y0 = 60e3)

# N: the hydrostatic model's T/S profiles (0.0174 1/s) or summer at mooring
# B1, north of Hatteras (0.0338 1/s). V: the speed reaching the shoal.
const configs = Dict(
    "DS_base"        => (; V = 0.10, N = 0.0174),
    "DS_summer"      => (; V = 0.10, N = 0.0338),
    "DS_summer_slow" => (; V = 0.04, N = 0.0338),
)
haskey(configs, config_name) || error("unknown config $config_name; one of $(keys(configs))")
cfg = configs[config_name]

Ro_h = cfg.V / (prototype_.f * prototype_.L)
Fr_h = cfg.V / (cfg.N * prototype_.H)

# ── model scales ─────────────────────────────────────────────────────────
const α_m = 0.1                     # model slope (real ~0.002)
const H_m = 40.0                    # model relief (m)
const L_m = H_m / α_m               # 400 m
const V_m = 0.1                     # model inflow speed (m/s); any value, by similarity
μ = H_m / prototype_.H              # vertical scale
λ = L_m / prototype_.L              # horizontal scale
f_m = V_m / (Ro_h * L_m)
N_m = V_m / (Fr_h * H_m)
N²_m = N_m^2
T_adv = L_m / V_m

g = (; Zs = μ * prototype_.Zs, Zsh = μ * prototype_.Zsh, tip_depth = μ * prototype_.tip_depth,
     h_deep = μ * prototype_.h_deep, h_coast = μ * prototype_.h_coast,
     crest_length = λ * prototype_.crest_length, taper_length = λ * prototype_.taper_length,
     sigma = λ * prototype_.sigma, shoreface_width = λ * prototype_.shoreface_width,
     break_distance = λ * prototype_.break_distance, rise_width = λ * prototype_.rise_width,
     cape_length = λ * prototype_.cape_length,
     Lx = λ * prototype_.Lx, Ly = λ * prototype_.Ly, y0 = λ * prototype_.y0)
Lz = -g.h_deep

@info @sprintf("Ro_h = %.3f, Fr_h = %.3f, S_h = %.2f, Gamma = alpha Fr_h = %.3f", Ro_h, Fr_h, Ro_h / Fr_h, α_m * Fr_h)
@info @sprintf("model: L = %.0f m, H = %.0f m, Lx x Ly x Lz = %.0f x %.0f x %.0f m, f = %.2e, N = %.2e, T_adv = %.0f s",
               L_m, H_m, g.Lx, g.Ly, Lz, f_m, N_m, T_adv)

# ── grid ─────────────────────────────────────────────────────────────────
# Full resolution (f1): 4 m horizontally, 1 m vertically down to just below
# the shelf, then stretched to ~5 m to the floor; Δx/Δz = 4 as in Chor &
# Wenegrat (4.2 and 3.5). fN divides every spacing by N.
Δh = 4.0 * factor
Δz0 = 1.0 * factor
z_fine = abs(g.Zsh) + 5.0
function z_faces(Lz, Δz0, z_fine)
    zf = collect(0.0:-Δz0:-z_fine)
    Δ = Δz0
    while zf[end] > -Lz
        Δ = min(1.12 * Δ, 5.0 * Δz0)
        push!(zf, zf[end] - Δ)
    end
    # The last step overshoots the floor: move that face onto it, and if the
    # cell above would then be thinner than half the previous one, merge them.
    # (Clipping alone left a bottom cell a few cm thick: a huge diffusive CFL.)
    zf[end] = -Lz
    if length(zf) > 2 && (zf[end-1] - zf[end]) < 0.5 * (zf[end-2] - zf[end-1])
        deleteat!(zf, length(zf) - 1)
    end
    return reverse(zf)
end
zf = z_faces(Lz, Δz0, z_fine)
Nx, Ny, Nz = round(Int, g.Lx / Δh), round(Int, g.Ly / Δh), length(zf) - 1
@info @sprintf("grid %d x %d x %d = %.1f M cells (Δh = %.1f m, Δz = %.1f m near the top)", Nx, Ny, Nz, Nx * Ny * Nz / 1e6, Δh, Δz0)

underlying = RectilinearGrid(arch; size = (Nx, Ny, Nz), halo = (4, 4, 4),
                             x = (0, g.Lx), y = (0, g.Ly), z = zf,
                             topology = (Bounded, Periodic, Bounded))

const bottom = dshoal_diamond_bottom(g.Ly; Zs = g.Zs, tip_depth = g.tip_depth,
    crest_length = g.crest_length, taper_length = g.taper_length, sigma = g.sigma,
    Zsh = g.Zsh, shoreface_width = g.shoreface_width, break_distance = g.break_distance,
    rise_width = g.rise_width, h_deep = g.h_deep, cape_length = g.cape_length,
    h_coast = g.h_coast, shoal_y0 = g.y0)
grid = ImmersedBoundaryGrid(underlying, GridFittedBottom(bottom))

# ── bottom drag: log law at the first grid point, natural roughness ─────
z₀ = 2.5e-3 * H_m
z₁ = Δz0 / 2
C_d = (0.4 / log(z₁ / z₀))^2
@info @sprintf("bottom drag: z0 = %.3f m, z1 = %.2f m, C_d = %.4f (alpha / C_d = %.2f; real shelf 0.7)", z₀, z₁, C_d, α_m / C_d)
drag = BulkDrag(coefficient = C_d)
u_bcs = FieldBoundaryConditions(bottom = drag, immersed = ImmersedBoundaryCondition(bottom = drag))
v_bcs = FieldBoundaryConditions(bottom = drag, immersed = ImmersedBoundaryCondition(bottom = drag))

# ── forcing ──────────────────────────────────────────────────────────────
const L_sponge = 0.1 * g.Ly
const σ_sponge = 40 * V_m / L_sponge
@inline sponge_mask(y) = ifelse(y <= L_sponge / 2, y / (L_sponge / 2),
                          ifelse(y <= L_sponge, (L_sponge - y) / (L_sponge / 2), 0.0))
@inline water_depth(x, y, h_min) = max(-bottom(x, y), h_min)

@inline Fu_(x, y, z, t, u, p) = -p.f * p.V - sponge_mask(y) * p.σ * u
@inline Fv_(x, y, z, t, v, p) = p.Cd * p.V^2 / water_depth(x, y, p.hmin) - sponge_mask(y) * p.σ * (v - p.V)
@inline Fw_(x, y, z, t, w, p) = -sponge_mask(y) * p.σ * w
@inline Fb_(x, y, z, t, b, p) = -sponge_mask(y) * p.σ * (b - p.N² * z)

fp = (; f = f_m, V = V_m, σ = σ_sponge, Cd = C_d, hmin = -g.h_coast, N² = N²_m)
forcing = (u = Forcing(Fu_, field_dependencies = :u, parameters = fp),
           v = Forcing(Fv_, field_dependencies = :v, parameters = fp),
           w = Forcing(Fw_, field_dependencies = :w, parameters = fp),
           b = Forcing(Fb_, field_dependencies = :b, parameters = fp))

# ── model ────────────────────────────────────────────────────────────────
closure = DynamicSmagorinsky(schedule = IterationInterval(5))
# Pressure: conjugate gradient, preconditioned by the FFT solver on the
# underlying grid. The FFT solver alone ignores the immersed boundary and
# leaves the velocity divergent next to it (Oceananigans warns); with a land
# cape that spans the full depth, that would let flow leak through the land.
pressure_solver = ConjugateGradientPoissonSolver(grid; maxiter = 20)
model = NonhydrostaticModel(grid;
                            pressure_solver,
                            advection = WENO(order = 5),
                            timestepper = :RungeKutta3,
                            buoyancy = BuoyancyTracer(),
                            tracers = :b,
                            coriolis = FPlane(f = f_m),
                            closure,
                            forcing,
                            boundary_conditions = (u = u_bcs, v = v_bcs))
@info "pressure solver: $(typeof(model.pressure_solver).name.name)"

# ── output and checkpoints ───────────────────────────────────────────────
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

# Stop a little before the job's walltime (submit_les_f1: 24 h, submit_les_f2:
# 12 h) so the final checkpoint below is written before PBS kills the job.
wall_limit = factor == 1 ? 23hours : 11.5hours
simulation = Simulation(model; Δt = 0.2 * Δz0 / V_m, stop_time = 80 * T_adv, wall_time_limit = wall_limit)
wizard = TimeStepWizard(cfl = 0.8, max_change = 1.05, min_change = 0.2, max_Δt = 1 / N_m)
simulation.callbacks[:wizard] = Callback(wizard, IterationInterval(4))
simulation.callbacks[:progress] = Callback(TimedMessenger(), IterationInterval(100))

u, v, w = model.velocities
b = model.tracers.b
ωz = Field(∂x(v) - ∂y(u))
fields2d = (; u, v, w, b, ωz)

k_top = Nz
# from zf (a CPU vector): on a GPU the grid's own znodes are a GPU array, and
# searching one element by element is the scalar indexing CUDA forbids
zc = (zf[1:end-1] .+ zf[2:end]) ./ 2
k_mid = searchsortedfirst(zc, g.Zsh / 2)                                 # mid-shelf depth
j_axis = round(Int, g.y0 / Δh)
i_crest = round(Int, (g.cape_length + g.crest_length / 2) / Δh)
Δt_slice = 0.5 * T_adv
ow = !pickup
simulation.output_writers[:top] = NetCDFWriter(model, fields2d; filename = joinpath(outdir, "top.nc"),
    indices = (:, :, k_top), schedule = TimeInterval(Δt_slice), overwrite_files = ow)
simulation.output_writers[:mid] = NetCDFWriter(model, fields2d; filename = joinpath(outdir, "midshelf.nc"),
    indices = (:, :, k_mid), schedule = TimeInterval(Δt_slice), overwrite_files = ow)
simulation.output_writers[:axis] = NetCDFWriter(model, fields2d; filename = joinpath(outdir, "axis_xz.nc"),
    indices = (:, j_axis, :), schedule = TimeInterval(Δt_slice), overwrite_files = ow)
simulation.output_writers[:crest] = NetCDFWriter(model, fields2d; filename = joinpath(outdir, "crest_yz.nc"),
    indices = (i_crest, :, :), schedule = TimeInterval(Δt_slice), overwrite_files = ow)

averages = (; u, v, w, b, uu = Field(u * u), vv = Field(v * v), ww = Field(w * w),
            uv = Field(u * v), ub = Field(u * b), vb = Field(v * b), wb = Field(w * b))
simulation.output_writers[:avg] = NetCDFWriter(model, averages; filename = joinpath(outdir, "time_avg_3d.nc"),
    schedule = AveragedTimeInterval(20T_adv, window = 20T_adv), overwrite_files = ow)
simulation.output_writers[:checkpointer] = Checkpointer(model; dir = outdir, prefix = "checkpoint",
    schedule = TimeInterval(10T_adv), cleanup = true)

# metadata: the scaling, so the analysis can map the model back to the shoal
NCDataset(joinpath(outdir, "metadata.nc"), "c") do ds
    for (k, v) in pairs((; config = config_name, factor, Ro_h, Fr_h, S_h = Ro_h / Fr_h, alpha = α_m,
                          H_m, L_m, V_m, f = f_m, N = N_m, T_adv, mu = μ, lambda = λ, C_d, z0 = z₀,
                          Lx = g.Lx, Ly = g.Ly, Lz, y0 = g.y0, cape_length = g.cape_length,
                          crest_length = g.crest_length, sigma = g.sigma, Zs = g.Zs, Zsh = g.Zsh,
                          tip_depth = g.tip_depth, Nx, Ny, Nz, dh = Δh, dz_top = Δz0,
                          proto_L = prototype_.L, proto_H = prototype_.H, proto_f = prototype_.f,
                          proto_V = cfg.V, proto_N = cfg.N))
        ds.attrib[String(k)] = v
    end
end

@info "Starting: $(80) advective periods ($(round(80T_adv / 86400, digits = 2)) days), pickup = $pickup"
run!(simulation; pickup)
if simulation.model.clock.time < 80 * T_adv
    Oceananigans.OutputWriters.checkpoint(simulation)      # resume from here next job
    @info "Stopped at the walltime limit after $(round(simulation.model.clock.time / T_adv, digits = 2)) advective periods; checkpoint written. Resubmit to continue."
else
    @info "Done: $(simulation.model.clock.time / T_adv) advective periods"
end
