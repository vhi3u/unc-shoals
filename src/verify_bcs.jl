using Oceananigans
using Oceananigans.Units
using Oceananigans.BoundaryConditions: OpenBoundaryCondition, FieldBoundaryConditions, FluxBoundaryCondition
using Oceananigans.TurbulenceClosures
using Oceananigans.Solvers: ConjugateGradientPoissonSolver
using Oceananigans.BuoyancyFormulations: buoyancy
using Oceananigans.OutputWriters
using Oceananigans.Forcings
using Statistics: mean
using Oceananigans.Diagnostics: CFL
using Oceanostics: RossbyNumber, ErtelPotentialVorticity
using SeawaterPolynomials.TEOS10
using Oceanostics.ProgressMessengers: SingleLineMessenger
using NCDatasets
using DataFrames
using Plots

include("dshoal_vn.jl")

@info "building domain"

# setup domain
Lx = 100e3
Ly = 200e3
Lz = 50.0
x, y, z = (0, Lx), (0, Ly), (-Lz, 0)

Nx, Ny, Nz = 30, 30, 10

# plan: closed boundary conditions on east and west, periodic flow from south to north.

grid = RectilinearGrid(CPU(); size=(Nx, Ny, Nz), halo=(4, 4, 4),
                       x, y, z, topology=(Bounded, Periodic, Bounded))

# bathymetry parameters
σ = 8.0         # [km] Gaussian width for shoal cross-section
Hs = 15.0       # [m] shoal height

# bathymetry
x_km, y_km, h = dshoal(Lx/1e3, Ly/1e3, σ, Hs, Nx)

# If x_km and y_km are 1D, create a 2D meshgrid
xgrid = repeat(x_km', length(y_km), 1)  # (Ny, Nx)
ygrid = repeat(y_km, 1, length(x_km))   # (Ny, Nx)

# Grid with immersed bathymetry
ib_grid = ImmersedBoundaryGrid(grid, GridFittedBottom(h))

# temperature and salinity profiles from B1 and B2 moorings

@info "loading B1 and B2 T/S profiles"
using CSV, Interpolations
ctd = CSV.read("src/ctd_smoothed.csv", DataFrame)
z_data = collect(ctd.z)
T_south = collect(ctd.T_B2)
T_north = collect(ctd.T_B1)
S_south = collect(ctd.S_B2)
S_north = collect(ctd.S_B1)

iT_south = extrapolate(interpolate((z_data,), T_south, Gridded(Linear())), Interpolations.Flat())
iT_north = extrapolate(interpolate((z_data,), T_north, Gridded(Linear())), Interpolations.Flat())
iS_south = extrapolate(interpolate((z_data,), S_south, Gridded(Linear())), Interpolations.Flat())
iS_north = extrapolate(interpolate((z_data,), S_north, Gridded(Linear())), Interpolations.Flat())

@inline tsbc(x, z, t) = iT_south(z)
@inline tnbc(x, z, t) = iT_north(z)
@inline ssbc(x, z, t) = iS_south(z)
@inline snbc(x, z, t) = iS_north(z)

# turn 3 argument to 4 argument (model seems to want this)
@inline tsbc(x, y, z, t) = tsbc(x, z, t)
@inline tnbc(x, y, z, t) = tnbc(x, z, t)
@inline ssbc(x, y, z, t) = ssbc(x, z, t)
@inline snbc(x, y, z, t) = snbc(x, z, t)

Tₑ(x, y, z) = 23.11
Sₑ(x, y, z) = 36.4


@info "setting up boundary conditions"

# prescribed inflow (we want 0.10 m/s flow from south to north i.e. positive)

v∞(x, z, t) = 0.10

# north and south sponges 

Lₛ = 10e3
τ = 6hours
τ_ts = 10days
params = (; Lx = Lx, Ly = Ly, Ls = Lₛ, τ = τ, τ_ts = τ_ts)

# linear masks 0 at interior 1 at boundary for north and south. 
# extent of this mask should be from y = 0 (south boundary) to y = Lₛ = 20 km 
@inline function south_mask(x, y, z, p)
    y0 = 0
    y1 = p.Ls
    if y0 <= y <= y1
        return 1 - y/y1
    else
        return 0.0
    end
end

# extent of this mask should be from Ly - Ls to y = Ly ? 
@inline function north_mask(x, y, z, p) 
    y0 = p.Ly - p.Ls
    y1 = p.Ly

    if y0 <= y <= y1
        return (y - y0) / (y1 - y0)
    else
        return 0.0
    end
end

@inline function east_nudge(x, y, z, p)
    xC = 3e3
    xS = 60e3
    Lw = p.Lx
    k1 = 80 / Lw
    k2 = 40 / Lw

    s1 = 1 / (1 + exp(-k1 * (x - xC)))
    s2 = 1 / (1 + exp(k2 * (x - xS)))
    s = (s1 - 1) + s2
    return clamp(s, 0.0, 1.0)
end

@inline function east_mask(x, y, z, p)
    x0 = p.Lx - p.Ls
    x1 = p.Lx

    if x0 <= x <= x1
        return (x - x0) / (x1 - x0)
    else
        return 0.0
    end
end

# sponge functions
@inline sponge_u(x, y, z, t, u, p) = -(south_mask(x, y, z, p) + east_nudge(x, y, z, p)) * u / p.τ 
@inline sponge_v(x, y, z, t, v, p) = -(south_mask(x, y, z, p) + east_nudge(x, y, z, p)) * (v - v∞(x, z, t)) / p.τ 
@inline sponge_w(x, y, z, t, w, p) = -(south_mask(x, y, z, p) + east_nudge(x, y, z, p)) * w / p.τ

# temperature and salinity: we need to nudge south to B2 mooring, and north to B1 mooring. 
@inline sponge_T(x, y, z, t, T, p) =
    -south_mask(x, y, z, p) / p.τ_ts * (T - tsbc(x, y, z, t)) -
    north_mask(x, y, z, p) / p.τ_ts * (T - tnbc(x, y, z, t)) -
    east_mask(x, y, z, p) / p.τ_ts * (T - Tₑ(x, y, z))
@inline sponge_S(x, y, z, t, S, p) =
    -south_mask(x, y, z, p) / p.τ_ts * (S - ssbc(x, y, z, t)) -
    north_mask(x, y, z, p) / p.τ_ts * (S - snbc(x, y, z, t)) -
    east_mask(x, y, z, p) / p.τ_ts * (S - Sₑ(x, y, z))

# add forcings
FT = Forcing(sponge_T, field_dependencies = :T, parameters = params)
FS = Forcing(sponge_S, field_dependencies = :S, parameters = params)
Fᵤ = Forcing(sponge_u, field_dependencies = :u, parameters = params)
Fᵥ = Forcing(sponge_v, field_dependencies = :v, parameters = params)
F_w = Forcing(sponge_w, field_dependencies = :w, parameters = params)

forcings = (u = Fᵤ, v = Fᵥ, w = F_w, T = FT, S = FS)

T_bcs = FieldBoundaryConditions()
S_bcs = FieldBoundaryConditions()

u_bcs = FieldBoundaryConditions()
v_bcs = FieldBoundaryConditions()
w_bcs = FieldBoundaryConditions()

bcs = (u = u_bcs, v = v_bcs, w = w_bcs, T = T_bcs, S = S_bcs)

model = NonhydrostaticModel(
    grid = ib_grid,
    timestepper = :RungeKutta3,
    advection = WENO(order=5),
    closure = AnisotropicMinimumDissipation(),
    pressure_solver = ConjugateGradientPoissonSolver(ib_grid),
    tracers = (:T, :S),
    buoyancy = SeawaterBuoyancy(),
    coriolis = FPlane(latitude = 35.2480),
    boundary_conditions = bcs,
    forcing = forcings
)

# output stuff
@info "creating output fields"

overwrite_existing = true

simulation = Simulation(model, Δt=10minutes, stop_time=2days)

conjure_time_step_wizard!(simulation, cfl = 0.7, diffusive_cfl = 0.8)

progress = SingleLineMessenger()
simulation.callbacks[:progress] = Callback(progress, TimeInterval(1hour))

u, v, w = model.velocities
T = model.tracers.T
S = model.tracers.S

outputs = (; u, v, w, T, S)
saved_output_filename = "verification_bcs.nc"

simulation.output_writers[:fields] = NetCDFWriter(model, outputs, 
                                                        schedule = TimeInterval(1hour),
                                                        filename = saved_output_filename,
                                                        overwrite_existing = overwrite_existing)

@info "imposing initial conditions"
# impose initial conditions
uᵢ = 0.005 * rand(size(u)...)
vᵢ = 0.005 * rand(size(v)...)
wᵢ = 0.005 * rand(size(w)...)

uᵢ .-= mean(uᵢ)
vᵢ .-= mean(vᵢ)
wᵢ .-= mean(wᵢ)
vᵢ .+= 0.10

@inline α_lin(y) = clamp(y / Ly, 0.0, 1.0)
@inline blend(a, b, α) = (1 - α) * a + α * b
@inline Tᵢ(x, y, z) = blend(iT_south(z), iT_north(z), α_lin(y))
@inline Sᵢ(x, y, z) = blend(iS_south(z), iS_north(z), α_lin(y))

set!(model, u = uᵢ, v = vᵢ, w = wᵢ, T = Tᵢ, S = Sᵢ)

@info "running verification simulation"
run!(simulation)

@info "analyzing results"

ds = NCDataset(saved_output_filename)
v_data = Array(ds["v"])
T_data = Array(ds["T"])
S_data = Array(ds["S"])
y_T = Array(ds["y_aca"])
y_v = Array(ds["y_afa"])
z_nodes = Array(ds["z_aac"])
t_nodes = Array(ds["time"])
close(ds)

println("Loaded data sizes:")
println("v: ", size(v_data))
println("T: ", size(T_data))
println("y_T: ", size(y_T))
println("y_v: ", size(y_v))

# Take the last time step
v_end = v_data[:, :, :, end]
T_end = T_data[:, :, :, end]
S_end = S_data[:, :, :, end]

# Extract profile at mid-x and mid-z (or near surface)
# x index: Nx/2
# z index: Nz/2
xi = div(Nx, 2)
zi = div(Nz, 2)

v_profile = v_end[xi, :, zi]
T_profile = T_end[xi, :, zi]
S_profile = S_end[xi, :, zi]

# Target values
z_val = z_nodes[zi]
T_target_south = iT_south(z_val)
T_target_north = iT_north(z_val)
v_target = 0.10

# Plotting
p1 = plot(y_v ./ 1e3, v_profile, label="v model", lw=2, title="Meridional Velocity", ylabel="v [m/s]")
plot!(p1, [0, Ly/1e3], [v_target, v_target], label="target", ls=:dash, color=:black)
vline!(p1, [Lₛ/1e3, (Ly-Lₛ)/1e3], label="sponge limits", color=:gray, alpha=0.5)

p2 = plot(y_T ./ 1e3, T_profile, label="T model", lw=2, title="Temperature", ylabel="T [°C]")
plot!(p2, [0, Lₛ/1e3], [T_target_south, T_target_south], label="South Target", lw=3, color=:red)
plot!(p2, [(Ly-Lₛ)/1e3, Ly/1e3], [T_target_north, T_target_north], label="North Target", lw=3, color=:blue)
vline!(p2, [Lₛ/1e3, (Ly-Lₛ)/1e3], label="sponge limits", color=:gray, alpha=0.5)

p_final = plot(p1, p2, layout=(2, 1), size=(800, 600), xlabel="y [km]")
savefig(p_final, "verification_bcs.png")

@info "Verification complete. Saved plot to verification_bcs.png"

# Print values for verification
println("Verification Results:")
println("South Boundary (y=0): T_model = $(T_profile[1]), T_target = $T_target_south")
println("North Boundary (y=Ly): T_model = $(T_profile[end]), T_target = $T_target_north")
println("South Sponge End (y=$(Lₛ/1e3)km): T_model = $(T_profile[findfirst(y_T .>= Lₛ)])")
println("North Sponge Start (y=$((Ly-Lₛ)/1e3)km): T_model = $(T_profile[findfirst(y_T .>= Ly-Lₛ)])")

