# Vendored from Oceanostics.jl v0.21.7, src/ProgressMessengers/ (all four files),
# because no Oceanostics release supports both Oceananigans 0.118 and Julia 1.11.
# Changes: the files are inlined into one, and the Crayons colouring is removed
# (ColoredNumber returns the plain string). Nothing else is changed.
#
# MIT License
#
# Copyright (c) 2021 Tomas Chor
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

module ProgressMessengers

using Printf

import Base: +, *

export AbstractProgressMessenger
export FunctionMessenger
export MaxUVelocity, MaxVVelocity, MaxWVelocity
export MaxVelocities
export Iteration, SimulationTime, TimeStep, PercentageProgress, Walltime, StepDuration, BasicTimeMessenger, TimeMessenger, StopwatchMessenger
export MaxViscosity, AdvectiveCFLNumber, DiffusiveCFLNumber, BasicStabilityMessenger, StabilityMessenger
export CourantNumber, NormalizedMaxViscosity
export BasicMessenger, SingleLineMessenger, TimedMessenger

abstract type AbstractProgressMessenger end

comma = ",  "
space = ""
indented_newline = "\n      "

#+++ ColoredNumber (plain: the colour codes are dropped; they are junk in PBS logs)
ColoredNumber(s::AbstractString) = String(s)
#---

#+++ FunctionMessenger
Base.@kwdef struct FunctionMessenger{F} <: AbstractProgressMessenger
    func :: F
end

function (fmessenger::FunctionMessenger)(sim)
    message = fmessenger.func(sim)
    return_or_print(message)
end
#---

#+++ Basic operations with functions and strings
@inline +(a::AbstractProgressMessenger,   b::AbstractProgressMessenger)   = FunctionMessenger(sim -> a(sim) * comma * b(sim))
@inline *(a::AbstractProgressMessenger,   b::AbstractProgressMessenger)   = FunctionMessenger(sim -> a(sim) * space * b(sim))

const FunctionOrProgressMessenger = Union{Function, AbstractProgressMessenger}
@inline +(a::AbstractProgressMessenger,   b::FunctionOrProgressMessenger) = FunctionMessenger(sim -> a(sim) * comma * b(sim))
@inline +(a::FunctionOrProgressMessenger, b::AbstractProgressMessenger)   = FunctionMessenger(sim -> a(sim) * comma * b(sim))
@inline *(a::AbstractProgressMessenger,   b::FunctionOrProgressMessenger) = FunctionMessenger(sim -> a(sim) * space * b(sim))
@inline *(a::FunctionOrProgressMessenger, b::AbstractProgressMessenger)   = FunctionMessenger(sim -> a(sim) * space * b(sim))

const StringOrProgressMessenger = Union{String, AbstractProgressMessenger}
@inline +(a::AbstractProgressMessenger, b::StringOrProgressMessenger) = FunctionMessenger(sim -> a(sim) * comma * b)
@inline +(a::StringOrProgressMessenger, b::AbstractProgressMessenger) = FunctionMessenger(sim -> a      * comma * b(sim))
@inline *(a::AbstractProgressMessenger, b::StringOrProgressMessenger) = FunctionMessenger(sim -> a(sim) * space * b)
@inline *(a::StringOrProgressMessenger, b::AbstractProgressMessenger) = FunctionMessenger(sim -> a      * space * b(sim))
#---

return_or_print(message, pm::AbstractProgressMessenger) = pm.print ? (@info message) : (return string(message))
return_or_print(message) = return string(message)

#+++ Basic definitions
Base.@kwdef struct MaxUVelocity <: AbstractProgressMessenger
    with_prefix :: Bool = true
    with_units  :: Bool = true
    print       :: Bool = false
end

@inline function (mu::MaxUVelocity)(sim)
    u_max = maximum(abs, sim.model.velocities.u)
    message = ColoredNumber(@sprintf("%.2e", u_max))
    mu.with_prefix     && (message = "|u|ₘₐₓ = " * message)
    mu.with_units && (message = message * " m/s")
    return_or_print(message, mu)
end

Base.@kwdef struct MaxVVelocity <: AbstractProgressMessenger
    with_prefix :: Bool = true
    with_units  :: Bool = true
    print       :: Bool = false
end

@inline function (mv::MaxVVelocity)(sim)
    v_max = maximum(abs, sim.model.velocities.v)
    message = ColoredNumber(@sprintf("%.2e", v_max))
    mv.with_prefix     && (message = "|v|ₘₐₓ = " * message)
    mv.with_units && (message = message * " m/s")
    return_or_print(message, mv)
end

Base.@kwdef struct MaxWVelocity <: AbstractProgressMessenger
    with_prefix :: Bool = true
    with_units  :: Bool = true
    print       :: Bool = false
end

@inline function (mw::MaxWVelocity)(sim)
    w_max = maximum(abs, sim.model.velocities.w)
    message = ColoredNumber(@sprintf("%.2e", w_max))
    mw.with_prefix     && (message = "|w|ₘₐₓ = " * message)
    mw.with_units && (message = message * " m/s")
    return_or_print(message, mw)
end
#---

#+++ MaxVelocities
struct MaxVelocities{PM <: AbstractProgressMessenger} <: AbstractProgressMessenger
    max_u       :: PM
    max_v       :: PM
    max_w       :: PM
    with_prefix :: Bool
    with_units  :: Bool
    print       :: Bool
end

MaxVelocities(; max_u = MaxUVelocity(with_prefix = false, with_units = false, print = false),
                max_v = MaxVVelocity(with_prefix = false, with_units = false, print = false),
                max_w = MaxWVelocity(with_prefix = false, with_units = false, print = false),
                with_prefix = true,
                with_units = true,
                print = false) = MaxVelocities{AbstractProgressMessenger}(max_u, max_v, max_w, with_prefix, with_units, print)

function (maxvel::MaxVelocities)(simulation)
    message = ("[" * maxvel.max_u + maxvel.max_v + maxvel.max_w * "]")(simulation)
    maxvel.with_prefix && (message = "|u⃗|ₘₐₓ = " * message)
    maxvel.with_units  && (message = message * " m/s")
    return_or_print(message, maxvel)
end
#---
using Oceananigans.Utils: prettytime
using Oceananigans.Simulations: iteration, time

#+++ Iteration
Base.@kwdef struct Iteration <: AbstractProgressMessenger
    with_prefix :: Bool = true
    print       :: Bool = false
end

@inline function (itm::Iteration)(simulation)
    iter = iteration(simulation)
    message = ColoredNumber(@sprintf("%6d", iter))
    itm.with_prefix && (message = "iter = " * message)
    return_or_print(message, itm)
end
#---

#+++ SimulationTime
Base.@kwdef struct SimulationTime <: AbstractProgressMessenger
    with_prefix :: Bool = true
    with_units ::  Bool = true
    print       :: Bool = false
end

@inline function (tm::SimulationTime)(simulation)
    t = time(simulation)
    message = ColoredNumber(tm.with_units ? prettytime(t) : @sprintf("%.2g", t))
    tm.with_prefix && (message = "time = " * message)
    return_or_print(message, tm)
end
#---

#+++ TimeStep
Base.@kwdef struct TimeStep <: AbstractProgressMessenger
    with_prefix :: Bool = true
    with_units ::  Bool = true
    print       :: Bool = false
end

@inline function (ts::TimeStep)(simulation)
    Δt = simulation.Δt
    message = ColoredNumber(ts.with_units ? prettytime(Δt) : @sprintf("%.2g", Δt))
    ts.with_prefix && (message = "Δt = " * message)
    return_or_print(message, ts)
end
#---

#+++ PercentageProgress
Base.@kwdef struct PercentageProgress <: AbstractProgressMessenger
    with_prefix :: Bool = true
    with_units  :: Bool = true
    print       :: Bool = false
end

@inline function (pp::PercentageProgress)(simulation)
    percentage_progress_time      = 100 * time(simulation) / simulation.stop_time
    percentage_progress_iteration = 100 * iteration(simulation) / simulation.stop_iteration
    if percentage_progress_time > percentage_progress_iteration
        message = ColoredNumber(@sprintf("%06.2f%%", percentage_progress_time))
        pp.with_units  && (message = message * " by simulation time")

    else
        message = ColoredNumber(@sprintf("%06.2f%%", percentage_progress_iteration))
        pp.with_units  && (message = message * " by iteration")
    end
    pp.with_prefix && (message = "progress = " * message)
    return_or_print(message, pp)
end
#---

#+++ Walltime
Base.@kwdef mutable struct Walltime{T} <: AbstractProgressMessenger
    wall_seconds⁰  :: T    = 1e-9*time_ns() # Wall time at previous calback
    with_prefix    :: Bool = true
    with_units     :: Bool = true
    print          :: Bool = false
end

function (wt::Walltime)(simulation)
    current_wall_seconds = 1e-9 * time_ns() - wt.wall_seconds⁰
    message = ColoredNumber(wt.with_units ? prettytime(current_wall_seconds) : @sprintf("%.2g", current_wall_seconds))
    wt.with_prefix && (message = "walltime = " * message)
    return_or_print(message, wt)
end
#---

#+++ StepDuration
Base.@kwdef mutable struct StepDuration{T, I} <: AbstractProgressMessenger
    wall_seconds⁻ :: T    = 1e-9*time_ns() # Wall time at previous calback
    iteration⁻    :: I    = 0 # Iteration at previous calback
    with_prefix   :: Bool = true
    with_units    :: Bool = true
    print         :: Bool = false
end

function (wpt::StepDuration)(simulation)
    iter = iteration(simulation)

    seconds_since_last_callback = 1e-9 * time_ns() - wpt.wall_seconds⁻
    iterations_since_last_callback = iter == 0 ? Inf : iter - wpt.iteration⁻

    wall_time_per_step = seconds_since_last_callback / iterations_since_last_callback
    wpt.wall_seconds⁻ = 1e-9 * time_ns()
    wpt.iteration⁻ = iter

    message = ColoredNumber(wpt.with_units ? prettytime(wall_time_per_step) : @sprintf("%.2g", wall_time_per_step))
    wpt.with_prefix && (message = "walltime / timestep = " * message)
    return_or_print(message, wpt)
end
#---

#+++ BasicTimeMessenger
struct BasicTimeMessenger{PM <: AbstractProgressMessenger} <: AbstractProgressMessenger
    percentage  :: PM
    time        :: PM
    Δt          :: PM
    walltime    :: PM
    print       :: Bool
end

BasicTimeMessenger(; percentage = PercentageProgress(with_prefix = false, with_units = false, print = false),
                     time = SimulationTime(with_prefix = true, with_units = true, print = false),
                     Δt = TimeStep(with_prefix = true, with_units = true, print = false),
                     walltime = Walltime(with_prefix = true, with_units = true, print = false),
                     print = true) = BasicTimeMessenger{AbstractProgressMessenger}(percentage, time, Δt, walltime, print)

function (stm::BasicTimeMessenger)(simulation)
    message = ("["*stm.percentage*"] " * stm.time + stm.Δt + stm.walltime)(simulation)
    return_or_print(message, stm)
end
#---

#+++ TimeMessenger
struct TimeMessenger{PM <: AbstractProgressMessenger} <: AbstractProgressMessenger
    iteration             :: PM
    simple_time_messenger :: PM
    print                 :: Bool
end

TimeMessenger(; iteration = Iteration(with_prefix = false, print = false),
                simple_time_messenger = BasicTimeMessenger(print = false),
                print = true) = TimeMessenger{AbstractProgressMessenger}(iteration, simple_time_messenger, print)

function (tm::TimeMessenger)(simulation)
    message = ("iter = " * tm.iteration + tm.simple_time_messenger)(simulation)
    return_or_print(message, tm)
end
#---

#+++ StopwatchMessenger
struct StopwatchMessenger{PM <: AbstractProgressMessenger} <: AbstractProgressMessenger
    time_messenger        :: PM
    walltime_per_timestep :: PM
    print                 :: Bool
end

StopwatchMessenger(; time_messenger = TimeMessenger(print = false),
                     walltime_per_timestep = StepDuration(with_prefix = true, with_units = true, print = false),
                     print = true) = StopwatchMessenger{AbstractProgressMessenger}(time_messenger, walltime_per_timestep, print)

function (swm::StopwatchMessenger)(simulation)
    message = (swm.time_messenger + swm.walltime_per_timestep)(simulation)
    return_or_print(message, swm)
end
#---
import Oceananigans.Diagnostics
using Oceananigans.TurbulenceClosures: viscosity, diffusivity

#+++ AdvectiveCFLNumber
Base.@kwdef struct AdvectiveCFLNumber <: AbstractProgressMessenger
    with_prefix :: Bool = true
    print       :: Bool = false
end

@inline function (acfl::AdvectiveCFLNumber)(simulation)
    cfl = Diagnostics.AdvectiveCFL(simulation.Δt)(simulation.model)
    message = ColoredNumber(@sprintf("%.2g", cfl))
    acfl.with_prefix && (message = "advective CFL = " * message)
    return_or_print(message, acfl)
end
#---

#+++ DiffusiveCFLNumber
Base.@kwdef struct DiffusiveCFLNumber <: AbstractProgressMessenger
    with_prefix :: Bool = true
    print       :: Bool = false
end

@inline function (dcfl::DiffusiveCFLNumber)(simulation)
    cfl = Diagnostics.DiffusiveCFL(simulation.Δt)(simulation.model)
    message = ColoredNumber(@sprintf("%.2g", cfl))
    dcfl.with_prefix && (message = "diffusive CFL = " * message)
    return_or_print(message, dcfl)
end
#---

#+++ MaxViscosity
Base.@kwdef struct MaxViscosity <: AbstractProgressMessenger
    with_prefix :: Bool = true
    with_units  :: Bool = true
    print       :: Bool = false
end

tuple_to_op(ν) = ν
tuple_to_op(::Nothing) = nothing
tuple_to_op(ν_tuple::Tuple) = sum(ν_tuple)

@inline function (maxν::MaxViscosity)(simulation)
    ν = tuple_to_op(viscosity(simulation.model.closure, simulation.model.closure_fields))
    ν_max = maximum(abs, ν)
    message = ColoredNumber(@sprintf("%.2g", ν_max))
    maxν.with_prefix && (message = "νₘₐₓ = " * message)
    maxν.with_units  && (message = message * " m²/s")
    return_or_print(message, maxν)
end
#---

#+++ BasicStabilityMessenger
struct BasicStabilityMessenger{PM <: AbstractProgressMessenger} <: AbstractProgressMessenger
    advective_cfl :: PM
    diffusive_cfl :: PM
    print         :: Bool
end

BasicStabilityMessenger(; advective_cfl = AdvectiveCFLNumber(with_prefix = true, print = false),
                           diffusive_cfl = DiffusiveCFLNumber(with_prefix = true, print = false),
                           print = true) = BasicStabilityMessenger{AbstractProgressMessenger}(advective_cfl, diffusive_cfl, print)

function (bsm::BasicStabilityMessenger)(simulation)
    message = (bsm.advective_cfl + bsm.diffusive_cfl)(simulation)
    return_or_print(message, bsm)
end
#---

#+++ StabilityMessenger
struct StabilityMessenger{PM <: AbstractProgressMessenger} <: AbstractProgressMessenger
    basic_stab_messenger :: PM
    max_viscosity        :: PM
    print                :: Bool
end

StabilityMessenger(; basic_stab_messenger = BasicStabilityMessenger(print = false),
                     max_viscosity = MaxViscosity(with_prefix = true, with_units = true, print = false),
                     print = true) = StabilityMessenger{AbstractProgressMessenger}(basic_stab_messenger, max_viscosity, print)

function (sm::StabilityMessenger)(simulation)
    message = (sm.basic_stab_messenger + sm.max_viscosity)(simulation)
    return_or_print(message, sm)
end
#---

const CourantNumber = AdvectiveCFLNumber
const NormalizedMaxViscosity = DiffusiveCFLNumber

#+++ BasicMessenger
struct BasicMessenger{PM <: AbstractProgressMessenger} <: AbstractProgressMessenger
    basic_time_messenger      :: PM
    basic_stability_messenger :: PM
    print                     :: Bool
end

BasicMessenger(; basic_time_messenger = BasicTimeMessenger(print = false),
                 basic_stability_messenger = BasicStabilityMessenger(print = false),
                 print = true) = BasicMessenger{AbstractProgressMessenger}(basic_time_messenger, basic_stability_messenger, print)

function (pm::BasicMessenger)(simulation)
    message = (pm.basic_time_messenger + pm.basic_stability_messenger)(simulation)
    return_or_print(message, pm)
end
#---

#+++ SingleLineMessenger
struct SingleLineMessenger{PM <: AbstractProgressMessenger} <: AbstractProgressMessenger
    time_messenger      :: PM
    stability_messenger :: PM
    print               :: Bool
end

SingleLineMessenger(; time_messenger = TimeMessenger(print = false),
                      stability_messenger = StabilityMessenger(print = false),
                      print = true) = SingleLineMessenger{AbstractProgressMessenger}(time_messenger, stability_messenger, print)

function (pm::SingleLineMessenger)(simulation)
    message = (pm.time_messenger + pm.stability_messenger)(simulation)
    return_or_print(message, pm)
end
#---

#+++ TimedMessenger
struct TimedMessenger{PM <: AbstractProgressMessenger} <: AbstractProgressMessenger
    stopwatch_messenger :: PM
    maxvels             :: PM
    stability_messenger :: PM
    print               :: Bool
end

TimedMessenger(; stopwatch_messenger = StopwatchMessenger(print = false),
                 maxvels = MaxVelocities(with_prefix = true, with_units = true, print = false),
                 stability_messenger = StabilityMessenger(print = false),
                 print = true) = TimedMessenger{AbstractProgressMessenger}(stopwatch_messenger, maxvels, stability_messenger, print)

function (pm::TimedMessenger)(simulation)
    message = (pm.stopwatch_messenger * indented_newline * pm.maxvels + pm.stability_messenger)(simulation)
    return_or_print(message, pm)
end
#---

# Checkpointing: a messenger holds no state worth keeping (only wall-clock
# timers), so it saves nothing and is left as it is on pickup. Without these,
# picking up a checkpointed simulation that uses one fails with a MethodError
# in restore_prognostic_state!.
import Oceananigans
Oceananigans.prognostic_state(::AbstractProgressMessenger) = nothing
Oceananigans.restore_prognostic_state!(m::AbstractProgressMessenger, from) = m
Oceananigans.restore_prognostic_state!(m::AbstractProgressMessenger, ::Nothing) = m   # resolves an ambiguity

end # module
