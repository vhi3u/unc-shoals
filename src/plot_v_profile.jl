using Plots

@inline function v∞(x, z, t, p)
    xC = 3e3
    xS = 60e3
    Lw = p.Lx
    k1 = 80 / Lw
    k2 = 40 / Lw

    s1 = 1 / (1 + exp(-k1 * (x - xC)))
    s2 = 1 / (1 + exp(k2 * (x - xS)))
    s = (s1 - 1) + s2
    sc = clamp(s, 0.0, 1.0)
    return 0.10 * sc
end

Lx = 100e3
p = (; Lx=Lx)
x = 0:100:Lx
v = [v∞(xi, 0, 0, p) for xi in x]

plt = plot(x, v, xlabel="x [m]", ylabel="v [m/s]", title="v∞ profile", label=false, size=(800, 400))
savefig(plt, "v_profile.png")
@info "Saved v_profile.png"
