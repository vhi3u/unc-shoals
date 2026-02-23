function dshoal_param(Lx, Ly, sigma, Hs, Nx, Ny;
    shelf_length=30.0,       # length of the shelf
    shelf_depth=-20.0,       # starting depth of the shelf
    shoal_length=30.0,       # length of the top of the shoal
    shoal_crest_depth=-5.0,  # shallowest depth of the shoal
    deep_ocean_depth=-50.0)  # depth of the deep ocean

    x = range(0, stop=Lx, length=Nx)
    y = range(0, stop=Ly, length=Ny)

    # --- BACKGROUND BATHYMETRY (HW) ---
    # You asked to verify: Yes, the shelf is indeed the "background" bathymetry (hw).

    # along-shoal
    hwx = zeros(length(x))

    # Scale the coordinates based on the kwarg sizes while keeping proportional shape
    aw, h_aw = shelf_length * (15.0 / 30.0), shelf_depth
    bw, h_bw = shelf_length, shelf_depth - 5.0
    cw, h_cw = shelf_length + 20.0, deep_ocean_depth

    m1w = h_aw / aw
    m2w = (h_bw - h_aw) / (bw - aw)
    m3w = (h_cw - h_bw) / (cw - bw)

    for i in eachindex(x)
        xi = x[i]
        if xi < aw
            hwx[i] = m1w * xi
        elseif xi < bw
            hwx[i] = h_aw + m2w * (xi - aw)
        elseif xi < cw
            hwx[i] = h_bw + m3w * (xi - bw)
        else
            hwx[i] = h_cw
        end
    end

    # across-shoal (flat)
    hwy = zeros(length(y))

    # Combine background
    hw = hwy .+ hwx'   # Outer addition (broadcast)
    hw .-= maximum(hw)

    # --- SHOAL BATHYMETRY (HS) ---
    # Width is defined by 'sigma', Height by 'Hs' cross-shore amplitude, & 'shoal_crest_depth'

    # along-shoal (hsx)
    hsx = zeros(length(x))

    # Scale shoal points proportionally
    as, h_as = shoal_length * (2.5 / 30.0), shoal_crest_depth

    # Make Hs directly increase the height of the bump from the default geometry
    bump_offset = Hs - 15.0

    bs, h_bs = shoal_length * (27.0 / 30.0), shoal_crest_depth - 10.0 + bump_offset
    cs, h_cs = shoal_length, shoal_crest_depth - 22.0 + bump_offset
    ds, h_ds = shoal_length * (36.0 / 30.0), deep_ocean_depth

    m1s = h_as / as
    m2s = (h_bs - h_as) / (bs - as)
    m3s = (h_cs - h_bs) / (cs - bs)
    m4s = (h_ds - h_cs) / (ds - cs)

    for i in eachindex(x)
        xi = x[i]
        if xi < as
            hsx[i] = m1s * xi
        elseif xi < bs
            hsx[i] = h_as + m2s * (xi - as)
        elseif xi < cs
            hsx[i] = h_bs + m3s * (xi - bs)
        elseif xi < ds
            hsx[i] = h_cs + m4s * (xi - cs)
        else
            hsx[i] = h_ds
        end
    end

    # across-shoal (hsy)
    y0 = Ly / 2
    # sigma controls the shoal width
    gauss = Hs .* exp.(-((y .- y0) .^ 2) ./ (2 * sigma^2))
    hsy = gauss # centered Gaussian

    # taper in x
    taper = ones(length(x))
    taper_len = max(1.0 * (shoal_length / 30.0), 0.1) # Scale the taper longitudinally
    for i in eachindex(x)
        if x[i] > cs
            taper[i] = 0.0
        elseif cs - taper_len <= x[i] <= cs
            taper[i] = 0.5 * (1 + cos(pi * (x[i] - (cs - taper_len))))
        end
    end

    # Combine shoal
    hs = hsy .* taper' .+ hsx'  # Outer addition
    hs .-= maximum(hs)

    # Combine background and shoal bathymetry
    h = max.(hw, hs)

    return x, y, h'
end
