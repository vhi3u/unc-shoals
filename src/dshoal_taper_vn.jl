function dshoal_taper(Lx, Ly, sigma, Hs, N)
    x = range(0, stop=Lx, length=N)
    y = range(0, stop=Ly, length=N)

    # --- BACKGROUND BATHYMETRY (HW) ---

    # along-shoal
    hwx = zeros(length(x))

    aw, h_aw = 15.0, -20.0
    bw, h_bw = 30.0, -25.0
    cw, h_cw = 50.0, -50.0

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

    # along-shoal (hsx)
    hsx = zeros(length(x))

    as, h_as = 2.5, -5.0
    bs, h_bs = 27.0, -15.0
    cs, h_cs = 30.0, -27.0
    ds, h_ds = 36.0, -50.0

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
    gauss = Hs .* exp.(-((y .- y0) .^ 2) ./ (2 * sigma^2))
    hsy = gauss # centered Gaussian

    # taper in x
    taper_x = ones(length(x))
    for i in eachindex(x)
        if x[i] > cs
            taper_x[i] = 0.0
        elseif cs - 1 <= x[i] <= cs
            taper_x[i] = 0.5 * (1 + cos(pi * (x[i] - (cs - 1))))
        end
    end

    # Combine shoal
    hs = hsy .* taper_x' .+ hsx'  # Outer addition
    hs .-= maximum(hs)

    # Combine background and shoal bathymetry
    h = max.(hw, hs)

    # --- Y-TAPER: blend to -50 m at y = 0 and y = Ly ---
    h_bottom = -50.0
    taper_width_y = 20.0  # 20 km taper width

    # Build y-taper array
    taper_y = ones(length(y))

    for j in eachindex(y)
        yj = y[j]
        if yj <= taper_width_y
            # Taper near y = 0: goes 0 -> 1
            taper_y[j] = 0.5 * (1 - cos(pi * yj / taper_width_y))
        elseif yj >= (Ly - taper_width_y)
            # Taper near y = Ly: goes 1 -> 0
            taper_y[j] = 0.5 * (1 + cos(pi * (yj - (Ly - taper_width_y)) / taper_width_y))
        end
    end

    # Apply taper: blend h towards -50 at y-boundaries
    # h_new = taper_y * h + (1 - taper_y) * h_bottom
    h = taper_y .* h .+ (1 .- taper_y) .* h_bottom

    return x, y, h'
end
