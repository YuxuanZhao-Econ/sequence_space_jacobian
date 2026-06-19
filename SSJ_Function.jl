# Auxiliary numerical and display helpers for the SSJ tutorial notebooks.

using LinearAlgebra

"""
    finite_difference_jacobian_at(f, x; h=1e-6)

Purpose: Compute a forward finite-difference Jacobian of vector-valued `f` at `x`.
Inputs: function `f`, point `x`, and step size `h`.
Output: matrix `J` with column `j` equal to the response to perturbing `x[j]`.
"""
function finite_difference_jacobian_at(f, x; h=1e-6)
    f0 = f(x)
    J = zeros(length(f0), length(x))
    for j in eachindex(x)
        xstep = copy(x)
        xstep[j] += h
        J[:, j] = (f(xstep) - f0) / h
    end
    return J
end

"""
    newton_solve(f, x0; tol=1e-12, max_iter=50, is_valid=x -> true)

Purpose: Solve a nonlinear system `f(x)=0` with finite-difference Newton steps.
Inputs: residual function `f`, initial guess `x0`, tolerance, iteration limit, and an optional feasibility check.
Output: root vector `x`.
"""
function newton_solve(f, x0; tol=1e-12, max_iter=50, is_valid=x -> true)
    x = copy(x0)
    fx = f(x)

    for it in 1:max_iter
        if maximum(abs.(fx)) < tol
            return x
        end

        J = finite_difference_jacobian_at(f, x)
        dx = -(J \ fx)

        step = 1.0
        accepted = false
        for bt in 1:30
            x_try = x + step * dx

            if is_valid(x_try)
                fx_try = f(x_try)
                if maximum(abs.(fx_try)) < maximum(abs.(fx))
                    x, fx = x_try, fx_try
                    accepted = true
                    break
                end
            end
            step *= 0.5
        end

        if !accepted
            error("Newton method failed to find an improving step.")
        end
    end

    error("Newton method did not converge after $max_iter iterations.")
end

if !isdefined(Main, :InlineSVG)
    @eval begin
        struct InlineSVG
            data::String
        end

        Base.show(io::IO, ::MIME"image/svg+xml", p::InlineSVG) = print(io, p.data)
        Base.show(io::IO, p::InlineSVG) = print(io, "InlineSVG")
    end
end

"""
    rbc_dag_svg()

Purpose: Draw the RBC tutorial DAG as an inline SVG.
Inputs: none.
Output: SVG string, usually displayed with `InlineSVG`.
"""
function rbc_dag_svg()
    return """
    <svg xmlns='http://www.w3.org/2000/svg' width='760' height='560' viewBox='0 0 760 560'>
      <defs>
        <marker id='arrow' markerWidth='10' markerHeight='10' refX='8' refY='3' orient='auto' markerUnits='strokeWidth'>
          <path d='M0,0 L0,6 L9,3 z' fill='#333333'/>
        </marker>
        <style>
          .node { fill: #ffffff; stroke: #2f3b52; stroke-width: 1.8; }
          .box { rx: 8; ry: 8; }
          .label { font-family: Georgia, serif; font-size: 17px; fill: #111827; text-anchor: middle; dominant-baseline: middle; }
          .small { font-family: Georgia, serif; font-size: 14px; fill: #374151; text-anchor: middle; }
          .edge { stroke: #333333; stroke-width: 1.7; fill: none; marker-end: url(#arrow); }
          .tag { font-family: Georgia, serif; font-size: 13px; fill: #111827; text-anchor: middle; }
        </style>
      </defs>

      <rect x='35' y='30' width='150' height='62' class='node box'/>
      <text x='110' y='55' class='label'>Exogenous</text>
      <text x='110' y='78' class='small'>Z</text>

      <rect x='550' y='30' width='150' height='62' class='node box'/>
      <text x='625' y='55' class='label'>Unknowns</text>
      <text x='625' y='78' class='small'>K<tspan baseline-shift='super'>+</tspan>, L</text>

      <ellipse cx='360' cy='155' rx='105' ry='38' class='node'/>
      <text x='360' y='155' class='label'>firm</text>
      <text x='360' y='181' class='small'>(K<tspan baseline-shift='sub'>t</tspan>, L, Z) -> (r, w, Y)</text>

      <ellipse cx='360' cy='285' rx='125' ry='40' class='node'/>
      <text x='360' y='282' class='label'>household</text>
      <text x='360' y='309' class='small'>(K<tspan baseline-shift='super'>+</tspan>, L, w) -> (C, I)</text>

      <ellipse cx='360' cy='420' rx='135' ry='42' class='node'/>
      <text x='360' y='416' class='label'>market clearing</text>
      <text x='360' y='442' class='small'>euler, goods_mkt, walras</text>

      <polygon points='360,510 470,535 360,560 250,535' class='node'/>
      <text x='360' y='535' class='label'>Targets</text>

      <path d='M145 92 C180 115, 235 140, 258 150' class='edge'/>
      <text x='205' y='126' class='tag'>Z</text>

      <path d='M585 92 C545 116, 490 140, 460 150' class='edge'/>
      <text x='530' y='121' class='tag'>L, K<tspan baseline-shift='sub'>t</tspan> from K<tspan baseline-shift='super'>+</tspan></text>

      <path d='M610 92 C565 185, 515 255, 465 278' class='edge'/>
      <text x='548' y='214' class='tag'>K<tspan baseline-shift='super'>+</tspan>, L</text>

      <path d='M625 92 C585 218, 540 340, 465 398' class='edge'/>
      <text x='565' y='302' class='tag'>K<tspan baseline-shift='super'>+</tspan>, L</text>

      <path d='M360 193 L360 244' class='edge'/>
      <text x='390' y='224' class='tag'>w</text>

      <path d='M282 180 C185 250, 170 345, 260 402' class='edge'/>
      <text x='195' y='312' class='tag'>r, w, Y</text>

      <path d='M360 325 L360 377' class='edge'/>
      <text x='393' y='356' class='tag'>C, I</text>

      <path d='M360 462 L360 508' class='edge'/>
      <text x='418' y='490' class='tag'>euler, goods_mkt</text>
    </svg>
    """
end

"""
    ks_dag_svg()

Purpose: Draw the Krusell-Smith sequence-space DAG as an inline SVG.
Inputs: none.
Output: SVG string, usually displayed with `InlineSVG`.
"""
function ks_dag_svg()
    return """
    <svg xmlns='http://www.w3.org/2000/svg' width='760' height='560' viewBox='0 0 760 560'>
      <defs>
        <marker id='ks_arrow' markerWidth='10' markerHeight='10' refX='8' refY='3' orient='auto' markerUnits='strokeWidth'>
          <path d='M0,0 L0,6 L9,3 z' fill='#333333'/>
        </marker>
        <style>
          .node { fill: #ffffff; stroke: #2f3b52; stroke-width: 1.8; }
          .box { rx: 8; ry: 8; }
          .label { font-family: Georgia, serif; font-size: 17px; fill: #111827; text-anchor: middle; dominant-baseline: middle; }
          .small { font-family: Georgia, serif; font-size: 14px; fill: #374151; text-anchor: middle; }
          .edge { stroke: #333333; stroke-width: 1.7; fill: none; marker-end: url(#ks_arrow); }
          .tag { font-family: Georgia, serif; font-size: 13px; fill: #111827; text-anchor: middle; }
        </style>
      </defs>

      <rect x='45' y='35' width='150' height='62' class='node box'/>
      <text x='120' y='60' class='label'>Exogenous</text>
      <text x='120' y='83' class='small'>Z</text>

      <rect x='565' y='35' width='150' height='62' class='node box'/>
      <text x='640' y='60' class='label'>Unknowns</text>
      <text x='640' y='83' class='small'>K<tspan baseline-shift='super'>+</tspan></text>

      <ellipse cx='380' cy='150' rx='112' ry='40' class='node'/>
      <text x='380' y='145' class='label'>firm</text>
      <text x='380' y='172' class='small'>(K<tspan baseline-shift='sub'>t</tspan>, Z) -> (r, w, Y)</text>

      <ellipse cx='380' cy='285' rx='145' ry='44' class='node'/>
      <text x='380' y='278' class='label'>household transition</text>
      <text x='380' y='306' class='small'>(r, w) -> (A<tspan baseline-shift='super'>+</tspan>, C)</text>

      <ellipse cx='380' cy='415' rx='135' ry='42' class='node'/>
      <text x='380' y='411' class='label'>asset market</text>
      <text x='380' y='437' class='small'>asset_mkt = A<tspan baseline-shift='super'>+</tspan> - K<tspan baseline-shift='super'>+</tspan></text>

      <polygon points='380,485 490,510 380,535 270,510' class='node'/>
      <text x='380' y='510' class='label'>Target</text>

      <path d='M160 97 C205 118, 250 136, 276 144' class='edge'/>
      <text x='222' y='123' class='tag'>Z</text>

      <path d='M605 97 C565 120, 515 140, 487 147' class='edge'/>
      <text x='550' y='123' class='tag'>K<tspan baseline-shift='sub'>t</tspan> from K<tspan baseline-shift='super'>+</tspan></text>

      <path d='M380 190 L380 239' class='edge'/>
      <text x='423' y='220' class='tag'>r, w</text>

      <path d='M300 183 C210 245, 210 360, 273 397' class='edge'/>
      <text x='205' y='298' class='tag'>Y</text>

      <path d='M640 97 C610 230, 555 355, 500 400' class='edge'/>
      <text x='580' y='287' class='tag'>K<tspan baseline-shift='super'>+</tspan></text>

      <path d='M380 329 L380 371' class='edge'/>
      <text x='420' y='356' class='tag'>A<tspan baseline-shift='super'>+</tspan>, C</text>

      <path d='M380 457 L380 483' class='edge'/>
      <text x='430' y='474' class='tag'>asset_mkt</text>
    </svg>
    """
end

# ---------------------------------------------------------------------
# Krusell-Smith notebook helpers
# ---------------------------------------------------------------------

# Structural and numerical parameters for the KS example.
Base.@kwdef struct KSParams
    beta::Float64 = 0.96
    eis::Float64 = 1.0
    delta::Float64 = 0.025
    alpha::Float64 = 0.36
    Z::Float64 = 1.0
    rho_e::Float64 = 0.966
    sd_e::Float64 = 0.05
    L::Float64 = 0.3043
    nE::Int = 7
    nA::Int = 500
    amin::Float64 = 0.0
    amax::Float64 = 200.0
end

# Container for all steady-state objects needed by the KS blocks.
struct KSSteadyState
    beta::Float64
    Z::Float64
    K::Float64
    L::Float64
    r::Float64
    w::Float64
    Y::Float64
    A::Float64
    C::Float64
    asset_mkt::Float64
    goods_mkt::Float64
    a_grid::Vector{Float64}
    e_grid::Vector{Float64}
    Pi::Matrix{Float64}
    Va::Matrix{Float64}
    a_policy::Matrix{Float64}
    c_policy::Matrix{Float64}
    D::Matrix{Float64}
end

"""
    rouwenhorst(n, rho, sigma)

Purpose: Approximate a persistent log income process with an `n`-state Rouwenhorst chain.
Inputs: number of states `n`, persistence `rho`, and innovation standard deviation `sigma`.
Output: raw income grid and transition matrix `Pi`.
"""
function rouwenhorst(n, rho, sigma)
    p = (1 + rho) / 2
    q = p
    Pi = [1.0]
    for m in 2:n
        old = Pi
        Pi = zeros(m, m)
        Pi[1:m-1, 1:m-1] .+= p .* old
        Pi[1:m-1, 2:m] .+= (1 - p) .* old
        Pi[2:m, 1:m-1] .+= (1 - q) .* old
        Pi[2:m, 2:m] .+= q .* old
        if m > 2
            Pi[2:m-1, :] ./= 2
        end
    end

    psi = sigma / sqrt(1 - rho^2) * sqrt(n - 1)
    log_grid = range(-psi, psi, length=n)
    return exp.(collect(log_grid)), Pi
end

"""
    stationary_markov(Pi; tol=1e-14, max_iter=10_000)

Purpose: Compute the invariant distribution of a finite Markov chain.
Inputs: transition matrix `Pi`, tolerance, and iteration limit.
Output: probability vector that sums to one.
"""
function stationary_markov(Pi; tol=1e-14, max_iter=10_000)
    n = size(Pi, 1)
    dist = fill(1 / n, n)
    for _ in 1:max_iter
        new_dist = Pi' * dist
        if maximum(abs.(new_dist .- dist)) < tol
            return new_dist ./ sum(new_dist)
        end
        dist = new_dist
    end
    return dist ./ sum(dist)
end

"""
    make_ks_grid(params)

Purpose: Build idiosyncratic income states and the asset grid.
Inputs: `KSParams`.
Output: asset grid, income grid with stationary mean `params.L`, and income transition matrix.
"""
function make_ks_grid(params::KSParams)
    e_grid_raw, Pi = rouwenhorst(params.nE, params.rho_e, params.sd_e)
    e_dist = stationary_markov(Pi)
    e_grid = params.L .* e_grid_raw ./ dot(e_dist, e_grid_raw)
    x = range(0.0, 1.0, length=params.nA)
    a_grid = params.amin .+ (params.amax - params.amin) .* (collect(x) .^ 2)
    return a_grid, e_grid, Pi
end

"""
    interp_linear(xgrid, ygrid, x)

Purpose: Evaluate a one-dimensional policy function by linear interpolation.
Inputs: grid `xgrid`, values `ygrid`, and query point `x`.
Output: interpolated scalar, clamped at the grid endpoints.
"""
function interp_linear(xgrid, ygrid, x)
    if x <= xgrid[1]
        return ygrid[1]
    elseif x >= xgrid[end]
        return ygrid[end]
    end
    j = searchsortedlast(xgrid, x)
    weight = (x - xgrid[j]) / (xgrid[j+1] - xgrid[j])
    return (1 - weight) * ygrid[j] + weight * ygrid[j+1]
end

"""
    egm_step(Va_next, a_grid, e_grid, Pi, r, w, beta, eis)

Purpose: Perform one endogenous-grid-method backward step for the household problem.
Inputs: next-period marginal value `Va_next`, grids, income transition matrix, prices, discount factor, and EIS.
Output: current marginal value, asset policy, and consumption policy.
"""
function egm_step(Va_next, a_grid, e_grid, Pi, r, w, beta, eis)
    nE, nA = length(e_grid), length(a_grid)
    EVa = Pi * Va_next
    a_policy = similar(Va_next)
    c_policy = similar(Va_next)
    Va = similar(Va_next)

    for ie in 1:nE
        c_nextgrid = (beta .* EVa[ie, :]) .^ (-eis)
        resources_endog = c_nextgrid .+ a_grid
        for ia in 1:nA
            coh = (1 + r) * a_grid[ia] + w * e_grid[ie]
            a = interp_linear(resources_endog, a_grid, coh)
            a = max(a, a_grid[1])
            c = coh - a
            a_policy[ie, ia] = a
            c_policy[ie, ia] = c
            Va[ie, ia] = (1 + r) * c^(-1 / eis)
        end
    end
    return Va, a_policy, c_policy
end

"""
    initial_Va(a_grid, e_grid, r, w, eis)

Purpose: Create a stable initial guess for household marginal value iteration.
Inputs: grids, prices, and EIS.
Output: initial marginal value array indexed by income and assets.
"""
function initial_Va(a_grid, e_grid, r, w, eis)
    Va = zeros(length(e_grid), length(a_grid))
    for ie in eachindex(e_grid), ia in eachindex(a_grid)
        coh = (1 + r) * a_grid[ia] + w * e_grid[ie]
        Va[ie, ia] = (1 + r) * (0.1 * coh)^(-1 / eis)
    end
    return Va
end

"""
    solve_policy_ss(a_grid, e_grid, Pi, r, w, beta, eis; tol=1e-10, max_iter=10_000)

Purpose: Solve the stationary household policy functions by iterating EGM steps.
Inputs: grids, income transition matrix, prices, discount factor, EIS, tolerance, and iteration limit.
Output: steady-state marginal value, asset policy, and consumption policy.
"""
function solve_policy_ss(a_grid, e_grid, Pi, r, w, beta, eis; tol=1e-10, max_iter=10_000)
    Va = initial_Va(a_grid, e_grid, r, w, eis)
    a_policy = similar(Va)
    c_policy = similar(Va)
    for _ in 1:max_iter
        Va_new, a_policy, c_policy = egm_step(Va, a_grid, e_grid, Pi, r, w, beta, eis)
        err = maximum(abs.(log.(Va_new) .- log.(Va)))
        Va = Va_new
        if err < tol
            return Va, a_policy, c_policy
        end
    end
    return Va, a_policy, c_policy
end

"""
    forward_distribution(D, a_policy, a_grid, Pi)

Purpose: Push the household distribution forward one period using the asset policy and income transitions.
Inputs: current distribution `D`, asset policy, asset grid, and transition matrix `Pi`.
Output: next-period distribution over income and assets.
"""
function forward_distribution(D, a_policy, a_grid, Pi)
    nE, nA = size(D)
    Dnext = zeros(nE, nA)
    for ie in 1:nE, ia in 1:nA
        mass = D[ie, ia]
        if mass == 0
            continue
        end
        ap = clamp(a_policy[ie, ia], a_grid[1], a_grid[end])
        j = searchsortedlast(a_grid, ap)
        if j >= nA
            j = nA - 1
            weight = 1.0
        else
            weight = (ap - a_grid[j]) / (a_grid[j+1] - a_grid[j])
        end
        for iep in 1:nE
            prob = Pi[ie, iep]
            Dnext[iep, j] += mass * prob * (1 - weight)
            Dnext[iep, j+1] += mass * prob * weight
        end
    end
    return Dnext ./ sum(Dnext)
end

"""
    stationary_distribution(a_policy, a_grid, Pi; tol=1e-12, max_iter=50_000)

Purpose: Find the invariant distribution implied by a stationary asset policy.
Inputs: asset policy, asset grid, income transition matrix, tolerance, and iteration limit.
Output: stationary joint distribution over income and assets.
"""
function stationary_distribution(a_policy, a_grid, Pi; tol=1e-12, max_iter=50_000)
    e_dist = stationary_markov(Pi)
    D = repeat(e_dist, 1, length(a_grid)) ./ length(a_grid)
    for _ in 1:max_iter
        Dnext = forward_distribution(D, a_policy, a_grid, Pi)
        if maximum(abs.(Dnext .- D)) < tol
            return Dnext ./ sum(Dnext)
        end
        D = Dnext
    end
    return D ./ sum(D)
end

"""
    aggregate_assets(D, a_policy)

Purpose: Aggregate household asset choices under distribution `D`.
Inputs: distribution `D` and asset policy.
Output: aggregate assets.
"""
function aggregate_assets(D, a_policy)
    return sum(D .* a_policy)
end

"""
    aggregate_consumption(D, c_policy)

Purpose: Aggregate household consumption under distribution `D`.
Inputs: distribution `D` and consumption policy.
Output: aggregate consumption.
"""
function aggregate_consumption(D, c_policy)
    return sum(D .* c_policy)
end

"""
    backward_policy_path(r_path, w_path, ss, params)

Purpose: Solve household policies backward along a price path.
Inputs: price paths, steady-state terminal object, and KS parameters.
Output: vectors of asset and consumption policy matrices.
"""
function backward_policy_path(r_path, w_path, ss, params)
    T = length(r_path)
    Va_next = ss.Va
    a_policies = Vector{Matrix{Float64}}(undef, T)
    c_policies = Vector{Matrix{Float64}}(undef, T)

    for t in T:-1:1
        Va, a_policy, c_policy = egm_step(
            Va_next, ss.a_grid, ss.e_grid, ss.Pi,
            r_path[t], w_path[t], ss.beta, params.eis
        )
        a_policies[t] = a_policy
        c_policies[t] = c_policy
        Va_next = Va
    end

    return (; a_policies, c_policies)
end

"""
    expectation_step(E, a_policy, a_grid, Pi)

Purpose: Apply the transpose of the distribution transition operator to a household object.
Inputs: household object `E`, asset policy, asset grid, and income transition matrix.
Output: expected next-period object conditional on today's state.
"""
function expectation_step(E, a_policy, a_grid, Pi)
    nE, nA = size(E)
    Enext = zeros(nE, nA)

    for ie in 1:nE, ia in 1:nA
        ap = clamp(a_policy[ie, ia], a_grid[1], a_grid[end])
        j = searchsortedlast(a_grid, ap)
        if j >= nA
            j = nA - 1
            weight = 1.0
        else
            weight = (ap - a_grid[j]) / (a_grid[j+1] - a_grid[j])
        end

        val = 0.0
        for iep in 1:nE
            val += Pi[ie, iep] * ((1 - weight) * E[iep, j] + weight * E[iep, j+1])
        end
        Enext[ie, ia] = val
    end

    return Enext
end

"""
    recover_fake_news_jacobian(first_row, first_col, F)

Purpose: Recover a full Jacobian from the first row, first column, and fake-news matrix.
Inputs: first row, first column, and shifted-index fake-news matrix.
Output: full sequence-space Jacobian.
"""
function recover_fake_news_jacobian(first_row, first_col, F)
    T = length(first_row)
    J = zeros(T, T)
    J[1, :] .= first_row
    J[:, 1] .= first_col

    for t in 2:T, s in 2:T
        J[t, s] = J[t-1, s-1] + F[t-1, s-1]
    end

    return J
end

"""
    household_steady_state(beta, r, w, params, a_grid, e_grid, Pi)

Purpose: Solve the stationary household block at fixed prices.
Inputs: discount factor, prices, parameters, grids, and income transition matrix.
Output: aggregate assets/consumption plus value, policy, and distribution objects.
"""
function household_steady_state(beta, r, w, params::KSParams, a_grid, e_grid, Pi)
    Va, a_policy, c_policy = solve_policy_ss(a_grid, e_grid, Pi, r, w, beta, params.eis)
    D = stationary_distribution(a_policy, a_grid, Pi)
    A = aggregate_assets(D, a_policy)
    C = aggregate_consumption(D, c_policy)
    return (; A, C, Va, a_policy, c_policy, D)
end

"""
    brent_bisect(f, lo, hi; tol=1e-10, max_iter=200)

Purpose: Find a scalar root in a bracket by bisection.
Inputs: scalar function `f`, lower/upper bracket, tolerance, and iteration limit.
Output: approximate root.
"""
function brent_bisect(f, lo, hi; tol=1e-10, max_iter=200)
    flo, fhi = f(lo), f(hi)
    if sign(flo) == sign(fhi)
        error("Root is not bracketed: f(lo)=$flo, f(hi)=$fhi")
    end
    for _ in 1:max_iter
        mid = (lo + hi) / 2
        fmid = f(mid)
        if abs(fmid) < tol || (hi - lo) / 2 < tol
            return mid
        elseif sign(fmid) == sign(flo)
            lo, flo = mid, fmid
        else
            hi, fhi = mid, fmid
        end
    end
    return (lo + hi) / 2
end

"""
    compute_ks_steady_state(params)

Purpose: Compute the KS steady state from structural parameters.
Inputs: `KSParams`.
Output: `KSSteadyState` containing prices, aggregates, grids, policies, and distribution.
"""
function compute_ks_steady_state(params::KSParams)
    a_grid, e_grid, Pi = make_ks_grid(params)
    firm_prices(K) = begin
        r = params.alpha * params.Z * (K / params.L)^(params.alpha - 1) - params.delta
        w = (1 - params.alpha) * params.Z * (K / params.L)^params.alpha
        Y = params.Z * K^params.alpha * params.L^(1 - params.alpha)
        return r, w, Y
    end

    excess_assets(K) = begin
        r, w, _ = firm_prices(K)
        household_steady_state(params.beta, r, w, params, a_grid, e_grid, Pi).A - K
    end

    lo = max(1e-4, params.amin + 1e-4)
    hi = min(params.amax * 0.95, 100.0)
    flo, fhi = excess_assets(lo), excess_assets(hi)
    while sign(flo) == sign(fhi) && hi < params.amax * 0.99
        hi = min(params.amax * 0.99, 1.5 * hi)
        fhi = excess_assets(hi)
    end
    if sign(flo) == sign(fhi)
        error("Could not bracket steady-state capital: excess_assets($lo)=$flo, excess_assets($hi)=$fhi")
    end

    K = brent_bisect(excess_assets, lo, hi)
    r, w, Y = firm_prices(K)
    hh = household_steady_state(params.beta, r, w, params, a_grid, e_grid, Pi)
    goods_mkt = Y - hh.C - params.delta * K

    return KSSteadyState(params.beta, params.Z, K, params.L, r, w, Y, hh.A, hh.C, hh.A - K, goods_mkt,
                         a_grid, e_grid, Pi, hh.Va, hh.a_policy, hh.c_policy, hh.D)
end
