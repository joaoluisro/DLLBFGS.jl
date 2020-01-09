export trust_region

function cauchy_point(g :: AbstractVector,
                      B :: AbstractMatrix,
                      Δ :: Real)
  nrmg = norm(g)
  gᵀBg = dot(g,B*g)
  p = Δ*(-g/nrmg)

  τ = gᵀBg > 0 ? min(1.0, (nrmg^3)/(Δ*gᵀBg)) : 1.0
  return τ*p
end

function dogleg(nlp :: AbstractNLPModel,
                g :: AbstractVector,
                B :: AbstractMatrix,
                pᵇ :: AbstractVector,
                Δ :: Real;
                Δₛ :: Real = 1e-5)

  # If the radius is too small, return the linear model minimizer
  Δ ≤ Δₛ && return -Δ*(g/norm(g))
  # Newton's direction is inside the region
  dot(pᵇ, pᵇ) ≤ Δ*Δ && return pᵇ
  # DogLeg's method
  pᵘ = -g * (dot(g,g)/dot(g,B*g))
  uᵀu = dot(pᵘ, pᵘ)
  vᵀv = dot(pᵇ - pᵘ, pᵇ - pᵘ)
  uᵀv = dot(pᵇ - pᵘ, pᵘ)

  a = vᵀv
  b = 2*(uᵀv - vᵀv)
  c = uᵀu - 2*uᵀv + vᵀv - Δ^2

  Γ = try
    sqrt(b^2 - 4*a*c)
  catch e
    0.0
  end
  δₖ = (Γ - b)/(2*a)
  p = pᵘ + (δₖ - 1)*(pᵇ - pᵘ)
  return p
end

function trust_region(nlp :: AbstractNLPModel;
                      x :: AbstractVector = copy(nlp.meta.x0),
                      max_iter :: Int = 100,
                      max_time :: Float64 = 30.0,
                      max_tol :: Real = √eps(eltype(x)),
                      radius_bound :: Real = 10.0,
                      sufficient_decrease :: Real = 0.0,
                      Δₖ :: Real = 1.0)

  B = Symmetric(hess(nlp, x), :L)
  g = grad(nlp, x)
  fₓ = obj(nlp, x)
  nrmgrad = norm(g)

  k = 0
  start_time = time()
  el_time = 0.0
  tired = k > max_iter || el_time > max_time
  optimal = nrmgrad < max_tol
  ρₖ = 1.0
  T = eltype(x)
  @info log_header([:iter, :f, :nrm, :radius, :rho], [Int, T, T, T], hdr_override=Dict(:f => "f(x)",
  :nrm => "‖∇f(x)‖", :radius => "Δₖ", :rho => "p"))

  while !(optimal || tired)

    @info log_row(Any[k, fₓ, nrmgrad, Δₖ, ρₖ])

    pᵇ = - B\g
    p = dot(pᵇ, g) < 0 ? dogleg(nlp, g, B, pᵇ,Δₖ) : cauchy_point(g, B, Δₖ)

    mₚ = fₓ + dot(g,p) + 0.5*dot(p, B*p)
    ρₖ = (fₓ - obj(nlp, x + p))/(fₓ - mₚ)
    if ρₖ < 0.25
      Δₖ *= 0.25
    elseif ρₖ > 0.75 && dot(p, p) == Δₖ*Δₖ
      Δₖ = min(2Δₖ, radius_bound)
    end
    if ρₖ ≥ sufficient_decrease
      x += p
    end

    B = Symmetric(hess(nlp, x), :L)
    g = grad(nlp, x)
    nrmgrad = norm(g)
    fₓ = obj(nlp, x)

    k += 1
    tired = k > max_iter || el_time > max_time
    optimal = nrmgrad < max_tol
    el_time = time() - start_time
  end

  if optimal
    status = :first_order
  elseif tired
    if el_time ≥ max_time
      status = :max_time
    else
      status = :max_iter
    end
  end

  return GenericExecutionStats(status, nlp, solution = x, objective = fₓ,
                               iter = k, elapsed_time = el_time)
end
