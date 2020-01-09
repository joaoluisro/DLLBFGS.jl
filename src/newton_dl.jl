export newton_dl
"""
  newton_dl(nlp)

  A Newton-DogLeg implementation.
"""
function dogleg(nlp :: AbstractNLPModel,
                g :: AbstractVector,
                B :: AbstractMatrix,
                Δ :: Real;
                Δₛ :: Real = 1e-5)

  # If the radius is too small, return the linear model minimizer
  pᵘ = -g * (dot(g,g)/dot(g,B*g))
  uᵀu = dot(pᵘ, pᵘ)
  (Δ ≤ Δₛ || uᵀu ≥ Δ*Δ) && return -Δ*(g/norm(g))
  # Newton's direction is inside the region
  pᵇ = - B\g
  dot(pᵇ, pᵇ) ≤ Δ*Δ && return pᵇ
  # DogLeg's method
  pᵘ = -g * (dot(g,g)/dot(g,B*g))
  pᵘ == pᵇ && return pᵇ
  vᵀv = dot(pᵇ - pᵘ, pᵇ - pᵘ)
  uᵀv = dot(pᵇ - pᵘ, pᵘ)

  a = vᵀv
  b = 2*(uᵀv - vᵀv)
  c = uᵀu - 2*uᵀv + vᵀv - Δ^2
  Γ = sqrt(b^2 - 4*a*c)

  δₖ = (Γ - b)/(2*a)
  p = pᵘ + (δₖ - 1)*(pᵇ - pᵘ)
  return p
end

function newton_dl(nlp :: AbstractNLPModel;
                   x :: AbstractVector = copy(nlp.meta.x0),
                   max_time :: Float64 = 30.0,
                   max_eval :: Int = - 1,
                   max_tol :: Real = √eps(eltype(x)))

  B = Symmetric(hess(nlp, x), :L)
  g = grad(nlp, x)
  fₓ = obj(nlp, x)
  nrmgrad = norm(g)
  T = eltype(x)
  tr = TrustRegion(min(max(nrmgrad/ 10, one(T)), T(100)))

  k = 0
  start_time = time()
  el_time = 0.0
  tired = el_time > max_time || neval_obj(nlp) ≥ max_eval ≥ 0
  optimal = nrmgrad < max_tol

  @info log_header([:iter, :f, :nrm, :radius, :rho], [Int, T, T, T], hdr_override=Dict(:f => "f(x)",
  :nrm => "‖∇f(x)‖", :radius => "Δₖ", :rho => "p"))

  while !(optimal || tired)

    @info log_row(Any[k, fₓ, nrmgrad, tr.radius, tr.ratio])

    p = dogleg(nlp, g, B, tr.radius)
    slope = dot(g, p)
    mₚ = fₓ + dot(g,p) + 0.5*dot(p, B*p)
    ared, pred = aredpred(nlp, obj(nlp, x + p), fₓ, (fₓ - mₚ), x + p, p, slope)
    tr.ratio = ared/pred

    if acceptable(tr)
      x += p
    end
    update!(tr, norm(p))

    B = Symmetric(hess(nlp, x), :L)
    g = grad(nlp, x)
    nrmgrad = norm(g)
    fₓ = obj(nlp, x)

    k += 1
    tired = el_time > max_time || neval_obj(nlp) ≥ max_eval ≥ 0
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