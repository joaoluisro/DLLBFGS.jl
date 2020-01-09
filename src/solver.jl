export dllbfgs
"""
  dllbfgs(nlp)

  nlp is a ADNLPModel
  this implementation follows
"""
function dllbfgs(nlp :: ADNLPModel;
                 x0 :: AbstractVector = copy(nlp.meta.x0),
                 max_tol :: Real = √eps(eltype(x)),
                 max_eval :: Int = 1e+4,
                 max_time :: Float64 = 30.0,
                 max_iter :: Int = 20
                 )

  k = 0
  start_time = time()
  el_time = 0.0
  tired = k > max_iter || el_time ≥ max_time || neval_obj(nlp) ≥ max_eval
  optimal = ∇fnorm ≤ max_tol

  while !(optimal || tired)

    k += 1

    optimal = ∇fnorm ≤ max_tol
    el_time = time() - start_time
    tired = k > max_iter || el_time ≥ max_time || neval_obj(nlp) ≥ max_eval
  end

  if optimal
    status = :first_order
  elseif tired
    if el_time >= max_time
      status = :max_time
    else
      status = :max_iter
    end
  end

  return GenericExecutionStats(status, nlp, solution = x, objective = fₓ,
                               iter = k, elapsed_time = el_time)
end
