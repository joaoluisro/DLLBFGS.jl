using DLLBFGS
# regular pkgs
using LinearAlgebra
# JSO
using JSOSolvers, SolverTools, Logging, NLPModels, Test, LinearOperators

function tests()
    problems = [(x->(x[1]^2 + x[2]^2)^2 + x[1]*x[2], [2.0,3.0]),
                (x->(1 - x[1])^2 + 100(x[2] - x[1]^2)^2, [2.0, 3.0]),
                (x->(x[1]^2 + x[2] - 11) + (x[1] + x[2]^2 - 7)^2, [2.0,3.0]),
                (x->0.26(x[1]^2 + x[2]^2) - 0.48*x[1]*x[2], [2.0,3.0])]

    nlps = [ADNLPModel(i[1],i[2]) for i in problems]
    solvers = [dllbfgs, newton_dl, lbfgs]
    res = []
    @testset "DL-LBFGS" begin
      for nlp in nlps
        with_logger(NullLogger()) do
          for s in solvers
            out = s(nlp)
            push!(res, out)
          end
          ans = minimum([i.objective for i in res])
          for r in res
            @test r.objective ≤ ans + abs(ans)*1e-1 + 1e-4
          end
          empty!(res)
        end
      end
    end
end

tests()
