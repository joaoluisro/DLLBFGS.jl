using DLLBFGS
# regular pkgs
using LinearAlgebra
# JSO
using JSOSolvers, SolverTools, Logging, NLPModels, Test, LinearOperators

function tests()
    problems = [(x->2*x[1]^2 - 1.05*x[1]^4 + (x[1]^6)/6 + x[1]*x[2] + x[2]^2),
                (x->(1 - x[1])^2 + 100(x[2] - x[1]^2)^2),
                (x->(x[1]^2 + x[2] - 11)^2 + (x[1] + x[2]^2 - 7)^2),
                (x->0.26(x[1]^2 + x[2]^2) - 0.48*x[1]*x[2])]

    nlps = [ADNLPModel(i, [2.0, 3.0]) for i in problems]
    solvers = [ dllbfgs]
    @testset "DL-LBFGS" begin
      for nlp in nlps
        with_logger(NullLogger()) do
          for s in solvers
            out = s(nlp)
            @test -1e-4 ≤ out.objective ≤ 1e-4
          end
        end
      end
    end
end

tests()
