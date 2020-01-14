# DLLBFGS (DogLeg Limited Memory BFGS)
A Julia implementation of a trust region solver for non-linear unconstrained optimization problems, that uses limited memory BFGS updates through the DogLeg method. The solver also makes use of the [JuliaSmoothOptimizers](https://github.com/JuliaSmoothOptimizers) packages.

## TODO

* Add multiple precision support
* Benchmark against `lbfgs` with CUTEst
* Code optimization

## References

* Numerical Optimization (Jorge Nocedal and Stephen J. Wright), Springer, 2006
