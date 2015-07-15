# micro-uq
Lightweight c++11 header-only library with commonly used UQ tools

### Dependencies

* eigen3 (header-only library included as git submodule in external/eigen)

### Overview

Lightweight (less than 500 lines of performant C++ code) implementations of PCE and MCMC codes that are useful for small projects. 

### Examples
```c++
#include "PCExpansion.h"

// construct PCE using c++11 lambda, std::function, or function pointer
auto f = [](const Eigen::VectorXd& x) { return pow(x(0), 2) + x(0) * x(1) + pow(x(1), 3); };
auto vars = {PCExpansion::GermType::Normal, PCExpansion::GermType::Normal};
auto pce = std::make_shared<PCExpansion>(f, vars, 5); // max total order = 5

// evaluate
std::cout << "f(0, 0) = " << pce->Evaluate(Eigen::VectorXd::Zero(2)) << std::endl;

// compute moments
auto moments = pce->GetMoments();
std::cout << "Mean: " << moments.first << " Variance: " << moments.second << std::endl;

// compute cross-covariance with another PCE
auto f2 = [](const Eigen::VectorXd& xi) { return xi(0) + xi(0) * pow(xi(1), 3); };
auto pce2 = std::make_shared<PCExpansion>(f2, vars, 5);
std::cout << "cross-covariance: " << pce->GetCrossCovariance(pce2) << std::endl;
```
