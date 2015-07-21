# micro-uq
Lightweight c++11 header-only library with commonly used UQ tools

### Dependencies

* Eigen3, a header-only linear-algebra library (included as git submodule)
  * After you clone the repository, run `git submodule init` and `git submodule update` to automatically grab the library.

### Overview

Lightweight (less than 500 lines of performant C++ code) implementations of PCE and MCMC codes that are useful for small projects. 

### Examples
```c++
#include "PCExpansion.h"

using namespace std;
using namespace Eigen;

// construct PCE using c++11 lambdas, std::function, or function pointer
auto f = [](const VectorXd& x) { return pow(x(0), 2) + x(0) * x(1) + pow(x(1), 3); };
auto vars = {PCExpansion::GermType::Normal, PCExpansion::GermType::Normal};
auto pce = make_shared<PCExpansion>(f, vars, 5); // max total order = 5

// evaluate
cout << "f(0, 0) = " << pce->Evaluate(VectorXd::Zero(2)) << endl;

// compute moments
auto moments = pce->GetMoments();
cout << "Mean: " << moments.first << " Variance: " << moments.second << endl;

// compute cross-covariance with another PCE
auto f2 = [](const VectorXd& xi) { return xi(0) + xi(0) * pow(xi(1), 3); };
auto pce2 = make_shared<PCExpansion>(f2, vars, 5);
cout << "cross-covariance: " << pce->GetCrossCovariance(pce2) << endl;
```
