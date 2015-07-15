#include <cmath>
#include <memory>
#include <iostream>
#include <random>

#include <Eigen/Core>

#include "pce/PCExpansion.h"

int main()
{
  // function handle, can be C++11 lambda, regular function pointer, or std::function
  auto f = [](const Eigen::VectorXd& x) { return pow(x(0), 2) + x(0) * x(1) + pow(x(1), 3); };

  // two iid normal distributed inputs
  auto vars = {PCExpansion::GermType::Normal, PCExpansion::GermType::Normal};

  // make shared pointer to PCExpansion
  auto pce = std::make_shared<PCExpansion>(f, vars, 5); // maxOrder = 5

  //
  // Below are example uses of a PCE
  //

  // get analytic moments from the PCE
  auto moments = pce->GetMoments();
  std::cout << "Mean: " << moments.first << " Variance: " << moments.second << std::endl;

  // Estimate MSE of PCE expansion using samples
  double mse = 0;
  int nsamps = 1000;
  auto rng = std::mt19937();
  std::normal_distribution<double> normal(0, 1);
  Eigen::VectorXd sample(2);
  for (int i = 0; i < nsamps; ++i) {
    sample << normal(rng), normal(rng);
    mse += pow(f(sample) - pce->Evaluate(sample), 2);
  }
  std::cout << "MSE: " << mse / nsamps << std::endl;

  // make a another PCE to get cross-covariance
  auto f_other = [](const Eigen::VectorXd& xi) { return xi(0) + xi(0) * pow(xi(1), 3); };
  auto pce_other = std::make_shared<PCExpansion>(f_other, vars, 5);
  std::cout << "cross-covariance: " << pce->GetCrossCovariance(pce_other) << std::endl;

  return 0;
}