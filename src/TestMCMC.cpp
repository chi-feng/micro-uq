#include <cmath>
#include <memory>
#include <iostream>
#include <random>

#include <Eigen/Core>

#include "RandomGenerator.h"
#include "mcmc/MCMC.h"

int main()
{
  // function handle, can be C++11 lambda, regular function pointer, or std::function
  auto logdens = [](const Eigen::VectorXd& x) {
    return -(x(0) * x(0)) - (x(1) * x(1));
  };

  auto rng = std::make_shared<RandomGenerator>();

  auto mcmc = std::make_shared<MCMC>(logdens, 2, rng);

  mcmc->Run(10000);

  auto chain = mcmc->GetChain();

  std::cout << "Mean: " << chain.colwise().mean() << std::endl;

  std::cout << "Autocorrelation: " << mcmc->GetAutocorrelation(10).transpose() << std::endl;

  return 0;
}