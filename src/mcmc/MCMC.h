#ifndef MCMC_h
#define MCMC_h

#include <memory>
#include <functional>
#include <Eigen/Core>
#include <Eigen/Cholesky>

#include "RandomGenerator.h"

class GaussianProposal {

private:

  int dim;
  Eigen::MatrixXd cov;
  Eigen::LLT<Eigen::MatrixXd> llt;
  double constant, logdet;

public:

  inline void SetCovariance(const Eigen::MatrixXd& cov) {
    this->cov = cov;
    llt = cov.llt();
    logdet = log(llt.matrixL().diagonal().array()).sum();
  }

  GaussianProposal(const Eigen::MatrixXd& cov)
  {
    dim = cov.rows();
    SetCovariance(cov);
    constant = -0.5 * log(2.0 * M_PI) * dim;
  }

  inline Eigen::VectorXd& GetProposal(std::shared_ptr<RandomGenerator> rng) {
    return llt.matrixL() * rng->GetNormalRandomVector(dim);
  }

  inline double LogDensity(const Eigen::VectorXd& x) {
    return constant - logdet - 0.5 * (llt.matrixL().solve(x)).squaredNorm();
  }

};

class MCMC {

  protected:

    std::function<double(const Eigen::VectorXd&)> LogDensityFuncHandle;
    int dim;
    std::shared_ptr<RandomGenerator> rng;

    Eigen::MatrixXd chain;        // column-wise
    Eigen::VectorXd logDensities;

    int accepted;

    std::unique_ptr<GaussianProposal> mhProposalDist;
    std::unique_ptr<GaussianProposal> amProposalDist;

    int adaptStride;
    double adaptProb;

    Eigen::VectorXd chainSum;     /// running sum for rank-1 covariance update
    Eigen::MatrixXd chainScatter; /// scatter matrix for rank-1 covariance

    int existing;

  public:

    inline void Reset()
    {
      chainScatter = Eigen::MatrixXd::Zero(dim, dim);
      chainSum   = Eigen::VectorXd::Zero(dim);
      accepted   = 0;
      existing   = 0;
    }

    MCMC(const std::function<double(const Eigen::VectorXd&)>& f, const int dim, std::shared_ptr<RandomGenerator> rng) : LogDensityFuncHandle(f), dim(dim), rng(rng)
    {
      adaptStride    = 10 * dim;
      adaptProb      = 0.9;
      mhProposalDist = std::make_unique<GaussianProposal>(Eigen::MatrixXd::Identity(dim, dim));
      amProposalDist = std::make_unique<GaussianProposal>(Eigen::MatrixXd::Identity(dim, dim));
      Reset();
    }

    inline void Run(const int N)
    {
      chain           = Eigen::MatrixXd::Zero(dim, N);
      logDensities    = Eigen::VectorXd::Zero(N);
      logDensities(0) = LogDensityFuncHandle(chain.col(0));
      for (int i = 0; i < N; ++i) {
        if (i > adaptStride && i % adaptStride == 0) {
          auto newSamps = chain.block(0, i - adaptStride, dim, adaptStride);
          chainScatter += newSamps * newSamps.transpose();
          chainSum   += newSamps.rowwise().sum();
          amProposalDist->SetCovariance(2.4 / dim * (chainScatter - chainSum * chainSum.transpose() / i) / i);
        }
        chain.col(i) = (RandomGenerator::GetUniform() < adaptProb)
                       ? chain.col(i - 1) + amProposalDist->GetProposal(rng)
                       : chain.col(i - 1) + mhProposalDist->GetProposal(rng);
        logDensities(i) = LogDensityFuncHandle(chain.col(i));
        if (log(RandomGenerator::GetUniform()) < logDensities(i) - logDensities(i - 1)) {
          accepted++;
        } else {
          chain.col(i)    = chain.col(i - 1);
          logDensities(i) = logDensities(i - 1);
        }
      }
    }

    inline Eigen::VectorXd GetAutocorrelation(const int lag)
    {
      // TODO: use FFT for fast calculation
      Eigen::VectorXd mean           = chain.rowwise().mean();
      Eigen::VectorXd autocovariance = Eigen::VectorXd::Zero(lag + 1);
      for (int k = 0; k < lag + 1; ++k)
        for (int i = k; i < chain.cols(); ++i)
          autocovariance(k) += (chain.col(i) - mean).dot(chain.col(i - k) - mean);
      return autocovariance / autocovariance(0);
    }

    inline Eigen::MatrixXd GetChain()           { return chain.transpose(); }
    inline Eigen::VectorXd GetLogDensities()    { return logDensities; }
    inline double          GetAcceptanceRatio() { return static_cast<double>(accepted) / chain.cols(); }
    inline Eigen::MatrixXd GetAdaptiveCov()     { return amProposalDist->GetCovariance(); }
};

#endif // ifndef MCMC_h