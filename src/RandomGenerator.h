#include <random>
#include <Eigen/Core>

class RandomGenerator
{
private:

  std::mt19937 engine;

public:

  RandomGenerator(size_t seed = std::mt19937::default_seed)
  {
    engine.seed(seed);
  }

  inline double GetNormal()
  {
    std::normal_distribution<double> dist(0, 1);
    return dist(engine);
  }

  inline double GetUniform()
  {
    std::uniform_real_distribution<double> dist(0, 1);
    return dist(engine);
  }

  inline Eigen::VectorXd GetNormalRandomVector(int const dim)
  {
    Eigen::VectorXd result(dim);
    for (int i = 0; i < dim; ++i)
      result(i) = GetNormal();
    return result;
  }
}