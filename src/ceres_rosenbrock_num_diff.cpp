#include "ceres/ceres.h"
#include "glog/logging.h"

// f(x,y) = (1-x)^2 + 100(y - x^2)^2;
// f(x1,x2,x3) = (100(x1^2 - x2)^2 + (x1 - 1)^2) + (100(x2^2 - x3)^2 + (x2 - 1)^2)
struct Rosenbrock
{
  bool operator()(const double *x, double *cost) const
  {
    constexpr int kNumParameters = 3;

    cost[0] = 0;
    for (int i = 1; i < kNumParameters; ++i)
    {
      cost[0] += (1.0 - x[i - 1]) * (1.0 - x[i - 1]) + 100.0 * (x[i] - x[i - 1] * x[i - 1]) * (x[i] - x[i - 1] * x[i - 1]);
    }
    return true;
  }

  static ceres::FirstOrderFunction *create()
  {
    constexpr int kNumParameters = 3;
    return new ceres::NumericDiffFirstOrderFunction<Rosenbrock,
                                                    ceres::CENTRAL,
                                                    kNumParameters>(
        new Rosenbrock);
  }
};

int main(int argc, char **argv)
{
  google::InitGoogleLogging(argv[0]);
  const double parameters[3] = {-1.0, -1.0, 1.0};
  const double *param = parameters;
  double cost = 0;
  double gradient[3];

  // ceres::GradientProblemSolver::Options options;
  // options.minimizer_progress_to_stdout = true;
  // ceres::GradientProblemSolver::Summary summary;
  // ceres::GradientProblem problem(Rosenbrock::create());
  // ceres::Solve(options, problem, parameters, &summary);
  // std::cout << summary.FullReport() << "\n";
  // std::cout << "Initial x: " << -1.2 << " y: " << 1.0 << "\n";
  // std::cout << "Final   x: " << parameters[0] << " y: " << parameters[1]
  //           << "\n";

  std::cout << Rosenbrock::create()->Evaluate(param, &cost, gradient) << std::endl;
  std::cout << "cost : " << cost << " gradient: " << gradient[0] << "," << gradient[1] << "," << gradient[2] << std::endl;

  return 0;
}
