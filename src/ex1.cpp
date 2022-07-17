#include <iostream>
#include <vector>
#include <string>

#include <Eigen/Core>

double RosenbrockEvaluate(const Eigen::VectorXd &x)
{
    const int order = x.rows();
    double sum = 0;
    for (int i = 1; i < order; ++i)
    {
        sum += (1.0 - x[i - 1]) * (1.0 - x[i - 1]) +
               100.0 * (x[i] - x[i - 1] * x[i - 1]) * (x[i] - x[i - 1] * x[i - 1]);
    }
    return sum;
}

Eigen::VectorXd RosenbrockFirstOrderDeriv(const Eigen::VectorXd &x)
{
    int order = x.rows();
    Eigen::VectorXd deriv(order);
    for (int i = 0; i < order; ++i)
    {
        if (i != 0 && i != order - 1)
        {
            deriv(i) = 100 * 2 * (x(i) * x(i) - x(i + 1)) * x(i) +
                       2 * (x(i) - 1) -
                       100 * 2 * (x(i - 1) * x(i - 1) - x(i));
        }
        else if (i == 0)
        {
            deriv(i) = 100 * 2 * (x(i) * x(i) - x(i + 1)) * x(i) +
                       2 * (x(i) - 1);
        }
        else
        {
            deriv(i) = -100 * 2 * (x(i - 1) * x(i - 1) - x(i));
        }
    }
    return deriv;
}

struct OptimizerConfig
{
    OptimizerConfig(double lr, double stop_delta, std::string method, double k, double c)
        : lr_(lr), stop_delta_(stop_delta), method_(method), k_(k), c_(c)
    {
    }
    double lr_ = 0.1;
    double stop_delta_ = 0.00001;
    std::string method_ = "inexact_line_search";
    double k_ = 2;
    double c_ = 0.5;
};

bool inexactLineSearchStep(Eigen::VectorXd &x,
                           Eigen::VectorXd &gradient,
                           double &cost_old,
                           double &lr,
                           const double &c,
                           std::vector<Eigen::VectorXd> &x_list)
{
    gradient = RosenbrockFirstOrderDeriv(x);
    // condition for armijo line search
    double armigo_cond = - c * gradient.squaredNorm();
    // gradient descend
    Eigen::VectorXd temp_new_x = x - lr * gradient;
    double cost_new = RosenbrockEvaluate(temp_new_x);
    // check condition loop
    int loop_iter = 0;
    while (cost_new - cost_old > armigo_cond * lr)
    {
        // learning rate decays
        lr /= 2;
        // update with new learning rate
        temp_new_x = x - lr * gradient;
        cost_new = RosenbrockEvaluate(temp_new_x);
        if (++loop_iter > 10)
        {
            std::cerr << "loop more than 10 times" << std::endl;
            return false;
        }
    }
    // finish one step
    x = temp_new_x;
    x_list.push_back(x);
    std::cout << "new x: (" << x(0);
    for (int i = 1; i < x.rows(); ++i)
    {
        std::cout << "," << x(i);
    }
    std::cout << ")\n";
    std::cout << "cost at this point: " << cost_new << std::endl;
    std::cout << "gradient norm: " << gradient.norm() << std::endl;

    return true;
}

bool gradientDescent(const Eigen::VectorXd &x_init,
                     OptimizerConfig &config)
{
    const int order = x_init.rows();
    int iter = 0;
    std::vector<Eigen::VectorXd> x_list;

    double lr = config.lr_;

    Eigen::VectorXd x = x_init;
    x_list.push_back(x);
    double cost_old = RosenbrockEvaluate(x);
    Eigen::VectorXd gradient = RosenbrockFirstOrderDeriv(x);

    while (gradient.norm() > config.stop_delta_)
    {
        std::cout << "Iteration " << ++iter << "-----------\n";
        if (config.method_ == "inexact_line_search")
        {
            if(!inexactLineSearchStep(x, gradient, cost_old, lr, config.c_, x_list))
            {
                return false;
            }
        }
    }
    return true;
}

int main(int argc, char **argv)
{
    Eigen::VectorXd x(3);
    x << -1.0, -1.0, -1.0;
    OptimizerConfig config(0.1, 0.00001, "inexact_line_search", 0, 0.1);
    gradientDescent(x, config);

    return 0;
}
