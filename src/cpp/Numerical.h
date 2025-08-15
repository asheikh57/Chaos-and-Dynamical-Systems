#ifndef NUMERICAL_H
#define NUMERICAL_H
#include <vector>
#include <functional>
namespace numerical {
    using StateVector = std::vector<double>;
    using DifferentialEquationFunction = std::function<std::vector<double>(double, std::vector<double>)>;
    using Time = double;
    std::vector<double> rk4(DifferentialEquationFunction &diff_eq, // takes t, y as input and outputs a y based off of it
           Time &t, 
           StateVector &y);

    std::vector<StateVector> solve_equation(
        DifferentialEquationFunction &diff_eq,
        std::vector<Time> &t_span,
        StateVector &y0,
        std::vector<Time> &t_eval,
    );

}
#endif