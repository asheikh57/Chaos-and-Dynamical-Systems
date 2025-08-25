#include <vector>
#include <functional>
#include <map>
#include <string>
#include "Numerical.h"
#include "StateVector.h"


namespace numerical {
    StateVector rk4(DifferentialEquationFunction &diff_eq, // takes t, y as input and outputs a y based off of it
           Time &t, 
           StateVector &y) 
    {
        // TODO: implement
        return StateVector(std::vector<double>());
    }

    std::vector<StateVector> solve_equation(
        DifferentialEquationFunction &diff_eq,
        std::vector<Time> &t_span,
        StateVector &y0,
        std::vector<Time> &t_eval,
        Parameters params) 
    {
        // TODO: implement
        return std::vector<StateVector>();
    }
}