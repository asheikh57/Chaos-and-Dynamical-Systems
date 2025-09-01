#ifndef NUMERICAL_H
#define NUMERICAL_H
#include <vector>
#include <functional>
#include <map>
#include <string>
#include "StateVector.h"
#include "StrongType.h"
#include "TypeTags.h"

namespace numerical {
    using StateElement = StrongType<double, StateElement>;
    using StateVector = StateVector<StateElement>;
    using Time = StrongType<double, TimeTag>;
    using Parameters = std::map<std::string, double>;
    using DifferentialEquationFunction = std::function<StateVector(Time, StateVector, Parameters)>;

    
    StateVector rk4(DifferentialEquationFunction &diff_eq, // takes t, y as input and outputs a y based off of it
           Time &t, 
           StateVector &y);

    std::vector<StateVector> solve_equation(
        DifferentialEquationFunction &diff_eq,
        std::vector<Time> &t_span,
        StateVector &y0,
        std::vector<Time> &t_eval,
        Parameters params
    );

}
#endif