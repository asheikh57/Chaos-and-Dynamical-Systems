#ifndef STATE_VECTOR_H
#define STATE_VECTOR_H

#include <cstdio>
#include <vector>
#include <algorithm>
#include <functional>

// wrapper class for std::vector with + operation
template<typename T>
class StateVector
{
    public:
      StateVector(std::vector<double> &value) : m_state(value) {};

      // Adding state vectors
      StateVector operator+(StateVector &other) const {
        if (m_state.size() != other.m_state.size()) {
            perror("StateVector: INVALID ADDITION, DIMENSION OF OPERANDS DO NOT MATCH");
            exit(1);
        }
        std::vector<T> result(other.m_state.size());
        std::transform(m_state.begin(), 
                       m_state.end(),
                       other.m_state.begin(), 
                       result.begin(),
                       std::plus<T>());
        return StateVector(result);    
      };

      // Scalar multiplication
      StateVector operator*(T &scalar) {
        std::vector<T> result(m_state.size());
        std::transform(m_state.begin(), 
                       m_state.end(), 
                       result.begin(), 
                       [scalar](T y) {return scalar * y;});
        return StateVector(result);
      }

      std::vector<T> getState() const { return m_state;} // breaks encapsulation :/
    private:
      std::vector<T> m_state;
};

#endif