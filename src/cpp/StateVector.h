#ifndef STATE_VECTOR_H
#define STATE_VECTOR_H

#include <cstdio>
#include <vector>
#include <algorithm>
#include <functional>

class StateVector // wrapper class for std::vector with + operation
{
    public:
      StateVector(std::vector<double> value) : m_state(value) {};

      // Rule of Five
      StateVector(StateVector &other) {
        m_state = other.m_state;
      }

      StateVector(StateVector&& other) {
        m_state = std::move(other.m_state);
      }

      StateVector& operator=(const StateVector &other) {
        m_state = other.m_state;
        return *this;
      }

      StateVector& operator=(const StateVector &&other) {
        m_state = std::move(other.m_state);
        return *this;
      }

      ~StateVector() {};

      // Adding state vectors
      StateVector operator+(StateVector &other) const {
        if (m_state.size() != other.m_state.size()) {
            perror("StateVector: INVALID ADDITION, DIMENSION OF OPERANDS DO NOT MATCH");
            exit(1);
        }
        std::vector<double> result(other.m_state.size());
        std::transform(m_state.begin(), 
                       m_state.end(),
                       other.m_state.begin(), 
                       result.begin(),
                       std::plus<double>());
        return StateVector(result);    
      };

      // Scalar multiplication
      StateVector operator*(double &scalar) {
        std::vector<double> result(m_state.size());
        std::transform(m_state.begin(), 
                       m_state.end(), 
                       result.begin(), 
                       [scalar](double y) {return scalar * y;});
        return StateVector(result);
      }

      std::vector<double> getState() const { return m_state;} // breaks encapsulation :/
    private:
      std::vector<double> m_state;
};

#endif