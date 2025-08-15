#ifndef STATE_VECTOR_H
#define STATE_VECTOR_H

#include <csignal>
#include <cstdio>
#include <iterator>
#include <vector>
#include <algorithm>
#include <functional>

class StateVector // wrapper class for std::vector with + operation
{
    public:
      StateVector(std::vector<double> value) : m_state(value) {};
      StateVector operator+(StateVector &other) const {
        if (m_state.size() != other.size()) {
            perror("StateVector: INVALID ADDITION, DIMENSION OF OPERANDS DO NOT MATCH");
            exit(1);
        }
        std::vector<double> result(other.size());
        std::transform(m_state.begin(), 
                       m_state.end(),
                       other.begin(), 
                       result.begin(),
                       std::plus<double>());
        return StateVector(result);    
      };
      StateVector& operator=(const StateVector &other) {
        
      }
      std::vector<double> getState() { return m_state;} // breaks encapsulation :/
    private:
      std::vector<double> m_state;
      int size() { return m_state.size();}
      std::vector<double>::iterator begin() {return m_state.begin();}
};

#endif