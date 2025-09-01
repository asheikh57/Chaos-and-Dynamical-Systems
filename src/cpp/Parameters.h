#ifndef PARAMETERS_H
#define PARAMETERS_H
#include <map>
#include <string>

class Parameters
{
    public:
        Parameters(std::map<std::string, double> &params) : m_parameters(params) {};
        double& operator[](const std::string &key)
        {
            return m_parameters[key];
        }
    private:
        std::map<std::string, double> m_parameters;
};


#endif