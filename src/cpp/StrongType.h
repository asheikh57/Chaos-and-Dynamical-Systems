#include "StateVector.h"
template<typename T, typename TypeName>
struct StrongType 
{
    T value;
    StrongType operator+(const StrongType &other) const 
    {
        return value + other.value;
    }

    bool operator==(const StrongType &other) const 
    {
        return value == other.value;
    }

    bool operator<(const StrongType &other) const
    {
        return value < other.value;
    }

    bool operator>(const StrongType &other) const
    {
        return value > other.value;
    }

    bool operator>=(const StrongType &other) const
    {
        return value >= other.value;
    }

    bool operator<=(const StrongType &other) const
    {
        return value <= other.value;
    }

    explicit StrongType(const T &val) : value(val) {};
    explicit StrongType(const T &&val) : value(std::move(val)) {};
};
