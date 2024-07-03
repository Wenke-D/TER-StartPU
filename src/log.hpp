#pragma once
#include <string>
#include <iostream>
#include <sstream>

class Log
{
private:
    static constexpr bool ON = true;

public:
    template <typename T>
    static void print(std::string location, T obj)
    {
        if (!ON)
            return;
        std::ostringstream oss;
        oss << "Log: [" << location << "]\n"
            << obj.toString() << std::endl;

        std::cout << oss.str();
    }
};