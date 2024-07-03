#pragma once

#include <string>
#include <chrono>

template <typename F>
void measureExecutionTime(std::string target, F func)
{
    int count = 1;
    std::vector<double> executionTimes;
    executionTimes.reserve(count);

    for (int i = 0; i < count; i++)
    {
        // Start time
        auto start = std::chrono::high_resolution_clock::now();

        // Execute the function with the provided arguments
        func();

        // End time
        auto end = std::chrono::high_resolution_clock::now();

        // Calculate elapsed time
        auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        executionTimes.push_back(elapsed.count());
    }

    double sum = 0;
    for (auto time : executionTimes)
    {
        sum += time;
    }
    double average = sum / count;

    // Print elapsed time
    std::cout << "Execution time"
              << "[" << target << "]: " << average << " microseconds" << std::endl;
}
