#include <iostream>
#include <fstream>
#include <vector>
#include <tuple>
#include <random>
#include <cmath>
#include <algorithm>
#include <numeric>

std::vector<std::tuple<int, int, int>> read_rpq_data(const std::string& filepath) {
    std::vector<std::tuple<int, int, int>> tasks;
    std::ifstream file(filepath);
    int r, p, q;

    while (file >> r >> p >> q) {
        tasks.push_back(std::make_tuple(r, p, q));
    }

    return tasks;
}

int calculate_makespan(const std::vector<int>& sequence, const std::vector<std::tuple<int, int, int>>& tasks) {
    int time = 0, end_time = 0;
    for (auto task_index : sequence) {
        int r = std::get<0>(tasks[task_index]);
        int p = std::get<1>(tasks[task_index]);
        int q = std::get<2>(tasks[task_index]);
        time = std::max(time + p, r + p);
        end_time = std::max(end_time, time + q);
    }
    return end_time;
}


std::vector<int> simulated_annealing(const std::vector<std::tuple<int, int, int>>& tasks, double initial_temp, double cooling_rate, int iteration_per_temp) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0, 1);

    double current_temp = initial_temp;
    std::vector<int> current_sequence(tasks.size());
    std::iota(current_sequence.begin(), current_sequence.end(), 0); // Fill with 0, 1, 2, ...
    std::shuffle(current_sequence.begin(), current_sequence.end(), gen);
    int current_makespan = calculate_makespan(current_sequence, tasks);

    while (current_temp > 1) {
        for (int i = 0; i < iteration_per_temp; ++i) {
            auto new_sequence = current_sequence;
            std::swap(new_sequence[std::uniform_int_distribution<>(0, new_sequence.size() - 1)(gen)],
                new_sequence[std::uniform_int_distribution<>(0, new_sequence.size() - 1)(gen)]);

            int new_makespan = calculate_makespan(new_sequence, tasks);
            if (new_makespan < current_makespan || dis(gen) < exp((current_makespan - new_makespan) / current_temp)) {
                current_sequence = std::move(new_sequence);
                current_makespan = new_makespan;
            }
        }
        current_temp *= cooling_rate;
    }

    return current_sequence;
}

int main() {
    std::string filepath = "rpq_100.txt"; // Change to your actual file path
    auto tasks = read_rpq_data(filepath);

    double initial_temp = 10000;
    double cooling_rate = 0.95;
    int iteration_per_temp = 500;

    auto best_sequence = simulated_annealing(tasks, initial_temp, cooling_rate, iteration_per_temp);
    int best_makespan = calculate_makespan(best_sequence, tasks);

    std::cout << "Najlepsza sekwencja: ";
    for (auto& task : best_sequence) {
        std::cout << task << " ";
    }
    std::cout << "\nMakespan: " << best_makespan << std::endl;

    return 0;
}