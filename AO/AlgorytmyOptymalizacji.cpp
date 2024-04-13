<<<<<<< HEAD
#include <iostream>
#include <fstream>
#include <vector>
#include <tuple>
#include <random>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <sstream>

using namespace std;

struct Job {
    int id;
    int release_time;
    int due_date;
    int processing_time;
};

int calculateCmax(const vector<Job>& schedule) {
    int currentTime = 0;
    int endTime = 0;
    for (const Job& job : schedule) {
        currentTime = max(currentTime + job.processing_time, job.release_time);
        endTime = max(endTime, currentTime + job.due_date);
    }
    return endTime;
}

vector<Job> simulatedAnnealing(const vector<Job>& jobs, int iteration_per_temp, double initialTemperature,
    double coolingRate, double minTemperature, vector<pair<double, int>>& annealingData) {
    random_device rd;
    mt19937 gen(rd());
    uniform_real_distribution<double> distribution(0.0, 1.0);

    double currentTemperature = initialTemperature;

    vector<Job> currentSchedule = jobs;
    vector<Job> bestSchedule = currentSchedule;

    while (currentTemperature > minTemperature) {
        for (int i = 0; i < iteration_per_temp; ++i) {
            vector<Job> newSchedule = currentSchedule;

            int index1 = uniform_int_distribution<int>(0, newSchedule.size() - 1)(gen);
            int index2 = uniform_int_distribution<int>(0, newSchedule.size() - 1)(gen);
            swap(newSchedule[index1], newSchedule[index2]);

            int delta = calculateCmax(newSchedule) - calculateCmax(currentSchedule);

            if (delta < 0 || distribution(gen) < exp(-static_cast<double>(delta) / currentTemperature)) {
                currentSchedule = newSchedule;
            }

            if (calculateCmax(currentSchedule) < calculateCmax(bestSchedule)) {
                bestSchedule = currentSchedule;
            }
        }
        annealingData.push_back(make_pair(currentTemperature, calculateCmax(bestSchedule)));
        currentTemperature *= coolingRate;
    }

    return bestSchedule;
}

void writeAnnealingDataToCSV(const vector<pair<double, int>>& annealingData, const string& filename) {
    ofstream outFile(filename);
    if (!outFile.is_open()) {
        cerr << "Error: Unable to open file for writing." << endl;
        return;
    }

    outFile << "Temperature;Cmax" << endl;                          // Naglowek

    for (const auto& data : annealingData) {                        // Dane
        outFile << -data.second << ";" << data.first << endl;
    }

    outFile.close();
    cout << "Annealing data written to " << filename << endl;
}

int main() {
    ifstream file("rpq_100.txt");
    if (!file.is_open()) {
        cerr << "Error: Unable to open file." << endl;
        return 1;
    }

    vector<Job> jobs;
    string line;
    int id = 1;

    while (getline(file, line)) {
        stringstream ss(line);
        Job job;
        ss >> job.release_time >> job.processing_time >> job.due_date;
        job.id = id++;
        jobs.push_back(job);
    }

    double initialTemperature = 10000.0;
    double coolingRate = 0.95;
    double minTemperature = 0.00001;
    int iteration_per_temp = 500;

    vector<pair<double, int>> annealingData;
    vector<Job> optimizedSchedule = simulatedAnnealing(jobs, iteration_per_temp, initialTemperature, coolingRate, minTemperature, annealingData);


    writeAnnealingDataToCSV(annealingData, "annealing_data.csv"); // Zapisz do CSV

    cout << "Optimal schedule:" << endl;
    cout << "Sequence: ";
    for (const auto& job : optimizedSchedule) {
        cout << job.id << " ";
    }
    cout << endl;
    cout << "Makespan (Cmax): " << calculateCmax(optimizedSchedule) << endl;

    return 0;
=======
#include <iostream>
#include <fstream>
#include <vector>
#include <tuple>
#include <random>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <sstream>

using namespace std;

struct Job {
    int id;
    int release_time;
    int due_date;
    int processing_time;
};

int calculateCmax(const vector<Job>& schedule) {
    int currentTime = 0;
    int endTime = 0;
    for (const Job& job : schedule) {
        currentTime = max(currentTime + job.processing_time, job.release_time);
        endTime = max(endTime, currentTime + job.due_date);
    }
    return endTime;
}

vector<Job> simulatedAnnealing(const vector<Job>& jobs, int iteration_per_temp, double initialTemperature,
    double coolingRate, double minTemperature, vector<pair<double, int>>& annealingData) {
    random_device rd;
    mt19937 gen(rd());
    uniform_real_distribution<double> distribution(0.0, 1.0);

    double currentTemperature = initialTemperature;

    vector<Job> currentSchedule = jobs;
    vector<Job> bestSchedule = currentSchedule;

    while (currentTemperature > minTemperature) {
        for (int i = 0; i < iteration_per_temp; ++i) {
            vector<Job> newSchedule = currentSchedule;

            int index1 = uniform_int_distribution<int>(0, newSchedule.size() - 1)(gen);
            int index2 = uniform_int_distribution<int>(0, newSchedule.size() - 1)(gen);
            swap(newSchedule[index1], newSchedule[index2]);

            int delta = calculateCmax(newSchedule) - calculateCmax(currentSchedule);

            if (delta < 0 || distribution(gen) < exp(-static_cast<double>(delta) / currentTemperature)) {
                currentSchedule = newSchedule;
            }

            if (calculateCmax(currentSchedule) < calculateCmax(bestSchedule)) {
                bestSchedule = currentSchedule;
            }
        }
        annealingData.push_back(make_pair(currentTemperature, calculateCmax(bestSchedule)));
        currentTemperature *= coolingRate;
    }

    return bestSchedule;
}

void writeAnnealingDataToCSV(const vector<pair<double, int>>& annealingData, const string& filename) {
    ofstream outFile(filename);
    if (!outFile.is_open()) {
        cerr << "Error: Unable to open file for writing." << endl;
        return;
    }

    outFile << "Temperature;Cmax" << endl;                          // Naglowek

    for (const auto& data : annealingData) {                        // Dane
        outFile << -data.second << ";" << data.first << endl;
    }

    outFile.close();
    cout << "Annealing data written to " << filename << endl;
}

int main() {
    ifstream file("rpq_100.txt");
    if (!file.is_open()) {
        cerr << "Error: Unable to open file." << endl;
        return 1;
    }

    vector<Job> jobs;
    string line;
    int id = 1;

    while (getline(file, line)) {
        stringstream ss(line);
        Job job;
        ss >> job.release_time >> job.processing_time >> job.due_date;
        job.id = id++;
        jobs.push_back(job);
    }

    double initialTemperature = 10000.0;
    double coolingRate = 0.95;
    double minTemperature = 0.00001;
    int iteration_per_temp = 500;

    vector<pair<double, int>> annealingData;
    vector<Job> optimizedSchedule = simulatedAnnealing(jobs, iteration_per_temp, initialTemperature, coolingRate, minTemperature, annealingData);


    writeAnnealingDataToCSV(annealingData, "annealing_data.csv"); // Zapisz do CSV

    cout << "Optimal schedule:" << endl;
    cout << "Sequence: ";
    for (const auto& job : optimizedSchedule) {
        cout << job.id << " ";
    }
    cout << endl;
    cout << "Makespan (Cmax): " << calculateCmax(optimizedSchedule) << endl;

    return 0;
>>>>>>> 674f4b1611ee4c46305483ad244c07ec57b9681f
}