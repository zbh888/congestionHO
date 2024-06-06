#include <iostream>
#include <cmath>
#include <vector>
#include <array>
#include <random>
#include <fstream>
#include <chrono>
#include <nlohmann/json.hpp>
using json = nlohmann::json;

// Define constants
constexpr double G = 6.674e-11;
constexpr double M_earth = 5.972e24;
constexpr double R_earth = 6.371e6;

// Function to compute velocity from height
double computeVFromH(double h) {
    double GM = G * M_earth;
    double r = h + R_earth;
    return std::sqrt(GM / r);
}

// Class for satellite
class Satellite {
private:
    double x, y, h, v, r, sind, cosd, tid;

public:
    Satellite(double h_, double d, double x_, double y_, double c_r, double t_id) :
            x(x_), y(y_), h(h_), r(c_r), tid(t_id) {
        v = computeVFromH(h * 1000) / 1000; //km/s
        sind = std::sin(M_PI * d / 180.0);
        cosd = std::cos(M_PI * d / 180.0);
    }

    std::pair<double, double> calculateLocation(double t, double t_unit) {
        double dis = v * t * t_unit; // km/s * s
        double y_new = dis * sind;
        double x_new = dis * cosd;
        return std::make_pair(x + x_new, y + y_new);
    }

    bool cover(std::pair<double, double> u_pos, double t, double t_unit) {
        auto s_pos = calculateLocation(t, t_unit);
        double dis = std::sqrt(std::pow(u_pos.first - s_pos.first, 2) + std::pow(u_pos.second - s_pos.second, 2));
        return dis < r;
    }

    json toJSON(int index) const {
        return {
            {"index", index},
            {"x", x},
            {"y", y},
            {"h", h},
            {"v", v},
            {"r", r},
            {"sind", sind},
            {"cosd", cosd},
            {"tid", tid}
        };
    }
};

// Class for UE (User Equipment)
class UE {
private:
    double x, y;

public:
    UE(double x_, double y_) : x(x_), y(y_) {}

    std::pair<double, double> position() {
        return std::make_pair(x, y);
    }

    json toJSON(int index) const {
        return {
            {"index", index},
            {"x", x},
            {"y", y},
        };
    }
};

// Function to generate C matrix
std::vector <std::vector<std::vector < int>>>

generateC(bool feasible, double t_unit, double t_total, std::vector <UE> &UEs, std::vector <Satellite> &satellites) {
    int T_SLOT_TOTAL = static_cast<int>(t_total / t_unit);
    std::vector < std::vector < std::vector < int>>> C;

    if (feasible) {
        C.resize(UEs.size(),
                 std::vector < std::vector < int >> (satellites.size() + 1, std::vector<int>(T_SLOT_TOTAL, 0)));
    } else {
        C.resize(UEs.size(), std::vector < std::vector < int >> (satellites.size(), std::vector<int>(T_SLOT_TOTAL, 0)));
    }

    for (size_t ue_i = 0; ue_i < UEs.size(); ++ue_i) {
        for (size_t s_i = 0; s_i < satellites.size(); ++s_i) {
            for (int t = 0; t < T_SLOT_TOTAL; ++t) {
                if (satellites[s_i].cover(UEs[ue_i].position(), t, t_unit)) {
                    C[ue_i][s_i][t] = 1;
                }
            }
        }

        if (feasible) {
            for (int t = 0; t < T_SLOT_TOTAL; ++t) {
                C[ue_i][satellites.size()][t] = 1;
            }
        }
    }

    return C;
}

// Function to generate UE (User Equipment)
UE generateUE(std::mt19937 &gen) {
    std::uniform_real_distribution<> dis(0, 1);
    double randomNum = dis(gen);
    if (randomNum < 0.5) {
        std::uniform_real_distribution<> disx(-5, 5);
        double x = disx(gen);
        std::uniform_real_distribution<> disy(-5, 5);
        double y = disy(gen);
        return UE(x, y);
    } else {
        std::uniform_real_distribution<> disx(-75, -70);
        double x = disx(gen);
        std::uniform_real_distribution<> disy(-5, 5);
        double y = disy(gen);
        return UE(x, y);
    }
}

// Class for trajectory
class Trajectory {
private:
    double d, inter_satellite_distance, lead_x, lead_y, h, c_r, tid;
public:
    Trajectory(double d_, double h_, double inter_d, double lead_x_, double lead_y_, double cov_r, double t_id) :
            d(d_), inter_satellite_distance(inter_d), lead_x(lead_x_), lead_y(lead_y_), h(h_), c_r(cov_r), tid(t_id) {}

    Satellite generateSatellite(int n) const {
        double dis = inter_satellite_distance * n;
        double y_ = lead_y + dis * std::sin(M_PI * (d + 180) / 180.0);
        double x_ = lead_x + dis * std::cos(M_PI * (d + 180) / 180.0);
        return Satellite(h, d, x_, y_, c_r, tid);
    }
};

int main() {
    int seed = 20702017;
    int N_satellites_one_trajectory = 8;
    int N_UE = 20;
    double T_TOTAL = 200;
    double T_UNIT2 = 0.01;

    bool feasible = true;

    std::vector<Trajectory> trajectories = {
        Trajectory(0, 300, 250, 0, 100, 200, 1),
        Trajectory(0, 300, 250, 100, -100, 200, 1),
        Trajectory(-45, 700, 250, -50, 90, 200, 2),
        Trajectory(-45, 700, 250, -50-50*1.414, 90-200*1.414+50, 200, 2),
        Trajectory(45, 1100, 250, -150, 0, 200, 3),
        Trajectory(45, 1100, 250, -150+50*1.414, 0-200*1.414+50, 200, 3)
    };

    std::vector <UE> UEs;
    std::vector <Satellite> satellites;

    // Create UEs
    std::mt19937 gen(seed);
    for (int i = 0; i < N_UE; ++i) {
        UEs.push_back(generateUE(gen));
    }

    // Create satellites based on trajectory
    for (int i = 0; i < N_satellites_one_trajectory; ++i) {
        for (const auto& trajectory_i : trajectories) {
            satellites.push_back(trajectory_i.generateSatellite(i));
        }
    }

    auto start = std::chrono::high_resolution_clock::now();
    auto data2 = generateC(feasible, T_UNIT2, T_TOTAL, UEs, satellites);
    auto end = std::chrono::high_resolution_clock::now();
    std::cout << "Time taken: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count()
              << " milliseconds" << std::endl;
    std::ofstream file("data_optimizer.bin", std::ios::binary);

    auto end2 = std::chrono::high_resolution_clock::now();

    size_t dim1_2 = data2.size();
    std::ofstream file2("data_simulation.bin", std::ios::binary);
    file2.write(reinterpret_cast<char *>(&dim1_2), sizeof(dim1_2));

    for (const auto &vec2d_2: data2) {
        size_t dim2_2 = vec2d_2.size();
        file2.write(reinterpret_cast<char *>(&dim2_2), sizeof(dim2_2));

        for (const auto &vec1d_2: vec2d_2) {
            size_t dim3_2 = vec1d_2.size();
            file2.write(reinterpret_cast<char *>(&dim3_2), sizeof(dim3_2));
            file2.write(reinterpret_cast<const char *>(vec1d_2.data()), dim3_2 * sizeof(int));
        }
    }

    json satellitesJson;
    for (int i = 0; i < satellites.size(); ++i) {
        satellitesJson.push_back(satellites[i].toJSON(i));
    }
    std::ofstream outputFileSAT("satellites.json");
    outputFileSAT << satellitesJson.dump(4); // Use 4 spaces for indentation
    outputFileSAT.close();

    json UEsJson;
    for (int i = 0; i < UEs.size(); ++i) {
        UEsJson.push_back(UEs[i].toJSON(i));
    }
    std::ofstream outputFileUE("UEs.json");
    outputFileUE << UEsJson.dump(4); // Use 4 spaces for indentation
    outputFileUE.close();

    std::ofstream outfileduration("duration.txt");
    outfileduration << T_TOTAL;
    outfileduration.close();

    std::cout << "Time taken to save: " << std::chrono::duration_cast<std::chrono::milliseconds>(end2 - end).count()
              << " milliseconds" << std::endl;
    return 0;
}
