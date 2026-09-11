#include "delivery/io.hpp"
#ifdef DELIVERY_HAS_TORCH
#include "delivery/gnn.hpp"
#endif
#include <iostream>
#include <fstream>
int main(int argc, char** argv) {
    try {
        if (argc != 5 && argc != 6) {
            std::cerr << "Usage: solve_delivery INSTANCE.delivery GA_CONFIG SEED OUTPUT.csv [MODEL.pt]\n";
            return 2;
        }
        auto record = delivery::read_record(argv[1]);
        auto config = delivery::read_config(argv[2]);
        auto seed = std::stoull(argv[3]);
        delivery::Prior prior;
        if (argc == 6) {
#ifdef DELIVERY_HAS_TORCH
            torch::set_num_threads(1);
            auto model = delivery::load_model(argv[5]);
            prior = model->predict(record.instance);
            std::ofstream p(std::string(argv[4]) + ".prior.csv");
            if (!p) throw std::runtime_error("cannot write prior");
            p << "slot,receiver,probability\n";
            for (size_t s = 0; s < prior.size(); ++s) {
                for (size_t r = 0; r < prior[s].size(); ++r) {
                    p << s << ',' << (r == record.instance.receivers.size() ? -1 : int(r))
                      << ',' << prior[s][r] << '\n';
                }
            }
            if (!p) throw std::runtime_error("prior write failed");
#else
            throw std::runtime_error("rebuild with DELIVERY_WITH_TORCH=ON for MODEL.pt");
#endif
        }
        auto result = delivery::run_ga(record.instance, config, seed, prior.empty() ? nullptr : &prior);
        if (!result.validation.feasible) {
            std::cerr << "No feasible allocation: " << result.validation.reason << '\n';
            return 1;
        }
        delivery::write_allocation(argv[4], record.instance, result.allocation);
        std::cout << "feasible=1 crossings=" << result.validation.crossings
                  << " certified=" << result.validation.certified()
                  << " evaluations=" << result.evaluations << '\n';
        // A feasible positive-crossing result is exported, but is explicitly not certified.
        return result.validation.certified() ? 0 : 3;
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
