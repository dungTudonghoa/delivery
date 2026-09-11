#include "delivery/io.hpp"
#include <iostream>
#include <fstream>
int main(int argc,char**argv){try{
 if(argc!=6){std::cerr<<"Usage: evaluate_ga DATA_DIR GA_CONFIG SEED RUNS OUTPUT.csv\n";return 2;}auto data=delivery::read_dataset(argv[1],false);auto cfg=delivery::read_config(argv[2]);auto seed=std::stoull(argv[3]);int runs=std::stoi(argv[4]);if(runs<1)throw std::invalid_argument("RUNS must be positive");std::ofstream f(argv[5]);if(!f)throw std::runtime_error("output open failed");
 f<<"id,method,seed,budget,evaluations,proposals,repairs,failures,feasible,success,best_crossings,first_zero,search_seconds\n";
 for(auto&rec:data)for(int k=0;k<runs;++k){auto x=delivery::run_ga(rec.instance,cfg,seed+k);f<<rec.instance.id<<",GA,"<<seed+k<<','<<cfg.budget<<','<<x.evaluations<<','<<x.proposals<<','<<x.repairs<<','<<x.failures<<','<<x.validation.feasible<<','<<x.validation.certified()<<','<<(x.validation.feasible?x.validation.crossings:-1)<<','<<x.first_zero<<','<<x.seconds<<'\n';}
 if(!f)throw std::runtime_error("output write failed");
}catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 1;}}
