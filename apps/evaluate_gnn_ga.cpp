#include "delivery/graph.hpp"
#include "delivery/gnn.hpp"
#include "delivery/io.hpp"
#include <chrono>
#include <fstream>
#include <iostream>
int main(int argc,char**argv){try{
 if(argc!=7){std::cerr<<"Usage: evaluate_gnn_ga DATA_DIR MODEL.pt GA_CONFIG SEED RUNS OUTPUT.csv\n";return 2;}torch::set_num_threads(1);auto model=delivery::load_model(argv[2]);auto data=delivery::read_dataset(argv[1],false);auto cfg=delivery::read_config(argv[3]);auto seed=std::stoull(argv[4]);int runs=std::stoi(argv[5]);if(runs<1)throw std::invalid_argument("RUNS must be positive");std::ofstream f(argv[6]);if(!f)throw std::runtime_error("output open failed");
 f<<"id,method,seed,budget,slots,receivers,edges,evaluations,proposals,repairs,failures,feasible,success,best_crossings,first_zero,inference_seconds,search_seconds,total_seconds\n";
 for(auto&rec:data)for(int k=0;k<runs;++k)for(int mode=0;mode<3;++mode){auto config=cfg;config.guided_mutation=mode==2;delivery::Prior prior;double inference=0;
  if(mode){auto start=std::chrono::steady_clock::now();prior=model->predict(rec.instance);inference=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();}
  auto x=delivery::run_ga(rec.instance,config,seed+k,mode?&prior:nullptr);const char*name=mode==0?"GA":mode==1?"GNN-GA-init":"GNN-GA-init-mutation";
  f<<rec.instance.id<<','<<name<<','<<seed+k<<','<<cfg.budget<<','<<rec.instance.slots.size()<<','<<rec.instance.receivers.size()<<','<<delivery::build_graph(rec.instance).edges.size()<<','<<x.evaluations<<','<<x.proposals<<','<<x.repairs<<','<<x.failures<<','<<x.validation.feasible<<','<<x.validation.certified()<<','<<(x.validation.feasible?x.validation.crossings:-1)<<','<<x.first_zero<<','<<inference<<','<<x.seconds<<','<<inference+x.seconds<<'\n';
 }if(!f)throw std::runtime_error("output write failed");
}catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 1;}}
