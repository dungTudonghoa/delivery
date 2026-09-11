#include "delivery/graph.hpp"
#include "delivery/io.hpp"
#include <iostream>
#include <fstream>
#include <numeric>
int main(int argc,char**argv){try{
 if(argc!=5){std::cerr<<"Usage: generate_dataset OUTPUT_DIR COUNT SEED GA_CONFIG\n";return 2;}
 std::filesystem::path root=argv[1];int count=std::stoi(argv[2]);auto seed=std::stoull(argv[3]);auto cfg=delivery::read_config(argv[4]);if(count<10)throw std::invalid_argument("COUNT must be >=10 for nonempty splits");std::filesystem::create_directories(root);
 if(std::filesystem::exists(root/"manifest.csv"))throw std::runtime_error("output already contains a dataset; choose a new directory");
 for(auto split:{"train","val","test","ood"}){auto dir=root/split;std::filesystem::create_directories(dir);if(!std::filesystem::is_empty(dir))throw std::runtime_error("split directory is not empty");}
 std::ofstream manifest(root/"manifest.csv");if(!manifest)throw std::runtime_error("manifest open failed");manifest<<"id,split,trucks,slots,receivers,edges,demand,seed,budget,certified,crossings,evaluations,seconds\n";int accepted=0;
 for(int n=0;n<count;++n){std::string split=n<count*6/10?"train":n<count*8/10?"val":n<count*9/10?"test":"ood";int T=split=="ood"?8:3+n%3,R=split=="ood"?10:3+n%4;auto instance=delivery::generate_instance(seed+n,T,R);auto result=delivery::run_ga(instance,cfg,seed+n);delivery::Record rec{instance,{},seed+n,cfg.budget,result.seconds};if(result.validation.certified()){rec.label=result.allocation;++accepted;}
  delivery::write_record(root/split/(instance.id+".delivery"),rec);int demand=0;for(auto&r:instance.receivers)demand+=std::accumulate(r.demand.begin(),r.demand.end(),0);
  manifest<<instance.id<<','<<split<<','<<T<<','<<instance.slots.size()<<','<<R<<','<<delivery::build_graph(instance).edges.size()<<','<<demand<<','<<rec.seed<<','<<cfg.budget<<','<<result.validation.certified()<<','<<(result.validation.feasible?result.validation.crossings:-1)<<','<<result.evaluations<<','<<result.seconds<<'\n';
 }if(!manifest)throw std::runtime_error("manifest write failed");std::cout<<"Certified "<<accepted<<" / "<<count<<". Splits fixed before label generation.\n";
}catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 1;}}
