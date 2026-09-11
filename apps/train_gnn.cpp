#include "delivery/gnn.hpp"
#include "delivery/io.hpp"
#include <iostream>
#include <fstream>
#include <set>
#include <random>
#include <numeric>
#include <algorithm>
static torch::Tensor targets(const delivery::Record&r){std::vector<int64_t> y;for(int d:r.label)y.push_back(d<0?int64_t(r.instance.receivers.size()):d);return torch::tensor(y,torch::kInt64);}
int main(int argc,char**argv){try{
 if(argc!=10){std::cerr<<"Usage: train_gnn TRAIN_DIR VAL_DIR MODEL.pt EPOCHS SEED HIDDEN LAYERS BATCH_SIZE LR\n";return 2;}
 torch::set_num_threads(1);auto seed=std::stoull(argv[5]);torch::manual_seed(seed);std::mt19937_64 rng(seed);auto train=delivery::read_dataset(argv[1]),val=delivery::read_dataset(argv[2]);
 std::set<std::string> ids;for(auto&r:train)if(!ids.insert(r.instance.id).second)throw std::runtime_error("duplicate train instance ID");for(auto&r:val)if(!ids.insert(r.instance.id).second)throw std::runtime_error("train/validation leakage or duplicate ID");
 int epochs=std::stoi(argv[4]),batch=std::stoi(argv[8]);double lr=std::stod(argv[9]);if(epochs<1||batch<1||!std::isfinite(lr)||lr<=0)throw std::invalid_argument("invalid training configuration");delivery::GNN model(delivery::GNNConfig{train.front().instance.types,std::stoi(argv[6]),std::stoi(argv[7])});torch::optim::Adam optimizer(model->parameters(),torch::optim::AdamOptions(lr));
 std::filesystem::path path=argv[3];if(!path.parent_path().empty())std::filesystem::create_directories(path.parent_path());std::ofstream log(path.string()+".csv");if(!log)throw std::runtime_error("cannot write training log");log<<"epoch,train_nll,val_nll,val_top1,val_top3\n";double best=std::numeric_limits<double>::infinity();std::vector<size_t> order(train.size());std::iota(order.begin(),order.end(),0);
 for(int epoch=1;epoch<=epochs;++epoch){model->train();std::shuffle(order.begin(),order.end(),rng);double train_loss=0;
  for(size_t b=0;b<order.size();b+=batch){optimizer.zero_grad();std::vector<torch::Tensor> losses;for(size_t j=b;j<std::min(order.size(),b+batch);++j){auto&r=train[order[j]];auto loss=torch::nn::functional::cross_entropy(model->forward(r.instance),targets(r));train_loss+=loss.item<double>();losses.push_back(loss);}torch::stack(losses).mean().backward();optimizer.step();}
  model->eval();torch::NoGradGuard guard;double nll=0,top1=0,top3=0;
  for(auto&r:val){auto logits=model->forward(r.instance),y=targets(r);nll+=torch::nn::functional::cross_entropy(logits,y).item<double>();top1+=logits.argmax(1).eq(y).to(torch::kFloat32).mean().item<double>();auto idx=std::get<1>(logits.topk(std::min<int64_t>(3,logits.size(1)),1));top3+=idx.eq(y.unsqueeze(1)).any(1).to(torch::kFloat32).mean().item<double>();}
  nll/=val.size();log<<epoch<<','<<train_loss/train.size()<<','<<nll<<','<<top1/val.size()<<','<<top3/val.size()<<'\n';if(nll<best){best=nll;delivery::save_model(model,path.string());}std::cout<<"epoch="<<epoch<<" val_nll="<<nll<<'\n';
 }if(!log)throw std::runtime_error("training log write failed");
}catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 1;}}
