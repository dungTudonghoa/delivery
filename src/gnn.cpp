#include "delivery/gnn.hpp"
#include <fstream>
#include <limits>
#include <stdexcept>
namespace delivery {
static torch::nn::Sequential mlp(int in,int hidden,int out){return torch::nn::Sequential(torch::nn::Linear(in,hidden),torch::nn::ReLU(),torch::nn::Linear(hidden,out));}
GNNImpl::GNNImpl(GNNConfig c):config(c){
 if(c.types<1||c.hidden<1||c.layers<1)throw std::invalid_argument("invalid GNN dimensions");
 slot_encoder=register_module("slot_encoder",mlp(7+c.types,c.hidden,c.hidden));receiver_encoder=register_module("receiver_encoder",mlp(2+c.types,c.hidden,c.hidden));scorer=register_module("scorer",mlp(2*c.hidden+6,c.hidden,1));unused=register_module("unused",mlp(c.hidden,c.hidden,1));
 for(int l=0;l<c.layers;++l){auto name=std::to_string(l);sr.push_back(register_module("sr"+name,mlp(2*c.hidden+6,c.hidden,c.hidden)));rs.push_back(register_module("rs"+name,mlp(2*c.hidden+6,c.hidden,c.hidden)));us.push_back(register_module("us"+name,mlp(2*c.hidden,c.hidden,c.hidden)));ur.push_back(register_module("ur"+name,mlp(2*c.hidden,c.hidden,c.hidden)));ns.push_back(register_module("ns"+name,torch::nn::LayerNorm(torch::nn::LayerNormOptions({c.hidden}))));nr.push_back(register_module("nr"+name,torch::nn::LayerNorm(torch::nn::LayerNormOptions({c.hidden}))));}
}
static torch::Tensor matrix(const std::vector<std::vector<float>>&rows,int width){std::vector<float> flat;for(auto&r:rows)flat.insert(flat.end(),r.begin(),r.end());if(flat.empty())return torch::empty({0,width},torch::kFloat32);return torch::from_blob(flat.data(),{int64_t(rows.size()),width},torch::kFloat32).clone();}
torch::Tensor GNNImpl::forward(const Instance&i){
 if(i.types!=config.types)throw std::invalid_argument("checkpoint goods-type vocabulary mismatch");auto g=build_graph(i);auto device=parameters().front().device();const int64_t S=i.slots.size(),R=i.receivers.size();
 auto hs=slot_encoder->forward(matrix(g.slots,7+config.types).to(device));auto hr=receiver_encoder->forward(matrix(g.receivers,2+config.types).to(device));auto ef=matrix(g.edges,6).to(device);
 std::vector<int64_t> sv(g.source.begin(),g.source.end()),rv(g.target.begin(),g.target.end());auto si=torch::tensor(sv,torch::TensorOptions().dtype(torch::kInt64)).to(device);auto ri=torch::tensor(rv,torch::TensorOptions().dtype(torch::kInt64)).to(device);
 for(int l=0;l<config.layers;++l){auto ms=torch::zeros_like(hs),mr=torch::zeros_like(hr);auto ds=torch::zeros({S,1},hs.options()),dr=torch::zeros({R,1},hr.options());
  if(!sv.empty()){auto se=hs.index_select(0,si),re=hr.index_select(0,ri);mr=mr.index_add(0,ri,sr[l]->forward(torch::cat({se,re,ef},1)));ms=ms.index_add(0,si,rs[l]->forward(torch::cat({re,se,ef},1)));auto ones=torch::ones({int64_t(sv.size()),1},hs.options());ds=ds.index_add(0,si,ones);dr=dr.index_add(0,ri,ones);}
  auto next_s=ns[l]->forward(hs+us[l]->forward(torch::cat({hs,ms/ds.clamp_min(1)},1)));auto next_r=nr[l]->forward(hr+ur[l]->forward(torch::cat({hr,mr/dr.clamp_min(1)},1)));hs=next_s;hr=next_r;
 }
 auto dense=torch::full({S*R},-std::numeric_limits<float>::infinity(),hs.options());
 if(!sv.empty()){auto logits=scorer->forward(torch::cat({hs.index_select(0,si),hr.index_select(0,ri),ef},1)).squeeze(1);dense=dense.index_copy(0,si*R+ri,logits);}
 return torch::cat({dense.reshape({S,R}),unused->forward(hs)},1);
}
Prior GNNImpl::predict(const Instance&i){torch::NoGradGuard guard;auto probs=torch::softmax(forward(i),1).to(torch::kCPU).contiguous();auto a=probs.accessor<float,2>();Prior p(i.slots.size(),std::vector<double>(i.receivers.size()+1));for(int64_t s=0;s<probs.size(0);++s)for(int64_t r=0;r<probs.size(1);++r)p[s][r]=a[s][r];return p;}
void save_model(GNN&m,const std::string&path){torch::save(m,path);std::ofstream f(path+".meta");if(!f)throw std::runtime_error("checkpoint metadata write failed");f<<"DELIVERY_GNN_V1 "<<m->config.types<<' '<<m->config.hidden<<' '<<m->config.layers<<'\n';if(!f)throw std::runtime_error("checkpoint metadata write failed");}
GNN load_model(const std::string&path){std::ifstream f(path+".meta");std::string magic;GNNConfig c;f>>magic>>c.types>>c.hidden>>c.layers;if(!f||magic!="DELIVERY_GNN_V1")throw std::runtime_error("invalid checkpoint metadata");GNN m(c);torch::load(m,path);m->eval();return m;}
}
