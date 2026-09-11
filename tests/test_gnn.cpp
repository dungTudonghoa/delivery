#include "delivery/gnn.hpp"
#include "delivery/io.hpp"
#include <iostream>
#include <stdexcept>
#include <algorithm>
#define DELIVERY_CHECK(x) do {if(!(x))throw std::runtime_error(std::string("failed: ")+#x);}while(false)
int main(){try{using namespace delivery;torch::set_num_threads(1);torch::manual_seed(42);GNN m(GNNConfig{2,16,2});
 for(auto dims:std::vector<std::pair<int,int>>{{3,4},{6,7}}){auto i=generate_instance(42,dims.first,dims.second);auto logits=m->forward(i);DELIVERY_CHECK(logits.size(0)==int64_t(i.slots.size()));DELIVERY_CHECK(logits.size(1)==int64_t(i.receivers.size()+1));auto p=m->predict(i);for(size_t s=0;s<p.size();++s){double sum=0;for(size_t r=0;r<p[s].size();++r){DELIVERY_CHECK(std::isfinite(p[s][r]));sum+=p[s][r];if(r<i.receivers.size()&&!eligible(i,s,r))DELIVERY_CHECK(p[s][r]==0);}DELIVERY_CHECK(std::abs(sum-1)<1e-5);}}
 auto i=generate_instance(9,3,4);auto before=m->forward(i).detach();auto perm=i;std::reverse(perm.slots.begin(),perm.slots.end());auto reversed=m->forward(perm).detach().flip({0});DELIVERY_CHECK(torch::allclose(before,reversed,1e-5,1e-5));
 auto receiver_perm=i;std::reverse(receiver_perm.receivers.begin(),receiver_perm.receivers.end());auto rp=m->forward(receiver_perm).detach();auto restored=torch::cat({rp.slice(1,0,int64_t(i.receivers.size())).flip({1}),rp.slice(1,int64_t(i.receivers.size()))},1);DELIVERY_CHECK(torch::allclose(before,restored,1e-5,1e-5));
 auto x=run_ga(i,GAConfig{},42);DELIVERY_CHECK(x.validation.certified());std::vector<int64_t> y;bool surplus=false;for(int d:x.allocation){surplus|=d<0;y.push_back(d<0?i.receivers.size():d);}DELIVERY_CHECK(surplus);auto target=torch::tensor(y,torch::kInt64);torch::optim::Adam opt(m->parameters(),torch::optim::AdamOptions(0.01));double initial=torch::nn::functional::cross_entropy(m->forward(i),target).item<double>();
 for(int epoch=0;epoch<40;++epoch){opt.zero_grad();auto loss=torch::nn::functional::cross_entropy(m->forward(i),target);loss.backward();opt.step();}DELIVERY_CHECK(torch::nn::functional::cross_entropy(m->forward(i),target).item<double>()<initial);
 DELIVERY_CHECK(m->slot_encoder->parameters().front().grad().defined());DELIVERY_CHECK(m->sr.front()->parameters().front().grad().defined());
 auto path=(std::filesystem::temp_directory_path()/"delivery-test-model.pt").string();save_model(m,path);auto loaded=load_model(path);DELIVERY_CHECK(torch::allclose(m->forward(i),loaded->forward(i)));std::filesystem::remove(path);std::filesystem::remove(path+".meta");
 // A graph with no candidate edges still has a finite, normalized UNUSED class.
 for(auto&t:i.trucks)t.max_range=0;auto p=m->predict(i);for(auto&row:p){DELIVERY_CHECK(row.back()==1);for(size_t r=0;r+1<row.size();++r)DELIVERY_CHECK(row[r]==0);}
 std::cout<<"gnn: all checks passed\n";
}catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 1;}}
