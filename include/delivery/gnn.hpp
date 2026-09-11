#pragma once
#include "delivery/domain.hpp"
#include <torch/torch.h>
namespace delivery {
struct GNNConfig { int types=2, hidden=64, layers=2; };
struct GNNImpl : torch::nn::Module {
 explicit GNNImpl(GNNConfig config);
 torch::Tensor forward(const Instance&); // [S, R+1] masked logits; last column UNUSED
 Prior predict(const Instance&);
 GNNConfig config;
 torch::nn::Sequential slot_encoder{nullptr},receiver_encoder{nullptr},scorer{nullptr},unused{nullptr};
 std::vector<torch::nn::Sequential> sr,rs,us,ur;
 std::vector<torch::nn::LayerNorm> ns,nr;
};
TORCH_MODULE(GNN);
void save_model(GNN&,const std::string& path);
GNN load_model(const std::string& path);
}
