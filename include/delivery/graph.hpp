#pragma once
#include "delivery/domain.hpp"
namespace delivery {
struct Graph { std::vector<std::vector<float>> slots, receivers, edges; std::vector<int> source, target; };
Graph build_graph(const Instance&);
}
