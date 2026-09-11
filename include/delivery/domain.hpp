#pragma once
#include <vector>
#include <string>
#include <cstdint>
namespace delivery {
struct Point { double x=0, y=0; };
struct DeliverTruck { Point position; double min_range=0, max_range=100, heading=0, half_angle=3.141592653589793; };
struct GoodsSlot { int truck=0, type=0; };
struct ReceivePoint { Point position; std::vector<int> demand; };
struct Instance { std::string id; int types=1; std::vector<DeliverTruck> trucks; std::vector<GoodsSlot> slots; std::vector<ReceivePoint> receivers; };
// One destination per slot; -1 means UNUSED. Indices are never neural features.
using Allocation = std::vector<int>;
using Prior = std::vector<std::vector<double>>; // [slot][receiver + UNUSED]
void check_instance(const Instance&);
bool eligible(const Instance&, int slot, int receiver);
bool interior_crossing(Point a, Point b, Point c, Point d);
struct Validation { bool feasible=false; int crossings=0; std::string reason; bool certified() const { return feasible && crossings==0; } };
Validation validate(const Instance&, const Allocation&);
struct Graph { std::vector<std::vector<float>> slots, receivers, edges; std::vector<int> source, target; };
Graph build_graph(const Instance&);
}
