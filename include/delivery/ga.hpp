#pragma once
#include "delivery/domain.hpp"
namespace delivery {
struct GAConfig { int population=48, budget=1000, max_proposals=20000; double mutation=0.15, guided_fraction=0.5, guidance=0.8; bool guided_mutation=true; };
struct GAResult { Allocation allocation; Validation validation; int evaluations=0, proposals=0, repairs=0, failures=0, first_zero=-1; double seconds=0; };
GAResult run_ga(const Instance&, const GAConfig&, std::uint64_t seed, const Prior* prior=nullptr);
}
