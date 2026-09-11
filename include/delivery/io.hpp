#pragma once
#include "delivery/ga.hpp"
#include <filesystem>
namespace delivery {
struct Record { Instance instance; Allocation label; std::uint64_t seed=0; int budget=0; double seconds=0; };
void write_record(const std::filesystem::path&,const Record&);
Record read_record(const std::filesystem::path&);
std::vector<Record> read_dataset(const std::filesystem::path&,bool certified_only=true);
Instance generate_instance(std::uint64_t seed,int trucks,int receivers,int types=2);
GAConfig read_config(const std::filesystem::path&);
void write_allocation(const std::filesystem::path&,const Instance&,const Allocation&);
}
