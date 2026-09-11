#include "delivery/domain.hpp"
#include "delivery/geometry.hpp"
#include <cmath>
#include <numbers>
#include <set>
#include <stdexcept>
#include <algorithm>
namespace delivery {
static bool finite(Point p) { return std::isfinite(p.x)&&std::isfinite(p.y); }
void check_instance(const Instance& i) {
 if(i.types<=0 || i.trucks.empty() || i.slots.empty() || i.receivers.empty()) throw std::invalid_argument("empty instance or invalid type count");
 for(auto& t:i.trucks) if(!finite(t.position)||!std::isfinite(t.min_range)||!std::isfinite(t.max_range)||!std::isfinite(t.heading)||!std::isfinite(t.half_angle)||t.min_range<0||t.max_range<t.min_range||t.half_angle<0||t.half_angle>std::numbers::pi) throw std::invalid_argument("invalid truck");
 for(auto s:i.slots) if(s.truck<0||s.truck>=int(i.trucks.size())||s.type<0||s.type>=i.types) throw std::invalid_argument("invalid slot");
 for(auto& r:i.receivers) if(!finite(r.position)||int(r.demand.size())!=i.types||std::any_of(r.demand.begin(),r.demand.end(),[](int n){return n<0;})) throw std::invalid_argument("invalid receiver");
}
bool eligible(const Instance& i,int s,int r) {
 const auto& slot=i.slots.at(s); const auto& t=i.trucks.at(slot.truck); const auto& p=i.receivers.at(r);
 const double dx=p.position.x-t.position.x,dy=p.position.y-t.position.y,d=std::hypot(dx,dy);
 const double angle=std::abs(std::remainder(std::atan2(dy,dx)-t.heading,2*std::numbers::pi));
 return p.demand.at(slot.type)>0 && d>=t.min_range-1e-9 && d<=t.max_range+1e-9 && (d<1e-9||angle<=t.half_angle+1e-9);
}
Validation validate(const Instance& i,const Allocation& a) {
 check_instance(i);
 if(a.size()!=i.slots.size()) return {false,0,"allocation size"};
 std::vector<std::vector<int>> counts(i.receivers.size(),std::vector<int>(i.types)); std::set<std::pair<int,int>> routes;
 for(int s=0;s<int(a.size());++s) { int r=a[s]; if(r==-1) continue;
  if(r<0||r>=int(i.receivers.size())||!eligible(i,s,r)) return {false,0,"ineligible destination"};
  ++counts[r][i.slots[s].type]; routes.emplace(i.slots[s].truck,r);
 }
 for(size_t r=0;r<counts.size();++r) if(counts[r]!=i.receivers[r].demand) return {false,0,"demand mismatch"};
 int crossings=0;
 for(auto x=routes.begin();x!=routes.end();++x) for(auto y=std::next(x);y!=routes.end();++y)
  crossings+=interior_crossing(i.trucks[x->first].position,i.receivers[x->second].position,i.trucks[y->first].position,i.receivers[y->second].position);
 return {true,crossings,""};
}
}
