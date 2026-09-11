#include "delivery/domain.hpp"
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
static long double orient(Point a,Point b,Point c) { return (static_cast<long double>(b.x)-a.x)*(static_cast<long double>(c.y)-a.y)-(static_cast<long double>(b.y)-a.y)*(static_cast<long double>(c.x)-a.x); }
bool interior_crossing(Point a,Point b,Point c,Point d) {
 // Proper interior intersections only: shared endpoints, collinear overlaps excluded.
 auto o1=orient(a,b,c),o2=orient(a,b,d),o3=orient(c,d,a),o4=orient(c,d,b);
 return ((o1>0&&o2<0)||(o1<0&&o2>0))&&((o3>0&&o4<0)||(o3<0&&o4>0));
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
Graph build_graph(const Instance& i) {
 check_instance(i); Graph g; double scale=1;
 for(auto& t:i.trucks) scale=std::max({scale,std::abs(t.position.x),std::abs(t.position.y),t.max_range});
 for(auto& r:i.receivers) scale=std::max({scale,std::abs(r.position.x),std::abs(r.position.y)});
 for(auto s:i.slots) {auto& t=i.trucks[s.truck]; std::vector<float> f{float(t.position.x/scale),float(t.position.y/scale),float(t.min_range/scale),float(t.max_range/scale),float(std::sin(t.heading)),float(std::cos(t.heading)),float(t.half_angle/std::numbers::pi)}; for(int k=0;k<i.types;++k) f.push_back(k==s.type?1.f:0.f); g.slots.push_back(f);}
 for(auto& r:i.receivers) {std::vector<float> f{float(r.position.x/scale),float(r.position.y/scale)};for(int n:r.demand) f.push_back(float(n)/i.slots.size());g.receivers.push_back(f);}
 for(int s=0;s<int(i.slots.size());++s) for(int r=0;r<int(i.receivers.size());++r) if(eligible(i,s,r)) {
  auto& t=i.trucks[i.slots[s].truck]; auto p=i.receivers[r].position;double dx=p.x-t.position.x,dy=p.y-t.position.y,d=std::hypot(dx,dy),b=std::atan2(dy,dx)-t.heading;
  g.source.push_back(s);g.target.push_back(r);g.edges.push_back({float(dx/scale),float(dy/scale),float(d/scale),float(d/std::max(t.max_range,1e-9)),float(std::sin(b)),float(std::cos(b))});
 } return g;
}
}
