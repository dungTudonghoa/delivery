#include "delivery/graph.hpp"
#include <cmath>
#include <algorithm>
#include <numbers>
namespace delivery {
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
