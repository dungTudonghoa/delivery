#include "delivery/geometry.hpp"
#include "delivery/graph.hpp"
#include "delivery/io.hpp"
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <numbers>
#include <limits>
#define CHECK(x) do { if(!(x)) throw std::runtime_error(std::string("failed: ")+ #x); } while(false)
template<class F> void rejects(F f){bool caught=false;try{f();}catch(const std::exception&){caught=true;}CHECK(caught);}
delivery::Instance fixture(){return {"cross",1,{{{0,0},0,10},{{2,0},0,10}},{{0,0},{0,0},{1,0},{1,0}},{{{0,2},{2}},{{2,2},{2}}}};}
int main(){try{using namespace delivery;
 auto i=fixture();check_instance(i);CHECK(eligible(i,0,0));i.trucks[0].min_range=3;CHECK(!eligible(i,0,0));i.trucks[0].min_range=0;i.trucks[0].max_range=1;CHECK(!eligible(i,0,0));i.trucks[0].max_range=10;i.trucks[0].half_angle=0.1;CHECK(!eligible(i,0,0));i=fixture();i.receivers[0].demand[0]=0;CHECK(!eligible(i,0,0));i=fixture();
 CHECK(interior_crossing({0,0},{2,2},{2,0},{0,2}));CHECK(!interior_crossing({0,0},{2,2},{0,0},{2,0}));CHECK(!interior_crossing({0,0},{2,2},{0,0},{2,2}));CHECK(!interior_crossing({0,0},{2,0},{1,0},{3,0}));CHECK(!interior_crossing({0,0},{2,0},{1,0},{1,1}));
 CHECK(validate(i,{1,1,0,0}).crossings==1);CHECK(validate(i,{0,0,1,1}).certified());CHECK(!validate(i,{0,0,0,1}).feasible);CHECK(!validate(i,{0,0,1}).feasible);CHECK(!validate(i,{0,0,1,2}).feasible);CHECK(!validate(i,{0,0,1,-2}).feasible);
 i.slots.push_back({0,0});CHECK(validate(i,{0,0,1,1,-1}).certified());auto g=build_graph(i);CHECK(g.slots.size()==5&&g.receivers.size()==2&&g.edges.size()==10);
 GAConfig c;c.budget=50;c.max_proposals=200;auto x=run_ga(i,c,42),y=run_ga(i,c,42);CHECK(x.validation.certified());CHECK(x.allocation==y.allocation&&x.evaluations==y.evaluations&&x.proposals==y.proposals);CHECK(x.evaluations<=c.budget);
 Prior p(i.slots.size(),std::vector<double>(3,1.0/3));for(bool mutation:{false,true}){c.guided_mutation=mutation;auto z=run_ga(i,c,42,&p);CHECK(z.validation.certified());CHECK(z.evaluations<=c.budget);}
 p[0][0]=std::numeric_limits<double>::quiet_NaN();rejects([&]{run_ga(i,c,42,&p);});
 // Hall bottleneck: total supply is adequate but no full matching exists.
 auto impossible=fixture();impossible.trucks[1].max_range=0;CHECK(!run_ga(impossible,c,42).validation.feasible);
 // Augmenting path: flexible slot must yield receiver 0 to the restricted slot.
 Instance hall{"hall",1,{{{0,0},0,10},{{0,0},0,1}},{{0,0},{1,0}},{{{0,1},{1}},{{2,0},{1}}}};
 for(int seed=0;seed<20;++seed){auto h=run_ga(hall,c,seed);CHECK(h.validation.certified());CHECK(h.allocation==Allocation({1,0}));}
 // A forced crossing can never be certified even if every hard constraint holds.
 Instance forced{"forced",2,{{{0,0},0,10},{{2,0},0,10}},{{0,0},{1,1}},{{{2,2},{1,0}},{{0,2},{0,1}}}};
 auto f=run_ga(forced,c,42);CHECK(f.validation.feasible&&!f.validation.certified());CHECK(f.validation.crossings==1&&f.evaluations==1&&f.first_zero==-1);
 auto tmp=std::filesystem::temp_directory_path()/"delivery-core-test.delivery";Record rec{i,x.allocation,42,50,0};write_record(tmp,rec);auto restored=read_record(tmp);CHECK(restored.label==x.allocation);CHECK(restored.instance.id==i.id);std::filesystem::remove(tmp);
 rejects([&]{write_record(tmp,{forced,f.allocation,42,50,0});});
 for(int seed=0;seed<10;++seed){auto generated=generate_instance(seed,3+seed%3,3+seed%4);check_instance(generated);auto result=run_ga(generated,c,seed);if(result.validation.feasible)CHECK(validate(generated,result.allocation).feasible);CHECK(result.evaluations<=c.budget);}
 std::cout<<"core: all checks passed\n";
}catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 1;}}
