#include "delivery/ga.hpp"
#include <algorithm>
#include <chrono>
#include <functional>
#include <numeric>
#include <random>
#include <set>
#include <stdexcept>
#include <cmath>
namespace delivery {
GAResult run_ga(const Instance& i,const GAConfig& c,std::uint64_t seed,const Prior* p) {
 check_instance(i);
 auto probability=[](double x){return std::isfinite(x)&&x>=0&&x<=1;};
 if(c.population<2||c.budget<1||c.max_proposals<1||!probability(c.mutation)||!probability(c.guided_fraction)||!probability(c.guidance)||c.guidance>=1) throw std::invalid_argument("invalid GA configuration (guidance must be <1)");
 const int S=i.slots.size(),R=i.receivers.size();
 if(p) {if(int(p->size())!=S) throw std::invalid_argument("prior slot dimension");for(int s=0;s<S;++s) {auto& row=p->at(s);if(int(row.size())!=R+1)throw std::invalid_argument("prior class dimension");double sum=0;for(int r=0;r<=R;++r){if(!probability(row[r])||(r<R&&!eligible(i,s,r)&&row[r]!=0))throw std::invalid_argument("invalid prior value/mask");sum+=row[r];}if(std::abs(sum-1)>1e-5)throw std::invalid_argument("prior normalization");}}
 std::mt19937_64 rng(seed); std::uniform_real_distribution<double> u(0,1); auto start=std::chrono::steady_clock::now(); GAResult out;
 auto sample=[&](int s,bool guided) {std::vector<double>w(R+1);int n=1;for(int r=0;r<R;++r)n+=eligible(i,s,r);for(int r=0;r<=R;++r)if(r==R||eligible(i,s,r))w[r]=(guided&&p)?c.guidance*p->at(s)[r]+(1-c.guidance)/n:1.0/n;int r=std::discrete_distribution<int>(w.begin(),w.end())(rng);return r==R?-1:r;};
 // Demand-token bipartite matching repairs proposals. Augmenting paths avoid greedy dead ends.
 auto repair=[&](Allocation& a) {
  std::vector<int> token_r,token_k;
  for(int r=0;r<R;++r)for(int k=0;k<i.types;++k)for(int n=0;n<i.receivers[r].demand[k];++n){token_r.push_back(r);token_k.push_back(k);}
  if(token_r.size()>i.slots.size())return false;
  std::vector<std::vector<int>> choices(token_r.size());
  for(int t=0;t<int(token_r.size());++t){auto& v=choices[t];for(int s=0;s<S;++s)if(i.slots[s].type==token_k[t]&&eligible(i,s,token_r[t]))v.push_back(s);std::shuffle(v.begin(),v.end(),rng);std::stable_sort(v.begin(),v.end(),[&](int x,int y){return (a[x]==token_r[t])>(a[y]==token_r[t]);});}
  std::vector<int> owner(S,-1),order(token_r.size());std::iota(order.begin(),order.end(),0);std::shuffle(order.begin(),order.end(),rng);
  std::function<bool(int,std::vector<bool>&)> augment=[&](int t,std::vector<bool>&seen){for(int s:choices[t])if(!seen[s]){seen[s]=true;if(owner[s]<0||augment(owner[s],seen)){owner[s]=t;return true;}}return false;};
  for(int t:order){std::vector<bool>seen(S);if(!augment(t,seen))return false;}
  Allocation fixed(S,-1);for(int s=0;s<S;++s)if(owner[s]>=0)fixed[s]=token_r[owner[s]];if(fixed!=a)++out.repairs;a=std::move(fixed);return true;
 };
 struct Individual{Allocation a;int score;};std::vector<Individual> pop;std::set<Allocation> seen;
 auto select=[&]()->const Allocation&{int x=rng()%pop.size(),y=rng()%pop.size();return pop[pop[x].score<pop[y].score?x:y].a;};
 while(out.evaluations<c.budget&&out.proposals<c.max_proposals){
  ++out.proposals;Allocation a(S,-1);
  if(int(pop.size())<c.population){bool guided=p&&u(rng)<c.guided_fraction;for(int s=0;s<S;++s)a[s]=sample(s,guided);}
  else {auto x=select(),y=select();for(int s=0;s<S;++s){a[s]=u(rng)<0.5?x[s]:y[s];if(u(rng)<c.mutation)a[s]=sample(s,p&&c.guided_mutation);}}
  if(!repair(a)){++out.failures;break;} // Matching proves demand infeasible on the eligibility graph.
  if(!seen.insert(a).second)continue;
  auto v=validate(i,a);if(!v.feasible){++out.failures;continue;}++out.evaluations;
  if(out.allocation.empty()||v.crossings<out.validation.crossings){out.allocation=a;out.validation=v;}
  if(v.certified()){out.first_zero=out.evaluations;break;}
  pop.push_back({a,v.crossings});std::stable_sort(pop.begin(),pop.end(),[](auto&x,auto&y){return x.score<y.score;});if(int(pop.size())>c.population)pop.pop_back();
 }
 if(out.allocation.empty())out.validation={false,0,"no feasible allocation found"};
 out.seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();return out;
}
}
