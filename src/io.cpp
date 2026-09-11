#include "delivery/io.hpp"
#include <fstream>
#include <iomanip>
#include <random>
#include <algorithm>
#include <stdexcept>
#include <numbers>
namespace delivery {
void write_record(const std::filesystem::path& path,const Record& v){
 check_instance(v.instance);if(!v.label.empty()&&!validate(v.instance,v.label).certified())throw std::invalid_argument("refusing uncertified label");
 if(!path.parent_path().empty())std::filesystem::create_directories(path.parent_path());
 std::ofstream o(path);if(!o)throw std::runtime_error("cannot write "+path.string());auto&i=v.instance;
 o<<std::setprecision(17)<<"DELIVERY_V1\n"<<std::quoted(i.id)<<' '<<i.types<<' '<<i.trucks.size()<<' '<<i.slots.size()<<' '<<i.receivers.size()<<'\n';
 for(auto&t:i.trucks)o<<t.position.x<<' '<<t.position.y<<' '<<t.min_range<<' '<<t.max_range<<' '<<t.heading<<' '<<t.half_angle<<'\n';
 for(auto s:i.slots)o<<s.truck<<' '<<s.type<<'\n';
 for(auto&r:i.receivers){o<<r.position.x<<' '<<r.position.y;for(int n:r.demand)o<<' '<<n;o<<'\n';}
 o<<v.seed<<' '<<v.budget<<' '<<v.seconds<<' '<<v.label.size()<<'\n';for(int r:v.label)o<<r<<' ';o<<'\n';if(!o)throw std::runtime_error("record write failed");
}
Record read_record(const std::filesystem::path&path){
 std::ifstream f(path);Record v;std::string magic;int T=0,S=0,R=0;f>>magic>>std::quoted(v.instance.id)>>v.instance.types>>T>>S>>R;
 if(!f||magic!="DELIVERY_V1"||T<1||S<1||R<1||T>100000||S>100000||R>100000||v.instance.types<1||v.instance.types>1024)throw std::runtime_error("invalid record header: "+path.string());
 auto&i=v.instance;i.trucks.resize(T);i.slots.resize(S);i.receivers.resize(R);
 for(auto&t:i.trucks)f>>t.position.x>>t.position.y>>t.min_range>>t.max_range>>t.heading>>t.half_angle;
 for(auto&s:i.slots)f>>s.truck>>s.type;
 for(auto&r:i.receivers){f>>r.position.x>>r.position.y;r.demand.resize(i.types);for(int&n:r.demand)f>>n;}
 int N=-1;f>>v.seed>>v.budget>>v.seconds>>N;if(!f||(N!=0&&N!=S))throw std::runtime_error("invalid label size");v.label.resize(N);for(int&r:v.label)f>>r;
 if(!f)throw std::runtime_error("truncated record");
 std::string tail;if(f>>tail)throw std::runtime_error("trailing record data");check_instance(i);
 if(!v.label.empty()&&!validate(i,v.label).certified())throw std::runtime_error("uncertified record label");
 return v;
}
std::vector<Record> read_dataset(const std::filesystem::path&dir,bool certified_only){
 std::vector<std::filesystem::path> paths;for(auto&e:std::filesystem::directory_iterator(dir))if(e.path().extension()==".delivery")paths.push_back(e.path());std::sort(paths.begin(),paths.end());std::vector<Record> records;
 for(auto&p:paths){auto r=read_record(p);if(!certified_only||!r.label.empty())records.push_back(std::move(r));}if(records.empty())throw std::runtime_error("empty dataset: "+dir.string());return records;
}
Instance generate_instance(std::uint64_t seed,int T,int R,int K){
 if(T<1||R<1||K<1)throw std::invalid_argument("generator dimensions");
 std::mt19937_64 rng(seed);std::uniform_real_distribution<double> u(0,1);Instance i;i.id="synthetic-"+std::to_string(seed);i.types=K;
 // Two parallel banks plus jitter: admits crossing and non-crossing configurations.
 for(int t=0;t<T;++t)i.trucks.push_back({{100.0*t/std::max(1,T-1),5*u(rng)},0,150,0,std::numbers::pi});
 for(int r=0;r<R;++r){ReceivePoint p{{100.0*r/std::max(1,R-1),80+5*u(rng)},std::vector<int>(K)};p.demand[rng()%K]=1;i.receivers.push_back(p);}
 // Every truck has one slot per type plus additional random stock: surplus is intentional.
 for(int t=0;t<T;++t)for(int k=0;k<K;++k)i.slots.push_back({t,k});
 for(int n=0;n<R;++n)i.slots.push_back({int(rng()%T),int(rng()%K)});
 return i;
}
GAConfig read_config(const std::filesystem::path&path){
 GAConfig c;std::ifstream f(path);if(!f)throw std::runtime_error("cannot open config");std::string key;double value;
 while(f>>key){if(key.starts_with('#')){std::getline(f,key);continue;}if(!(f>>value))throw std::runtime_error("invalid config value");
 auto integer=[&](){if(!std::isfinite(value)||value!=std::floor(value)||value<0||value>100000000)throw std::runtime_error("invalid integer config");return int(value);};
 if(key=="population")c.population=integer();else if(key=="budget")c.budget=integer();else if(key=="max_proposals")c.max_proposals=integer();else if(key=="mutation")c.mutation=value;else if(key=="guided_fraction")c.guided_fraction=value;else if(key=="guidance")c.guidance=value;else if(key=="guided_mutation"){if(value!=0&&value!=1)throw std::runtime_error("guided_mutation must be 0/1");c.guided_mutation=value!=0;}else throw std::runtime_error("unknown config key: "+key);
 }return c;
}
void write_allocation(const std::filesystem::path&p,const Instance&i,const Allocation&a){auto v=validate(i,a);if(!v.feasible)throw std::invalid_argument("invalid final allocation");std::ofstream f(p);if(!f)throw std::runtime_error("cannot write allocation");f<<"slot,truck,type,receiver\n";for(size_t s=0;s<a.size();++s)f<<s<<','<<i.slots[s].truck<<','<<i.slots[s].type<<','<<a[s]<<'\n';if(!f)throw std::runtime_error("allocation write failed");}
}
