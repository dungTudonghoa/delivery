#include "delivery/geometry.hpp"
namespace delivery {
static long double orient(Point a,Point b,Point c) { return (static_cast<long double>(b.x)-a.x)*(static_cast<long double>(c.y)-a.y)-(static_cast<long double>(b.y)-a.y)*(static_cast<long double>(c.x)-a.x); }
bool interior_crossing(Point a,Point b,Point c,Point d) {
 // Proper interior intersections only: shared endpoints, collinear overlaps excluded.
 auto o1=orient(a,b,c),o2=orient(a,b,d),o3=orient(c,d,a),o4=orient(c,d,b);
 return ((o1>0&&o2<0)||(o1<0&&o2>0))&&((o3>0&&o4<0)||(o3<0&&o4>0));
}
}
