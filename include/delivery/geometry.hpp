#pragma once
#include "delivery/domain.hpp"
namespace delivery {
// Proper interior intersection; shared endpoints and collinear overlaps are allowed.
bool interior_crossing(Point a, Point b, Point c, Point d);
}
