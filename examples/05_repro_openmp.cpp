#include <keisan/keisan.hpp>
#include <iostream>
#include <vector>

int main() {
    std::vector<double> x(4096, 0.1), y(4096, 0.2);

    double fast_dot  = keisan::dot(x, y, keisan::ReductionPolicy::FAST);
    double repro_dot = keisan::dot_repro(x, y);
    double par_dot   = keisan::dot_par(x, y);

    std::cout << "fast=" << fast_dot << " repro=" << repro_dot << " par=" << par_dot << "\n";
    return 0;
}
