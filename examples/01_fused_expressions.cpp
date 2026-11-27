#include <keisan/keisan.hpp>
#include <iostream>

int main() {
    using namespace keisan::ops;

    keisan::aligned_vector<double> a(1024, 2.0), b(1024, 3.0), c(1024, 1.5), out(1024);

    // Lazy expression AST: zero temporary allocations, single fused pass
    keisan::assign(out, keisan::clamp(keisan::fma(a, b, -c), 0.0, 10.0));
    out += 0.5 * c;

    std::cout << "out[0]=" << out[0] << " norm=" << keisan::norm(out) << "\n";
    return 0;
}
