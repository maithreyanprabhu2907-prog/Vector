#include <keisan/keisan.hpp>
#include <iostream>
#include <span>

int main() {
    using namespace keisan::ops;

    constexpr std::size_t n = 512;
    keisan::aligned_vector<double> u(n, 1.0), rhs(n, 0.25);

    // Pre-allocate once before the time loop
    keisan::WorkspaceArena arena(64 * 1024);

    for (int step = 0; step < 3; ++step) {
        arena.reset();
        std::span<double> k1 = arena.allocate<double>(n);
        std::span<double> tmp = arena.allocate<double>(n);

        keisan::eval_into(k1.data(), n, rhs * 2.0);
        keisan::eval_into(tmp.data(), n, u + 0.5 * k1);
        u += 0.1 * tmp;
    }

    std::cout << "after 3 RK steps u[0]=" << u[0] << "\n";
    return 0;
}
