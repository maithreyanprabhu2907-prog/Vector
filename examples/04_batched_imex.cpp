#include <keisan/keisan.hpp>
#include <array>
#include <iostream>
#include <vector>

int main() {
    constexpr std::size_t N = 3;
    constexpr std::size_t E = 16;

    std::vector<double> A(N * N * E, 0.0), b(N * E, 0.0), x(N * E, 0.0);
    std::array<const double*, N * N> Ap{};
    std::array<const double*, N> bp{};
    std::array<double*, N> xp{};

    for (std::size_t i = 0; i < N * N; ++i) Ap[i] = A.data() + i * E;
    for (std::size_t i = 0; i < N; ++i) {
        bp[i] = b.data() + i * E;
        xp[i] = x.data() + i * E;
    }

    for (std::size_t e = 0; e < E; ++e) {
        for (std::size_t r = 0; r < N; ++r) {
            A[(r * N + r) * E + e] = 4.0 + static_cast<double>(r);
            b[r * E + e] = (4.0 + static_cast<double>(r)) * static_cast<double>(r + 1);
        }
    }

    keisan::solve_nxn_batched<N>(Ap.data(), bp.data(), xp.data(), E);
    std::cout << "x[0][0]=" << x[0] << " x[2][0]=" << x[2 * E] << "\n";
    return 0;
}
