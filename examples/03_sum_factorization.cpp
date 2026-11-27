#include <keisan/keisan.hpp>
#include <iostream>
#include <vector>

int main() {
    constexpr std::size_t N = 4;
    std::vector<double> D(N * N, 0.0), in(N * N * N, 1.0), tmp(N * N * N), out(N * N * N);
    for (std::size_t i = 0; i < N; ++i) D[i * N + i] = 2.0;

    // 3D tensor-product sum-factorization along X, Y, Z
    keisan::contract_axis<keisan::Axis::X, N>(D.data(), in.data(), out.data(), N, N, N);
    keisan::contract_axis<keisan::Axis::Y, N>(D.data(), out.data(), tmp.data(), N, N, N);
    keisan::contract_axis<keisan::Axis::Z, N>(D.data(), tmp.data(), out.data(), N, N, N);

    std::cout << "contracted out[0]=" << out[0] << "\n";
    return 0;
}
