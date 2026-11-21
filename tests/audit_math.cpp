#include <vec_math/vector_math.hpp>
#include <iostream>
#include <vector>
#include <span>
#include <array>
#include <cmath>
#include <cassert>
#include <iomanip>
#include <numeric>
#include <random>

#define AUDIT_ASSERT(cond, msg) \
    do { \
        if (!(cond)) { \
            std::cerr << "FAILED: " << msg << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
            std::exit(1); \
        } \
    } while (false)

void audit_contract_axis() {
    std::cout << "[AUDIT] 1. Testing contract_axis..." << std::endl;
    using namespace vec_math;

    // Helper to evaluate tensor indexing (X-fastest): index(x, y, z) = x + y*nx + z*nx*ny
    auto idx3d = [](std::size_t x, std::size_t y, std::size_t z, std::size_t nx, std::size_t ny) {
        return x + y * nx + z * nx * ny;
    };

    // Test anisotropic sizes
    std::vector<std::tuple<std::size_t, std::size_t, std::size_t>> test_dims = {
        {1, 1, 1},
        {1, 4, 3},
        {4, 1, 3},
        {4, 3, 1},
        {3, 4, 5},
        {5, 2, 4},
        {8, 8, 8}
    };

    for (auto [nx, ny, nz] : test_dims) {
        const std::size_t total = nx * ny * nz;

        // --- Axis 0 (X) ---
        {
            std::vector<double> D(nx * nx);
            for (std::size_t i = 0; i < nx; ++i)
                for (std::size_t j = 0; j < nx; ++j)
                    D[i * nx + j] = (i == j) ? 2.0 : 0.5 * (i + j + 1);

            std::vector<double> in(total);
            for (std::size_t i = 0; i < total; ++i) in[i] = static_cast<double>(i + 1);

            std::vector<double> out_ref(total, 0.0);
            for (std::size_t z = 0; z < nz; ++z) {
                for (std::size_t y = 0; y < ny; ++y) {
                    for (std::size_t x = 0; x < nx; ++x) {
                        double sum = 0.0;
                        for (std::size_t q = 0; q < nx; ++q) {
                            sum += D[x * nx + q] * in[idx3d(q, y, z, nx, ny)];
                        }
                        out_ref[idx3d(x, y, z, nx, ny)] = sum;
                    }
                }
            }

            std::vector<double> out_test(total, 0.0);
            contract_axis<Axis::X>(D.data(), in.data(), out_test.data(), nx, ny, nz);

            for (std::size_t i = 0; i < total; ++i) {
                AUDIT_ASSERT(std::abs(out_test[i] - out_ref[i]) < 1e-12, "Axis 0 mismatch");
            }

            // Test aliased in-place
            std::vector<double> in_out = in;
            contract_axis<Axis::X>(D.data(), in_out.data(), in_out.data(), nx, ny, nz);
            for (std::size_t i = 0; i < total; ++i) {
                AUDIT_ASSERT(std::abs(in_out[i] - out_ref[i]) < 1e-12, "Axis 0 aliased mismatch");
            }
        }

        // --- Axis 1 (Y) ---
        {
            std::vector<double> D(ny * ny);
            for (std::size_t i = 0; i < ny; ++i)
                for (std::size_t j = 0; j < ny; ++j)
                    D[i * ny + j] = (i == j) ? 3.0 : 0.25 * (i - j + 2);

            std::vector<double> in(total);
            for (std::size_t i = 0; i < total; ++i) in[i] = static_cast<double>(i + 1);

            std::vector<double> out_ref(total, 0.0);
            for (std::size_t z = 0; z < nz; ++z) {
                for (std::size_t y = 0; y < ny; ++y) {
                    for (std::size_t x = 0; x < nx; ++x) {
                        double sum = 0.0;
                        for (std::size_t q = 0; q < ny; ++q) {
                            sum += D[y * ny + q] * in[idx3d(x, q, z, nx, ny)];
                        }
                        out_ref[idx3d(x, y, z, nx, ny)] = sum;
                    }
                }
            }

            std::vector<double> out_test(total, 0.0);
            contract_axis<Axis::Y>(D.data(), in.data(), out_test.data(), nx, ny, nz);

            for (std::size_t i = 0; i < total; ++i) {
                AUDIT_ASSERT(std::abs(out_test[i] - out_ref[i]) < 1e-12, "Axis 1 mismatch");
            }

            // Test aliased in-place
            std::vector<double> in_out = in;
            contract_axis<Axis::Y>(D.data(), in_out.data(), in_out.data(), nx, ny, nz);
            for (std::size_t i = 0; i < total; ++i) {
                AUDIT_ASSERT(std::abs(in_out[i] - out_ref[i]) < 1e-12, "Axis 1 aliased mismatch");
            }
        }

        // --- Axis 2 (Z) ---
        {
            std::vector<double> D(nz * nz);
            for (std::size_t i = 0; i < nz; ++i)
                for (std::size_t j = 0; j < nz; ++j)
                    D[i * nz + j] = (i == j) ? 1.5 : 0.1 * (i + 2 * j);

            std::vector<double> in(total);
            for (std::size_t i = 0; i < total; ++i) in[i] = static_cast<double>(i + 1);

            std::vector<double> out_ref(total, 0.0);
            for (std::size_t z = 0; z < nz; ++z) {
                for (std::size_t y = 0; y < ny; ++y) {
                    for (std::size_t x = 0; x < nx; ++x) {
                        double sum = 0.0;
                        for (std::size_t q = 0; q < nz; ++q) {
                            sum += D[z * nz + q] * in[idx3d(x, y, q, nx, ny)];
                        }
                        out_ref[idx3d(x, y, z, nx, ny)] = sum;
                    }
                }
            }

            std::vector<double> out_test(total, 0.0);
            contract_axis<Axis::Z>(D.data(), in.data(), out_test.data(), nx, ny, nz);

            for (std::size_t i = 0; i < total; ++i) {
                AUDIT_ASSERT(std::abs(out_test[i] - out_ref[i]) < 1e-12, "Axis 2 mismatch");
            }

            // Test aliased in-place
            std::vector<double> in_out = in;
            contract_axis<Axis::Z>(D.data(), in_out.data(), in_out.data(), nx, ny, nz);
            for (std::size_t i = 0; i < total; ++i) {
                AUDIT_ASSERT(std::abs(in_out[i] - out_ref[i]) < 1e-12, "Axis 2 aliased mismatch");
            }
        }
    }

    // Static template parameter N tests (N > 0)
    {
        constexpr std::size_t N = 4;
        std::vector<double> D(N * N, 0.5);
        for (std::size_t i = 0; i < N; ++i) D[i * N + i] = 2.0;

        std::size_t nx = 4, ny = 3, nz = 2;
        std::vector<double> in(nx * ny * nz, 1.0);
        std::vector<double> out(nx * ny * nz, 0.0);
        contract_axis<Axis::X, N>(D.data(), in.data(), out.data(), nx, ny, nz);
        for (auto v : out) AUDIT_ASSERT(std::abs(v - 3.5) < 1e-12, "Static N Axis X failed");

        nx = 3; ny = 4; nz = 2;
        in.assign(nx * ny * nz, 1.0);
        out.assign(nx * ny * nz, 0.0);
        contract_axis<Axis::Y, N>(D.data(), in.data(), out.data(), nx, ny, nz);
        for (auto v : out) AUDIT_ASSERT(std::abs(v - 3.5) < 1e-12, "Static N Axis Y failed");

        nx = 3; ny = 2; nz = 4;
        in.assign(nx * ny * nz, 1.0);
        out.assign(nx * ny * nz, 0.0);
        contract_axis<Axis::Z, N>(D.data(), in.data(), out.data(), nx, ny, nz);
        for (auto v : out) AUDIT_ASSERT(std::abs(v - 3.5) < 1e-12, "Static N Axis Z failed");
    }

    // Large buffer aliasing test (> 1024 elements) to check heap fallback
    {
        std::size_t nx = 16, ny = 16, nz = 8; // total = 2048 > 1024
        std::vector<double> D(nx * nx, 0.1);
        for (std::size_t i = 0; i < nx; ++i) D[i * nx + i] = 1.0;
        std::vector<double> in_out(nx * ny * nz, 1.0);
        contract_axis<Axis::X>(D.data(), in_out.data(), in_out.data(), nx, ny, nz);
        for (auto v : in_out) {
            AUDIT_ASSERT(std::abs(v - 2.5) < 1e-12, "Large aliased contract_axis failed");
        }
    }
    std::cout << "  -> contract_axis: PASSED all tests!" << std::endl;
}

void audit_matvec() {
    std::cout << "[AUDIT] 2. Testing matvec & matvec_fixed..." << std::endl;
    using namespace vec_math;

    std::vector<std::pair<std::size_t, std::size_t>> dims = {
        {1, 1}, {1, 5}, {5, 1}, {3, 3}, {4, 4}, {5, 5}, {7, 13}, {13, 7},
        {36, 36}, {37, 37}, {40, 10}, {10, 40}, {130, 20}
    };

    for (auto [rows, cols] : dims) {
        std::vector<double> A(rows * cols);
        std::vector<double> x(cols);
        std::vector<double> y_ref(rows, 0.0);
        std::vector<double> y_test(rows, 0.0);

        for (std::size_t i = 0; i < A.size(); ++i) A[i] = std::sin(static_cast<double>(i + 1));
        for (std::size_t i = 0; i < x.size(); ++i) x[i] = std::cos(static_cast<double>(i + 1));

        // Ref matvec
        for (std::size_t r = 0; r < rows; ++r) {
            double sum = 0.0;
            for (std::size_t c = 0; c < cols; ++c) {
                sum += A[r * cols + c] * x[c];
            }
            y_ref[r] = sum;
        }

        matvec(A.data(), x.data(), y_test.data(), rows, cols);
        for (std::size_t r = 0; r < rows; ++r) {
            AUDIT_ASSERT(std::abs(y_test[r] - y_ref[r]) < 1e-12, "matvec value mismatch");
        }

        // Aliased test for square matrices
        if (rows == cols) {
            std::vector<double> x_aliased = x;
            matvec(A.data(), x_aliased.data(), x_aliased.data(), rows, cols);
            for (std::size_t r = 0; r < rows; ++r) {
                AUDIT_ASSERT(std::abs(x_aliased[r] - y_ref[r]) < 1e-12, "matvec aliased mismatch");
            }
        }

        // Transposed matvec
        std::vector<double> xt(rows);
        std::vector<double> yt_ref(cols, 0.0);
        std::vector<double> yt_test(cols, 0.0);
        for (std::size_t i = 0; i < xt.size(); ++i) xt[i] = std::cos(static_cast<double>(i + 1));

        for (std::size_t c = 0; c < cols; ++c) {
            double sum = 0.0;
            for (std::size_t r = 0; r < rows; ++r) {
                sum += A[r * cols + c] * xt[r];
            }
            yt_ref[c] = sum;
        }

        matvec_transposed(A.data(), xt.data(), yt_test.data(), rows, cols);
        for (std::size_t c = 0; c < cols; ++c) {
            AUDIT_ASSERT(std::abs(yt_test[c] - yt_ref[c]) < 1e-12, "matvec_transposed value mismatch");
        }

        if (rows == cols) {
            std::vector<double> xt_aliased = xt;
            matvec_transposed(A.data(), xt_aliased.data(), xt_aliased.data(), rows, cols);
            for (std::size_t c = 0; c < cols; ++c) {
                AUDIT_ASSERT(std::abs(xt_aliased[c] - yt_ref[c]) < 1e-12, "matvec_transposed aliased mismatch");
            }
        }
    }

    // Compile-time fixed tests
    {
        constexpr std::size_t M = 4, N = 6;
        double A[M * N], x[N], y[M], y_ref[M];
        for (int i = 0; i < M * N; ++i) A[i] = i * 0.5 + 1.0;
        for (int i = 0; i < N; ++i) x[i] = 2.0 - i * 0.1;
        for (int r = 0; r < M; ++r) {
            double sum = 0;
            for (int c = 0; c < N; ++c) sum += A[r * N + c] * x[c];
            y_ref[r] = sum;
        }
        matvec_fixed<M, N>(A, x, y);
        for (int r = 0; r < M; ++r) AUDIT_ASSERT(std::abs(y[r] - y_ref[r]) < 1e-12, "matvec_fixed mismatch");

        // Fixed square aliased
        double Asq[M * M], x_alias[M];
        for (int i = 0; i < M * M; ++i) Asq[i] = i * 0.25 + 0.5;
        for (int i = 0; i < M; ++i) x_alias[i] = i + 1.0;
        double ref_alias[M];
        for (int r = 0; r < M; ++r) {
            double sum = 0;
            for (int c = 0; c < M; ++c) sum += Asq[r * M + c] * x_alias[c];
            ref_alias[r] = sum;
        }
        matvec_fixed<M, M>(Asq, x_alias, x_alias);
        for (int r = 0; r < M; ++r) AUDIT_ASSERT(std::abs(x_alias[r] - ref_alias[r]) < 1e-12, "matvec_fixed aliased mismatch");
    }

    std::cout << "  -> matvec: PASSED all tests!" << std::endl;
}

void audit_solve_4x4() {
    std::cout << "[AUDIT] 3. Testing solve_4x4_batched..." << std::endl;
    using namespace vec_math;

    const std::size_t num_elements = 6;
    std::vector<std::vector<double>> A_soa_storage(16, std::vector<double>(num_elements));
    std::vector<std::vector<double>> b_soa_storage(4, std::vector<double>(num_elements));
    std::vector<std::vector<double>> x_soa_storage(4, std::vector<double>(num_elements, 0.0));

    std::vector<const double*> A_ptrs(16);
    std::vector<const double*> b_ptrs(4);
    std::vector<double*> x_ptrs(4);
    for (int i = 0; i < 16; ++i) A_ptrs[i] = A_soa_storage[i].data();
    for (int i = 0; i < 4; ++i) b_ptrs[i] = b_soa_storage[i].data();
    for (int i = 0; i < 4; ++i) x_ptrs[i] = x_soa_storage[i].data();

    // Element 0: Standard invertible matrix
    // [4, 1, 0, 0] [x0] = [5]  -> x = [1, 1, 1, 1]
    // [1, 4, 1, 0] [x1] = [6]
    // [0, 1, 4, 1] [x2] = [6]
    // [0, 0, 1, 4] [x3] = [5]
    double A0[4][4] = {
        {4, 1, 0, 0},
        {1, 4, 1, 0},
        {0, 1, 4, 1},
        {0, 0, 1, 4}
    };
    double b0[4] = {5, 6, 6, 5};
    for (int r = 0; r < 4; ++r) {
        for (int c = 0; c < 4; ++c) A_soa_storage[r * 4 + c][0] = A0[r][c];
        b_soa_storage[r][0] = b0[r];
    }

    // Element 1: Pivoting required (zero on diagonal)
    // [0, 2, 0, 1] [x] = [3]  sol: [1, 1, 1, 1] -> b = [3, 4, 1, 6]
    // [3, 0, 1, 0] [x] = [4]
    // [0, 0, 0, 1] [x] = [1]
    // [1, 2, 3, 0] [x] = [6]
    double A1[4][4] = {
        {0, 2, 0, 1},
        {3, 0, 1, 0},
        {0, 0, 0, 1},
        {1, 2, 3, 0}
    };
    double b1[4] = {3, 4, 1, 6};
    for (int r = 0; r < 4; ++r) {
        for (int c = 0; c < 4; ++c) A_soa_storage[r * 4 + c][1] = A1[r][c];
        b_soa_storage[r][1] = b1[r];
    }

    // Element 2: Singular matrix (all zeros)
    for (int r = 0; r < 4; ++r) {
        for (int c = 0; c < 4; ++c) A_soa_storage[r * 4 + c][2] = 0.0;
        b_soa_storage[r][2] = 1.0;
    }

    // Element 3: Rank-deficient matrix (row 2 = row 0 + row 1)
    double A3[4][4] = {
        {1, 2, 3, 4},
        {2, 1, 0, 1},
        {3, 3, 3, 5},
        {0, 1, 2, 3}
    };
    for (int r = 0; r < 4; ++r) {
        for (int c = 0; c < 4; ++c) A_soa_storage[r * 4 + c][3] = A3[r][c];
        b_soa_storage[r][3] = 1.0;
    }

    // Element 4: Near-zero diagonal / tiny entries below threshold 1e-20 (e.g. 1e-22)
    for (int r = 0; r < 4; ++r) {
        for (int c = 0; c < 4; ++c) A_soa_storage[r * 4 + c][4] = (r == c) ? 1e-22 : 0.0;
        b_soa_storage[r][4] = 1.0;
    }

    // Element 5: Ill-conditioned Hilbert 4x4
    // H_ij = 1.0 / (i + j + 1)
    double b5[4] = {0.0, 0.0, 0.0, 0.0};
    double x5_exact[4] = {1.0, 2.0, 3.0, 4.0};
    for (int r = 0; r < 4; ++r) {
        for (int c = 0; c < 4; ++c) {
            double h = 1.0 / (r + c + 1);
            A_soa_storage[r * 4 + c][5] = h;
            b5[r] += h * x5_exact[c];
        }
        b_soa_storage[r][5] = b5[r];
    }

    solve_4x4_batched(A_ptrs.data(), b_ptrs.data(), x_ptrs.data(), num_elements);

    // Verify Element 0: x = [1, 1, 1, 1]
    for (int r = 0; r < 4; ++r) {
        AUDIT_ASSERT(std::abs(x_soa_storage[r][0] - 1.0) < 1e-12, "Element 0 solution incorrect");
    }

    // Verify Element 1: x = [1, 1, 1, 1]
    for (int r = 0; r < 4; ++r) {
        AUDIT_ASSERT(std::abs(x_soa_storage[r][1] - 1.0) < 1e-12, "Element 1 pivot solution incorrect");
    }

    // Verify Element 2: All zeros matrix should produce sol = [0, 0, 0, 0] without NaN
    for (int r = 0; r < 4; ++r) {
        AUDIT_ASSERT(!std::isnan(x_soa_storage[r][2]) && !std::isinf(x_soa_storage[r][2]), "Element 2 produced NaN/Inf");
        AUDIT_ASSERT(x_soa_storage[r][2] == 0.0, "Element 2 singular solution not zero");
    }

    // Verify Element 3: Rank deficient should not produce NaN or Inf
    for (int r = 0; r < 4; ++r) {
        AUDIT_ASSERT(!std::isnan(x_soa_storage[r][3]) && !std::isinf(x_soa_storage[r][3]), "Element 3 produced NaN/Inf");
    }

    // Verify Element 4: Pivot below 1e-20 threshold should safely regularize to 0, no NaN/Inf
    for (int r = 0; r < 4; ++r) {
        AUDIT_ASSERT(!std::isnan(x_soa_storage[r][4]) && !std::isinf(x_soa_storage[r][4]), "Element 4 produced NaN/Inf");
        AUDIT_ASSERT(x_soa_storage[r][4] == 0.0, "Element 4 below threshold did not regularize to zero");
    }

    // Verify Element 5: Hilbert 4x4 solution accuracy
    for (int r = 0; r < 4; ++r) {
        AUDIT_ASSERT(std::abs(x_soa_storage[r][5] - x5_exact[r]) < 1e-8, "Element 5 Hilbert matrix accuracy failed");
    }

    std::cout << "  -> solve_4x4_batched: PASSED all tests!" << std::endl;
}

void audit_sum_dot_repro() {
    std::cout << "[AUDIT] 4. Testing sum_repro & dot_repro..." << std::endl;
    using namespace vec_math;

    // Boundary sizes around BLOCK_SIZE (512)
    std::vector<std::size_t> sizes = {0, 1, 2, 511, 512, 513, 1023, 1024, 1025, 2048, 5000};
    for (auto n : sizes) {
        std::vector<double> v(n, 1.0);
        double s = sum_repro(v);
        AUDIT_ASSERT(std::abs(s - static_cast<double>(n)) < 1e-12, "sum_repro size test failed");

        std::vector<double> u(n, 2.0);
        double d = dot_repro(u, v);
        AUDIT_ASSERT(std::abs(d - 2.0 * static_cast<double>(n)) < 1e-12, "dot_repro size test failed");
    }

    // FP32 Rounding error behavior: Swallowing test
    // 1e7f followed by 100,000 instances of 0.1f.
    // In naive sequential reduction:
    // (1e7 + 0.1f) == 1e7 in FP32 because 0.1f is smaller than the precision gap (ulp = 1e7 * 2^-24 = ~0.6).
    // So naive summation completely loses all 100,000 small additions!
    // But in sum_repro, each block of 512 accumulates 512 * 0.1f = 51.2f before adding to total!
    // 51.2f is larger than 0.6, so it is NOT completely swallowed!
    {
        const std::size_t count = 100000;
        std::vector<float> data(count + 1);
        data[0] = 1000000.0f; // 1e6
        for (std::size_t i = 1; i <= count; ++i) {
            data[i] = 0.01f; // ulp of 1e6 is 0.0625, so 0.01 is smaller than ulp
        }

        float naive_sum = sum_deterministic(data);
        float repro_sum = sum_repro(data);

        // In naive sum, 0.01f is swallowed immediately:
        // naive_sum stays 1000000.0f!
        // In repro_sum, each 512 block sums to 5.12f, which is > 0.0625 and gets preserved!
        std::cout << "    [FP32 swallow test] True sum: 1001000.0f" << std::endl;
        std::cout << "    [FP32 swallow test] Naive sequential sum: " << naive_sum << std::endl;
        std::cout << "    [FP32 swallow test] sum_repro:            " << repro_sum << std::endl;

        AUDIT_ASSERT(naive_sum == 1000000.0f, "Naive sum did not exhibit expected floating point absorption");
        AUDIT_ASSERT(repro_sum > naive_sum, "sum_repro failed to preserve precision over naive reduction");
    }

    // Test bit-identical determinism across multiple invocations
    {
        std::mt19937_64 rng(12345);
        std::uniform_real_distribution<float> dist(-100.0f, 100.0f);
        std::vector<float> a(10000), b(10000);
        for (std::size_t i = 0; i < 10000; ++i) {
            a[i] = dist(rng);
            b[i] = dist(rng);
        }

        float s1 = sum_repro(a);
        float s2 = sum_repro(a);
        AUDIT_ASSERT(s1 == s2, "sum_repro is not bit-identical");

        float d1 = dot_repro(a, b);
        float d2 = dot_repro(a, b);
        AUDIT_ASSERT(d1 == d2, "dot_repro is not bit-identical");
    }

    std::cout << "  -> sum_repro & dot_repro: PASSED all tests!" << std::endl;
}

void audit_workspace_arena() {
    std::cout << "[AUDIT] 5. Testing WorkspaceArena..." << std::endl;
    using namespace vec_math;

    WorkspaceArena arena(64 * 1024); // 64 KB
    AUDIT_ASSERT(arena.capacity() == 64 * 1024, "Capacity mismatch");
    AUDIT_ASSERT(arena.allocated_bytes() == 0, "Initial offset non-zero");

    // Multiple allocations of irregular byte sizes
    std::vector<std::size_t> counts = {1, 3, 7, 15, 64, 127, 33};
    for (auto c : counts) {
        auto span_double = arena.allocate<double>(c);
        uintptr_t addr = reinterpret_cast<uintptr_t>(span_double.data());
        AUDIT_ASSERT(addr % 64 == 0, "Span not 64-byte aligned");
        AUDIT_ASSERT(span_double.size() == c, "Span size mismatch");

        // Write and read
        for (std::size_t i = 0; i < c; ++i) span_double[i] = static_cast<double>(i + 1);
        for (std::size_t i = 0; i < c; ++i) AUDIT_ASSERT(span_double[i] == static_cast<double>(i + 1), "Value corrupted");
    }

    // Check allocated bytes calculation
    AUDIT_ASSERT(arena.allocated_bytes() > 0, "Allocated bytes not updated");

    // Reset test
    arena.reset();
    AUDIT_ASSERT(arena.allocated_bytes() == 0, "Reset did not reset offset");
    auto first = arena.allocate<float>(10);
    AUDIT_ASSERT(reinterpret_cast<uintptr_t>(first.data()) % 64 == 0, "Post-reset alignment failed");

    // Capacity overflow test
    bool caught = false;
    try {
        [[maybe_unused]] auto s = arena.allocate<double>(100000); // 800 KB > 64 KB
    } catch (const std::length_error&) {
        caught = true;
    }
    AUDIT_ASSERT(caught, "Capacity overflow did not throw length_error");

    // Move semantics test
    WorkspaceArena arena2 = std::move(arena);
    AUDIT_ASSERT(arena2.capacity() == 64 * 1024, "Moved arena capacity mismatch");
    AUDIT_ASSERT(arena.capacity() == 0, "Moved-from arena capacity non-zero");

    std::cout << "  -> WorkspaceArena: PASSED all tests!" << std::endl;
}

void audit_primitives() {
    std::cout << "[AUDIT] 6. Testing StridedSpan, broadcast_multiply, gather, scatter_add..." << std::endl;
    using namespace vec_math;

    // StridedSpan
    {
        std::vector<int> data = {10, 20, 30, 40, 50, 60, 70, 80};
        StridedSpan<int> s(data.data(), 4, 2);
        AUDIT_ASSERT(s.size() == 4, "StridedSpan size mismatch");
        AUDIT_ASSERT(s.stride() == 2, "StridedSpan stride mismatch");
        AUDIT_ASSERT(!s.empty(), "StridedSpan empty mismatch");
        AUDIT_ASSERT(s[0] == 10, "s[0] mismatch");
        AUDIT_ASSERT(s[1] == 30, "s[1] mismatch");
        AUDIT_ASSERT(s[2] == 50, "s[2] mismatch");
        AUDIT_ASSERT(s[3] == 70, "s[3] mismatch");

        // Modification through span
        s[2] = 999;
        AUDIT_ASSERT(data[4] == 999, "StridedSpan write failed");

        // Const StridedSpan
        const std::vector<int>& cdata = data;
        StridedSpan<const int> cs(cdata.data(), 4, 2);
        AUDIT_ASSERT(cs[2] == 999, "Const StridedSpan read mismatch");
    }

    // broadcast_multiply
    {
        std::vector<double> v = {1.0, 2.0, 3.0, 4.0};
        broadcast_multiply(v.data(), v.size(), 2.5);
        AUDIT_ASSERT(v[0] == 2.5 && v[1] == 5.0 && v[2] == 7.5 && v[3] == 10.0, "broadcast_multiply raw failed");

        std::span<double> sp(v);
        broadcast_multiply(sp, 2.0);
        AUDIT_ASSERT(v[0] == 5.0 && v[1] == 10.0 && v[2] == 15.0 && v[3] == 20.0, "broadcast_multiply span failed");
    }

    // gather
    {
        std::vector<double> src = {100.0, 200.0, 300.0, 400.0, 500.0};
        std::vector<int> indices = {4, 1, 3, 1, 0};
        std::vector<double> dst(5, 0.0);

        gather(dst.data(), src.data(), indices.data(), 5);
        AUDIT_ASSERT(dst[0] == 500.0, "gather[0] failed");
        AUDIT_ASSERT(dst[1] == 200.0, "gather[1] failed");
        AUDIT_ASSERT(dst[2] == 400.0, "gather[2] failed");
        AUDIT_ASSERT(dst[3] == 200.0, "gather[3] failed");
        AUDIT_ASSERT(dst[4] == 100.0, "gather[4] failed");

        // span gather
        std::vector<double> dst2(5, 0.0);
        gather(std::span(dst2), std::span<const double>(src), std::span<const int>(indices));
        for (int i = 0; i < 5; ++i) AUDIT_ASSERT(dst[i] == dst2[i], "span gather mismatch");
    }

    // scatter_add
    {
        std::vector<double> dst = {0.0, 0.0, 0.0};
        std::vector<int> indices = {1, 0, 1, 2, 1}; // repeated index 1 three times
        std::vector<double> src = {10.0, 5.0, 20.0, 3.0, 2.0};

        scatter_add(dst.data(), indices.data(), src.data(), 5);
        // dst[0] = 5.0
        // dst[1] = 10.0 + 20.0 + 2.0 = 32.0
        // dst[2] = 3.0
        AUDIT_ASSERT(dst[0] == 5.0, "scatter_add[0] failed");
        AUDIT_ASSERT(dst[1] == 32.0, "scatter_add[1] failed");
        AUDIT_ASSERT(dst[2] == 3.0, "scatter_add[2] failed");

        // span scatter_add
        std::vector<double> dst2 = {1.0, 2.0, 3.0};
        scatter_add(std::span(dst2), std::span<const int>(indices), std::span<const double>(src));
        AUDIT_ASSERT(dst2[0] == 6.0, "span scatter_add[0] failed");
        AUDIT_ASSERT(dst2[1] == 34.0, "span scatter_add[1] failed");
        AUDIT_ASSERT(dst2[2] == 6.0, "span scatter_add[2] failed");
    }

    std::cout << "  -> Primitives: PASSED all tests!" << std::endl;
}

int main() {
    std::cout << "======================================================" << std::endl;
    std::cout << "Starting Mathematical & Numerical Rigorous Audit Tests" << std::endl;
    std::cout << "======================================================" << std::endl;

    audit_contract_axis();
    audit_matvec();
    audit_solve_4x4();
    audit_sum_dot_repro();
    audit_workspace_arena();
    audit_primitives();

    std::cout << "======================================================" << std::endl;
    std::cout << "ALL MATHEMATICAL AUDIT VERIFICATIONS PASSED 100%!" << std::endl;
    std::cout << "======================================================" << std::endl;
    return 0;
}
