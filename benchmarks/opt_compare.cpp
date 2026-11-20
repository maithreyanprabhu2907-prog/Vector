#if defined(_MSC_VER) && defined(_DEBUG)
#define _ITERATOR_DEBUG_LEVEL 0
#endif

#include <vec_math/vector_math.hpp>

#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>

using namespace vec_math;
using namespace vec_math::ops;

static volatile void* g_escape_sink = nullptr;

template <typename T>
inline void do_not_optimize(const T& val) {
    g_escape_sink = const_cast<void*>(static_cast<const void*>(&val));
}

struct BenchResult {
    std::string opt_level;
    std::size_t size;
    double time_us;
    double gflops;
    double gb_per_sec;
};

// Benchmark fused expression: out <<= 2.0 * a + b / c - 1.5 + (+a) - (-c);
// 5 FLOPs per element
BenchResult benchmark_fused_expr(const std::string& opt_level, std::size_t n, int warmup_runs, int timed_runs) {
    aligned_vector<double> a(n, 1.25);
    aligned_vector<double> b(n, 2.50);
    aligned_vector<double> c(n, 0.75);
    aligned_vector<double> out(n, 0.0);

    for (int i = 0; i < warmup_runs; ++i) {
        out <<= 2.0 * a + b / c - 1.5 + (+a) - (-c);
        do_not_optimize(out.data());
    }

    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < timed_runs; ++i) {
        out <<= 2.0 * a + b / c - 1.5 + (+a) - (-c);
        do_not_optimize(out.data());
    }
    auto end = std::chrono::high_resolution_clock::now();

    double total_us = std::chrono::duration<double, std::micro>(end - start).count();
    double avg_us = total_us / timed_runs;

    // 5 floating point operations per element
    double gflops = (static_cast<double>(n) * 5.0) / (avg_us * 1000.0);
    // 3 vector reads (a, b, c) + 1 vector write (out) = 4 * n * 8 bytes
    double total_bytes = static_cast<double>(n) * 4.0 * sizeof(double);
    double gb_per_sec = (total_bytes / (avg_us * 1e-6)) / (1024.0 * 1024.0 * 1024.0);

    return {opt_level, n, avg_us, gflops, gb_per_sec};
}

// Benchmark AXPY: y += alpha * x; (2 FLOPs/elem)
BenchResult benchmark_axpy(const std::string& opt_level, std::size_t n, int warmup_runs, int timed_runs) {
    aligned_vector<double> x(n, 1.5);
    aligned_vector<double> y(n, 3.0);
    double alpha = 2.0;

    for (int i = 0; i < warmup_runs; ++i) {
        y += alpha * x;
        do_not_optimize(y.data());
    }

    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < timed_runs; ++i) {
        y += alpha * x;
        do_not_optimize(y.data());
    }
    auto end = std::chrono::high_resolution_clock::now();

    double total_us = std::chrono::duration<double, std::micro>(end - start).count();
    double avg_us = total_us / timed_runs;

    // 2 FLOPs per element
    double gflops = (static_cast<double>(n) * 2.0) / (avg_us * 1000.0);
    // 1 read x + 1 read y + 1 write y = 3 * n * 8 bytes
    double total_bytes = static_cast<double>(n) * 3.0 * sizeof(double);
    double gb_per_sec = (total_bytes / (avg_us * 1e-6)) / (1024.0 * 1024.0 * 1024.0);

    return {opt_level, n, avg_us, gflops, gb_per_sec};
}

// Benchmark Dot Product: d = (a | b); (2 FLOPs/elem)
BenchResult benchmark_dot(const std::string& opt_level, std::size_t n, int warmup_runs, int timed_runs) {
    aligned_vector<double> a(n, 1.0001);
    aligned_vector<double> b(n, 0.9999);
    double total = 0.0;

    for (int i = 0; i < warmup_runs; ++i) {
        total += (a | b);
        do_not_optimize(total);
    }

    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < timed_runs; ++i) {
        total += (a | b);
        do_not_optimize(total);
    }
    auto end = std::chrono::high_resolution_clock::now();

    double total_us = std::chrono::duration<double, std::micro>(end - start).count();
    double avg_us = total_us / timed_runs;

    double gflops = (static_cast<double>(n) * 2.0) / (avg_us * 1000.0);
    double total_bytes = static_cast<double>(n) * 2.0 * sizeof(double);
    double gb_per_sec = (total_bytes / (avg_us * 1e-6)) / (1024.0 * 1024.0 * 1024.0);

    return {opt_level, n, avg_us, gflops, gb_per_sec};
}

int main(int argc, char* argv[]) {
    std::string opt_tag = "UNKNOWN";
#if defined(OPT_OFAST)
    opt_tag = "-Ofast";
#elif defined(OPT_O3)
    opt_tag = "-O3";
#elif defined(OPT_OS)
    opt_tag = "-Os";
#endif

    if (argc > 1) {
        opt_tag = argv[1];
    }

    std::string outfile = "bench_opt_" + opt_tag.substr(opt_tag.find_first_not_of('-')) + ".txt";
    if (argc > 2) {
        outfile = argv[2];
    }

    std::ofstream out(outfile);
    if (!out.is_open()) {
        std::cerr << "Failed to open output file: " << outfile << std::endl;
        return 1;
    }

    const std::vector<std::size_t> sizes = {1000, 10000, 100000, 1000000, 5000000};

    std::cout << "Running benchmark for " << opt_tag << "..." << std::endl;
    out << "# OPT_LEVEL SIZE OP AVG_US GFLOPS GB_PER_SEC\n";

    for (std::size_t n : sizes) {
        int warmup = 10;
        int timed = 100;
        if (n >= 1000000) {
            warmup = 3;
            timed = 20;
        } else if (n >= 100000) {
            warmup = 5;
            timed = 50;
        }

        auto res_fused = benchmark_fused_expr(opt_tag, n, warmup, timed);
        out << opt_tag << " " << n << " fused " << std::fixed << std::setprecision(4)
            << res_fused.time_us << " " << res_fused.gflops << " " << res_fused.gb_per_sec << "\n";

        auto res_axpy = benchmark_axpy(opt_tag, n, warmup, timed);
        out << opt_tag << " " << n << " axpy " << std::fixed << std::setprecision(4)
            << res_axpy.time_us << " " << res_axpy.gflops << " " << res_axpy.gb_per_sec << "\n";

        auto res_dot = benchmark_dot(opt_tag, n, warmup, timed);
        out << opt_tag << " " << n << " dot " << std::fixed << std::setprecision(4)
            << res_dot.time_us << " " << res_dot.gflops << " " << res_dot.gb_per_sec << "\n";

        std::cout << "  N=" << n << " | Fused: " << res_fused.gflops << " GFLOPS, "
                  << res_fused.time_us << " us | AXPY: " << res_axpy.gflops << " GFLOPS | Dot: "
                  << res_dot.gflops << " GFLOPS" << std::endl;
    }

    out.close();
    std::cout << "Done! Results saved to " << outfile << std::endl;
    return 0;
}
