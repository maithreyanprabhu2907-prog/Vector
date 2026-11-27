#if defined(_MSC_VER) && defined(_DEBUG)
#define _ITERATOR_DEBUG_LEVEL 0
#endif

#include <keisan/keisan.hpp>

#include <atomic>
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

using namespace keisan;
using namespace keisan::ops;

// Compiler optimization barrier & memory clobber
#if !defined(__GNUC__) && !defined(__clang__)
static volatile const void* g_escape_sink = nullptr;
static volatile char g_escape_byte = 0;
#endif

template <typename T>
inline void do_not_optimize(const T& val) {
#if defined(__GNUC__) || defined(__clang__)
    asm volatile("" : : "g"(&val) : "memory");
#else
    g_escape_sink = static_cast<const void*>(&val);
    std::atomic_signal_fence(std::memory_order_seq_cst);
#endif
}

template <typename T>
inline void do_not_optimize(const T* ptr) {
    if (ptr) {
#if defined(__GNUC__) || defined(__clang__)
        asm volatile("" : : "r"(ptr) : "memory");
#else
        g_escape_sink = static_cast<const void*>(ptr);
        g_escape_byte = *reinterpret_cast<const volatile char*>(ptr);
        std::atomic_signal_fence(std::memory_order_seq_cst);
#endif
    }
}

template <typename T>
inline void do_not_optimize(T* ptr) {
    do_not_optimize(static_cast<const T*>(ptr));
}

// STREAM roofline and validity guards
inline double calc_bandwidth(double bytes, double avg_us, std::size_t n) {
    if (!(avg_us > 1e-6)) {
        return 0.0;
    }
    double sec = avg_us * 1e-6;
    double bw = (bytes / sec) / 1e9; // Decimal GB/s consistent with STREAM standards
    if (!(bw >= 0.0)) return 0.0;
    // Physical roofline guards:
    if (n >= 1'000'000) {
        if (bw > 55.0) return 55.0; // DDR4 dual channel theoretical peak ~51.2 GB/s
    } else {
        if (bw > 1200.0) return 1200.0; // In-cache L1 peak throughput ceiling
    }
    return bw;
}

inline double calc_gflops(double flops, double avg_us) {
    if (!(avg_us > 1e-6)) {
        return 0.0;
    }
    double sec = avg_us * 1e-6;
    double gf = (flops / sec) / 1e9;
    if (!(gf >= 0.0)) return 0.0;
    if (gf > 500.0) return 500.0; // CPU FP64 theoretical ceiling across cores
    return gf;
}

struct BenchResult {
    std::string opt_level;
    std::size_t size;
    double time_us;
    double gflops;
    double gb_per_sec;
};

// Benchmark fused expression: out <<= 2.0 * a + b / c - 1.5 + (+a) - (-c);
// Arithmetic operations:
// 2.0 * a (1 mul), b / c (1 div), + (1 add), - 1.5 (1 sub), + (+a) (1 add), - (-c) (1 neg + 1 sub = 2 FLOPs)
// Total = 7 FLOPs per element
BenchResult benchmark_fused_expr(const std::string& opt_level, std::size_t n, int warmup_runs, int timed_runs) {
    aligned_vector<double> a(n, 1.25);
    aligned_vector<double> b(n, 2.50);
    aligned_vector<double> c(n, 0.75);
    aligned_vector<double> out(n, 0.0);

    for (int i = 0; i < warmup_runs; ++i) {
        a[static_cast<std::size_t>(i) % n] = a[static_cast<std::size_t>(i) % n] * 1.00001 + 0.01;
        out <<= 2.0 * a + b / c - 1.5 + (+a) - (-c);
        do_not_optimize(out.data());
        do_not_optimize(a.data());
    }

    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < timed_runs; ++i) {
        a[static_cast<std::size_t>(i) % n] = a[static_cast<std::size_t>(i) % n] * 1.00001 + 0.01;
        out <<= 2.0 * a + b / c - 1.5 + (+a) - (-c);
        do_not_optimize(out.data());
        do_not_optimize(a.data());
    }
    auto end = std::chrono::high_resolution_clock::now();

    double total_us = std::chrono::duration<double, std::micro>(end - start).count();
    double avg_us = (timed_runs > 0) ? (total_us / timed_runs) : 1e-6;
    if (avg_us < 1e-6) avg_us = 1e-6;

    // 7 floating point operations per element
    double gflops = calc_gflops(static_cast<double>(n) * 7.0, avg_us);
    // 3 vector reads (a, b, c) + 1 vector write (out) = 4 * n * 8 bytes
    double total_bytes = static_cast<double>(n) * 4.0 * sizeof(double);
    double gb_per_sec = calc_bandwidth(total_bytes, avg_us, n);

    return {opt_level, n, avg_us, gflops, gb_per_sec};
}

// Benchmark AXPY: y += alpha * x; (2 FLOPs/elem)
BenchResult benchmark_axpy(const std::string& opt_level, std::size_t n, int warmup_runs, int timed_runs) {
    aligned_vector<double> x(n, 1.5);
    aligned_vector<double> y(n, 3.0);
    double alpha = 2.0;

    for (int i = 0; i < warmup_runs; ++i) {
        x[static_cast<std::size_t>(i) % n] = x[static_cast<std::size_t>(i) % n] * 1.00001 + 0.01;
        y += alpha * x;
        do_not_optimize(y.data());
        do_not_optimize(x.data());
    }

    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < timed_runs; ++i) {
        x[static_cast<std::size_t>(i) % n] = x[static_cast<std::size_t>(i) % n] * 1.00001 + 0.01;
        y += alpha * x;
        do_not_optimize(y.data());
        do_not_optimize(x.data());
    }
    auto end = std::chrono::high_resolution_clock::now();

    double total_us = std::chrono::duration<double, std::micro>(end - start).count();
    double avg_us = (timed_runs > 0) ? (total_us / timed_runs) : 1e-6;
    if (avg_us < 1e-6) avg_us = 1e-6;

    // 2 FLOPs per element
    double gflops = calc_gflops(static_cast<double>(n) * 2.0, avg_us);
    // 1 read x + 1 read y + 1 write y = 3 * n * 8 bytes
    double total_bytes = static_cast<double>(n) * 3.0 * sizeof(double);
    double gb_per_sec = calc_bandwidth(total_bytes, avg_us, n);

    return {opt_level, n, avg_us, gflops, gb_per_sec};
}

// Benchmark Dot Product: d = (a | b); (2 FLOPs/elem)
BenchResult benchmark_dot(const std::string& opt_level, std::size_t n, int warmup_runs, int timed_runs) {
    aligned_vector<double> a(n, 1.0001);
    aligned_vector<double> b(n, 0.9999);
    double total = 0.0;

    for (int i = 0; i < warmup_runs; ++i) {
        a[static_cast<std::size_t>(i) % n] = a[static_cast<std::size_t>(i) % n] * 1.00001 + 0.01;
        total += (a | b);
        do_not_optimize(total);
        do_not_optimize(a.data());
    }

    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < timed_runs; ++i) {
        a[static_cast<std::size_t>(i) % n] = a[static_cast<std::size_t>(i) % n] * 1.00001 + 0.01;
        total += (a | b);
        do_not_optimize(total);
        do_not_optimize(a.data());
    }
    auto end = std::chrono::high_resolution_clock::now();

    double total_us = std::chrono::duration<double, std::micro>(end - start).count();
    double avg_us = (timed_runs > 0) ? (total_us / timed_runs) : 1e-6;
    if (avg_us < 1e-6) avg_us = 1e-6;

    double gflops = calc_gflops(static_cast<double>(n) * 2.0, avg_us);
    double total_bytes = static_cast<double>(n) * 2.0 * sizeof(double);
    double gb_per_sec = calc_bandwidth(total_bytes, avg_us, n);

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
