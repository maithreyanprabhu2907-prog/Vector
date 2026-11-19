#if defined(_MSC_VER) && defined(_DEBUG)
#define _ITERATOR_DEBUG_LEVEL 0
#endif

#include <vec_math/vector_math.hpp>

using namespace vec_math;
using namespace vec_math::ops;

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <new>
#include <numeric>
#include <sstream>
#include <string>
#include <vector>

// ============================================================================
// Allocation Tracking
// ============================================================================

namespace {
    bool g_track_allocs = false;
    std::size_t g_alloc_count = 0;
    std::size_t g_alloc_bytes = 0;
}

void* operator new(std::size_t size) {
    if (g_track_allocs) {
        ++g_alloc_count;
        g_alloc_bytes += size;
    }
    void* p = std::malloc(size == 0 ? 1 : size);
    if (!p) throw std::bad_alloc{};
    return p;
}

void* operator new[](std::size_t size) {
    return ::operator new(size);
}

void operator delete(void* ptr) noexcept {
    std::free(ptr);
}

void operator delete(void* ptr, std::size_t) noexcept {
    std::free(ptr);
}

void operator delete[](void* ptr) noexcept {
    std::free(ptr);
}

void operator delete[](void* ptr, std::size_t) noexcept {
    std::free(ptr);
}

// Compiler optimization barrier
static volatile void* g_escape_sink = nullptr;

template <typename T>
void do_not_optimize(const T& val) {
    g_escape_sink = const_cast<void*>(static_cast<const void*>(&val));
}

// ============================================================================
// Benchmark Implementations
// ============================================================================

// 1. Naive std::vector arithmetic (creates temporary allocations per op)
namespace naive {
    template <typename T>
    std::vector<T> mul(const std::vector<T>& v, T s) {
        std::vector<T> r(v.size());
        for (std::size_t i = 0; i < v.size(); ++i) {
            r[i] = v[i] * s;
        }
        return r;
    }

    template <typename T>
    std::vector<T> add(const std::vector<T>& a, const std::vector<T>& b) {
        std::vector<T> r(a.size());
        for (std::size_t i = 0; i < a.size(); ++i) {
            r[i] = a[i] + b[i];
        }
        return r;
    }

    template <typename T>
    std::vector<T> sub(const std::vector<T>& a, const std::vector<T>& b) {
        std::vector<T> r(a.size());
        for (std::size_t i = 0; i < a.size(); ++i) {
            r[i] = a[i] - b[i];
        }
        return r;
    }

    // Evaluates: 2.5 * a + 1.5 * b - 0.5 * c
    // Creates 5 heap allocations per call!
    template <typename T>
    void evaluate_fused(std::vector<T>& out, const std::vector<T>& a, const std::vector<T>& b, const std::vector<T>& c) {
        auto t1 = mul(a, T(2.5));
        auto t2 = mul(b, T(1.5));
        auto t3 = add(t1, t2);
        auto t4 = mul(c, T(0.5));
        out = sub(t3, t4);
    }
}

// 2. Hand-written scalar loop
namespace scalar {
    template <typename T>
    void evaluate_fused(std::vector<T>& out, const std::vector<T>& a, const std::vector<T>& b, const std::vector<T>& c) {
        const std::size_t n = out.size();
        const T* pa = a.data();
        const T* pb = b.data();
        const T* pc = c.data();
        T* pout = out.data();
        for (std::size_t i = 0; i < n; ++i) {
            pout[i] = T(2.5) * pa[i] + T(1.5) * pb[i] - T(0.5) * pc[i];
        }
    }
}

// 3. vec_math zero-allocation expression template & in-place operations
namespace vm_expr {
    template <typename T>
    void evaluate_fused(std::vector<T>& out, const std::vector<T>& a, const std::vector<T>& b, const std::vector<T>& c) {
        out <<= T(2.5) * a + T(1.5) * b - T(0.5) * c;
    }

    template <typename T>
    void evaluate_compound(std::vector<T>& out, const std::vector<T>& a, const std::vector<T>& b) {
        out += a;
        out *= T(1.5);
        out -= b;
    }
}

// 4. vec_math SYCL-compatible device view evaluation
namespace vm_device {
    template <typename T>
    void evaluate_fused(vec_math::device_view<T> vout,
                        vec_math::device_view<const T> va,
                        vec_math::device_view<const T> vb,
                        vec_math::device_view<const T> vc) {
        vout <<= T(2.5) * va + T(1.5) * vb - T(0.5) * vc;
    }

    template <typename T>
    void evaluate_kernel_sim(vec_math::device_view<T> vout,
                             vec_math::device_view<const T> va,
                             vec_math::device_view<const T> vb,
                             vec_math::device_view<const T> vc) {
        auto expr = T(2.5) * va + T(1.5) * vb - T(0.5) * vc;
        const std::size_t n = vout.size();
        T* VEC_MATH_RESTRICT p = vout.data();
        VEC_MATH_PRAGMA_SIMD
        for (std::size_t i = 0; i < n; ++i) {
            vec_math::device_eval(p, expr, i);
        }
    }
}

// ============================================================================
// Benchmark Runner Structures & Utilities
// ============================================================================

struct BenchmarkResult {
    std::string name;
    std::size_t vector_size;
    std::size_t iterations;
    double time_ms;              // total time for all iterations
    double avg_latency_us;        // average latency per run in microseconds
    double speedup;              // vs Naive std::vector
    std::size_t alloc_count;     // heap allocations per run
    std::size_t alloc_bytes;     // bytes allocated per run
    double gflops;               // throughput in GFLOPS
    double bandwidth_gb_s;       // throughput in GB/s
    std::string visual_bar;      // ASCII progress bar
};

struct SizeBenchmarkGroup {
    std::size_t size;
    std::string size_label;
    std::vector<BenchmarkResult> results;
};

// Formats memory bytes into human-readable string
std::string format_bytes(std::size_t bytes) {
    std::ostringstream oss;
    if (bytes == 0) {
        return "0 B";
    } else if (bytes < 1024) {
        oss << bytes << " B";
    } else if (bytes < 1024 * 1024) {
        oss << std::fixed << std::setprecision(1) << (static_cast<double>(bytes) / 1024.0) << " KB";
    } else {
        oss << std::fixed << std::setprecision(1) << (static_cast<double>(bytes) / (1024.0 * 1024.0)) << " MB";
    }
    return oss.str();
}

// Generates visual ASCII progress bar
std::string make_ascii_bar(double speedup, double max_speedup, int bar_width = 20) {
    if (max_speedup <= 0.0) max_speedup = 1.0;
    double fraction = std::clamp(speedup / max_speedup, 0.0, 1.0);
    int filled = static_cast<int>(std::round(fraction * bar_width));
    if (filled < 1 && speedup > 0.0) filled = 1;

    std::string bar;
    for (int i = 0; i < filled; ++i) {
        bar += "█";
    }
    for (int i = filled; i < bar_width; ++i) {
        bar += "░";
    }
    return bar;
}

// ============================================================================
// Core Benchmark Execution
// ============================================================================

SizeBenchmarkGroup run_size_benchmark(std::size_t n, const std::string& label, std::size_t iters) {
    SizeBenchmarkGroup group;
    group.size = n;
    group.size_label = label;

    std::vector<double> a(n, 1.25);
    std::vector<double> b(n, 2.75);
    std::vector<double> c(n, 0.50);
    std::vector<double> out(n, 0.0);

    // Warmup
    naive::evaluate_fused(out, a, b, c);
    scalar::evaluate_fused(out, a, b, c);
    vm_expr::evaluate_fused(out, a, b, c);

    vec_math::device_view<double> vout = vec_math::make_device_view(out.data(), out.size());
    vec_math::device_view<const double> va = vec_math::make_device_view(a.data(), a.size());
    vec_math::device_view<const double> vb = vec_math::make_device_view(b.data(), b.size());
    vec_math::device_view<const double> vc = vec_math::make_device_view(c.data(), c.size());
    vm_device::evaluate_fused(vout, va, vb, vc);

    // FLOP count: 5 FLOPs per element (3 mul, 1 add, 1 sub)
    const double total_flops = 5.0 * static_cast<double>(n) * static_cast<double>(iters);
    // Data transferred: 3 inputs (reads) + 1 output (write) = 4 * sizeof(double) = 32 bytes per element
    const double total_bytes = 32.0 * static_cast<double>(n) * static_cast<double>(iters);

    // -------------------------------------------------------------
    // Test 1: Naive std::vector
    // -------------------------------------------------------------
    std::size_t naive_allocs = 0;
    std::size_t naive_alloc_bytes = 0;
    {
        g_alloc_count = 0;
        g_alloc_bytes = 0;
        g_track_allocs = true;
        naive::evaluate_fused(out, a, b, c);
        g_track_allocs = false;
        naive_allocs = g_alloc_count;
        naive_alloc_bytes = g_alloc_bytes;
        do_not_optimize(out.data());
    }

    auto t0 = std::chrono::high_resolution_clock::now();
    for (std::size_t i = 0; i < iters; ++i) {
        naive::evaluate_fused(out, a, b, c);
        do_not_optimize(out.data());
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    double naive_sec = std::chrono::duration<double>(t1 - t0).count();
    double naive_ms = naive_sec * 1000.0;
    double naive_us = (naive_sec / static_cast<double>(iters)) * 1e6;

    BenchmarkResult r_naive;
    r_naive.name = "Naive std::vector";
    r_naive.vector_size = n;
    r_naive.iterations = iters;
    r_naive.time_ms = naive_ms;
    r_naive.avg_latency_us = naive_us;
    r_naive.speedup = 1.0;
    r_naive.alloc_count = naive_allocs;
    r_naive.alloc_bytes = naive_alloc_bytes;
    r_naive.gflops = (total_flops / naive_sec) / 1e9;
    r_naive.bandwidth_gb_s = (total_bytes / naive_sec) / 1e9;

    // -------------------------------------------------------------
    // Test 2: Hand-written scalar loop
    // -------------------------------------------------------------
    std::size_t scalar_allocs = 0;
    std::size_t scalar_alloc_bytes = 0;
    {
        g_alloc_count = 0;
        g_alloc_bytes = 0;
        g_track_allocs = true;
        scalar::evaluate_fused(out, a, b, c);
        g_track_allocs = false;
        scalar_allocs = g_alloc_count;
        scalar_alloc_bytes = g_alloc_bytes;
        do_not_optimize(out.data());
    }

    t0 = std::chrono::high_resolution_clock::now();
    for (std::size_t i = 0; i < iters; ++i) {
        scalar::evaluate_fused(out, a, b, c);
        do_not_optimize(out.data());
    }
    t1 = std::chrono::high_resolution_clock::now();
    double scalar_sec = std::chrono::duration<double>(t1 - t0).count();
    double scalar_ms = scalar_sec * 1000.0;
    double scalar_us = (scalar_sec / static_cast<double>(iters)) * 1e6;

    // -------------------------------------------------------------
    // Test 3: vec_math (Expression Template)
    // -------------------------------------------------------------
    std::size_t expr_allocs = 0;
    std::size_t expr_alloc_bytes = 0;
    {
        g_alloc_count = 0;
        g_alloc_bytes = 0;
        g_track_allocs = true;
        vm_expr::evaluate_fused(out, a, b, c);
        g_track_allocs = false;
        expr_allocs = g_alloc_count;
        expr_alloc_bytes = g_alloc_bytes;
        do_not_optimize(out.data());
    }

    t0 = std::chrono::high_resolution_clock::now();
    for (std::size_t i = 0; i < iters; ++i) {
        vm_expr::evaluate_fused(out, a, b, c);
        do_not_optimize(out.data());
    }
    t1 = std::chrono::high_resolution_clock::now();
    double expr_sec = std::chrono::duration<double>(t1 - t0).count();
    double expr_ms = expr_sec * 1000.0;
    double expr_us = (expr_sec / static_cast<double>(iters)) * 1e6;

    // -------------------------------------------------------------
    // Test 4: vec_math (SYCL Device View)
    // -------------------------------------------------------------
    // Re-create views so vout points to out's current valid buffer
    vout = vec_math::make_device_view(out.data(), out.size());
    va = vec_math::make_device_view(a.data(), a.size());
    vb = vec_math::make_device_view(b.data(), b.size());
    vc = vec_math::make_device_view(c.data(), c.size());

    std::size_t dev_allocs = 0;
    std::size_t dev_alloc_bytes = 0;
    {
        g_alloc_count = 0;
        g_alloc_bytes = 0;
        g_track_allocs = true;
        vm_device::evaluate_fused(vout, va, vb, vc);
        g_track_allocs = false;
        dev_allocs = g_alloc_count;
        dev_alloc_bytes = g_alloc_bytes;
        do_not_optimize(vout.data());
    }

    t0 = std::chrono::high_resolution_clock::now();
    for (std::size_t i = 0; i < iters; ++i) {
        vm_device::evaluate_fused(vout, va, vb, vc);
        do_not_optimize(vout.data());
    }
    t1 = std::chrono::high_resolution_clock::now();
    double dev_sec = std::chrono::duration<double>(t1 - t0).count();
    double dev_ms = dev_sec * 1000.0;
    double dev_us = (dev_sec / static_cast<double>(iters)) * 1e6;

    BenchmarkResult r_scalar;
    r_scalar.name = "Scalar Loop";
    r_scalar.vector_size = n;
    r_scalar.iterations = iters;
    r_scalar.time_ms = scalar_ms;
    r_scalar.avg_latency_us = scalar_us;
    r_scalar.speedup = naive_sec / scalar_sec;
    r_scalar.alloc_count = scalar_allocs;
    r_scalar.alloc_bytes = scalar_alloc_bytes;
    r_scalar.gflops = (total_flops / scalar_sec) / 1e9;
    r_scalar.bandwidth_gb_s = (total_bytes / scalar_sec) / 1e9;

    BenchmarkResult r_expr;
    r_expr.name = "vec_math (Expr Template)";
    r_expr.vector_size = n;
    r_expr.iterations = iters;
    r_expr.time_ms = expr_ms;
    r_expr.avg_latency_us = expr_us;
    r_expr.speedup = naive_sec / expr_sec;
    r_expr.alloc_count = expr_allocs;
    r_expr.alloc_bytes = expr_alloc_bytes;
    r_expr.gflops = (total_flops / expr_sec) / 1e9;
    r_expr.bandwidth_gb_s = (total_bytes / expr_sec) / 1e9;

    BenchmarkResult r_device;
    r_device.name = "vec_math (SYCL Device View)";
    r_device.vector_size = n;
    r_device.iterations = iters;
    r_device.time_ms = dev_ms;
    r_device.avg_latency_us = dev_us;
    r_device.speedup = naive_sec / dev_sec;
    r_device.alloc_count = dev_allocs;
    r_device.alloc_bytes = dev_alloc_bytes;
    r_device.gflops = (total_flops / dev_sec) / 1e9;
    r_device.bandwidth_gb_s = (total_bytes / dev_sec) / 1e9;

    // Calculate ASCII bars
    double max_speedup = std::max({r_naive.speedup, r_scalar.speedup, r_expr.speedup, r_device.speedup});
    r_naive.visual_bar = make_ascii_bar(r_naive.speedup, max_speedup);
    r_scalar.visual_bar = make_ascii_bar(r_scalar.speedup, max_speedup);
    r_expr.visual_bar = make_ascii_bar(r_expr.speedup, max_speedup);
    r_device.visual_bar = make_ascii_bar(r_device.speedup, max_speedup);

    group.results.push_back(r_naive);
    group.results.push_back(r_scalar);
    group.results.push_back(r_expr);
    group.results.push_back(r_device);

    return group;
}

// ============================================================================
// Extended Benchmarks: Reductions & Compound In-Place
// ============================================================================

struct ExtendedResult {
    std::string operation;
    std::string baseline_name;
    double baseline_ms;
    std::string vec_math_name;
    double vec_math_ms;
    double speedup;
    double gflops;
};

std::vector<ExtendedResult> run_extended_benchmarks(std::size_t n, std::size_t iters) {
    std::vector<ExtendedResult> ext_results;
    std::vector<double> a(n, 1.0001);
    std::vector<double> b(n, 0.9999);
    std::vector<double> out(n, 0.5);

    // 1. Dot Product: std::inner_product vs vec_math::dot
    {
        auto t0 = std::chrono::high_resolution_clock::now();
        double sum_base = 0.0;
        for (std::size_t i = 0; i < iters; ++i) {
            sum_base += std::inner_product(a.begin(), a.end(), b.begin(), 0.0);
            do_not_optimize(sum_base);
        }
        auto t1 = std::chrono::high_resolution_clock::now();
        double base_sec = std::chrono::duration<double>(t1 - t0).count();

        t0 = std::chrono::high_resolution_clock::now();
        double sum_vm = 0.0;
        for (std::size_t i = 0; i < iters; ++i) {
            sum_vm += vec_math::dot(a, b);
            do_not_optimize(sum_vm);
        }
        t1 = std::chrono::high_resolution_clock::now();
        double vm_sec = std::chrono::duration<double>(t1 - t0).count();

        ExtendedResult res;
        res.operation = "Dot Product (dot)";
        res.baseline_name = "std::inner_product";
        res.baseline_ms = base_sec * 1000.0;
        res.vec_math_name = "vec_math::dot (8x unrolled)";
        res.vec_math_ms = vm_sec * 1000.0;
        res.speedup = base_sec / vm_sec;
        // 2 FLOPs per element (mul + add)
        res.gflops = ((2.0 * static_cast<double>(n) * static_cast<double>(iters)) / vm_sec) / 1e9;
        ext_results.push_back(res);
    }

    // 2. Sum Reduction: std::accumulate vs vec_math::sum
    {
        auto t0 = std::chrono::high_resolution_clock::now();
        double sum_base = 0.0;
        for (std::size_t i = 0; i < iters; ++i) {
            sum_base += std::accumulate(a.begin(), a.end(), 0.0);
            do_not_optimize(sum_base);
        }
        auto t1 = std::chrono::high_resolution_clock::now();
        double base_sec = std::chrono::duration<double>(t1 - t0).count();

        t0 = std::chrono::high_resolution_clock::now();
        double sum_vm = 0.0;
        for (std::size_t i = 0; i < iters; ++i) {
            sum_vm += vec_math::sum(a);
            do_not_optimize(sum_vm);
        }
        t1 = std::chrono::high_resolution_clock::now();
        double vm_sec = std::chrono::duration<double>(t1 - t0).count();

        ExtendedResult res;
        res.operation = "Sum Reduction (sum)";
        res.baseline_name = "std::accumulate";
        res.baseline_ms = base_sec * 1000.0;
        res.vec_math_name = "vec_math::sum (8x tree)";
        res.vec_math_ms = vm_sec * 1000.0;
        res.speedup = base_sec / vm_sec;
        // 1 FLOP per element
        res.gflops = ((1.0 * static_cast<double>(n) * static_cast<double>(iters)) / vm_sec) / 1e9;
        ext_results.push_back(res);
    }

    // 3. Fused Multiply-Add (FMA): scalar loop vs vec_math::fma
    {
        auto t0 = std::chrono::high_resolution_clock::now();
        for (std::size_t it = 0; it < iters; ++it) {
            for (std::size_t i = 0; i < n; ++i) {
                out[i] = a[i] * b[i] + 1.5;
            }
            do_not_optimize(out.data());
        }
        auto t1 = std::chrono::high_resolution_clock::now();
        double base_sec = std::chrono::duration<double>(t1 - t0).count();

        t0 = std::chrono::high_resolution_clock::now();
        for (std::size_t it = 0; it < iters; ++it) {
            out <<= vec_math::fma(a, b, 1.5);
            do_not_optimize(out.data());
        }
        t1 = std::chrono::high_resolution_clock::now();
        double vm_sec = std::chrono::duration<double>(t1 - t0).count();

        ExtendedResult res;
        res.operation = "Fused Multiply-Add (fma)";
        res.baseline_name = "Scalar a*b + c";
        res.baseline_ms = base_sec * 1000.0;
        res.vec_math_name = "vec_math::fma (Ternary)";
        res.vec_math_ms = vm_sec * 1000.0;
        res.speedup = base_sec / vm_sec;
        // 2 FLOPs per element
        res.gflops = ((2.0 * static_cast<double>(n) * static_cast<double>(iters)) / vm_sec) / 1e9;
        ext_results.push_back(res);
    }

    return ext_results;
}

// ============================================================================
// Markdown & Mermaid Output Generation
// ============================================================================

std::string generate_markdown_report(const std::vector<SizeBenchmarkGroup>& groups,
                                     const std::vector<ExtendedResult>& ext_results) {
    std::ostringstream ss;

    ss << "## Comprehensive Performance Benchmarks\n\n";
    ss << "Benchmark configuration:\n";
    ss << "- **Compiler**: MSVC /O2 /arch:AVX2 (64-bit)\n";
    ss << "- **Workload**: Multi-vector linear combination (`out = 2.5 * a + 1.5 * b - 0.5 * c`)\n";
    ss << "- **Arithmetic Precision**: `double` (64-bit IEEE-754 floating point)\n";
    ss << "- **Memory Traffic**: 32 bytes/element (3 reads + 1 write)\n";
    ss << "- **FLOP Intensity**: 5 FLOPs/element (3 multiplications, 1 addition, 1 subtraction)\n\n";

    // 1. Mermaid Charts (xychart-beta)
    ss << "### Throughput & Speedup Visualizations\n\n";

    // Chart 1: Throughput (GFLOPS)
    ss << "```mermaid\n";
    ss << "xychart-beta\n";
    ss << "    title \"Compute Throughput (GFLOPS) across Vector Sizes\"\n";
    ss << "    x-axis [";
    for (std::size_t i = 0; i < groups.size(); ++i) {
        ss << "\"" << groups[i].size_label << "\"" << (i + 1 < groups.size() ? ", " : "");
    }
    ss << "]\n";

    // Determine max GFLOPS for scale
    double max_gflops = 0.0;
    for (const auto& g : groups) {
        for (const auto& r : g.results) {
            if (r.gflops > max_gflops) max_gflops = r.gflops;
        }
    }
    int y_max = static_cast<int>(std::ceil(max_gflops * 1.15));
    if (y_max < 10) y_max = 10;
    ss << "    y-axis \"GFLOPS (Higher is better)\" 0 --> " << y_max << "\n";

    // Series for Naive
    ss << "    line [";
    for (std::size_t i = 0; i < groups.size(); ++i) {
        ss << std::fixed << std::setprecision(2) << groups[i].results[0].gflops << (i + 1 < groups.size() ? ", " : "");
    }
    ss << "]\n";

    // Series for Scalar
    ss << "    line [";
    for (std::size_t i = 0; i < groups.size(); ++i) {
        ss << std::fixed << std::setprecision(2) << groups[i].results[1].gflops << (i + 1 < groups.size() ? ", " : "");
    }
    ss << "]\n";

    // Series for vec_math Expr
    ss << "    line [";
    for (std::size_t i = 0; i < groups.size(); ++i) {
        ss << std::fixed << std::setprecision(2) << groups[i].results[2].gflops << (i + 1 < groups.size() ? ", " : "");
    }
    ss << "]\n";

    // Series for vec_math Device View
    ss << "    line [";
    for (std::size_t i = 0; i < groups.size(); ++i) {
        ss << std::fixed << std::setprecision(2) << groups[i].results[3].gflops << (i + 1 < groups.size() ? ", " : "");
    }
    ss << "]\n";
    ss << "```\n\n";
    ss << "> **Chart Legend (Throughput)**: Line 1 (bottom) = Naive `std::vector` (temporary heap mallocs) | Line 2 = Scalar Loop | Line 3 = `vec_math` Expression Templates (zero mallocs) | Line 4 = `vec_math` SYCL Device View (zero mallocs)\n\n";

    // Chart 2: Speedup comparison
    ss << "```mermaid\n";
    ss << "xychart-beta\n";
    ss << "    title \"Speedup Factor Relative to Naive std::vector (x-fold)\"\n";
    ss << "    x-axis [";
    for (std::size_t i = 0; i < groups.size(); ++i) {
        ss << "\"" << groups[i].size_label << "\"" << (i + 1 < groups.size() ? ", " : "");
    }
    ss << "]\n";

    double max_speedup = 0.0;
    for (const auto& g : groups) {
        for (const auto& r : g.results) {
            if (r.speedup > max_speedup) max_speedup = r.speedup;
        }
    }
    int sp_max = static_cast<int>(std::ceil(max_speedup * 1.15));
    if (sp_max < 5) sp_max = 5;
    ss << "    y-axis \"Speedup Factor (x)\" 0 --> " << sp_max << "\n";

    // Scalar speedup
    ss << "    line [";
    for (std::size_t i = 0; i < groups.size(); ++i) {
        ss << std::fixed << std::setprecision(2) << groups[i].results[1].speedup << (i + 1 < groups.size() ? ", " : "");
    }
    ss << "]\n";

    // Expr speedup
    ss << "    line [";
    for (std::size_t i = 0; i < groups.size(); ++i) {
        ss << std::fixed << std::setprecision(2) << groups[i].results[2].speedup << (i + 1 < groups.size() ? ", " : "");
    }
    ss << "]\n";

    // Device view speedup
    ss << "    line [";
    for (std::size_t i = 0; i < groups.size(); ++i) {
        ss << std::fixed << std::setprecision(2) << groups[i].results[3].speedup << (i + 1 < groups.size() ? ", " : "");
    }
    ss << "]\n";
    ss << "```\n\n";
    ss << "> **Chart Legend (Speedup vs Naive)**: Line 1 = Scalar Loop | Line 2 = `vec_math` Expression Templates | Line 3 = `vec_math` SYCL Device View\n\n";

    // 2. Formatted Markdown Tables with Visual ASCII Progress Bars
    ss << "### Benchmark Results Across Vector Scales\n\n";

    for (const auto& g : groups) {
        ss << "#### Vector Size: " << g.size_label << " (" << g.size << " elements)\n\n";
        ss << "| Implementation | Latency / op | Speedup | Heap Allocs | Throughput | Bandwidth | Visual Speedup Bar |\n";
        ss << "| :--- | :--- | :--- | :--- | :--- | :--- | :--- |\n";

        for (const auto& r : g.results) {
            ss << "| **" << r.name << "** | ";
            if (r.avg_latency_us < 1000.0) {
                ss << std::fixed << std::setprecision(2) << r.avg_latency_us << " µs | ";
            } else {
                ss << std::fixed << std::setprecision(2) << (r.avg_latency_us / 1000.0) << " ms | ";
            }

            ss << "**" << std::fixed << std::setprecision(2) << r.speedup << "x** | ";

            if (r.alloc_count == 0) {
                ss << "`0 allocs (0 B)` | ";
            } else {
                ss << r.alloc_count << " allocs (" << format_bytes(r.alloc_bytes) << ") | ";
            }

            ss << std::fixed << std::setprecision(2) << r.gflops << " GFLOPS | ";
            ss << std::fixed << std::setprecision(2) << r.bandwidth_gb_s << " GB/s | ";
            ss << "`" << r.visual_bar << "` |\n";
        }
        ss << "\n";
    }

    // 3. Extended Reductions & Math Utilities Table
    ss << "### Reductions & Extended Math Utilities (1M Elements)\n\n";
    ss << "| Operation | Baseline | `vec_math` | Baseline Time | `vec_math` Time | Speedup | Throughput |\n";
    ss << "| :--- | :--- | :--- | :--- | :--- | :--- | :--- |\n";
    for (const auto& ext : ext_results) {
        ss << "| **" << ext.operation << "** | " << ext.baseline_name << " | **" << ext.vec_math_name << "** | "
           << std::fixed << std::setprecision(2) << ext.baseline_ms << " ms | **"
           << std::fixed << std::setprecision(2) << ext.vec_math_ms << " ms** | **"
           << std::fixed << std::setprecision(2) << ext.speedup << "x** | "
           << std::fixed << std::setprecision(2) << ext.gflops << " GFLOPS |\n";
    }
    ss << "\n";

    // 4. SYCL Compatibility Documentation
    ss << "## SYCL & Heterogeneous Compute Compatibility\n\n";
    ss << "`vec_math` provides full SYCL 2020 device compatibility **without requiring any SYCL headers** (`<sycl/sycl.hpp>`). This guarantees clean compilation in pure host C++20 code while functioning seamlessly inside SYCL device kernels.\n\n";
    ss << "### Key Architectural Guarantees\n";
    ss << "1. **Zero Header Dependency**: The library uses predefined macros (`__SYCL_DEVICE_ONLY__`, `SYCL_LANGUAGE_VERSION`) to map annotations (`VEC_MATH_DEVICE`, `VEC_MATH_HOST_DEVICE`, `SYCL_EXTERNAL`) without polluting translation units with vendor headers.\n";
    ss << "2. **TriviallyCopyable Kernel Capture**: All expression nodes (`BinExpr`, `UnaryExpr`, `TernaryExpr`) and memory views (`vec_span<T>`, `device_view<T>`) satisfy `std::is_trivially_copyable_v<T>`. This allows safe by-value capture inside device kernel lambdas (`cgh.parallel_for(...)`) across GPU/accelerator memory spaces.\n";
    ss << "3. **Unified Memory & USM Support**: `vec_span<T>` / `device_view<T>` seamlessly wraps raw pointers from host memory, Unified Shared Memory (`sycl::malloc_shared`), device memory (`sycl::malloc_device`), or `sycl::buffer` accessors.\n";
    ss << "4. **Kernel Evaluation Helpers**: `vec_math::device_eval(dst, expr, i)` provides zero-allocation element-wise evaluation directly inside GPU work-items.\n\n";

    ss << "### SYCL Usage Examples\n\n";
    ss << "#### Example 1: Expression Evaluation with Unified Shared Memory (USM)\n";
    ss << "```cpp\n";
    ss << "#include <sycl/sycl.hpp>\n";
    ss << "#include <vec_math/vector_math.hpp>\n\n";
    ss << "using namespace vec_math;\n\n";
    ss << "void run_usm_example(sycl::queue& q, std::size_t N) {\n";
    ss << "    // Allocate USM shared or device memory\n";
    ss << "    double* a_ptr   = sycl::malloc_shared<double>(N, q);\n";
    ss << "    double* b_ptr   = sycl::malloc_shared<double>(N, q);\n";
    ss << "    double* out_ptr = sycl::malloc_shared<double>(N, q);\n\n";
    ss << "    // Wrap in vec_math device views (TriviallyCopyable, 0 bytes heap overhead)\n";
    ss << "    vec_math::device_view<const double> va = vec_math::make_device_view(a_ptr, N);\n";
    ss << "    vec_math::device_view<const double> vb = vec_math::make_device_view(b_ptr, N);\n";
    ss << "    vec_math::device_view<double> vout     = vec_math::make_device_view(out_ptr, N);\n\n";
    ss << "    // Submit SYCL kernel\n";
    ss << "    q.parallel_for(sycl::range<1>(N), [=](sycl::id<1> idx) {\n";
    ss << "        const std::size_t i = idx[0];\n";
    ss << "        // Construct expression graph (stored on kernel registers / stack)\n";
    ss << "        auto expr = 2.5 * va + 1.5 * vb - 0.5;\n";
    ss << "        // Evaluate element with zero allocations\n";
    ss << "        vec_math::device_eval(vout, expr, i);\n";
    ss << "    }).wait();\n\n";
    ss << "    sycl::free(a_ptr, q);\n";
    ss << "    sycl::free(b_ptr, q);\n";
    ss << "    sycl::free(out_ptr, q);\n";
    ss << "}\n";
    ss << "```\n\n";

    ss << "#### Example 2: SYCL Buffer Accessor Integration\n";
    ss << "```cpp\n";
    ss << "#include <sycl/sycl.hpp>\n";
    ss << "#include <vec_math/vector_math.hpp>\n\n";
    ss << "using namespace vec_math;\n\n";
    ss << "void run_buffer_example(sycl::queue& q, std::vector<float>& host_a, std::vector<float>& host_b, std::vector<float>& host_out) {\n";
    ss << "    const std::size_t N = host_a.size();\n";
    ss << "    sycl::buffer<float, 1> buf_a(host_a.data(), sycl::range<1>(N));\n";
    ss << "    sycl::buffer<float, 1> buf_b(host_b.data(), sycl::range<1>(N));\n";
    ss << "    sycl::buffer<float, 1> buf_out(host_out.data(), sycl::range<1>(N));\n\n";
    ss << "    q.submit([&](sycl::handler& cgh) {\n";
    ss << "        auto acc_a   = buf_a.get_access<sycl::access::mode::read>(cgh);\n";
    ss << "        auto acc_b   = buf_b.get_access<sycl::access::mode::read>(cgh);\n";
    ss << "        auto acc_out = buf_out.get_access<sycl::access::mode::write>(cgh);\n\n";
    ss << "        cgh.parallel_for(sycl::range<1>(N), [=](sycl::id<1> idx) {\n";
    ss << "            const std::size_t i = idx[0];\n";
    ss << "            // Wrap accessor pointers inside work-item\n";
    ss << "            auto va = vec_math::make_device_view(acc_a.get_pointer(), N);\n";
    ss << "            auto vb = vec_math::make_device_view(acc_b.get_pointer(), N);\n";
    ss << "            auto vout = vec_math::make_device_view(acc_out.get_pointer(), N);\n\n";
    ss << "            auto expr = vec_math::fma(va, vb, 10.0f);\n";
    ss << "            vec_math::device_eval(vout, expr, i);\n";
    ss << "        });\n";
    ss << "    }).wait();\n";
    ss << "}\n";
    ss << "```\n";

    return ss.str();
}

// Updates README.md by inserting or updating the Benchmark & Performance section
void update_readme_file(const std::string& benchmark_md) {
    std::ifstream in("README.md");
    if (!in.is_open()) {
        std::cerr << "Warning: Could not open README.md for reading.\n";
        return;
    }
    std::string content((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    in.close();

    const std::string marker_start = "<!-- BEGIN_BENCHMARK_SECTION -->";
    const std::string marker_end = "<!-- END_BENCHMARK_SECTION -->";

    std::string new_content;
    auto pos_start = content.find(marker_start);
    auto pos_end = content.find(marker_end);

    if (pos_start != std::string::npos && pos_end != std::string::npos) {
        new_content = content.substr(0, pos_start + marker_start.length()) + "\n\n"
                    + benchmark_md + "\n"
                    + content.substr(pos_end);
    } else {
        // Append benchmark section to the end of README.md
        new_content = content + "\n\n" + marker_start + "\n\n" + benchmark_md + "\n" + marker_end + "\n";
    }

    std::ofstream out("README.md");
    if (!out.is_open()) {
        std::cerr << "Warning: Could not open README.md for writing.\n";
        return;
    }
    out << new_content;
    out.close();
    std::cout << "\nSuccessfully updated README.md with benchmark results and performance graphs!\n";
}

// ============================================================================
// Main
// ============================================================================

int main() {
    try {
        std::cout << "======================================================================\n";
        std::cout << " vec_math High-Performance Zero-Allocation & SYCL Benchmark Suite\n";
        std::cout << "======================================================================\n";

        struct SizeConfig {
            std::size_t size;
            std::string label;
            std::size_t iters;
        };

        std::vector<SizeConfig> configs = {
            {1'000,     "1K",   10'000},
            {10'000,    "10K",  2'000},
            {100'000,   "100K", 200},
            {1'000'000, "1M",   30},
            {5'000'000, "5M",   10}
        };

        std::vector<SizeBenchmarkGroup> groups;
        for (const auto& cfg : configs) {
            std::cout << "Running benchmark for vector size " << cfg.label
                      << " (" << cfg.size << " elements, " << cfg.iters << " iterations)...\n" << std::flush;
            auto g = run_size_benchmark(cfg.size, cfg.label, cfg.iters);
            groups.push_back(g);

            // Print immediate summary
            std::cout << "  --------------------------------------------------------------------------------\n";
            std::cout << "  " << std::left << std::setw(30) << "Implementation"
                      << std::right << std::setw(12) << "Latency"
                      << std::setw(10) << "Speedup"
                      << std::setw(14) << "Throughput"
                      << std::setw(14) << "Allocs" << "\n";
            std::cout << "  --------------------------------------------------------------------------------\n";
            for (const auto& r : g.results) {
                std::cout << "  " << std::left << std::setw(30) << r.name << std::right;
                if (r.avg_latency_us < 1000.0) {
                    std::cout << std::setw(9) << std::fixed << std::setprecision(2) << r.avg_latency_us << " µs";
                } else {
                    std::cout << std::setw(9) << std::fixed << std::setprecision(2) << (r.avg_latency_us / 1000.0) << " ms";
                }
                std::cout << std::setw(9) << std::fixed << std::setprecision(2) << r.speedup << "x"
                          << std::setw(8) << std::fixed << std::setprecision(2) << r.gflops << " GF"
                          << std::setw(14) << (r.alloc_count == 0 ? "0 B" : format_bytes(r.alloc_bytes))
                          << "\n";
            }
            std::cout << "  --------------------------------------------------------------------------------\n\n" << std::flush;
        }

        std::cout << "Running extended reductions and math utilities benchmarks (1M elements)...\n" << std::flush;
        auto ext_results = run_extended_benchmarks(1'000'000, 50);

        // Generate markdown report
        std::string md_report = generate_markdown_report(groups, ext_results);

        // Save to benchmark_results.md
        std::ofstream out_md("benchmark_results.md");
        if (out_md.is_open()) {
            out_md << md_report;
            out_md.close();
            std::cout << "Wrote detailed benchmark report to benchmark_results.md\n";
        }

        // Always update README.md with generated performance graphs and benchmark data
        update_readme_file(md_report);

        std::cout << "======================================================================\n";
        std::cout << " All Benchmarks Completed Successfully!\n";
        std::cout << "======================================================================\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "EXCEPTION CAUGHT: " << e.what() << "\n";
        return 1;
    } catch (...) {
        std::cerr << "UNKNOWN EXCEPTION CAUGHT\n";
        return 2;
    }
}
