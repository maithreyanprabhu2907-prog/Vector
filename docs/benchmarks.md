# Performance & Benchmarking

This document details the benchmarking methodology, empirical results, and performance characteristics of the `vec_math` library.

---

## 1. Benchmark Methodology

The benchmark suite is implemented in [`benchmarks/benchmark.cpp`](../benchmarks/benchmark.cpp) and configured as a first-class CMake target.

### Workload Characteristics
- **Operation**: Linear combination across three vectors:
  $$\mathbf{out} = 2.5 \cdot \mathbf{a} + 1.5 \cdot \mathbf{b} - 0.5 \cdot \mathbf{c}$$
- **Data Type**: `double` (64-bit IEEE 754 floating point)
- **Memory Footprint**: 32 bytes per element (3 vector reads + 1 vector write)
- **Arithmetic Intensity**: 5 FLOPs per element (3 multiplications, 1 addition, 1 subtraction)
- **Tested Vector Sizes**: 1K ($10^3$), 10K ($10^4$), 100K ($10^5$), 1M ($10^6$), and 5M ($5 \times 10^6$) elements
- **Hardware Platform**: Windows x64, MSVC 19.51 (`/O2 /arch:AVX2`)

### Compared Implementations
1. **Naive `std::vector` Arithmetic**: Direct operator overloading creating temporary `std::vector` instances for intermediate sub-expressions.
2. **Hand-Written Scalar Loop**: Direct pointer traversal with SIMD auto-vectorization.
3. **`vec_math` (Zero-Allocation Expression Templates)**: Expression template evaluation executed via `out <<= expr`.

---

## 2. Empirical Performance Results

### Summary Across Scales

| Vector Size | Naive `std::vector` Latency | `vec_math` Latency | Speedup vs Naive | Throughput (GFLOPS) | Memory Bandwidth | Heap Allocs |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **1K** | 2.51 µs | **0.38 µs** | **6.59x** | 13.15 GFLOPS | 84.14 GB/s | **0 allocs (0 B)** |
| **10K** | 144.34 µs | **4.03 µs** | **35.80x** | 12.40 GFLOPS | 79.37 GB/s | **0 allocs (0 B)** |
| **100K** | 1.04 ms | **41.01 µs** | **25.35x** | 12.19 GFLOPS | 78.03 GB/s | **0 allocs (0 B)** |
| **1M** | 10.46 ms | **892.33 µs** | **11.73x** | 5.60 GFLOPS | 35.86 GB/s | **0 allocs (0 B)** |
| **5M** | 52.44 ms | **7.13 ms** | **7.36x** | 3.51 GFLOPS | 22.45 GB/s | **0 allocs (0 B)** |

> **Allocation Contrast**: Naive `std::vector` performed **5 heap allocations (39 KB to 190 MB)** per evaluation. `vec_math` performed strictly **0 allocations (0 bytes)** across all scales.

---

## 3. Reductions & Math Utilities Benchmark (1M Elements)

Reductions were benchmarked with 1,000,000 double-precision elements:

| Operation | Baseline | `vec_math` Implementation | Baseline Time | `vec_math` Time | Speedup | Throughput |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Dot Product** | `std::inner_product` | `vec_math::dot` (8x unrolled tree) | 39.26 ms | **13.12 ms** | **2.99x** | 7.62 GFLOPS |
| **Sum** | `std::accumulate` | `vec_math::sum` (8x unrolled tree) | 36.61 ms | **5.34 ms** | **6.85x** | 9.36 GFLOPS |
| **FMA** | Scalar loop ($a \cdot b + c$) | `vec_math::fma` (ternary AST) | 31.57 ms | **40.66 ms** | **0.78x** | 2.46 GFLOPS |

---

## 4. Performance Visualizations

### Compute Throughput (GFLOPS)
```mermaid
xychart-beta
    title "Compute Throughput (GFLOPS) across Vector Sizes"
    x-axis ["1K", "10K", "100K", "1M", "5M"]
    y-axis "GFLOPS (Higher is better)" 0 --> 58
    line [2.00, 0.35, 0.48, 0.48, 0.48]
    line [49.74, 20.52, 16.37, 5.79, 3.51]
    line [13.15, 12.40, 12.19, 5.60, 3.51]
```
*(Line 1: Naive `std::vector` | Line 2: Hand-written Scalar Loop | Line 3: `vec_math` Expression Templates)*

### Speedup Factor Relative to Naive `std::vector`
```mermaid
xychart-beta
    title "Speedup Factor Relative to Naive std::vector (x-fold)"
    x-axis ["1K", "10K", "100K", "1M", "5M"]
    y-axis "Speedup Factor (x)" 0 --> 69
    line [24.92, 59.24, 34.03, 12.12, 7.36]
    line [6.59, 35.80, 25.35, 11.73, 7.36]
```
*(Line 1: Hand-written Scalar Loop | Line 2: `vec_math` Expression Templates)*

---

## 5. Architectural Insights

### Why Small & Medium Scales (1K - 100K) Gain 25x - 36x Speedup
At sizes fitting within CPU cache (L1/L2/L3), memory allocation overhead dominates execution time. In naive vector math, allocating memory locks the global allocator heap, calls OS kernel memory management routines, zero-initializes memory, and triggers frequent cache evictions. By operating on pre-allocated buffers with single-pass expression fusion, `vec_math` executes entirely within cache lines at full SIMD speeds.

### Why Large Scales (1M - 5M) Saturate Memory Bandwidth
At 5,000,000 elements, each vector consumes 40 MB ($160\text{ MB}$ total across 4 vectors), significantly exceeding typical CPU L3 cache capacities ($16\text{ MB} - 32\text{ MB}$). The bottleneck shifts from compute latency to DDR main memory bandwidth. `vec_math` achieves **22.45 - 35.86 GB/s**, matching the hand-written scalar loop and saturating available memory bus throughput.

---

## 6. How to Reproduce Benchmarks

```bash
# Configure with Release optimization
cmake -B build -DCMAKE_BUILD_TYPE=Release

# Build the benchmark executable
cmake --build build --config Release --target benchmark

# Execute the benchmark suite
./build/Release/benchmark
```
*Note: Running `benchmark` will print live metrics to stdout and automatically refresh the Markdown tables and Mermaid charts in `README.md`.*
