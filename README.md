# vec_math

Zero-allocation, header-only C++20 vector math operators (`+`, `-`, `*`, `/`, `+=`, `-=`, `*=`, `/=`, `<<=`) for `std::vector<T, Alloc>`.

## Repository Layout

Place the files into this structure:

```text
vec_math/
├── CMakeLists.txt
├── README.md
├── benchmarks/
│   └── benchmark.cpp
├── cmake/
│   └── vec_mathConfig.cmake.in
├── docs/
│   ├── README.md
│   ├── api_reference.md
│   ├── architecture.md
│   ├── benchmarks.md
│   ├── integration.md
│   └── sycl_guide.md
├── include/
│   └── vec_math/
│       └── vector_math.hpp
└── tests/
    └── vector_math_test.cpp
```

## Documentation

Comprehensive guides and references are available in the [`docs/`](docs/README.md) folder:
* **[Architecture & Design Principles](docs/architecture.md)**: Zero-allocation expression templates, AST lazy evaluation, SIMD unrolling, and memory optimization.
* **[API Reference](docs/api_reference.md)**: Exhaustive catalog of concepts, operators, expressions, views, and math utilities.
* **[SYCL Compatibility Guide](docs/sycl_guide.md)**: Zero-header SYCL compiler compatibility, TriviallyCopyable expressions, and device execution.
* **[Performance & Benchmarking](docs/benchmarks.md)**: In-depth metrics, GFLOPS, memory bandwidth, and scalability analysis.
* **[Integration & Build Guide](docs/integration.md)**: CMake FetchContent, system installation, compiler tuning, and test instructions.

## Integration

### 1. CMake `FetchContent`
```cmake
include(FetchContent)
FetchContent_Declare(
    vec_math
    GIT_REPOSITORY https://github.com/<you>/vec_math.git
    GIT_TAG v1.0.0
)
FetchContent_MakeAvailable(vec_math)

target_link_libraries(your_target PRIVATE vec_math::vec_math)
```

### 2. System Install + `find_package`
```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
ctest --test-dir build --output-on-failure
sudo cmake --install build
```
Then in any project:
```cmake
find_package(vec_math 1.0 REQUIRED)
target_link_libraries(your_target PRIVATE vec_math::vec_math)
```

### 3. Single-Header Drop-In
Copy `vector_math.hpp` into your project's `include/` folder and compile with `-std=c++20` (or `/std:c++20`).

## Usage

### 1. Zero-Allocation Fused Operator Overloading
```cpp
#include <vec_math/vector_math.hpp>
#include <vector>

int main() {
    using namespace vec_math;

    std::vector<double> a = {1.0, 2.0, 3.0};
    std::vector<double> b = {4.0, 5.0, 6.0};
    std::vector<double> c = {0.5, 1.5, 2.5};
    std::vector<double> out(3);

    // Zero-malloc fused multi-term evaluation into pre-allocated buffer
    out <<= 2.0 * a + b / 2.0 - 0.5 * c;

    // Zero-malloc in-place compound operations
    out += a;
    out *= 0.5;

    // Zero-malloc rvalue buffer recycling
    auto res = 3.0 * std::move(a) + b;
}
```

### 2. Discontinuous Galerkin (DG) Scientific Math Kit
Designed specifically for high-order DG time integration, CFL calculation, and slope/positivity limiters:

```cpp
using namespace vec_math;

// 1. BLAS-1 Equivalent Operations via Pure Operator Overloading
// AXPY (y += alpha * x) with zero heap allocation:
y += 0.5 * x;

// AXPBY / Runge-Kutta Stage Update (y = alpha * x + beta * y) with zero heap allocation:
y <<= 0.75 * x + 0.25 * y;

// 2. Inner Product / Dot Product via operator|
double inner_prod = u | v; // or vec_math::dot(u, v)

// 3. CFL & Time-Stepping Reductions
double dt_cfl   = min_val(dx / sound_speed); // minimum element
double max_wave = max_val(wave_speeds);      // maximum element
double l_inf    = norm_inf(residual);        // L_infinity norm (max |v_i|)
double l1       = norm_l1(error);            // L_1 norm (sum |v_i|)

// 4. Element-wise Limiters (DG Positivity & WENO Limiters)
out <<= clamp(u, 0.0, 1.0);  // element-wise clamping: lo <= u_i <= hi
out <<= min(u, u_max);       // element-wise minimum
out <<= max(u, u_min);       // element-wise maximum

// 5. Element-wise Transcendental & Math Functions
out <<= sqrt(density);
out <<= exp(-energy);
out <<= pow(pressure, 1.4);
out <<= sin(phase) + cos(phase);
```

### 3. 64-Byte Cache-Line Aligned Memory (`aligned_vector`)
Eliminates unaligned split-cache-line penalties across AVX2 (256-bit) and AVX-512 vector registers:

```cpp
#include <vec_math/vector_math.hpp>

// 64-byte aligned vector type
vec_math::aligned_vector<double> u(1'000'000);
vec_math::aligned_vector<double> v(1'000'000);
vec_math::aligned_vector<double> w(1'000'000);

// Seamlessly interoperates with all overloaded operators
w <<= 0.5 * u + 0.5 * v;
double d = u | v;
```

### 4. Non-Temporal Streaming Stores (`assign_stream`)
When vector sizes exceed the CPU L3 cache (e.g. 1M - 5M `double` elements on Ryzen 5 5500 with 16 MB L3 cache), standard cache writes incur costly Read-For-Ownership (RFO) bus transactions. `assign_stream` bypasses cache allocation using non-temporal AVX streaming stores (`_mm256_stream_pd` / `_mm256_stream_ps`):

```cpp
// Streams directly to DRAM, bypassing CPU cache pollution and RFO
vec_math::assign_stream(dst, 2.5 * a + 1.5 * b);
```

### 5. Multi-Core Parallel Domain Acceleration (OpenMP)
All default operators are deliberately single-threaded SIMD so that outer element-quadrature loops in DG solvers do not incur OpenMP thread pool contention. When performing global mesh-wide updates or whole-domain reductions, explicit parallel helpers scale across all CPU cores:

```cpp
// Parallel expression evaluation across all CPU cores
vec_math::assign_par(out, 2.5 * a + 1.5 * b);

// Multi-core parallel reductions
double total_sum  = vec_math::sum_par(energy);
double global_dot = vec_math::dot_par(u, v);
```

### 6. Discontinuous Galerkin (DG) Endgame Primitives

A specialized suite of high-performance primitives designed specifically for high-order DG discretizations, Runge-Kutta time steppers, face Riemann flux exchange, and Implicit-Explicit (IMEX) relaxation:

#### 1. Compile-Time Fixed $N_p$ Matrix-Vector (`matvec_fixed`, `matvec_transposed_fixed`)
Flattens matrix-vector products into vector registers with 4-way independent accumulators (`acc0..acc3`) for DG differentiation:
```cpp
// Order p=3 triangle (Np = 10 nodes): dudr = Dr * u
alignas(64) double Dr[100], u[10], dudr[10];
vec_math::matvec_fixed<10, 10>(Dr, u, dudr);

// Safe self-aliased in-place execution:
vec_math::matvec_fixed<10, 10>(Dr, u, u);
```

#### 2. Full In-Place Compound Assignment on `std::span` and `std::array`
Perform zero-allocation in-place arithmetic (`+=`, `-=`, `*=`, `/=`) directly on slices or stack arrays:
```cpp
// std::span slice accumulation
std::span<double> elem_u(domain.data(), 64);
std::span<const double> elem_flux(fluxes.data(), 64);
elem_u += elem_flux;
elem_u *= 0.5;

// std::array state update (e.g. Nvar=4 state vectors)
std::array<double, 4> state = {1.0, 2.0, 0.0, 2.5};
state += dstate;
```

#### 3. Bit-Identical Reproducible Reductions (`sum_repro`, `dot_repro`, `ReductionPolicy`)
Guarantees bit-identical reductions across thread counts and compilers via blocked 512-element tree accumulation:
```cpp
// Reproducible dot product for Krylov solvers (GMRES / BiCGSTAB)
double d = vec_math::dot_repro(r, p);

// Policy-guided reductions:
double s1 = vec_math::sum(res, vec_math::ReductionPolicy::REPRO); // Bit-identical
double s2 = vec_math::sum(res, vec_math::ReductionPolicy::FAST);  // Max throughput
```

#### 4. Non-Contiguous Strided Views (`StridedSpan<T>`)
Zero-copy 1D random-access views over non-contiguous multi-variable or interleaved arrays:
```cpp
// Slicing y-coordinates from interleaved [x0, y0, z0, x1, y1, z1, ...]
vec_math::StridedSpan<double> y_coords(coords + 1, num_nodes, 3);
double y1 = y_coords[1];
```

#### 5. Elemental Metric Scaling (`broadcast_multiply`)
Scales quadrature node arrays by elemental Jacobians $J_e$ or face metric determinants with SIMD vectorization:
```cpp
vec_math::broadcast_multiply(std::span<double>(quad_weights, 8), J_elem);
```

#### 6. Zero-Malloc Runge-Kutta Scratchpad (`WorkspaceArena`)
64-byte cache-aligned linear bump allocator for scratch buffers across RK stages ($k_1 \dots k_s$):
```cpp
vec_math::WorkspaceArena arena(16 * 1024 * 1024); // 16 MB pre-allocated
arena.reset();
auto k1 = arena.allocate<double>(num_dofs);
auto k2 = arena.allocate<double>(num_dofs);
```

#### 7. Surface Riemann Flux Exchange (`gather` and `scatter_add`)
Extracts traces from element volume nodes onto faces and scatters flux residuals back:
```cpp
// Gather trace values onto face quadrature nodes
vec_math::gather(std::span(u_face), std::span(u_vol), std::span(face_node_map));

// Scatter-add numerical fluxes to volume residual
vec_math::scatter_add(std::span(residual), std::span(face_node_map), std::span(flux));
```

#### 8. Point-Implicit IMEX Solve (`solve_4x4_batched`) & Layout Transpositions
Solves decoupled $4 \times 4$ linear systems across all elements simultaneously in SoA layout with partial pivoting:
```cpp
// Solves A_soa * x_soa = b_soa across 1000 elements concurrently
vec_math::solve_4x4_batched(A_ptrs, b_ptrs, x_ptrs, 1000);

// AoS <-> SoA layout conversions
vec_math::transpose_aos_to_soa(aos_data, soa_data, num_elements, 4);
vec_math::transpose_soa_to_aos(soa_data, aos_data, num_elements, 4);
```

#### 9. Sum-Factorization Directional Tensor Contractions (`contract_axis`)
Applies 1D directional differentiation operators across multi-dimensional tensor-product quad/hex elements in $\mathcal{O}(d \cdot N^{d+1})$ operations instead of dense $\mathcal{O}(N^{2d})$:
```cpp
// 3D Hex element with N=4 (64 nodes): directional derivative along X, Y, and Z
vec_math::contract_axis<vec_math::Axis::X, 4>(D_1d, u, du_dx, 4, 4, 4);
vec_math::contract_axis<vec_math::Axis::Y, 4>(D_1d, u, du_dy, 4, 4, 4);
vec_math::contract_axis<vec_math::Axis::Z, 4>(D_1d, u, du_dz, 4, 4, 4);
```

## Compile-Time Options

* **`VEC_MATH_NO_EXCEPTIONS`**: Define before including `vector_math.hpp` when building with `-fno-exceptions` to use `assert` + `std::abort()` on dimension/capacity errors instead of throwing.


<!-- BEGIN_BENCHMARK_SECTION -->

## Comprehensive Performance Benchmarks (-Ofast Optimization)

Benchmark configuration & hardware environment:
- **CPU Microarchitecture**: AMD Ryzen 5 5500 (6 Cores / 12 Hardware Threads, AVX2, FMA3, 16 MB L3 Cache)
- **Compiler Optimization**: `-Ofast` mode (MSVC `/O2 /fp:fast /openmp /arch:AVX2` | GCC/Clang `-Ofast -fopenmp -mavx2`)
- **Core Workload**: Multi-vector linear combination (`out = 2.5 * a + 1.5 * b - 0.5 * c`)
- **Arithmetic Precision**: `double` (64-bit IEEE-754 floating point with fast math enabled)
- **Memory Traffic**: 32 bytes/element (3 reads + 1 write)
- **FLOP Intensity**: 5 FLOPs/element (3 multiplications, 1 addition, 1 subtraction)

### Throughput & Speedup Visualizations

```mermaid
xychart-beta
    title "Compute Throughput (GFLOPS) across Vector Sizes"
    x-axis ["1K", "10K", "100K", "1M", "5M"]
    y-axis "GFLOPS (Higher is better)" 0 --> 52
    line [2.14, 0.36, 0.48, 0.51, 0.52]
    line [44.95, 19.53, 16.41, 4.80, 3.42]
    line [42.35, 19.40, 16.77, 4.81, 3.41]
```

> **Chart Legend (Throughput)**: Line 1 = Naive `std::vector` (temporary heap mallocs) | Line 2 = Hand-written Scalar Loop | Line 3 = `vec_math` Expression Templates (zero mallocs)

```mermaid
xychart-beta
    title "Speedup Factor Relative to Naive std::vector (x-fold)"
    x-axis ["1K", "10K", "100K", "1M", "5M"]
    y-axis "Speedup Factor (x)" 0 --> 63
    line [21.04, 53.95, 34.34, 9.42, 6.54]
    line [19.82, 53.57, 35.11, 9.45, 6.52]
```

> **Chart Legend (Speedup vs Naive)**: Line 1 = Hand-written Scalar Loop | Line 2 = `vec_math` Expression Templates (zero mallocs)

### Core Multi-Scale Benchmark Results (1K to 5M Elements)

#### Vector Size: 1K (1000 elements)

| Implementation | Latency / op | Speedup | Heap Allocs | Throughput | Bandwidth | Visual Speedup Bar |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Naive std::vector** | 2.34 µs | **1.00x** | 5 allocs (39.3 KB) | 2.14 GFLOPS | 13.67 GB/s | `█░░░░░░░░░░░░░░░░░░░` |
| **Scalar Loop** | 0.11 µs | **21.04x** | `0 allocs (0 B)` | 44.95 GFLOPS | 287.67 GB/s | `████████████████████` |
| **vec_math (Expr Template)** | 0.12 µs | **19.82x** | `0 allocs (0 B)` | 42.35 GFLOPS | 271.05 GB/s | `███████████████████░` |

#### Vector Size: 10K (10000 elements)

| Implementation | Latency / op | Speedup | Heap Allocs | Throughput | Bandwidth | Visual Speedup Bar |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Naive std::vector** | 138.10 µs | **1.00x** | 5 allocs (390.8 KB) | 0.36 GFLOPS | 2.32 GB/s | `█░░░░░░░░░░░░░░░░░░░` |
| **Scalar Loop** | 2.56 µs | **53.95x** | `0 allocs (0 B)` | 19.53 GFLOPS | 125.00 GB/s | `████████████████████` |
| **vec_math (Expr Template)** | 2.58 µs | **53.57x** | `0 allocs (0 B)` | 19.40 GFLOPS | 124.14 GB/s | `████████████████████` |

#### Vector Size: 100K (100000 elements)

| Implementation | Latency / op | Speedup | Heap Allocs | Throughput | Bandwidth | Visual Speedup Bar |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Naive std::vector** | 1.05 ms | **1.00x** | 5 allocs (3.8 MB) | 0.48 GFLOPS | 3.06 GB/s | `█░░░░░░░░░░░░░░░░░░░` |
| **Scalar Loop** | 30.47 µs | **34.34x** | `0 allocs (0 B)` | 16.41 GFLOPS | 105.02 GB/s | `████████████████████` |
| **vec_math (Expr Template)** | 29.81 µs | **35.11x** | `0 allocs (0 B)` | 16.77 GFLOPS | 107.35 GB/s | `████████████████████` |

#### Vector Size: 1M (1000000 elements)

| Implementation | Latency / op | Speedup | Heap Allocs | Throughput | Bandwidth | Visual Speedup Bar |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Naive std::vector** | 9.83 ms | **1.00x** | 5 allocs (38.1 MB) | 0.51 GFLOPS | 3.26 GB/s | `██░░░░░░░░░░░░░░░░░░` |
| **Scalar Loop** | 1.04 ms | **9.42x** | `0 allocs (0 B)` | 4.80 GFLOPS | 30.69 GB/s | `████████████████████` |
| **vec_math (Expr Template)** | 1.04 ms | **9.45x** | `0 allocs (0 B)` | 4.81 GFLOPS | 30.78 GB/s | `████████████████████` |

#### Vector Size: 5M (5000000 elements)

| Implementation | Latency / op | Speedup | Heap Allocs | Throughput | Bandwidth | Visual Speedup Bar |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Naive std::vector** | 47.75 ms | **1.00x** | 5 allocs (190.7 MB) | 0.52 GFLOPS | 3.35 GB/s | `███░░░░░░░░░░░░░░░░░` |
| **Scalar Loop** | 7.30 ms | **6.54x** | `0 allocs (0 B)` | 3.42 GFLOPS | 21.90 GB/s | `████████████████████` |
| **vec_math (Expr Template)** | 7.32 ms | **6.52x** | `0 allocs (0 B)` | 3.41 GFLOPS | 21.85 GB/s | `████████████████████` |

### Discontinuous Galerkin Runge-Kutta Time Step Update (1M Elements)

Workload: High-order DG stage accumulation `y = 0.75 * x + 0.25 * y` (AXPBY operation).

| Implementation | Latency / op | Speedup vs Naive | Heap Allocs | Compute Throughput | Memory Bandwidth |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Naive std::vector (3 mallocs/step)** | 6.44 ms | **1.00x** | 3 allocs | 0.47 GFLOPS | 3.73 GB/s |
| **Manual Scalar Loop** | 633.56 µs | **10.16x** | `0 allocs (0 B)` | 4.74 GFLOPS | 37.88 GB/s |
| **vec_math Operator `y <<= a*x + b*y`** | 797.57 µs | **8.07x** | `0 allocs (0 B)` | 3.76 GFLOPS | 30.09 GB/s |
| **vec_math `aligned_vector` (64-byte)** | 868.12 µs | **7.42x** | `0 allocs (0 B)` | 3.46 GFLOPS | 27.65 GB/s |

### Non-Temporal Streaming Stores (`assign_stream` vs Cache-Allocating Stores)

Evaluates non-temporal streaming stores (`_mm256_stream_pd`) that bypass Read-For-Ownership (RFO) on vectors exceeding CPU L3 cache:

| Vector Size | Normal Assign (`<<=`) | Streaming (`assign_stream`) | Normal Bandwidth | Stream Bandwidth | Streaming Speedup |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **1M Elements (8 MB)** | 21.60 ms | **8.70 ms** | 14.81 GB/s | **36.80 GB/s** | **2.48x** |
| **5M Elements (40 MB)** | 67.49 ms | **43.43 ms** | 8.89 GB/s | **13.82 GB/s** | **1.55x** |

### Multi-Core Parallel Scaling Across Ryzen 5 5500 Cores (`assign_par`)

Evaluates explicit OpenMP domain-parallel evaluation (`assign_par`) across 12 hardware threads:

| Vector Scale | Single-Threaded Time | Parallel Time (`assign_par`) | Single GFLOPS | Parallel GFLOPS | Parallel Speedup |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **100K (100000 elem)** | 14.02 ms | **2.42 ms** | 6.42 GF | **37.12 GF** | **5.78x** |
| **1M (1000000 elem)** | 36.19 ms | **24.83 ms** | 3.32 GF | **4.83 GF** | **1.46x** |
| **5M (5000000 elem)** | 84.70 ms | **90.23 ms** | 2.66 GF | **2.49 GF** | **0.94x** |

### Reductions & Extended DG Math Kit (1M Elements)

| Operation | Baseline | `vec_math` Implementation | Baseline Time | `vec_math` Time | Speedup | Throughput |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Dot Product (a | b)** | std::inner_product | **vec_math `a | b` (8x SIMD)** | 31.02 ms | **22.79 ms** | **1.36x** | 4.39 GFLOPS |
| **Parallel Dot Product** | vec_math::dot (Single) | **vec_math::dot_par (OpenMP)** | 14.03 ms | **3.64 ms** | **3.86x** | 27.48 GFLOPS |
| **Sum Reduction (sum)** | std::accumulate | **vec_math::sum (8x tree)** | 4.76 ms | **6.99 ms** | **0.68x** | 7.15 GFLOPS |
| **Parallel Sum (sum_par)** | vec_math::sum (Single) | **vec_math::sum_par (OpenMP)** | 4.50 ms | **1.46 ms** | **3.09x** | 34.27 GFLOPS |
| **L_inf Norm (norm_inf)** | Scalar std::abs + max | **vec_math::norm_inf** | 4.62 ms | **4.26 ms** | **1.09x** | 23.49 GFLOPS |
| **Min Element (min_val)** | std::min_element | **vec_math::min_val** | 13.09 ms | **5.44 ms** | **2.41x** | 9.20 GFLOPS |
| **Fused Multiply-Add (fma)** | Scalar a*b + c | **vec_math::fma (Ternary)** | 39.86 ms | **59.17 ms** | **0.67x** | 1.69 GFLOPS |
| **WENO Limiter (clamp)** | Scalar std::clamp | **vec_math::clamp (Ternary)** | 44.95 ms | **29.10 ms** | **1.54x** | 3.44 GFLOPS |

### `-Ofast` Microarchitectural Optimization Analysis

Building with `-Ofast` (or MSVC `/fp:fast /O2 /openmp /arch:AVX2`) yields critical performance enhancements on modern x86-64 processors:

1. **Hardware FMA Fusing**: Allows the compiler to fuse multiplication and addition operations into single-cycle hardware `vfmadd213pd` / `vfmadd231pd` instructions, doubling theoretical peak floating-point throughput.
2. **Tree Reassociation**: Enables out-of-order algebraic reassociation across `vec_math`'s 8-way unrolled reduction trees (`sum`, `dot`), breaking serial accumulator dependency chains and saturating CPU execution ports.
3. **64-Byte Cache Alignment**: `aligned_vector<T>` ensures zero split-cache-line loads and unaligned penalties across AVX2 256-bit registers.
4. **Streaming Stores Bypassing RFO**: `assign_stream` utilizes `_mm256_stream_pd` non-temporal stores to eliminate cache write-allocate reads when vectors exceed CPU L3 cache size (16 MB).
5. **Selective Domain Parallelism**: Single-threaded default operators eliminate OpenMP thread pool overhead in fine-grained DG element quadrature loops, while `assign_par` accelerates bulk global mesh updates across all CPU cores.

### Discontinuous Galerkin (DG) Math Core & Endgame Primitives

`vec_math` provides an endgame mathematical foundation tailored for high-order Discontinuous Galerkin solvers:

| Primitive | Category | Key Characteristics | DG Solver Application |
| :--- | :--- | :--- | :--- |
| **`matvec_fixed<Rows, Cols>`** | Kernel | Static dimensions, 4-way ILP unrolling, self-aliasing safe | DG volume/surface numerical differentiation ($N_p \le 36$) |
| **`matvec_transposed_fixed<Rows, Cols>`** | Kernel | Static dimensions, register-pinned transposed matvec | Mass matrix inverse application, adjoint operators |
| **Span/Array In-Place Assign** | Syntax | `+=`, `-=`, `*=`, `/=` for `std::span` and `std::array` | Sub-element state and local flux accumulation |
| **`ReductionPolicy` (`FAST`, `REPRO`)** | Reductions | Blocked 512-element tree vs 8-way multi-accumulator SIMD | Bit-identical Krylov solvers vs high-speed CFL reductions |
| **`sum_repro`, `dot_repro`** | Reductions | Guaranteed bit-identical results across thread counts | Orthogonalization in GMRES / BiCGSTAB, regression tests |
| **`StridedSpan<T>`** | View | 1D non-contiguous view with stride and random-access iterator | Slicing multi-variable fields in interleaved AoS/SoA data |
| **`broadcast_multiply`** | Metric | Vectorized scaling of spans/pointers by scalar factors | Scaling quadrature nodes by elemental Jacobians $J_e$ |
| **`WorkspaceArena`** | Memory | 64-byte aligned linear bump allocator with instant reset | Zero-malloc Runge-Kutta stages ($k_1 \dots k_s$) & scratch pads |
| **`gather`** | Surface | SIMD indirect gather from nodal to face quadrature points | Extracting trace values $Q^-, Q^+$ at element interfaces |
| **`scatter_add`** | Surface | SIMD indirect scatter-addition into residual vectors | Accumulating Riemann numerical fluxes into element residuals |
| **`solve_4x4_batched`** | Linear Solve | Batched $4 \times 4$ LU solve with partial pivoting in SoA | Point-implicit IMEX relaxation for stiff acoustic modes |
| **`transpose_aos_to_soa` / `transpose_soa_to_aos`** | Layout | High-throughput AoS $\leftrightarrow$ SoA multi-variable array conversions | Interfacing nodal quadrature loops with batched SIMD solves |
| **`contract_axis<Axis, N>`** | Tensor / SF | Directional 1D tensor contraction along $X, Y, Z$ with unrolling | 2D/3D Quad/Hex sum-factorization in $\mathcal{O}(d \cdot N^{d+1})$ operations |

1. **Sub-Range Spans (`std::span`)**: Zero-copy, zero-allocation operations on sub-vectors and element blocks (`elem_u <<= elem_u + dt * elem_k`, `elem_u += elem_flux`).
2. **Fixed-Size State Vectors (`std::array`)**: 4-variable state blocks (e.g. Euler $[\rho, \rho u, \rho v, E]$) processed on the stack with zero heap allocation.
3. **Dense Matrix-Vector Operations (`matvec` & `matvec_fixed`)**: Small dense matvec with runtime dimensions or compile-time unrolling (`matvec_fixed<Rows, Cols>`) optimized for DG polynomial orders ($N_p = 4 \dots 36$) for volume, surface, and mass-inverse operators.
4. **Reproducible Reductions (`sum_repro`, `dot_repro`)**: Invariant 512-element chunk tree accumulation ensuring bit-identical results regardless of OpenMP thread count or compiler optimizations.
5. **Direct Buffer Write (`eval_into`)**: Evaluates expressions directly into raw pointers or device buffers without container allocations (`eval_into(ptr, n, expr)`).
6. **Compile-Time Include Optimization**: Defining `VEC_MATH_NO_INTRINSICS` skips `<immintrin.h>`, reducing translation unit compile times by ~800 ms.
7. **Directional Sum-Factorization (`contract_axis`)**: Directional tensor contractions along axes 0, 1, and 2 for fast spectral element/DG operator evaluation.

## Compiler Flag Recommendations

- **GCC / Clang**: `-std=c++20 -O3 -mavx2 -mfma` (note: `-mfma` is required with `-mavx2` for hardware FMA instructions).
- **MSVC**: `/std:c++20 /O2 /fp:precise /arch:AVX2 /openmp`.

## SYCL Compatibility

`vec_math` is engineered for full compatibility with modern heterogeneous C++ frameworks, including SYCL compilers (such as Intel oneAPI DPC++ and AdaptiveCpp):

- **Zero SYCL Headers Included**: The library contains strictly zero `#include <sycl/sycl.hpp>` or proprietary vendor header dependencies. Host code compiles cleanly under standard C++20 with no foreign runtime or SDK overhead.
- **TriviallyCopyable Expression Templates**: All expression nodes (`VecRef`, `ScalarVal`, `BinExpr`, `UnaryExpr`, `TernaryExpr`) are standard-layout and satisfy `std::is_trivially_copyable_v<T>`. This allows expressions to be captured by value in SYCL device kernel lambdas (`cgh.parallel_for`) without violating device capture rules.
- **Device-Callable Core Annotations**: All core evaluation operators and expression evaluators are decorated with `VEC_MATH_DEVICE`, `VEC_MATH_HOST_DEVICE`, and `SYCL_EXTERNAL` macros when compiled under SYCL device compilation modes (`__SYCL_DEVICE_ONLY__` or `SYCL_LANGUAGE_VERSION`), enabling direct device execution.


<!-- END_BENCHMARK_SECTION -->
