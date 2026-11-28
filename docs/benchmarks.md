# `keisan++` Benchmarks & Roofline

`keisan` uses [benchmarks/regression_bench.cpp](file:///home/void_zlr/Dev/keisanpp/benchmarks/regression_bench.cpp) as its standardized, anti-hoisted performance and integrity suite.

---

## Running the Benchmark Suite

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release -DKEISAN_BUILD_BENCHMARKS=ON -DCMAKE_CXX_FLAGS="-O3 -mavx2 -mfma"
cmake --build build --target regression_bench -j
./build/regression_bench --quick
```

Every run verifies:
- **`MUTATION=OK`**: Output buffers are genuinely mutated (preventing dead-code elimination).
- **`IMPOSSIBLE_COUNT=0`**: Measured FLOP/s and GB/s stay within physical hardware roofline limits (`PEAK_GFLOPS`, `PEAK_COPY_GBPS`).
- **Naive-reference cross-checks**: Accuracy is verified before timing begins.

---

## Measured Results (GCC 16.2.1 `-std=c++26 -O3 -mavx2 -mfma`, Ryzen 5 5500 6C/12T)

Bandwidth-bound kernels behave very differently depending on whether the working set fits in the 16 MB L3 cache (`--quick`, `n = 1 << 20`) or spills to DRAM (full run, `n = 1 << 22`).

### Full Run (`n = 1 << 22`, DRAM-bound)

| Kernel / Metric | Measured Throughput | Notes |
|---|---:|---|
| `PEAK_GFLOPS` (AVX2 FMA roofline) | **54.29 GFLOPS** | 8×4 lanes |
| `PEAK_COPY_GBPS` (STREAM copy) | **25.73 GB/s** | DRAM-bound (> 16 MB L3) |
| `STREAM_AXPY` | **40.20 GB/s** | 2 read + 1 write (DRAM) |
| `STREAM_NON_TEMPORAL_GBPS` (`assign_stream`) | **38.20 GB/s** | Cache-bypassing stores avoid RFO |
| `DOT_PAR` (12 threads, OpenMP) | **29.19 GB/s** | DRAM-saturated across cores |
| `DOT_REPRO` (pairwise tree) | **15.58 GB/s** | Bit-identical IEEE-754 |
| `MATVEC_NP64` (runtime cols) | **24.48 GFLOPS** | Dual-accumulator `std::simd` |
| `MATVEC_FIXED_NP20` | **21.43 GFLOPS** | Compile-time `Cols` |
| `SOLVE3_MELEM_S` (`solve_nxn_batched<3>`) | **207.62 MElem/s** | `std::simd` SoA batching |
| `SOLVE4_MELEM_S` (`solve_nxn_batched<4>`) | **61.68 MElem/s** | `std::simd` SoA batching |
| `SOLVE6_MELEM_S` (`solve_nxn_batched<6>`) | **15.57 MElem/s** | `std::simd` SoA batching |

### Quick Run (`--quick`, `n = 1 << 20`, L3-resident)

| Kernel / Metric | Measured Throughput | Notes |
|---|---:|---|
| `PEAK_GFLOPS` (AVX2 FMA roofline) | **54.41 GFLOPS** | 8×4 lanes |
| `STREAM_AXPY` | **86.61 GB/s** | L3-resident |
| `DOT_PAR` (12 threads, OpenMP) | **211.53 GB/s** | L3-resident per-thread `std::simd` |
| `STREAM_NON_TEMPORAL_GBPS` (`assign_stream`) | **2.66 GB/s** | Non-temporal stores pay penalty on L3-sized arrays |
