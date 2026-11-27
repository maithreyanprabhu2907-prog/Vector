# `keisan++` Benchmarks & Roofline

`keisan` uses [benchmarks/regression_bench.cpp](file:///home/void_zlr/Dev/Vector/benchmarks/regression_bench.cpp) as its standardized, anti-hoisted performance and integrity suite.

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

| Kernel / Metric | Measured Throughput | Notes |
|---|---:|---|
| `PEAK_GFLOPS` (AVX2 FMA roofline) | **54.42 GFLOPS** | 8×4 lanes |
| `PEAK_COPY_GBPS` (STREAM copy) | **61.11 GB/s** | — |
| `STREAM_AXPY` | **62.41 GB/s** | 2 read + 1 write |
| `DOT` (1 thread, `std::simd`) | **67.53 GB/s** | Dual-accumulator FMA |
| `DOT_PAR` (12 threads, OpenMP) | **217.66 GB/s** | Per-thread `std::simd` |
| `DOT_REPRO` (pairwise tree) | **15.67 GB/s** | Bit-identical IEEE-754 |
| `MATVEC_NP64` (runtime cols) | **24.07 GFLOPS** | ~5× faster than pre-optimization |
| `MATVEC_FIXED_NP20` | **21.46 GFLOPS** | Compile-time `Cols` |
| `SF_N4` / `SF_N6` (`contract_axis`) | **13.93** / **19.57 GFLOPS** | 1.45× / 2.43× speedup over dense |
| `SOLVE3_MELEM_S` (`solve_nxn_batched<3>`) | **191.37 MElem/s** | `std::simd` SoA batching |
| `SOLVE4_MELEM_S` (`solve_nxn_batched<4>`) | **56.70 MElem/s** (1.70 GFLOPS) | 2.38× faster than scalar |
| `SOLVE6_MELEM_S` (`solve_nxn_batched<6>`) | **17.59 MElem/s** | `std::simd` SoA batching |
