# `keisan++` (`keisan`) Detailed API & Architectural Reference

> Per-public-function specification and systems-design rationale.  
> For a quick 1-page lookup, see [cheatsheet.md](cheatsheet.md). For the high-level blueprint, see [architecture.md](architecture.md).

---

## 1. `vcore` — Expression Engine, Math Ops & Evaluators

### `eval_into(T* dst, std::size_t n, const Src& src)`
- **Signature**: `template<typename T, VectorOperand Src> KEISAN_HD_INLINE void eval_into(T* dst, std::size_t n, const Src& src)`
- **Behavior**: Evaluates a lazy `VectorOperand` (expression tree, `VecRef`, or `StridedSpan`) into `dst[0..n)`.
- **Architectural Rationale**: Isolates the hot loop behind a raw pointer + size boundary so it works identically inside `std::vector`, `std::span`, `std::array`, and SYCL `parallel_for` device kernels without carrying container metadata.
- **Invariants & Pitfalls**: Caller must ensure `dst` has at least `n` elements and `src.size() == n` (checked by higher-level `assign`).

### `assign(std::vector<T, Alloc>& dst, const Src& src)`
- **Behavior**: Resizes `dst` if needed (unless `KEISAN_STRICT_ZERO_MALLOC` is defined, in which case a size mismatch throws `std::bad_alloc`), validates operand sizes, and dispatches to `eval_into`.
- **Architectural Rationale**: Separating *construction* of expressions (`+`, `-`, `*`, `/`, `fma`) from *evaluation* (`assign`) guarantees a single fused pass over memory with zero temporary heap allocations.

### `assign_stream(std::vector<T, Alloc>& dst, const Src& src)`
- **Behavior**: Evaluates `src` into `dst` using non-temporal (cache-bypassing) streaming stores (`_mm256_stream_pd`/`_mm256_stream_ps` on AVX, `_mm_stream_*` on SSE2, followed by `_mm_sfence()`), falling back to `std::simd::unchecked_store` or scalar loops on non-x86 / `KEISAN_NO_INTRINSICS`.
- **Architectural Rationale**: Standard stores (`vmovupd`) trigger Read-For-Ownership (RFO) cache-line fills, wasting 33% of DRAM bandwidth on write-only arrays larger than L3 cache. Non-temporal stores bypass L1/L2/L3 and write directly to DRAM write-combining buffers.
- **Pitfalls**: Requires 32-byte alignment for AVX (`aligned_vector<double>` / `aligned_vector<float>` handles this automatically). For small L1/L2-resident arrays, `assign()` is faster because it keeps `dst` hot in cache.

### `assign_par(std::vector<T, Alloc>& dst, const Src& src)`
- **Behavior**: Evaluates `src` into `dst` across OpenMP threads (`#pragma omp parallel for`) when compiled with `_OPENMP`.
- **Architectural Rationale**: Memory-bandwidth-bound kernels saturate a single core's memory controller around ~45–60 GB/s; multi-core OpenMP scaling helps on multi-socket/NUMA or compute-heavy transcendental expressions (`sin`, `cos`, `exp`, `pow`).

### Compound Assignment (`+=`, `-=`, `*=`, `/=`)
- **Behavior**: In-place elementwise updates on `std::vector`, `std::span`, and `std::array` against any `VectorOperand` or `Scalar`.
- **Architectural Rationale**: `u += dt * rhs` evaluates without allocating a temporary `dt * rhs` vector and handles self-aliasing (`u += u * 0.5`).

### `fma(a, b, c)` & `clamp(v, lo, hi)`
- **Behavior**: Constructs a `TernaryExpr` (or `BinExpr` specialization) supporting any mix of `VectorOperand` and `Scalar` arguments.
- **Architectural Rationale**: Explicitly modeling ternary FMA and clamp avoids nested binary-tree depth and maps cleanly to hardware `vfmadd` / `vmin`/`vmax` instructions.

### Elementwise Math (`abs`, `sqrt`, `sin`, `cos`, `exp`, `log`, `pow`)
- **Behavior**: Returns a lazy `UnaryExpr` / `BinExpr` evaluated at the sink (`assign` / `sum` / `dot`).

---

## 2. `vmem` — Zero-Allocation Memory & Views

### `aligned_allocator<T, Alignment = 64>` & `aligned_vector<T>`
- **Behavior**: C++20/26 stateful-free allocator using `std::aligned_alloc` (POSIX) / `_aligned_malloc` (Windows) with 64-byte cache-line alignment. `aligned_vector<T>` is `std::vector<T, aligned_allocator<T, 64>>`.
- **Architectural Rationale**: 64-byte alignment guarantees zero cache-line splits for AVX-512 (64B) and AVX2 (32B) loads/stores and satisfies hardware alignment rules for `_mm256_stream_*`.

### `StridedSpan<T>`
- **Signature**: `StridedSpan(T* ptr, std::size_t size, std::size_t stride)`
- **Behavior**: Non-owning strided view where `span[i]` accesses `ptr[i * stride]`. `TriviallyCopyable` and accepted by all `VectorOperand` expressions and reductions.
- **Architectural Rationale**: DG solvers frequently slice a single field component out of AoS layouts or extract a face/slice without copying into a contiguous temporary. `OperandStorage` detects `is_strided_span_v` and stores `StridedSpan` directly while disabling contiguous `std::simd::unchecked_load` fast paths.

### `WorkspaceArena`
- **Behavior**: Pre-allocates a single 64B-aligned byte buffer (`WorkspaceArena(capacity_bytes)`). `allocate<T>(count)` returns a 64B-aligned `std::span<T>` in $O(1)$ via bump-pointer arithmetic; `reset()` rewinds the offset to `0` without freeing memory. Throws `std::bad_alloc` on overflow (or aborts under `KEISAN_NO_EXCEPTIONS`).
- **Architectural Rationale**: Runge-Kutta time-stepping cannot afford `malloc`/`free` syscalls or page faults inside inner stages. `WorkspaceArena` gives deterministic, zero-fragmentation scratchpad memory.

---

## 3. `vreduce` — Reductions, Norms & Reproducibility

### `ReductionPolicy` (`FAST`, `REPRO`, `DETERMINISTIC`)
- **`FAST`** (default): Uses dual-accumulator `std::simd::vec<T>` (`acc0`, `acc1`) + `std::simd::reduce` for maximum throughput. Non-associative across different SIMD widths.
- **`REPRO` / `DETERMINISTIC`**: Executes an iterative pairwise binary-tree summation over a 64-slot stack buffer. Produces **bit-identical IEEE-754 results** regardless of CPU SIMD width or OpenMP thread count.

### `sum(op, policy)` / `sum_repro(op)` / `sum_par(op)`
- **Behavior**: Returns the sum of all elements in `op` (`T{0}` if empty).
- **Architectural Rationale**:
  - `sum()` uses `if !consteval` so `static_assert(sum(arr) == ...)` works at compile time while runtime contiguous `double`/`float` paths execute dual-accumulator `std::simd::vec` loops (hiding 4-cycle FMA/ADD latency).
  - `sum_par()` uses a serial threshold (`n < 16384`) to avoid OpenMP fork/join overhead on small vectors, and delegates each thread's contiguous slice to `sum()` so every core runs `std::simd`.

### `dot(lhs, rhs, policy)` / `dot_repro(lhs, rhs)` / `dot_par(lhs, rhs)`
- **Behavior**: Computes $\sum_i \text{lhs}[i] \cdot \text{rhs}[i]$ (throws `std::invalid_argument` on size mismatch).
- **Architectural Rationale**: Same dual-accumulator `std::simd` FMA pattern as `sum()`, with `dot_par` delegating thread slices to `dot()`.

### `norm(op)`, `norm_sq(op)`, `norm_l1(op)`, `norm_inf(op)`
- **Behavior**: Computes $\|x\|_2 = \sqrt{\text{norm\_sq}(x)}$, $\|x\|_2^2$, $\|x\|_1$, and $\|x\|_\infty = \max_i |x_i|$ (with `std::isnan(val) || val > res` so `NaN` is never silently dropped).

### `min_val(op)` / `max_val(op)`
- **Behavior**: Returns minimum / maximum element; throws `std::invalid_argument` on empty input.

---

## 4. `vdg` — Discontinuous Galerkin & Tensor Kernels

### `matvec(A, x, y, rows, cols)` & `matvec_transposed(A, x, y, rows, cols)`
- **Behavior**: Row-major dense matrix-vector product $y = Ax$ and $y = A^T x$.
- **Architectural Rationale**:
  - Public functions check for pointer overlap between `y` and `A`/`x` and dispatch to `detail::matvec_noalias` / `detail::matvec_transposed_noalias` marked with `KEISAN_RESTRICT`.
  - Uses dual-accumulator `std::simd::vec` across `cols` for all matrix sizes (eliminating scalar tail cliffs for $N_p > 36$).

### `matvec_fixed<Rows, Cols>(A, x, y)` & `matvec_transposed_fixed<Rows, Cols>(A, x, y)`
- **Behavior**: Compile-time-sized $y = Ax$ and $y = A^T x$. Handles `Rows == 0 || Cols == 0` cleanly without zero-length stack arrays, and buffers `y` on the stack if `y` aliases `x` or `A`.
- **Architectural Rationale**: Known polynomial dimensions ($N_p = 10, 20, 36, 64$) let GCC 16 unroll and register-allocate inner loops, reaching ~37–43 GFLOPS on AVX2+FMA.

### `contract_axis<AxisParam, N>(D, in, out, nx, ny, nz)`
- **Behavior**: 3D tensor-product sum-factorization contraction along `Axis::X`, `Axis::Y`, or `Axis::Z` (supports raw pointers and `std::span`, with optional compile-time `N` for unrolling).
- **Architectural Rationale**: Replaces $O(N^6)$ 3D dense operator application with three $O(N^4)$ 1D contractions. `Axis::Y` and `Axis::Z` place the `nx` (`i`/`p`) dimension innermost so all loads/stores are unit-stride (`KEISAN_RESTRICT` via `detail::contract_axis_noalias`).

### `solve_nxn_batched<N>(A_soa, b_soa, x_soa, num_elements)` & `solve_4x4_batched(...)`
- **Behavior**: Solves `num_elements` independent $N\times N$ linear systems stored in Structure-of-Arrays (SoA) layout (`A_soa[r * N + c][e]`). Supports in-place `x_soa == b_soa`. `solve_4x4_batched` delegates to `solve_nxn_batched<4>`.
- **Architectural Rationale**:
  - Vectorizes **across systems** (`W = std::simd::vec<T>::size()` elements in lockstep) rather than within a single small matrix.
  - Uses branchless `std::simd::select` for partial pivoting.
  - Computes `max_abs = max(|A[r][c]|)` in registers and checks `!(abs_diag > max_abs * rel_eps)` (`1e-6` for `float`, `1e-14` for `double`), accumulating a `singular_flag` mask and throwing **after** the SIMD loop—preserving full vectorization while catching both `float` numerical singularities and `NaN` pivots without false positives on small-scale well-conditioned systems.

### `gather` & `scatter_add`
- **Behavior**: Indirect surface flux gather (`dst[i] = src[indices[i]]`) and accumulation (`dst[indices[i]] += src[i]`) for raw pointers and `std::span`.

### `broadcast_multiply(data, count, factor)`
- **Behavior**: In-place elementwise scaling (`data[i] *= factor`) for elemental metric/Jacobian scaling.

### `transpose_aos_to_soa` & `transpose_soa_to_aos`
- **Behavior**: Converts between `[elem][comp]` (AoS) and `[comp][elem]` (SoA) layouts so `solve_nxn_batched` and `std::simd` kernels achieve unit-stride loads.

---

## 5. `vsimd` — Concepts & Compile-Time Macros

- **`Scalar<T>`**: Valid arithmetic or `std::complex<float|double>` scalar.
- **`NumericVector<T>`**: Contiguous container (`std::vector`, `aligned_vector`, `std::span`, `std::array`) or `StridedSpan` over a `Scalar`.
- **`VecExpression<T>` / `VectorOperand<T>`**: Lazy expression AST nodes and valid operands.
- **`sycl_compatible<T>`**: `std::is_trivially_copyable_v<std::remove_cvref_t<T>>`.
- **`KEISAN_NO_INTRINSICS`**: Disables `<simd>` and x86 streaming intrinsics for pure portable scalar fallback.
- **`KEISAN_STRICT_ZERO_MALLOC`**: Throws `std::bad_alloc` if `assign()` attempts to resize a `std::vector`.
- **`KEISAN_NO_EXCEPTIONS`**: Replaces `throw` with `std::abort()` (host) or `__builtin_trap()` (device).
