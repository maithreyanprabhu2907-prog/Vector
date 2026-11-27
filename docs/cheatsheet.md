# `keisan++` (`keisan`) API Cheatsheet

> One-page reference grouped by module, modeled after raylib’s cheatsheet.
> Click any section header for deep architectural rationale in [api_reference.md](api_reference.md).

---

## `kcore` — Expressions, Math Ops & Evaluators

| Symbol | Signature / Form | Purpose |
|---|---|---|
| `assign` | `assign(dst, src)` / `dst = expr` (via `eval_into`) | Fused single-pass evaluation into `std::vector`, `aligned_vector`, or `std::span` |
| `eval_into` | `eval_into(T* dst, size_t n, const Src& src)` | Low-level fused `std::simd` (`eval_simd`) & scalar evaluator; `constexpr` & SYCL device-safe |
| `assign_stream` | `assign_stream(dst, src)` | Non-temporal (`_mm256_stream_*`) cache-bypassing store for multi-MB writes |
| `assign_par` | `assign_par(dst, src)` | OpenMP parallel fused evaluation (`n < 16384` falls back to `assign`; resizes unless `KEISAN_STRICT_ZERO_MALLOC`) |
| `+=`, `-=`, `*=`, `/=` | `dst op= expr` | In-place `std::simd`-fused compound assignment on `std::vector`, `std::span`, `std::array` |
| `+`, `-`, `*`, `/` | `a op b` | Lazy `BinExpr` / `UnaryExpr` nodes (zero heap allocations) |
| `fma` | `fma(a, b, c)` | Fused multiply-add expression (`a * b + c`, supports vector/scalar mixes) |
| `clamp` | `clamp(v, lo, hi)` | Elementwise clamp expression (vector or scalar bounds) |
| `abs`, `sqrt`, `sin`, `cos`, `exp`, `log`, `pow` | `fn(expr)` / `pow(base, exp)` | Lazy elementwise math expressions |

---

## `kmem` — Zero-Allocation Memory & Views

| Symbol | Signature / Form | Purpose |
|---|---|---|
| `aligned_allocator<T, 64>` | `aligned_allocator<T, Align>` | 64-byte cache-line aligned C++ allocator (`aligned_alloc` / `_aligned_malloc`) |
| `aligned_vector<T>` | `std::vector<T, aligned_allocator<T, 64>>` | Owning 64B-aligned container for SIMD & non-temporal streaming |
| `StridedSpan<T>` | `StridedSpan(T* ptr, size_t size, size_t stride)` | Non-owning strided view (`ptr[i * stride]`), `TriviallyCopyable` |
| `WorkspaceArena` | `WorkspaceArena(size_t capacity_bytes)` | 64B-aligned bump allocator for zero-malloc RK stages (`allocate<T>(n)`, `reset()`) |

---

## `kreduce` — Reductions, Norms & Reproducibility

| Symbol | Signature / Form | Purpose |
|---|---|---|
| `ReductionPolicy` | `FAST`, `REPRO`, `DETERMINISTIC` | Controls speed vs. bit-identical IEEE-754 pairwise tree reduction |
| `sum` | `sum(expr, policy = FAST)` | Elementwise sum (`std::simd` dual-accumulator in `FAST`; pairwise tree in `REPRO`) |
| `sum_repro` / `sum_par` | `sum_repro(expr)` / `sum_par(expr)` | Explicit bit-reproducible or OpenMP multi-core sum |
| `dot` | `dot(lhs, rhs, policy = FAST)` | Fused inner product (`std::simd` FMA in `FAST`; pairwise tree in `REPRO`) |
| `dot_repro` / `dot_par` | `dot_repro(lhs, rhs)` / `dot_par(lhs, rhs)` | Explicit bit-reproducible or OpenMP multi-core dot product |
| `norm` / `norm_sq` / `norm_l1` / `norm_inf` | `norm*(expr)` | `std::simd` $L_2$, squared $L_2$, dual-acc $L_1$, and `NaN`-propagating $L_\infty$ norms |
| `min_val` / `max_val` | `min_val(expr)` / `max_val(expr)` | `std::simd`-accelerated minimum / maximum element (throws on empty input) |

---

## `ktensor` — Matrices, Tensors & Batched Solvers

| Symbol | Signature / Form | Purpose |
|---|---|---|
| `matvec` | `matvec(A, x, y, rows, cols)` | Row-major $y = Ax$ (`std::simd` dual-accumulator + alias guard) |
| `matvec_transposed` | `matvec_transposed(A, x, y, rows, cols)` | Row-major $y = A^T x$ (unit-stride inner FMAs) |
| `matvec_fixed<R, C>` | `matvec_fixed<Rows, Cols>(A, x, y)` | Compile-time sized $y = Ax$ with alias-safe stack buffering |
| `matvec_transposed_fixed<R, C>` | `matvec_transposed_fixed<Rows, Cols>(A, x, y)` | Compile-time sized $y = A^T x$ |
| `contract_axis<Axis, N>` | `contract_axis<Axis::X\|Y\|Z, N>(D, in, out, nx, ny, nz)` | 3D tensor sum-factorization contraction along X, Y, or Z (`std::simd` Y/Z slices) |
| `solve_nxn_batched<N>` | `solve_nxn_batched<N>(A_soa, b_soa, x_soa, count)` | `std::simd` SoA batched $N\times N$ solve with partial pivoting & scale-invariant singularity check |
| `solve_4x4_batched` | `solve_4x4_batched(A_soa, b_soa, x_soa, count)` | Convenience wrapper for `solve_nxn_batched<4>` |
| `gather` / `scatter_add` | `gather(dst, src, idx)` / `scatter_add(dst, idx, src)` | Indirect surface flux gather and accumulation (`ptr` or `std::span`) |
| `broadcast_multiply` | `broadcast_multiply(data, count, factor)` | In-place scalar scaling (`ptr` or `std::span`) |
| `transpose_aos_to_soa` / `transpose_soa_to_aos` | `transpose_*(src, dst, num_elem, num_comp)` | AoS ↔ SoA layout conversion for vectorization |

---

## `ksimd` — Concepts & Compile-Time Macros

| Symbol | Kind | Purpose |
|---|---|---|
| `Scalar<T>`, `NumericVector<T>`, `VecExpression<T>`, `VectorOperand<T>` | C++26 Concepts | Constrain valid numeric types, views, and lazy expressions |
| `sycl_compatible<T>` / `is_sycl_compatible_v<T>` | Concept / Trait | Verifies `std::is_trivially_copyable_v` for device kernel capture |
| `real_of_t<T>` | Type Trait | Extracts underlying real scalar type (`std::complex<T>` → `T`) |
| `KEISAN_NO_INTRINSICS` | Macro | Disables `<simd>` & x86 intrinsics, forcing pure portable scalar loops |
| `KEISAN_STRICT_ZERO_MALLOC` | Macro | Throws `std::bad_alloc` if `assign` / `assign_par` would resize a `std::vector` (`capacity() < n`) |
| `KEISAN_NO_EXCEPTIONS` | Macro | Replaces `throw` with `abort()` / `__builtin_trap()` for device/noexcept builds |
