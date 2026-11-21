#!/usr/bin/env bash
# tools/metrics.sh — run the per-commit measurement protocol against a vec_math checkout on Linux/macOS
#
# Usage:
#   ./tools/metrics.sh                  # Tier 1 (default, ~8 min)
#   ./tools/metrics.sh --tier 0         # Tier 0  (~2 min, per commit)
#   ./tools/metrics.sh --tier 2         # Tier 2  (~20 min, per release)
#   ./tools/metrics.sh --quick          # Run benchmark with --quick
#
# Writes docs/metrics/<yyyy-MM-dd>-<sha>.md and prints the same table to stdout.
# Exit code: 0 if no FAIL, 1 if any FAIL-level check failed.

set -o pipefail

TIER=1
QUICK=-1
FULL=0
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
UPSTREAM="$(cd "${SCRIPT_DIR}/.." && pwd)"
REPO="$UPSTREAM"
OUTDIR=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --tier)
            TIER="$2"
            shift 2
            ;;
        --tier=*)
            TIER="${1#*=}"
            shift 1
            ;;
        --quick)
            QUICK=1
            shift 1
            ;;
        --full)
            FULL=1
            QUICK=0
            shift 1
            ;;
        --upstream)
            UPSTREAM="$2"
            shift 2
            ;;
        --repo)
            REPO="$2"
            shift 2
            ;;
        --outdir)
            OUTDIR="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

if [[ $QUICK -eq -1 ]]; then
    if [[ $TIER -le 1 && $FULL -eq 0 ]]; then
        QUICK=1
    else
        QUICK=0
    fi
fi

if [[ -z "$OUTDIR" ]]; then
    OUTDIR="${REPO}/docs/metrics"
fi
mkdir -p "$OUTDIR"

TMP_DIR="${TMPDIR:-/tmp}/vecmath_metrics_$$"
mkdir -p "$TMP_DIR"
trap 'rm -rf "$TMP_DIR"' EXIT

CXX="${CXX:-g++}"
CLANGXX="${CLANGXX:-clang++}"
CMAKE="${CMAKE:-cmake}"
INCLUDE_DIR="${UPSTREAM}/include"

STAMP="$(date +%Y-%m-%d)"
SHA="$(git -C "$UPSTREAM" rev-parse --short HEAD 2>/dev/null || echo "unknown")"
REPORT="${OUTDIR}/${STAMP}-${SHA}.md"

ARCH="$(uname -m)"
ARCH_FLAGS=""
if [[ "$ARCH" =~ ^(x86_64|amd64)$ ]]; then
    ARCH_FLAGS="-mavx2 -mfma"
fi

ANY_FAIL=0
declare -a RESULTS_GATE
declare -a RESULTS_NAME
declare -a RESULTS_VALUE
declare -a RESULTS_STATUS
declare -a RESULTS_NOTE

add_result() {
    local gate="$1"
    local name="$2"
    local value="$3"
    local status="$4"
    local note="${5:-}"

    RESULTS_GATE+=("$gate")
    RESULTS_NAME+=("$name")
    RESULTS_VALUE+=("$value")
    RESULTS_STATUS+=("$status")
    RESULTS_NOTE+=("$note")

    if [[ "$status" == "FAIL" ]]; then
        ANY_FAIL=1
    fi

    local tag=" -- "
    case "$status" in
        PASS) tag="PASS" ;;
        WARN) tag="WARN" ;;
        FAIL) tag="FAIL" ;;
    esac

    printf "  [%s] %-34s %s\n" "$tag" "$name" "$value"
}

echo ""
echo -e "\033[36mvec_math metrics  sha=${SHA}  tier=${TIER}\033[0m"
echo "upstream: ${UPSTREAM}"
echo "report:   ${REPORT}"
echo ""

# --------------------------------------------------------------------- 0. env
echo -e "\033[33m0. environment\033[0m"
CPU="unknown"
if [[ -f /proc/cpuinfo ]]; then
    CPU="$(grep -m1 "model name" /proc/cpuinfo | cut -d: -f2 | sed 's/^[ \t]*//')"
elif command -v sysctl &>/dev/null; then
    CPU="$(sysctl -n machdep.cpu.brand_string 2>/dev/null || echo "unknown")"
fi

CORES=1
if [[ -f /proc/cpuinfo ]]; then
    PHYS_CORES="$(grep "core id" /proc/cpuinfo | sort -u | wc -l)"
    if [[ "$PHYS_CORES" -gt 0 ]]; then
        CORES="$PHYS_CORES"
    elif command -v nproc &>/dev/null; then
        CORES="$(nproc)"
    fi
elif command -v sysctl &>/dev/null; then
    CORES="$(sysctl -n hw.physicalcpu 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 1)"
elif command -v nproc &>/dev/null; then
    CORES="$(nproc)"
fi

CXX_VER="$($CXX --version 2>/dev/null | head -n 1 || echo "not found")"
HEADER_LINES=$(wc -l < "${INCLUDE_DIR}/vec_math/vector_math.hpp" 2>/dev/null || echo 0)

add_result "env" "upstream sha" "$SHA" "INFO" "$UPSTREAM"
add_result "env" "cpu" "${CPU} / ${CORES} cores" "INFO"
add_result "env" "compiler" "$CXX_VER" "INFO"
add_result "env" "header lines" "$HEADER_LINES" "INFO"

# --------------------------------------------------------------- 1. G1 license
echo -e "\n\033[33m1. G1 license gate\033[0m"
LIC_PATH="${UPSTREAM}/LICENSE"
if [[ -f "$LIC_PATH" ]]; then
    WORDS=$(wc -w < "$LIC_PATH")
    HAS_GRANT=0
    HAS_WARRANTY=0
    if grep -q "Permission is hereby granted" "$LIC_PATH"; then HAS_GRANT=1; fi
    if grep -q "WITHOUT WARRANTY" "$LIC_PATH"; then HAS_WARRANTY=1; fi

    if [[ $HAS_GRANT -eq 1 && $HAS_WARRANTY -eq 1 ]]; then
        add_result "G1" "license text" "MIT grant present (${WORDS} words)" "PASS"
    elif [[ $WORDS -lt 40 ]]; then
        add_result "G1" "license text" "stub only (${WORDS} words, no grant)" "FAIL" "GitHub reports 'no license'"
    else
        add_result "G1" "license text" "partial (${WORDS} words)" "WARN"
    fi
else
    add_result "G1" "license text" "LICENSE missing" "FAIL"
fi

# ------------------------------------------------------------------ 2. G2 gate
declare -A BENCH_KV
PEAK_GFLOPS=50.0

test_published_numbers() {
    echo -e "\n\033[33m2. G2 published-numbers gate\033[0m"
    local peak="${BENCH_KV[PEAK_GFLOPS]:-50.0}"
    local impossible=$(awk "BEGIN {print $peak * 10.0}")
    local bad=()

    # (a) Check live benchmark output
    local exe=""
    if [[ -f "${TMP_DIR}/upstream_bench_o3" ]]; then
        exe="${TMP_DIR}/upstream_bench_o3"
    elif [[ -f "${UPSTREAM}/build/bench_o3" ]]; then
        exe="${UPSTREAM}/build/bench_o3"
    else
        local liveSrc="${UPSTREAM}/benchmarks/opt_compare.cpp"
        if [[ -f "$liveSrc" ]]; then
            $CXX -std=c++20 -O3 $ARCH_FLAGS -DOPT_O3 -I "$INCLUDE_DIR" "$liveSrc" -o "${TMP_DIR}/upstream_bench_o3" &>/dev/null || true
            if [[ -f "${TMP_DIR}/upstream_bench_o3" ]]; then
                exe="${TMP_DIR}/upstream_bench_o3"
            fi
        fi
    fi

    if [[ -n "$exe" && -x "$exe" ]]; then
        local out
        out="$("$exe" 2>&1 || true)"
        while IFS= read -r line; do
            if [[ "$line" =~ [Dd]ot:[[:space:]]*([0-9.eE+-]+)[[:space:]]*GFLOPS ]]; then
                local v="${BASH_REMATCH[1]}"
                local cmp=$(awk "BEGIN {print ($v > $impossible) ? 1 : 0}")
                if [[ $cmp -eq 1 ]]; then
                    bad+=("live $(basename "$exe"): Dot = $v GFLOPS")
                fi
            fi
        done <<< "$out"
    fi

    # (b) Committed result files
    while IFS= read -r f; do
        [[ -f "$f" ]] || continue
        while IFS= read -r line; do
            if [[ "$line" =~ ^[[:space:]]*[-a-zA-Z0-9_]+[[:space:]]+([0-9]+)[[:space:]]+dot[[:space:]]+[0-9.eE+-]+[[:space:]]+([0-9.eE+-]+) ]]; then
                local n="${BASH_REMATCH[1]}"
                local v="${BASH_REMATCH[2]}"
                local cmp=$(awk "BEGIN {print ($v > $impossible) ? 1 : 0}")
                if [[ $cmp -eq 1 ]]; then
                    bad+=("$(basename "$f"): N=$n Dot = $v GFLOPS")
                fi
            elif [[ "$line" =~ [Dd]ot:[[:space:]]*([0-9.eE+-]+)[[:space:]]*GFLOPS ]]; then
                local v="${BASH_REMATCH[1]}"
                local cmp=$(awk "BEGIN {print ($v > $impossible) ? 1 : 0}")
                if [[ $cmp -eq 1 ]]; then
                    bad+=("$(basename "$f"): Dot = $v GFLOPS")
                fi
            fi
        done < "$f"
    done < <(find "$UPSTREAM" -maxdepth 2 -name "*.txt" 2>/dev/null)

    if [[ ${#bad[@]} -eq 0 ]]; then
        add_result "G2" "published GFLOPS values" "none above ${impossible} GFLOPS (10x peak)" "PASS"
    else
        add_result "G2" "published GFLOPS values" "${#bad[@]} value(s) above ${impossible} GFLOPS" "FAIL" "${bad[0]}"
    fi
}

# ------------------------------------------------- 3. build + run test matrix
if [[ $TIER -ge 1 ]]; then
    echo -e "\n\033[33m3. build + run matrix (GCC)\033[0m"
    TEST_SRC="${UPSTREAM}/tests/vector_math_test.cpp"

    configs=(
        "O0:-O0"
        "O2:-O2"
        "O3:-O3"
        "Ofast:-Ofast"
        "O2 arch:-O2 ${ARCH_FLAGS}"
        "O3 arch:-O3 ${ARCH_FLAGS}"
        "NO_INTRINSICS:-O2 ${ARCH_FLAGS} -DVEC_MATH_NO_INTRINSICS"
        "NO_CHECKS:-O3 ${ARCH_FLAGS} -DVEC_MATH_NO_CHECKS"
        "O2 openmp:-O2 -fopenmp"
        "Wall Werror:-O3 -Wall -Wextra -Werror"
    )

    for item in "${configs[@]}"; do
        cfg_name="${item%%:*}"
        cfg_flags="${item#*:}"
        clean_name=$(echo "$cfg_name" | tr -c '[:alnum:]' '_')
        exe="${TMP_DIR}/t_${clean_name}"

        # compile
        # shellcheck disable=SC2086
        if ! $CXX -std=c++20 $cfg_flags -I "$INCLUDE_DIR" "$TEST_SRC" -o "$exe" 2>"${TMP_DIR}/build.err"; then
            err_line=$(head -n 1 "${TMP_DIR}/build.err" | sed 's/^[ \t]*//')
            add_result "build" "$cfg_name" "build failed: $err_line" "FAIL"
            continue
        fi

        # run
        run_out="$("$exe" 2>&1 || true)"
        last_line=$(echo "$run_out" | grep -v '^[[:space:]]*$' | tail -n 1)

        if [[ "$run_out" =~ PASSED ]]; then
            add_result "build" "$cfg_name" "tests pass" "PASS"
        elif [[ "$cfg_name" == "NO_CHECKS" ]]; then
            add_result "build" "$cfg_name" "tests abort: $last_line" "FAIL" "config changes error semantics and is untested"
        else
            add_result "build" "$cfg_name" "run failed: $last_line" "FAIL"
        fi
    done
fi

# ----------------------------------------------------------- 4. Clang builds
if [[ $TIER -ge 2 ]] && command -v "$CLANGXX" &>/dev/null; then
    echo -e "\n\033[33m4. Clang builds\033[0m"
    TEST_SRC="${UPSTREAM}/tests/vector_math_test.cpp"
    exe="${TMP_DIR}/t_clang_release"

    # shellcheck disable=SC2086
    if $CLANGXX -std=c++20 -O3 $ARCH_FLAGS -Wno-pass-failed -I "$INCLUDE_DIR" "$TEST_SRC" -o "$exe" 2>"${TMP_DIR}/clang_build.err"; then
        run_out="$("$exe" 2>&1 || true)"
        if [[ "$run_out" =~ PASSED ]]; then
            add_result "build" "Clang Release" "tests pass" "PASS"
        else
            add_result "build" "Clang Release" "run failed" "FAIL"
        fi
    else
        add_result "build" "Clang Release" "build failed" "FAIL"
    fi
fi

# ----------------------------------------------------------- 5. CMake + ctest
if [[ $TIER -ge 2 ]] && command -v "$CMAKE" &>/dev/null; then
    echo -e "\n\033[33m5. CMake + ctest\033[0m"
    BUILD_DIR="${TMP_DIR}/cmake_build"
    rm -rf "$BUILD_DIR"
    mkdir -p "$BUILD_DIR"

    if $CMAKE -S "$UPSTREAM" -B "$BUILD_DIR" -DCMAKE_BUILD_TYPE=Release -DVEC_MATH_BUILD_TESTS=ON -DVEC_MATH_BUILD_BENCHMARKS=OFF &>/dev/null; then
        if $CMAKE --build "$BUILD_DIR" -j "$CORES" &>/dev/null; then
            ctest_out="$(ctest --test-dir "$BUILD_DIR" --output-on-failure 2>&1 || true)"
            if [[ "$ctest_out" =~ 100%[[:space:]]+tests[[:space:]]+passed ]]; then
                add_result "build" "ctest" "100% tests passed" "PASS"
            else
                add_result "build" "ctest" "tests failed in ctest" "FAIL"
            fi
        else
            add_result "build" "ctest" "build failed" "FAIL"
        fi
    else
        add_result "build" "ctest" "cmake config failed" "FAIL"
    fi
fi

# ------------------------------------------------------------------- 6. probe
echo -e "\n\033[33m6. conformance probe\033[0m"
PROBE_SRC="${REPO}/tools/probe.cpp"
PROBE_EXE="${TMP_DIR}/probe"

# shellcheck disable=SC2086
if ! $CXX -std=c++20 -O2 $ARCH_FLAGS -I "$INCLUDE_DIR" "$PROBE_SRC" -o "$PROBE_EXE" 2>"${TMP_DIR}/probe.err"; then
    add_result "probe" "compile" "probe.cpp did not compile" "FAIL"
else
    probe_out="$("$PROBE_EXE" 2>&1 || true)"
    if [[ "$probe_out" =~ PASS=([0-9]+)[[:space:]]+FAIL=([0-9]+)[[:space:]]+MISSING=([0-9]+) ]]; then
        p="${BASH_REMATCH[1]}"
        f="${BASH_REMATCH[2]}"
        m="${BASH_REMATCH[3]}"
        status="PASS"
        if [[ $f -ne 0 || $m -ne 0 ]]; then status="FAIL"; fi
        add_result "probe" "checks" "PASS=${p} FAIL=${f} MISSING=${m}" "$status"

        while IFS= read -r line; do
            if [[ "$line" =~ ^(FAIL|MISSING)[[:space:]] ]]; then
                add_result "probe" "finding" "$line" "FAIL"
            fi
        done <<< "$probe_out"
    else
        add_result "probe" "run" "no summary line" "FAIL"
    fi
fi

# ------------------------------------------------------------------- 7. bench
echo -e "\n\033[33m8. benchmark\033[0m"
BENCH_SRC="${REPO}/tools/bench.cpp"
BENCH_EXE="${TMP_DIR}/dgbench_o3"

# shellcheck disable=SC2086
if ! $CXX -std=c++20 -O3 $ARCH_FLAGS -fopenmp -I "$INCLUDE_DIR" "$BENCH_SRC" -o "$BENCH_EXE" 2>"${TMP_DIR}/bench.err"; then
    add_result "bench" "compile" "bench.cpp did not compile" "FAIL"
else
    bench_args=()
    if [[ $QUICK -eq 1 ]]; then bench_args+=("--quick"); fi

    bench_out="$("$BENCH_EXE" "${bench_args[@]}" 2>&1 || true)"
    BENCH_LOG="$bench_out"

    while IFS= read -r line; do
        if [[ "$line" =~ ^([A-Z0-9_]+)=([-+0-9.eE]+) ]]; then
            BENCH_KV["${BASH_REMATCH[1]}"]="${BASH_REMATCH[2]}"
        elif [[ "$line" =~ ^([A-Z0-9_]+)=(OK|NOOP|SKIPPED|IMPOSSIBLE) ]]; then
            BENCH_KV["${BASH_REMATCH[1]}"]="${BASH_REMATCH[2]}"
        fi
    done <<< "$bench_out"

    if [[ "${BENCH_KV[MUTATION]}" == "OK" ]]; then
        add_result "bench" "in-loop mutation" "mutation exceeds half-ULP and is observed" "PASS"
    else
        add_result "bench" "in-loop mutation" "NOOP — every timing is a hoisted loop" "FAIL"
    fi

    if [[ -n "${BENCH_KV[IMPOSSIBLE_COUNT]}" ]]; then
        ic="${BENCH_KV[IMPOSSIBLE_COUNT]}"
        st="PASS"
        if [[ "$ic" -ne 0 ]]; then st="FAIL"; fi
        add_result "bench" "impossible numbers" "${ic} kernel(s) > 10x measured peak" "$st"
    fi

    for k in $(echo "${!BENCH_KV[@]}" | tr ' ' '\n' | sort); do
        if [[ "$k" =~ _SPREAD$ ]]; then
            v="${BENCH_KV[$k]}"
            st="PASS"
            cmp_warn=$(awk "BEGIN {print ($v > 1.5) ? 1 : 0}")
            cmp_fail=$(awk "BEGIN {print ($v > 2.0) ? 1 : 0}")
            if [[ $cmp_fail -eq 1 ]]; then st="FAIL"; elif [[ $cmp_warn -eq 1 ]]; then st="WARN"; fi
            if [[ "$k" != "PEAK_COPY_GBPS_SPREAD" ]]; then
                add_result "bench" "$k" "$v" "$st" "max/min over trials"
            fi
        fi
    done

    for k in PEAK_GFLOPS PEAK_COPY_GBPS DOT_GBPS DOT_PAR_GBPS DOT_REPRO_GBPS \
             MATVEC_NP10_GFLOPS MATVEC_NP36_GFLOPS MATVEC_FIXED_NP36_GFLOPS \
             SF_N4_SPEEDUP SF_N6_SPEEDUP SOLVE4_MELEM_S; do
        if [[ -n "${BENCH_KV[$k]}" ]]; then
            add_result "bench" "$k" "${BENCH_KV[$k]}" "INFO"
        fi
    done

    # scaling
    if [[ -n "${BENCH_KV[DOT_GBPS]}" && -n "${BENCH_KV[DOT_PAR_GBPS]}" ]]; then
        scale=$(awk "BEGIN {d=(${BENCH_KV[DOT_GBPS]} > 1e-9) ? ${BENCH_KV[DOT_GBPS]} : 1e-9; printf \"%.2f\", ${BENCH_KV[DOT_PAR_GBPS]} / d}")
        st="WARN"
        cmp_pass=$(awk "BEGIN {print ($scale >= 1.5) ? 1 : 0}")
        if [[ $cmp_pass -eq 1 ]]; then st="PASS"; fi
        add_result "bench" "dot_par scaling at max threads" "${scale}x" "$st"
    fi
fi

# G2 depends on benchmark peak
test_published_numbers

# ------------------------------------------------- 9. codegen sensitivity
if [[ $TIER -ge 1 && -f "$BENCH_EXE" ]]; then
    echo -e "\n\033[33m9. codegen sensitivity (-O2 vs -O3)\033[0m"
    BENCH_O2="${TMP_DIR}/dgbench_o2"
    # shellcheck disable=SC2086
    if $CXX -std=c++20 -O2 $ARCH_FLAGS -fopenmp -I "$INCLUDE_DIR" "$BENCH_SRC" -o "$BENCH_O2" &>/dev/null; then
        bench_args=()
        if [[ $QUICK -eq 1 ]]; then bench_args+=("--quick"); fi
        b2_out="$("$BENCH_O2" "${bench_args[@]}" 2>&1 || true)"
        declare -A BENCH_O2_KV
        while IFS= read -r line; do
            if [[ "$line" =~ ^([A-Z0-9_]+)=([-+0-9.eE]+) ]]; then
                BENCH_O2_KV["${BASH_REMATCH[1]}"]="${BASH_REMATCH[2]}"
            fi
        done <<< "$b2_out"

        unstable=()
        for k in MATVEC_NP10_GFLOPS MATVEC_NP20_GFLOPS MATVEC_NP36_GFLOPS MATVEC_NP64_GFLOPS \
                 MATVEC_FIXED_NP10_GFLOPS MATVEC_FIXED_NP20_GFLOPS MATVEC_FIXED_NP36_GFLOPS MATVEC_FIXED_NP64_GFLOPS \
                 SF_N4_GFLOPS SF_N6_GFLOPS; do
            if [[ -n "${BENCH_KV[$k]}" && -n "${BENCH_O2_KV[$k]}" ]]; then
                a="${BENCH_KV[$k]}"
                b="${BENCH_O2_KV[$k]}"
                ratio=$(awk "BEGIN {
                    max = ($a > $b) ? $a : $b;
                    min = ($a < $b) ? $a : $b;
                    if (min < 1e-9) min = 1e-9;
                    printf \"%.2f\", max / min;
                }")
                cmp=$(awk "BEGIN {print ($ratio > 1.5) ? 1 : 0}")
                if [[ $cmp -eq 1 ]]; then
                    unstable+=("${k} ${a}/${b} = ${ratio}x")
                fi
            fi
        done

        if [[ ${#unstable[@]} -eq 0 ]]; then
            add_result "codegen" "O2 vs O3" "all kernels within 1.5x" "PASS"
        else
            joined=$(IFS="; "; echo "${unstable[*]}")
            add_result "codegen" "O2 vs O3" "${#unstable[@]} kernel(s) swing >1.5x" "WARN" "$joined"
        fi
    fi
fi

# ------------------------------------------------------------ 10. determinism
if [[ $TIER -ge 1 ]]; then
    echo -e "\n\033[33m10. determinism\033[0m"
    DET_SRC="${TMP_DIR}/det.cpp"
    cat << 'EOF' > "$DET_SRC"
#include <vec_math/vector_math.hpp>
#include <vector>
#include <cstdio>
using namespace vec_math;
int main() {
    std::vector<double> v(1000003);
    for (std::size_t i = 0; i < v.size(); ++i) v[i] = 1.0 / double(i + 1.0) * (i % 2 ? -1 : 1);
    std::printf("sum_repro=%.17g\n", (double)sum_repro(v));
    std::printf("dot_repro=%.17g\n", (double)dot_repro(v, v));
    std::printf("sum_det=%.17g\n", (double)sum_deterministic(v));
    return 0;
}
EOF
    DET_O2="${TMP_DIR}/det_o2"
    DET_OFAST="${TMP_DIR}/det_ofast"

    # shellcheck disable=SC2086
    $CXX -std=c++20 -O2 -fopenmp -I "$INCLUDE_DIR" "$DET_SRC" -o "$DET_O2" &>/dev/null
    # shellcheck disable=SC2086
    $CXX -std=c++20 -Ofast -fopenmp -I "$INCLUDE_DIR" "$DET_SRC" -o "$DET_OFAST" &>/dev/null

    declare -A THREAD_SUM_REPRO
    declare -A THREAD_DOT_REPRO
    declare -A THREAD_SUM_DET

    for t in 1 2 4 8 12; do
        out="$(OMP_NUM_THREADS=$t "$DET_O2" 2>/dev/null || true)"
        while IFS= read -r line; do
            if [[ "$line" =~ ^sum_repro=([0-9.eE+-]+) ]]; then THREAD_SUM_REPRO[$t]="${BASH_REMATCH[1]}"; fi
            if [[ "$line" =~ ^dot_repro=([0-9.eE+-]+) ]]; then THREAD_DOT_REPRO[$t]="${BASH_REMATCH[1]}"; fi
            if [[ "$line" =~ ^sum_det=([0-9.eE+-]+) ]]; then THREAD_SUM_DET[$t]="${BASH_REMATCH[1]}"; fi
        done <<< "$out"
    done

    # Count unique
    uniq_sum_repro=$(echo "${THREAD_SUM_REPRO[@]}" | tr ' ' '\n' | sort -u | wc -l)
    st="PASS"
    if [[ $uniq_sum_repro -gt 1 ]]; then st="FAIL"; fi
    add_result "determinism" "sum_repro across threads 1/2/4/8/12" "${uniq_sum_repro} distinct value(s)" "$st"

    uniq_dot_repro=$(echo "${THREAD_DOT_REPRO[@]}" | tr ' ' '\n' | sort -u | wc -l)
    st="PASS"
    if [[ $uniq_dot_repro -gt 1 ]]; then st="FAIL"; fi
    add_result "determinism" "dot_repro across threads 1/2/4/8/12" "${uniq_dot_repro} distinct value(s)" "$st"

    uniq_sum_det=$(echo "${THREAD_SUM_DET[@]}" | tr ' ' '\n' | sort -u | wc -l)
    st="PASS"
    if [[ $uniq_sum_det -gt 1 ]]; then st="WARN"; fi
    add_result "determinism" "sum_det across threads 1/2/4/8/12" "${uniq_sum_det} distinct value(s)" "$st"

    # Compare -O2 vs -Ofast
    out_o2="$("$DET_O2" 2>/dev/null || true)"
    out_ofast="$("$DET_OFAST" 2>/dev/null || true)"

    s_rep_o2="$(echo "$out_o2" | grep -m1 '^sum_repro=' | cut -d= -f2)"
    s_rep_fa="$(echo "$out_ofast" | grep -m1 '^sum_repro=' | cut -d= -f2)"
    d_rep_o2="$(echo "$out_o2" | grep -m1 '^dot_repro=' | cut -d= -f2)"
    d_rep_fa="$(echo "$out_ofast" | grep -m1 '^dot_repro=' | cut -d= -f2)"

    if [[ "$s_rep_o2" == "$s_rep_fa" ]]; then
        add_result "determinism" "sum_repro across -O2/-Ofast" "identical" "PASS"
    else
        add_result "determinism" "sum_repro across -O2/-Ofast" "-O2 ${s_rep_o2} vs -Ofast ${s_rep_fa}" "FAIL"
    fi

    if [[ "$d_rep_o2" == "$d_rep_fa" ]]; then
        add_result "determinism" "dot_repro across -O2/-Ofast" "identical" "PASS"
    else
        add_result "determinism" "dot_repro across -O2/-Ofast" "-O2 ${d_rep_o2} vs -Ofast ${d_rep_fa}" "FAIL"
    fi
fi

# ----------------------------------------------------------- 11. include cost
if [[ $TIER -ge 1 ]]; then
    echo -e "\n\033[33m11. include cost\033[0m"
    EMPTY_SRC="${TMP_DIR}/empty.cpp"
    echo "" > "$EMPTY_SRC"
    HDR="${INCLUDE_DIR}/vec_math/vector_math.hpp"

    best_ms=999999
    for _ in 1 2 3; do
        t0=$(python3 -c 'import time; print(int(time.time() * 1000))' 2>/dev/null || date +%s%N | cut -b1-13)
        $CXX -std=c++20 -O2 -I "$INCLUDE_DIR" -include "$HDR" -c "$EMPTY_SRC" -o "${TMP_DIR}/e.o" 2>/dev/null
        t1=$(python3 -c 'import time; print(int(time.time() * 1000))' 2>/dev/null || date +%s%N | cut -b1-13)
        diff_ms=$(( t1 - t0 ))
        if [[ $diff_ms -lt $best_ms ]]; then best_ms=$diff_ms; fi
    done

    st="PASS"
    if [[ $best_ms -gt 2500 ]]; then st="FAIL"; elif [[ $best_ms -gt 1500 ]]; then st="WARN"; fi
    add_result "build" "header compile cost (min of 3)" "${best_ms} ms" "$st"
fi

# -------------------------------------------------------- 12. doc snippets
if [[ $TIER -ge 2 ]]; then
    echo -e "\n\033[33m12. doc snippet compile check\033[0m"
    doc_files=()
    if [[ -f "${UPSTREAM}/README.md" ]]; then doc_files+=("${UPSTREAM}/README.md"); fi
    while IFS= read -r f; do
        [[ -f "$f" ]] && doc_files+=("$f")
    done < <(find "${UPSTREAM}/docs" -name "*.md" 2>/dev/null)

    n_ok=0
    n_fail=0
    api_errors=()

    for d in "${doc_files[@]}"; do
        d_name="$(basename "$d")"
        python3 -c '
import sys, re
with open(sys.argv[1], "r", encoding="utf-8") as f:
    text = f.read()
snippets = re.findall(r"```(?:cpp|c\+\+)\n(.*?)```", text, re.DOTALL)
for i, s in enumerate(snippets, 1):
    with open(f"'"$TMP_DIR"'/{sys.argv[2]}_{i}.snip", "w", encoding="utf-8") as out:
        out.write(s)
' "$d" "$d_name" 2>/dev/null || true

        for snip in "$TMP_DIR"/"$d_name"_*.snip; do
            [[ -f "$snip" ]] || continue
            cat << EOF > "${TMP_DIR}/test_snip.cpp"
#include <vec_math/vector_math.hpp>
#include <vector>
#include <span>
#include <array>
using namespace vec_math;
void __snippet() {
$(cat "$snip")
}
EOF
            if $CXX -std=c++20 -fsyntax-only -I "$INCLUDE_DIR" "${TMP_DIR}/test_snip.cpp" 2>"${TMP_DIR}/snip.err"; then
                (( n_ok++ ))
            else
                if grep -Eq 'vec_math|no matching function|no member named' "${TMP_DIR}/snip.err"; then
                    (( n_fail++ ))
                    err_msg="$(grep -Em1 'vec_math|no matching function|no member named' "${TMP_DIR}/snip.err" | sed 's/^[ \t]*//')"
                    api_errors+=("${d_name}: ${err_msg}")
                fi
            fi
        done
    done

    if [[ $n_fail -eq 0 ]]; then
        add_result "docs" "snippets" "${n_ok} compiled, 0 API errors" "PASS"
    else
        add_result "docs" "snippets" "${n_fail} API error(s)" "WARN" "${api_errors[0]}"
    fi
fi

# ----------------------------------------------------------- 13. flamegraph
echo -e "\n\033[33m13. flamegraph\033[0m"
FLAME_PY="${REPO}/tools/generate_flamegraph.py"
[[ -f "$FLAME_PY" ]] || FLAME_PY="${UPSTREAM}/tools/generate_flamegraph.py"

PY_CMD=""
if command -v python3 &>/dev/null; then
    PY_CMD="python3"
elif command -v python &>/dev/null; then
    PY_CMD="python"
fi

FLAME_HTML_NAME=""
FLAME_SVG_NAME=""

if [[ -f "$FLAME_PY" && -n "$PY_CMD" ]]; then
    FLAME_HTML="${OUTDIR}/${STAMP}-${SHA}-flamegraph.html"
    FLAME_SVG="${OUTDIR}/${STAMP}-${SHA}-flamegraph.svg"
    BENCH_LOG_FILE="${TMP_DIR}/bench_log.txt"
    if [[ -n "$BENCH_LOG" ]]; then
        echo "$BENCH_LOG" > "$BENCH_LOG_FILE"
    fi

    flame_args=("$FLAME_PY" "--repo" "$REPO" "--out-html" "$FLAME_HTML" "--out-svg" "$FLAME_SVG" "--title" "vec_math CPU Flamegraph (${SHA})")
    if [[ -f "$BENCH_LOG_FILE" ]]; then
        flame_args+=("--bench-log" "$BENCH_LOG_FILE")
    fi

    if $PY_CMD "${flame_args[@]}" &>/dev/null; then
        if [[ -f "$FLAME_HTML" ]]; then
            FLAME_HTML_NAME="$(basename "$FLAME_HTML")"
            FLAME_SVG_NAME="$(basename "$FLAME_SVG")"
            add_result "flamegraph" "interactive HTML" "generated: ${FLAME_HTML_NAME}" "PASS"
        fi
    else
        add_result "flamegraph" "interactive HTML" "generation failed" "WARN"
    fi
fi

# ------------------------------------------------------------------- report
OVERALL_STATUS="PASS"
if [[ $ANY_FAIL -eq 1 ]]; then OVERALL_STATUS="FAIL"; fi

{
    echo "# vec_math metrics — ${STAMP} @ \`${SHA}\`"
    echo ""
    echo "Generated by \`tools/metrics.sh --tier ${TIER}\`. Machine: ${CPU} / ${CORES} cores."
    echo "Compiler: \`${CXX_VER}\`. Overall: **${OVERALL_STATUS}**."
    echo ""
    echo "## Results"
    echo ""
    echo "| gate | check | value | status | note |"
    echo "|---|---|---|---|---|"
    for i in "${!RESULTS_GATE[@]}"; do
        echo "| ${RESULTS_GATE[$i]} | ${RESULTS_NAME[$i]} | ${RESULTS_VALUE[$i]} | ${RESULTS_STATUS[$i]} | ${RESULTS_NOTE[$i]} |"
    done
    echo ""
    if [[ -n "$FLAME_HTML_NAME" ]]; then
        echo "## Visualizations"
        echo ""
        echo "- [Interactive CPU Flamegraph (HTML)](./${FLAME_HTML_NAME})"
        echo "- [CPU Flamegraph (SVG)](./${FLAME_SVG_NAME})"
        echo ""
    fi
    echo "## Raw benchmark output (-O3)"
    echo ""
    echo '```'
    echo "$BENCH_LOG"
    echo '```'
    echo ""
    echo "## Thresholds"
    echo ""
    echo "See \`docs/metrics.md\` §4. This file only records; the thresholds live there."
} > "$REPORT"

echo ""
if [[ "$OVERALL_STATUS" == "PASS" ]]; then
    echo -e "\033[32mOverall: PASS\033[0m"
else
    echo -e "\033[31mOverall: FAIL\033[0m"
fi
echo "Report:  ${REPORT}"
fail_count=0
for st in "${RESULTS_STATUS[@]}"; do
    if [[ "$st" == "FAIL" ]]; then (( fail_count++ )); fi
done
echo "Checks:  ${#RESULTS_GATE[@]} total, ${fail_count} FAIL"

if [[ $ANY_FAIL -eq 1 ]]; then
    exit 1
else
    exit 0
fi
