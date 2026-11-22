# tools/metrics.ps1 — run the per-commit measurement protocol against a vec_math checkout
#
#   .\tools\metrics.ps1                 # Tier 1 (default, ~8 min)
#   .\tools\metrics.ps1 -Tier 0         # Tier 0  (~2 min, per commit)
#   .\tools\metrics.ps1 -Tier 2         # Tier 2  (~20 min, per release)
#
# Writes docs/metrics/<yyyy-MM-dd>-<sha>.md and prints the same table to stdout.
# Exit code: 0 if no FAIL, 1 if any FAIL-level check failed.
#
# This script does not decide whether vec_math is good. docs/backend-rubric.md
# decides that. This script measures the same 16 properties every commit and
# records the numbers so a score change can be traced to a cause.

param(
    [string]$Upstream = "C:\dev\Vector",
    [string]$Repo     = "C:\dev\Vector",
    [ValidateSet(0, 1, 2)][int]$Tier = 1,
    [string]$OutDir   = "",
    [switch]$Quick,
    [switch]$Full
)

$ErrorActionPreference = "Continue"
$ProgressPreference    = "SilentlyContinue"

# ------------------------------------------------------------------ toolchain
$Gpp  = "C:\Program Files\JetBrains\CLion 2026.3.0.1\bin\mingw\bin\g++.exe"
if (-not (Test-Path $Gpp)) {
    $foundGpp = Get-Command g++ -ErrorAction SilentlyContinue
    if ($foundGpp) { $Gpp = $foundGpp.Source }
}
$Env:PATH = [System.IO.Path]::GetDirectoryName($Gpp) + ";" + $Env:PATH

$Cmake = "C:\Program Files\CMake\bin\cmake.exe"
if (-not (Test-Path $Cmake)) {
    $Cmake = "C:\Program Files\Microsoft Visual Studio\18\Community\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe"
    if (-not (Test-Path $Cmake)) {
        $foundCmake = Get-Command cmake -ErrorAction SilentlyContinue
        if ($foundCmake) { $Cmake = $foundCmake.Source }
    }
}

$VsDevCmd = "C:\Program Files\Microsoft Visual Studio\18\Community\Common7\Tools\VsDevCmd.bat"
$Include  = Join-Path $Upstream "include"
$Tmp      = Join-Path $env:TEMP "vecmath_metrics"
New-Item -ItemType Directory -Force -Path $Tmp | Out-Null
if (-not $OutDir) { $OutDir = Join-Path $Repo "docs\metrics" }
New-Item -ItemType Directory -Force -Path $OutDir | Out-Null

$Stamp  = Get-Date -Format "yyyy-MM-dd"
$Sha    = (& git -C $Upstream rev-parse --short HEAD 2>$null)
if (-not $Sha) { $Sha = "unknown" }
$Report = Join-Path $OutDir "$Stamp-$Sha.md"

$script:Results   = New-Object System.Collections.Generic.List[object]
$script:Bench     = @{}
$script:BenchO2   = @{}
$script:AnyFail   = $false

function Add-Result {
    param([string]$Gate, [string]$Name, [string]$Value, [string]$Status, [string]$Note = "")
    $script:Results.Add([pscustomobject]@{ Gate = $Gate; Check = $Name; Value = $Value; Status = $Status; Note = $Note })
    if ($Status -eq "FAIL") { $script:AnyFail = $true }
    $tag = switch ($Status) { "PASS" { "PASS" } "WARN" { "WARN" } "FAIL" { "FAIL" } default { " -- " } }
    Write-Host ("  [{0}] {1,-34} {2}" -f $tag, $Name, $Value)
}

function Invoke-Native {
    param([string]$Exe, [string[]]$ArgList)
    $out = & $Exe @ArgList 2>&1
    return [pscustomobject]@{ Output = $out; Code = $LASTEXITCODE }
}

function Parse-KeyValues {
    param([string[]]$Lines)
    $h = @{}
    foreach ($l in $Lines) {
        if ($l -match '^([A-Z0-9_]+)=([-+0-9.eE]+)') { $h[$Matches[1]] = [double]$Matches[2] }
        elseif ($l -match '^([A-Z0-9_]+)=(OK|NOOP|SKIPPED|IMPOSSIBLE)') { $h[$Matches[1]] = $Matches[2] }
    }
    return $h
}

Write-Host ""
Write-Host "vec_math metrics  sha=$Sha  tier=$Tier" -ForegroundColor Cyan
Write-Host ("upstream: {0}" -f $Upstream)
Write-Host ("report:   {0}" -f $Report)
Write-Host ""

# --------------------------------------------------------------------- 0. env
Write-Host "0. environment" -ForegroundColor Yellow
$Cpu = (Get-CimInstance Win32_Processor | Select-Object -First 1).Name
$Cores = (Get-CimInstance Win32_Processor | Select-Object -First 1).NumberOfCores
$GppVer = (& $Gpp --version | Select-Object -First 1)
Add-Result "env" "upstream sha" $Sha "INFO" $Upstream
Add-Result "env" "cpu" ("{0} / {1} cores" -f $Cpu.Trim(), $Cores) "INFO"
Add-Result "env" "compiler" $GppVer "INFO"
Add-Result "env" "header lines" ((Get-Content (Join-Path $Include "vec_math\vector_math.hpp") | Measure-Object -Line).Lines) "INFO"

# --------------------------------------------------------------- 1. G1 license
Write-Host "`n1. G1 license gate" -ForegroundColor Yellow
$licPath = Join-Path $Upstream "LICENSE"
if (Test-Path $licPath) {
    $lic = Get-Content $licPath -Raw
    $words = ($lic -split '\s+' | Where-Object { $_ }).Count
    $hasGrant = $lic -match 'Permission is hereby granted'
    $hasWarranty = $lic -match 'WITHOUT WARRANTY'
    if ($hasGrant -and $hasWarranty) {
        Add-Result "G1" "license text" "MIT grant present ($words words)" "PASS"
    } elseif ($words -lt 40) {
        Add-Result "G1" "license text" "stub only ($words words, no grant)" "FAIL" "GitHub reports 'no license'"
    } else {
        Add-Result "G1" "license text" "partial ($words words)" "WARN"
    }
} else {
    Add-Result "G1" "license text" "LICENSE missing" "FAIL"
}

# ------------------------------------------------------------------ 2. G2 gate
# Runs after the benchmark, because "implausible" is defined relative to *our*
# measured peak on this machine, not to a constant.
function Test-PublishedNumbers {
    Write-Host "`n2. G2 published-numbers gate" -ForegroundColor Yellow
    $peak = if ($script:Bench.ContainsKey('PEAK_GFLOPS')) { $script:Bench['PEAK_GFLOPS'] } else { 50.0 }
    $impossible = $peak * 10.0
    $bad = @()

    # (a) the live benchmark — the ground truth. Build it if no binary exists:
    #     the committed .txt files are stale, the fabricated numbers only appear
    #     when the benchmark actually runs.
    $exe = Get-ChildItem -Path $Upstream, $Tmp -Filter "bench_o3.exe" -Recurse -File -ErrorAction SilentlyContinue | Select-Object -First 1
    if (-not $exe) {
        $liveSrc = Join-Path $Upstream "benchmarks\opt_compare.cpp"
        if (Test-Path $liveSrc) {
            $liveExe = Join-Path $Tmp "upstream_bench_o3.exe"
            Remove-Item $liveExe -ErrorAction SilentlyContinue
            & $Gpp -std=c++20 -O3 -mavx2 -mfma -DOPT_O3 -I $Include $liveSrc -o $liveExe 2>&1 | Out-Null
            if (Test-Path $liveExe) { $exe = Get-Item $liveExe }
        }
    }
    if ($exe) {
        Push-Location $Tmp
        try { $r = Invoke-Native $exe.FullName @() } finally { Pop-Location }
        foreach ($line in $r.Output) {
            if ($line -match '(?i)dot:\s*([0-9.e+]+)\s*GFLOPS') {
                $v = [double]$Matches[1]
                if ($v -gt $impossible) { $bad += ("live {0}: Dot = {1:G4} GFLOPS" -f $exe.Name, $v) }
            }
        }
        if (-not ($r.Output | Select-String '(?i)dot:')) {
            Add-Result "G2" "live benchmark" "ran but produced no dot line" "WARN"
        }
    } else {
        Add-Result "G2" "live benchmark" "could not build opt_compare.cpp" "WARN"
    }

    # (b) committed result files, in either format:
    #     "-O3 5000000 dot 5977 1e+06 12345"      (space separated)
    #     "Dot: 1e+06 GFLOPS"                      (live output pasted in)
    foreach ($f in (Get-ChildItem -Path $Upstream -Filter "*.txt" -File -ErrorAction SilentlyContinue)) {
        foreach ($line in (Get-Content $f.FullName -ErrorAction SilentlyContinue)) {
            if ($line -match '^\s*[-\w]+\s+(\d+)\s+dot\s+[0-9.e+-]+\s+([0-9.eE+-]+)') {
                $v = [double]$Matches[2]
                if ($v -gt $impossible) { $bad += ("{0}: N={1} Dot = {2:G4} GFLOPS" -f $f.Name, $Matches[1], $v) }
            }
            elseif ($line -match '(?i)dot:\s*([0-9.e+]+)\s*GFLOPS') {
                $v = [double]$Matches[1]
                if ($v -gt $impossible) { $bad += ("{0}: Dot = {1:G4} GFLOPS" -f $f.Name, $v) }
            }
        }
    }

    if ($bad.Count -eq 0) {
        Add-Result "G2" "published GFLOPS values" ("none above {0:N0} GFLOPS (10x peak)" -f $impossible) "PASS"
    } else {
        Add-Result "G2" "published GFLOPS values" ("{0} value(s) above {1:N0} GFLOPS" -f $bad.Count, $impossible) "FAIL" $bad[0]
    }
}

# ------------------------------------------------- 3. build + run test matrix
if ($Tier -ge 1) {
    Write-Host "`n3. build + run matrix (GCC)" -ForegroundColor Yellow
    $TestSrc = Join-Path $Upstream "tests\vector_math_test.cpp"
    $configs = @(
        @{ N = "O0";             F = @("-O0") },
        @{ N = "O2";             F = @("-O2") },
        @{ N = "O3";             F = @("-O3") },
        @{ N = "Ofast";          F = @("-Ofast") },
        @{ N = "O2 avx2 fma";    F = @("-O2", "-mavx2", "-mfma") },
        @{ N = "O3 avx2 fma";    F = @("-O3", "-mavx2", "-mfma") },
        @{ N = "NO_INTRINSICS";  F = @("-O2", "-mavx2", "-mfma", "-DVEC_MATH_NO_INTRINSICS") },
        @{ N = "NO_CHECKS";      F = @("-O3", "-mavx2", "-mfma", "-DVEC_MATH_NO_CHECKS") },
        @{ N = "O2 openmp";      F = @("-O2", "-fopenmp") },
        @{ N = "Wall Werror";    F = @("-O3", "-Wall", "-Wextra", "-Werror") }
    )
    foreach ($c in $configs) {
        $exe = Join-Path $Tmp ("t_" + ($c.N -replace '[^A-Za-z0-9]', '_') + ".exe")
        Remove-Item $exe -ErrorAction SilentlyContinue
        $b = Invoke-Native $Gpp (@("-std=c++20") + $c.F + @("-I", $Include, $TestSrc, "-o", $exe))
        $errs = @($b.Output | Select-String 'error:')
        if ($errs.Count -gt 0) {
            Add-Result "build" $c.N ("build failed: " + $errs[0].Line.Trim()) "FAIL"
            continue
        }
        $r = Invoke-Native $exe @()
        $last = ($r.Output | Where-Object { $_ -match '\S' } | Select-Object -Last 1)
        if ($r.Code -eq 0 -and $last -match 'PASSED') {
            Add-Result "build" $c.N "tests pass" "PASS"
        } elseif ($c.N -eq "NO_CHECKS") {
            Add-Result "build" $c.N ("tests abort: " + $last) "FAIL" "config changes error semantics and is untested"
        } else {
            Add-Result "build" $c.N ("run failed: " + $last) "FAIL"
        }
    }
}

# ----------------------------------------------------------- 4. MSVC + ctest
if ($Tier -ge 2) {
    Write-Host "`n4. MSVC builds" -ForegroundColor Yellow
    foreach ($cfg in @(@{ N = "MSVC Release /arch:AVX2"; Flags = "/arch:AVX2"; Bt = "Release" },
                       @{ N = "MSVC Debug"; Flags = ""; Bt = "Debug" })) {
        $d = Join-Path $Tmp ("msvc_" + ($cfg.Bt))
        Remove-Item -Recurse -Force $d -ErrorAction SilentlyContinue
        & $Cmake -S $Upstream -B $d -DCMAKE_CXX_FLAGS="$($cfg.Flags)" -DVEC_MATH_BUILD_TESTS=ON -DVEC_MATH_BUILD_BENCHMARKS=OFF 2>&1 | Out-Null
        $out = cmd /c "`"$VsDevCmd`" -arch=x64 -no_logo -host_arch=x64 && `"$Cmake`" --build `"$d`" --config $($cfg.Bt) --target vector_math_test" 2>&1
        $errs = @($out | Select-String 'error')
        $exe = Join-Path $d ("{0}\vector_math_test.exe" -f $cfg.Bt)
        if ($errs.Count -gt 0) { Add-Result "build" $cfg.N "build failed" "FAIL" }
        elseif (Test-Path $exe) {
            $r = Invoke-Native $exe @()
            $last = ($r.Output | Select-Object -Last 1)
            if ($last -match 'PASSED') { Add-Result "build" $cfg.N "tests pass" "PASS" }
            else { Add-Result "build" $cfg.N "tests failed" "FAIL" }
        } else { Add-Result "build" $cfg.N "no binary produced" "FAIL" }
    }

    Write-Host "`n4b. upstream benchmark binary (for the G2 live check)" -ForegroundColor Yellow
    $ub = Join-Path $Tmp "upstream_bench"
    & $Cmake -S $Upstream -B $ub -G "MinGW Makefiles" -DCMAKE_CXX_COMPILER="$Gpp" -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_FLAGS="-mavx2 -mfma" 2>&1 | Out-Null
    & $Cmake --build $ub -j 8 --target bench_o3 2>&1 | Out-Null
    if (Test-Path (Join-Path $ub "bench_o3.exe")) {
        Add-Result "G2" "live benchmark binary" "upstream bench_o3.exe built" "INFO"
    } else {
        Add-Result "G2" "live benchmark binary" "bench_o3 target did not build" "WARN"
    }

    Write-Host "`n5. CMake + ctest" -ForegroundColor Yellow
    $d = Join-Path $Tmp "cmake"
    Remove-Item -Recurse -Force $d -ErrorAction SilentlyContinue
    & $Cmake -S $Upstream -B $d -G "MinGW Makefiles" -DCMAKE_CXX_COMPILER="$Gpp" -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_FLAGS="-mavx2 -mfma" 2>&1 | Out-Null
    & $Cmake --build $d -j 8 2>&1 | Out-Null
    $ct = Invoke-Native $Cmake @("--build", $d, "--target", "test")
    $passLine = ($ct.Output | Select-String 'tests passed|Failed' | Select-Object -First 1)
    if ($passLine -and $passLine.Line -match '100% tests passed') {
        Add-Result "build" "ctest" $passLine.Line.Trim() "PASS"
    } else {
        Add-Result "build" "ctest" "see report" "WARN"
    }
}

# ------------------------------------------------------------------- 6. probe
Write-Host "`n6. conformance probe" -ForegroundColor Yellow
$probeSrc = Join-Path $Repo "tools\probe.cpp"
$probeExe = Join-Path $Tmp "probe.exe"
& $Gpp -std=c++20 -O2 -mavx2 -mfma -I $Include $probeSrc -o $probeExe 2>&1 | Out-Null
if (-not (Test-Path $probeExe)) {
    Add-Result "probe" "compile" "probe.cpp did not compile" "FAIL"
} else {
    $r = Invoke-Native $probeExe @()
    $sum = $r.Output | Select-String 'PASS=\d+ FAIL=\d+ MISSING=\d+'
    if ($sum) {
        $s = $sum.Matches[0].Value
        $p = [int]($s -replace '.*PASS=(\d+).*', '$1')
        $f = [int]($s -replace '.*FAIL=(\d+).*', '$1')
        $m = [int]($s -replace '.*MISSING=(\d+).*', '$1')
        $st = if ($f -eq 0 -and $m -eq 0) { "PASS" } else { "FAIL" }
        Add-Result "probe" "checks" ("PASS={0} FAIL={1} MISSING={2}" -f $p, $f, $m) $st
        foreach ($line in ($r.Output | Select-String '^(FAIL|MISSING) ')) {
            Add-Result "probe" "finding" $line.Line.Trim() "FAIL"
        }
    } else {
        Add-Result "probe" "run" "no summary line" "FAIL"
    }
}
$script:ProbeLog = @()

# ------------------------------------------------------------------- 7. bench
Write-Host "`n8. benchmark" -ForegroundColor Yellow
$benchSrc = Join-Path $Repo "tools\bench.cpp"
function Build-Bench([string]$opt, [string]$outName) {
    $exe = Join-Path $Tmp $outName
    Remove-Item $exe -ErrorAction SilentlyContinue
    & $Gpp -std=c++20 $opt -mavx2 -mfma -fopenmp -I $Include $benchSrc -o $exe 2>&1 | Out-Null
    return $exe
}
$benchExe = Build-Bench "-O3" "dgbench_o3.exe"
if (-not (Test-Path $benchExe)) {
    Add-Result "bench" "compile" "bench.cpp did not compile" "FAIL"
} else {
    $isQuick = $Quick -or ($Tier -le 1 -and -not $Full)
    $quickArg = if ($isQuick) { @("--quick") } else { @() }
    $r = Invoke-Native $benchExe $quickArg
    $script:Bench = Parse-KeyValues $r.Output
    $script:BenchLog = $r.Output
    if ($script:Bench['MUTATION'] -eq 'OK') {
        Add-Result "bench" "in-loop mutation" "mutation exceeds half-ULP and is observed" "PASS"
    } else {
        Add-Result "bench" "in-loop mutation" "NOOP — every timing is a hoisted loop" "FAIL"
    }
    if ($script:Bench.ContainsKey('IMPOSSIBLE_COUNT')) {
        $ic = [int]$script:Bench['IMPOSSIBLE_COUNT']
        Add-Result "bench" "impossible numbers" ("{0} kernel(s) > 10x measured peak" -f $ic) $(if ($ic -eq 0) { "PASS" } else { "FAIL" })
    }
    foreach ($k in ($script:Bench.Keys | Sort-Object)) {
        if ($k -match '_SPREAD$') {
            $v = $script:Bench[$k]
            $st = if ($v -le 1.5) { "PASS" } elseif ($v -le 2.0) { "WARN" } else { "FAIL" }
            if ($k -ne 'PEAK_COPY_GBPS_SPREAD') { Add-Result "bench" $k ("{0:N2}" -f $v) $st "max/min over 5 trials" }
        }
    }
    foreach ($k in @('PEAK_GFLOPS', 'PEAK_COPY_GBPS', 'DOT_GBPS', 'DOT_PAR_GBPS', 'DOT_REPRO_GBPS',
                     'MATVEC_NP10_GFLOPS', 'MATVEC_NP36_GFLOPS', 'MATVEC_FIXED_NP36_GFLOPS',
                     'SF_N4_SPEEDUP', 'SF_N6_SPEEDUP', 'SOLVE4_MELEM_S')) {
        if ($script:Bench.ContainsKey($k)) {
            Add-Result "bench" $k ("{0:N3}" -f $script:Bench[$k]) "INFO"
        }
    }

    # stability as a quality signal: 12-thread dot_par vs 1-thread dot
    if ($script:Bench.ContainsKey('DOT_GBPS') -and $script:Bench.ContainsKey('DOT_PAR_GBPS')) {
        $scale = $script:Bench['DOT_PAR_GBPS'] / [Math]::Max(1e-9, $script:Bench['DOT_GBPS'])
        $st = if ($scale -ge 1.5) { "PASS" } elseif ($scale -ge 1.1) { "WARN" } else { "WARN" }
        Add-Result "bench" "dot_par scaling at max threads" ("{0:N2}x" -f $scale) $st "1.0x means the kernel is bandwidth-bound already"
    }
}

# G2 depends on the measured peak, so it runs after the benchmark.
Test-PublishedNumbers

# ------------------------------------------------- 9. codegen sensitivity
if ($Tier -ge 1 -and (Test-Path (Join-Path $Tmp "dgbench_o3.exe"))) {
    Write-Host "`n9. codegen sensitivity (-O2 vs -O3)" -ForegroundColor Yellow
    $o2 = Build-Bench "-O2" "dgbench_o2.exe"
    if (Test-Path $o2) {
        $r2 = Invoke-Native $o2 $quickArg
        $script:BenchO2 = Parse-KeyValues $r2.Output
        $unstable = @()
        foreach ($k in @('MATVEC_NP10_GFLOPS', 'MATVEC_NP20_GFLOPS', 'MATVEC_NP36_GFLOPS', 'MATVEC_NP64_GFLOPS',
                         'MATVEC_FIXED_NP10_GFLOPS', 'MATVEC_FIXED_NP20_GFLOPS', 'MATVEC_FIXED_NP36_GFLOPS', 'MATVEC_FIXED_NP64_GFLOPS',
                         'SF_N4_GFLOPS', 'SF_N6_GFLOPS')) {
            if ($script:Bench.ContainsKey($k) -and $script:BenchO2.ContainsKey($k)) {
                $a = $script:Bench[$k]; $b = $script:BenchO2[$k]
                $ratio = [Math]::Max($a, $b) / [Math]::Max(1e-9, [Math]::Min($a, $b))
                if ($ratio -gt 1.5) { $unstable += ("{0} {1:N1}/{2:N1} = {3:N1}x" -f $k, $a, $b, $ratio) }
            }
        }
        if ($unstable.Count -eq 0) {
            Add-Result "codegen" "O2 vs O3" "all kernels within 1.5x" "PASS"
        } else {
            Add-Result "codegen" "O2 vs O3" ("{0} kernel(s) swing >1.5x" -f $unstable.Count) "WARN" ($unstable -join "; ")
        }
    }
}

# ------------------------------------------------------------ 10. determinism
if ($Tier -ge 1) {
    Write-Host "`n10. determinism" -ForegroundColor Yellow
    $detSrc = Join-Path $Tmp "det.cpp"
    @'
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
'@ | Set-Content $detSrc -Encoding utf8
    $detExe = Join-Path $Tmp "det_o2.exe"
    & $Gpp -std=c++20 -O2 -fopenmp -I $Include $detSrc -o $detExe 2>&1 | Out-Null
    $byThreads = @{}
    foreach ($t in @(1, 2, 4, 8, 12)) {
        $Env:OMP_NUM_THREADS = "$t"
        $r = Invoke-Native $detExe @()
        $h = Parse-KeyValues $r.Output
        $byThreads[$t] = $h
    }
    $Env:OMP_NUM_THREADS = $null
    foreach ($k in @('sum_repro', 'dot_repro', 'sum_det')) {
        $vals = $byThreads.Keys | Sort-Object | ForEach-Object { $byThreads[$_][$k] }
        $uniq = @($vals | Select-Object -Unique)
        $st = if ($uniq.Count -le 1) { "PASS" } else { "FAIL" }
        if ($k -eq 'sum_det' -and $uniq.Count -gt 1) { $st = "WARN" }
        Add-Result "determinism" "$k across threads 1/2/4/8/12" ("{0} distinct value(s)" -f $uniq.Count) $st
    }
    $detFast = Join-Path $Tmp "det_ofast.exe"
    & $Gpp -std=c++20 -Ofast -fopenmp -I $Include $detSrc -o $detFast 2>&1 | Out-Null
    $rO2 = Invoke-Native $detExe @(); $rFa = Invoke-Native $detFast @()
    $hO2 = Parse-KeyValues $rO2.Output; $hFa = Parse-KeyValues $rFa.Output
    foreach ($k in @('sum_repro', 'dot_repro')) {
        $same = ($hO2[$k] -eq $hFa[$k])
        Add-Result "determinism" "$k across -O2/-Ofast" $(if ($same) { "identical" } else { "-O2 {0:N17} vs -Ofast {1:N17}" -f $hO2[$k], $hFa[$k] }) $(if ($same) { "PASS" } else { "FAIL" })
    }
}

# ----------------------------------------------------------- 11. include cost
if ($Tier -ge 1) {
    Write-Host "`n11. include cost" -ForegroundColor Yellow
    $empty = Join-Path $Tmp "empty.cpp"
    "" | Set-Content $empty -Encoding utf8
    $hdr = Join-Path $Include "vec_math\vector_math.hpp"
    $best = 1e9
    foreach ($i in 1..3) {
        $sw = [System.Diagnostics.Stopwatch]::StartNew()
        & $Gpp -std=c++20 -O2 -I $Include -include $hdr -c $empty -o (Join-Path $Tmp "e.o") 2>$null
        $ms = $sw.ElapsedMilliseconds
        if ($ms -lt $best) { $best = $ms }
    }
    $st = if ($best -le 1500) { "PASS" } elseif ($best -le 2500) { "WARN" } else { "FAIL" }
    Add-Result "build" "header compile cost (min of 3)" ("{0} ms" -f $best) $st
}

# -------------------------------------------------------- 12. doc snippets
if ($Tier -ge 2) {
    Write-Host "`n12. doc snippet compile check" -ForegroundColor Yellow
    $docs = @()
    $readme = Join-Path $Upstream "README.md"
    if (Test-Path $readme) { $docs += $readme }
    $docs += (Get-ChildItem (Join-Path $Upstream "docs") -Filter "*.md" -ErrorAction SilentlyContinue | ForEach-Object { $_.FullName })
    $nOk = 0; $nApiFail = 0; $apiErrors = @()
    foreach ($d in $docs) {
        $txt = Get-Content $d -Raw
        $matches2 = [regex]::Matches($txt, '(?ms)^```(?:cpp|c\+\+)\r?\n(.*?)^```')
        $i = 0
        foreach ($m in $matches2) {
            $i++
            $code = $m.Groups[1].Value
            $snip = Join-Path $Tmp ("snip_{0}_{1}.cpp" -f [IO.Path]::GetFileNameWithoutExtension($d), $i)
            ("#include <vec_math/vector_math.hpp>`n#include <vector>`n#include <span>`n#include <array>`nusing namespace vec_math;`nvoid __snippet() {`n" + $code + "`n}`n") | Set-Content $snip -Encoding utf8
            $res = Invoke-Native $Gpp @("-std=c++20", "-fsyntax-only", "-I", $Include, $snip)
            if ($res.Code -eq 0) { $nOk++ }
            else {
                $err = @($res.Output | Select-String 'error:' | Select-Object -First 1)
                if ($err.Count -gt 0) {
                    $line = $err[0].Line
                    if ($line -match 'vec_math|no matching function|no member named') {
                        $nApiFail++
                        $apiErrors += ("{0}#{1}: {2}" -f [IO.Path]::GetFileName($d), $i, ($line -replace '^.*error:\s*', ''))
                    }
                }
            }
        }
    }
    if ($nApiFail -eq 0) {
        Add-Result "docs" "snippets" ("{0} compiled, 0 API errors" -f $nOk) "PASS"
    } else {
        Add-Result "docs" "snippets" ("{0} API error(s)" -f $nApiFail) "WARN" $apiErrors[0]
    }
}

# ----------------------------------------------------------- 13. flamegraph
$script:FlamegraphHtml = $null
$script:FlamegraphSvg  = $null
Write-Host "`n13. flamegraph" -ForegroundColor Yellow
$flamePy = Join-Path $Repo "tools\generate_flamegraph.py"
if (-not (Test-Path $flamePy)) {
    $flamePy = Join-Path $Upstream "tools\generate_flamegraph.py"
}
$Py = "python"
if (-not (Get-Command $Py -ErrorAction SilentlyContinue)) {
    $foundPy = Get-Command python3 -ErrorAction SilentlyContinue
    if ($foundPy) { $Py = $foundPy.Source }
}

if ((Test-Path $flamePy) -and (Get-Command $Py -ErrorAction SilentlyContinue)) {
    $flameHtml = Join-Path $OutDir ("{0}-{1}-flamegraph.html" -f $Stamp, $Sha)
    $flameSvg  = Join-Path $OutDir ("{0}-{1}-flamegraph.svg" -f $Stamp, $Sha)
    $benchLogFile = Join-Path $Tmp "bench_log.txt"
    if ($script:BenchLog) {
        $script:BenchLog | Set-Content $benchLogFile -Encoding utf8
    }

    $flameArgs = @($flamePy, "--repo", $Repo, "--out-html", $flameHtml, "--out-svg", $flameSvg, "--title", "vec_math CPU Flamegraph ($Sha)")
    if (Test-Path $benchLogFile) {
        $flameArgs += @("--bench-log", $benchLogFile)
    }

    $res = Invoke-Native $Py $flameArgs
    if (Test-Path $flameHtml) {
        $script:FlamegraphHtml = [System.IO.Path]::GetFileName($flameHtml)
        $script:FlamegraphSvg  = [System.IO.Path]::GetFileName($flameSvg)
        Add-Result "flamegraph" "interactive HTML" ("generated: {0}" -f $script:FlamegraphHtml) "PASS"
    } else {
        Add-Result "flamegraph" "interactive HTML" "generation failed" "WARN"
    }
}

# ------------------------------------------------------------------- report
$status = if ($script:AnyFail) { "FAIL" } else { "PASS" }
$md = New-Object System.Text.StringBuilder
[void]$md.AppendLine("# vec_math metrics — $Stamp @ ``$Sha``")
[void]$md.AppendLine()
[void]$md.AppendLine("Generated by ``tools/metrics.ps1 -Tier $Tier``. Machine: $($Cpu.Trim()) / $Cores cores.")
[void]$md.AppendLine("Compiler: ``$GppVer``. Overall: **$status**.")
[void]$md.AppendLine()
[void]$md.AppendLine("## Results")
[void]$md.AppendLine()
[void]$md.AppendLine("| gate | check | value | status | note |")
[void]$md.AppendLine("|---|---|---|---|---|")
foreach ($r in $script:Results) {
    [void]$md.AppendLine(("| {0} | {1} | {2} | {3} | {4} |" -f $r.Gate, $r.Check, $r.Value, $r.Status, $r.Note))
}
[void]$md.AppendLine()
if ($script:FlamegraphHtml) {
    [void]$md.AppendLine("## Visualizations")
    [void]$md.AppendLine()
    [void]$md.AppendLine(("- [Interactive CPU Flamegraph (HTML)](./{0})" -f $script:FlamegraphHtml))
    [void]$md.AppendLine(("- [CPU Flamegraph (SVG)](./{0})" -f $script:FlamegraphSvg))
    [void]$md.AppendLine()
}
[void]$md.AppendLine("## Raw benchmark output (-O3)")
[void]$md.AppendLine()
[void]$md.AppendLine("``````")
foreach ($l in $script:BenchLog) { [void]$md.AppendLine($l) }
[void]$md.AppendLine("``````")
[void]$md.AppendLine()
[void]$md.AppendLine("## Thresholds")
[void]$md.AppendLine()
[void]$md.AppendLine("See ``docs/metrics.md`` §4. This file only records; the thresholds live there.")
$md.ToString() | Set-Content $Report -Encoding utf8

Write-Host ""
Write-Host ("Overall: {0}" -f $status) -ForegroundColor $(if ($status -eq "PASS") { "Green" } else { "Red" })
Write-Host ("Report:  {0}" -f $Report)
Write-Host ("Checks:  {0} total, {1} FAIL" -f $script:Results.Count, (@($script:Results | Where-Object { $_.Status -eq 'FAIL' }).Count))
exit $(if ($script:AnyFail) { 1 } else { 0 })
