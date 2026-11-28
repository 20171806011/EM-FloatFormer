param(
    [Parameter(Mandatory = $true, Position = 0)]
    [string]$TargetFile,
    [switch]$AllowCustomAxioms
)

$ErrorActionPreference = 'Stop'
$LabRoot = "C:\lean4\math_lab"
$env:ELAN_HOME = "C:\lean4\elan"
$env:Path = "C:\lean4\elan\bin;$env:Path"
$env:TMP = "C:\lean4\tmp"
$env:TEMP = "C:\lean4\tmp"

if (-not [System.IO.Path]::IsPathRooted($TargetFile)) {
    $ResolvedPath = Join-Path $LabRoot $TargetFile
} else {
    $ResolvedPath = $TargetFile
}

if (-not (Test-Path -LiteralPath $ResolvedPath)) {
    Write-Error "Target Lean file not found: $ResolvedPath"
    exit 2
}

$FullPath = (Resolve-Path -LiteralPath $ResolvedPath).Path

# Check UTF-8 BOM (Lean 4 rejects BOM-prefixed source files)
$RawBytes = [System.IO.File]::ReadAllBytes($FullPath)
if ($RawBytes.Length -ge 3 -and $RawBytes[0] -eq 0xEF -and $RawBytes[1] -eq 0xBB -and $RawBytes[2] -eq 0xBF) {
    Write-Error "UTF-8 BOM detected in $FullPath. Lean 4 requires UTF-8 WITHOUT BOM."
    exit 4
}

$Content = [System.IO.File]::ReadAllText($FullPath, [System.Text.UTF8Encoding]::new($false))

# Strip block comments /- ... -/ and line comments -- ... for static audit
$Stripped = [regex]::Replace($Content, '/-[\s\S]*?-/', '')
$Stripped = [regex]::Replace($Stripped, '--[^\r\n]*', '')

$Flags = New-Object System.Collections.Generic.List[string]
if ($Stripped -match '\bsorry\b') { $Flags.Add("contains 'sorry'") }
if ($Stripped -match '\badmit\b') { $Flags.Add("contains 'admit'") }
if ($Stripped -match '\bnative_decide\b') { $Flags.Add("contains 'native_decide'") }
if ($Stripped -match '\bimplemented_by\b') { $Flags.Add("contains 'implemented_by'") }
if (-not $AllowCustomAxioms -and ($Stripped -match '(?m)^\s*(?:private\s+|protected\s+)?axiom\b')) {
    $Flags.Add("declares custom 'axiom' (pass -AllowCustomAxioms if intentional)")
}

Write-Output "=== Lean 4 Formal Verification ==="
Write-Output "File: $FullPath"

$sw = [System.Diagnostics.Stopwatch]::StartNew()
Push-Location $LabRoot
try {
    $LeanOut = & lake env lean $FullPath 2>&1
    $ExitCode = $LASTEXITCODE
} finally {
    Pop-Location
}
$sw.Stop()

$SawSorryAx = $false
if ($LeanOut) {
    $LeanOut | ForEach-Object {
        $line = "$_"
        if ($line -match '\bsorryAx\b') { $SawSorryAx = $true }
        Write-Output $line
    }
}
if ($SawSorryAx) { $Flags.Add("theorem depends on 'sorryAx'") }

Write-Output "=== Audit Summary ==="
Write-Output ("Elapsed: {0:N2}s" -f $sw.Elapsed.TotalSeconds)
Write-Output "CompilerExitCode: $ExitCode"

if ($Flags.Count -gt 0) {
    Write-Output ("StaticAudit: FAILED ({0})" -f ($Flags -join "; "))
    if ($ExitCode -eq 0) {
        exit 3
    }
    exit $ExitCode
} else {
    Write-Output "StaticAudit: CLEAN (no sorry / admit / native_decide / custom axioms)"
}

if ($ExitCode -ne 0) {
    Write-Output "VERDICT: REJECTED (compiler error)"
    exit $ExitCode
}

Write-Output "VERDICT: VERIFIED BY LEAN 4 KERNEL"
exit 0
