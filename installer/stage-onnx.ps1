param(
    [Parameter(Mandatory=$true)][string]$Stage,
    [string]$AppDir,
    [string]$AssetsDir,
    [string]$RuntimeDir,
    [string]$RedistDir,
    [string]$DeiterisLicense
)
$ErrorActionPreference = 'Stop'
$repo = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
$destination = [IO.Path]::GetFullPath($Stage)
$allowed = (Join-Path $repo 'dist-work') + [IO.Path]::DirectorySeparatorChar
if (!$destination.StartsWith($allowed, [StringComparison]::OrdinalIgnoreCase) -or (Test-Path -LiteralPath $destination)) {
    throw 'Stage must be a new directory inside dist-work/'
}
$app = if ($AppDir) { $AppDir } else { Join-Path $repo 'crates/rvc-app/target/release' }
$assets = if ($AssetsDir) { $AssetsDir } else { Join-Path $repo 'PoC/assets/app' }
$runtime = if ($RuntimeDir) { $RuntimeDir } else { Join-Path $repo 'PoC/01-rust-ort-official-rt/runtime' }
$redist = if ($RedistDir) { $RedistDir } else { Join-Path $repo 'dist-work/stage' }
$license = if ($DeiterisLicense) { $DeiterisLicense } else { Join-Path $repo 'PoC/assets/research-20260919/deiteris/LICENSE' }
New-Item -ItemType Directory -Path $destination | Out-Null
foreach ($file in @('rvc-app.exe','DirectML.dll','onnxruntime_providers_cuda.dll','onnxruntime_providers_shared.dll')) {
    Copy-Item -LiteralPath (Join-Path $app $file) -Destination $destination
}
foreach ($file in @('msvcp140.dll','msvcp140_1.dll','vcruntime140.dll','vcruntime140_1.dll')) {
    Copy-Item -LiteralPath (Join-Path $redist $file) -Destination $destination
}
$runtimeOut = New-Item -ItemType Directory -Path (Join-Path $destination 'runtime')
foreach ($file in @('cudart64_13.dll','cublas64_13.dll','cublasLt64_13.dll','cufft64_12.dll','libportaudio64bit.dll')) {
    Copy-Item -LiteralPath (Join-Path $runtime $file) -Destination $runtimeOut.FullName
}
Get-ChildItem -LiteralPath $runtime -Filter 'cudnn*64_9.dll' -File | Copy-Item -Destination $runtimeOut.FullName
$assetsOut = New-Item -ItemType Directory -Path (Join-Path $destination 'assets')
foreach ($file in @('contentvec.onnx','rmvpe.onnx','fcpe.onnx')) {
    Copy-Item -LiteralPath (Join-Path $assets $file) -Destination $assetsOut.FullName
}
Copy-Item -LiteralPath (Join-Path $assets 'voices') -Destination $assetsOut.FullName -Recurse
foreach ($family in @('', 'deiteris')) {
    $source = Join-Path $assets $family
    $target = Join-Path $assetsOut.FullName $family
    foreach ($rate in @('32k','40k','48k')) {
        $template = New-Item -ItemType Directory -Path (Join-Path $target "templates/$rate") -Force
        foreach ($file in @('generator.onnx','template.json')) {
            Copy-Item -LiteralPath (Join-Path $source "templates/$rate/$file") -Destination $template.FullName
        }
    }
}
foreach ($bundle in @('tg-fast-v1','tg-gpu-pitch-v2')) {
    Copy-Item -LiteralPath (Join-Path $assets "deiteris/$bundle") -Destination (Join-Path $assetsOut.FullName 'deiteris') -Recurse
}
Copy-Item -LiteralPath (Join-Path $PSScriptRoot 'THIRD_PARTY_NOTICES.txt') -Destination $destination
Copy-Item -LiteralPath $license -Destination (Join-Path $destination 'DEITERIS_LICENSE.txt')
$files = @(Get-ChildItem -LiteralPath $destination -Recurse -File)
if ($files.Name -match '^(torch|c10|python)' -or $files.Extension -contains '.pt' -or $files.Name -contains 'generator.weights') {
    throw 'Unexpected Torch, Python or reference weights in stage'
}
[pscustomobject]@{stage=$destination; files=$files.Count; bytes=($files | Measure-Object Length -Sum).Sum;
    app_sha256=(Get-FileHash -LiteralPath (Join-Path $destination 'rvc-app.exe')).Hash;
    scope='local validation payload; upstream asset redistribution conditions remain unresolved'} | ConvertTo-Json
