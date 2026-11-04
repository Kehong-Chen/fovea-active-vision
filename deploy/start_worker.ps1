# FOVEA worker launcher with watchdog (restart on crash; exit 66 = token invalid -> stop)
$ErrorActionPreference = 'Continue'

$env:FOVEA_BASE_URL     = 'https://api.siliconflow.cn/v1'
$env:FOVEA_API_KEY      = '<YOUR_SILICONFLOW_API_KEY>'
$env:FOVEA_MODEL        = 'Qwen/Qwen3-VL-8B-Instruct'
$env:FOVEA_ANSWER_MODEL = 'Qwen/Qwen3-VL-32B-Instruct'
$env:FOVEA_TIMEOUT      = '60'
$env:FOVEA_RETRIES      = '2'
$env:PYTHONUNBUFFERED   = '1'

$work = 'C:/Users/29046/Documents/deepseek-harness/default-workspace'
$py   = 'C:/Users/29046/AppData/Local/Programs/Python/Python310/python.exe'
$log  = Join-Path $env:USERPROFILE '.fovea_worker.log'

Set-Location $work

while ($true) {
    $stamp = Get-Date -Format 'yyyy-MM-dd HH:mm:ss'
    "[$stamp][watchdog] starting worker concurrency=2" | Out-File -FilePath $log -Append -Encoding utf8
    & $py -m fovea_service.worker --server http://115.159.34.79:8765 --token <WORKER_TOKEN> --backend vlm --concurrency 2 2>&1 | Out-File -FilePath $log -Append -Encoding utf8
    $code = $LASTEXITCODE
    $stamp = Get-Date -Format 'yyyy-MM-dd HH:mm:ss'
    if ($code -eq 66) {
        "[$stamp][watchdog] token invalid, stop restarting" | Out-File -FilePath $log -Append -Encoding utf8
        break
    }
    "[$stamp][watchdog] worker exited code=$code, restart in 5s" | Out-File -FilePath $log -Append -Encoding utf8
    Start-Sleep -Seconds 5
}
