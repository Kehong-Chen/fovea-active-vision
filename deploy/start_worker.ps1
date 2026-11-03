# FOVEA worker 启动脚本（由计划任务调用，不依赖任何终端会话）
$ErrorActionPreference = 'Continue'

$env:FOVEA_BASE_URL     = 'https://api.siliconflow.cn/v1'
$env:FOVEA_API_KEY      = '<YOUR_SILICONFLOW_API_KEY>'
$env:FOVEA_MODEL        = 'Qwen/Qwen3-VL-8B-Instruct'
$env:FOVEA_ANSWER_MODEL = 'Qwen/Qwen3-VL-32B-Instruct'
$env:FOVEA_TIMEOUT      = '60'
$env:FOVEA_RETRIES      = '2'
$env:PYTHONUNBUFFERED   = '1'

$work = 'C:\Users\29046\Documents\deepseek-harness\default-workspace'
$py   = 'C:\Users\29046\AppData\Local\Programs\Python\Python310\python.exe'
$log  = Join-Path $env:USERPROFILE '.fovea_worker.log'

Set-Location $work
& $py -m fovea_service.worker --server http://115.159.34.79:8765 --token <WORKER_TOKEN> --backend vlm *>> $log
