' 开机/登录时自动拉起 FOVEA worker（隐藏窗口，单实例由 worker 自己的锁保证）
Set sh = CreateObject("WScript.Shell")
sh.Run "powershell.exe -NoProfile -WindowStyle Hidden -ExecutionPolicy Bypass -File ""C:\Users\29046\Documents\deepseek-harness\default-workspace\fovea_deploy\start_worker.ps1""", 0, False
