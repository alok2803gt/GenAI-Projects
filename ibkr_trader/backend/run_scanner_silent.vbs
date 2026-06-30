Set oShell = CreateObject("WScript.Shell")
oShell.Run "powershell.exe -NonInteractive -WindowStyle Hidden -ExecutionPolicy Bypass -File ""C:\Projects\GenAI-Projects\ibkr_trader\backend\run_scanner.ps1""", 0, False
