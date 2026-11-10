param([switch]$NoBrowser)
# Start-Live-System.ps1 - finds the DroidCam phone on the network, starts the
# camera bridge (which the TI board pulls frames from), and opens the dashboard.
# Safe to run any time - it kills any previous bridge instance first. ASCII only.

$ErrorActionPreference = "SilentlyContinue"
# this pc-tools folder: camera_bridge.py + bridge_config.txt live here
$sysDir  = Split-Path -Parent $MyInvocation.MyCommand.Path
$py      = "D:\samsung\BIRD-DETECTION\venv\Scripts\python.exe"
$bridge  = Join-Path $sysDir "camera_bridge.py"
$cfgPath = Join-Path $sysDir "bridge_config.txt"
$dash    = "http://localhost:9000/view"

function Test-Port([string]$ip, [int]$port, [int]$ms) {
    $c = New-Object Net.Sockets.TcpClient
    $ok = $c.ConnectAsync($ip, $port).Wait($ms)
    $c.Close()
    if ($ok) { return $true }
    $c.Dispose()
    return $false
}

function Save-Config([string]$url) {
    if ($url -match 'http://[^ ]+/video') {
        Set-Content -Path $cfgPath -Value $Matches[0] -Encoding Ascii
    }
}

function Find-Phone {
    # 1) last-known-good URL from config
    if (Test-Path $cfgPath) {
        $raw = ($cfgPath | Get-Content -Raw).Trim()
        if ($raw -match 'http://[^ \r\n]+/video') {
            $url = $Matches[0]
            if ($url -match 'http://([^:/]+):(\d+)') {
                $ip = $Matches[1]; $port = [int]$Matches[2]
                if (Test-Port $ip $port 400) {
                    Write-Host "[phone] found from saved config: $url"
                    return $url
                }
            }
        }
    }

    # 2) device IPs from the ARP table (things this PC has talked to)
    $cands = @()
    foreach ($line in (arp -a)) {
        $tok = ($line -split '\s+') | Where-Object { $_ }
        if ($tok.Count -gt 0 -and $tok[0] -match '^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$') {
            if ($tok[0] -notmatch '^192\.168\.1\.(1|5|22)$') { $cands += $tok[0] }
        }
    }
    $cands = $cands | Select-Object -Unique
    foreach ($ip in $cands) {
        if (Test-Port $ip 4747 300) {
            Write-Host "[phone] found via ARP table: ${ip}:4747"
            return "http://${ip}:4747/video"
        }
        if (Test-Port $ip 8080 300) {
            Write-Host "[phone] found via ARP table: ${ip}:8080"
            return "http://${ip}:8080/video"
        }
    }

    # 3) full subnet sweep, all sockets in parallel (fast)
    Write-Output "[phone] scanning 192.168.1.x for DroidCam ports 4747/8080..."
    foreach ($port in 4747, 8080) {
        $list = @()
        foreach ($i in 2..254) {
            $ip = "192.168.1.$i"
            $c = New-Object Net.Sockets.TcpClient
            $list += @{ ip = $ip; c = $c; t = $c.ConnectAsync($ip, $port) }
        }
        [void][System.Threading.Tasks.Task]::WaitAll(@($list | ForEach-Object { $_.t }), 1800)
        foreach ($e in $list) {
            if ($e.t.Status -eq 'RanToCompletion' -and $e.t.Result) {
                $e.c.Close()
                Write-Host "[phone] found via subnet sweep: $($e.ip):$port"
                return "http://$($e.ip):$port/video"
            }
            $e.c.Dispose()
        }
    }
    return $null
}

# --- 1) stop any previous bridge
Get-CimInstance Win32_Process -Filter "Name='python.exe'" |
    Where-Object { $_.CommandLine -like '*camera_bridge.py*' } |
    ForEach-Object { Stop-Process -Id $_.ProcessId -Force }

# --- 2) find the phone (falls back to laptop webcam if not found)
$url = Find-Phone
if ($url) {
    Save-Config $url
} else {
    Write-Host "[phone] NOT FOUND - falling back to the laptop webcam as the camera source"
    Remove-Item $cfgPath -ErrorAction SilentlyContinue
}

# --- 3) start the bridge (hidden)
if (-not (Test-Path $py)) { Write-Output "ERROR: python venv not found at $py"; exit 1 }
$args_ = '-u "%BRIDGE%"'
if ($url) { $args_ = '-u "%BRIDGE%" "%URL%"' }
$args_ = $args_.Replace('%BRIDGE%', $bridge)
if ($url) { $args_ = $args_.Replace('%URL%', $url) }
$null = Start-Process -WindowStyle Hidden -FilePath $py -ArgumentList $args_

# --- 4) wait for it to come up
$up = $false
for ($i = 0; $i -lt 14; $i++) {
    Start-Sleep -Milliseconds 600
    try {
        $r = Invoke-WebRequest -Uri "http://127.0.0.1:9000/ping" -TimeoutSec 2
        if ($r.Content -match 'bridge-ok') { $up = $true; break }
    } catch { }
}
if ($up) {
    Write-Output "[bridge] RUNNING on port 9000"
    Write-Output "[page]   dashboard: http://192.168.1.5:9000/view  (open from any device on the Wi-Fi)"
    if (-not $NoBrowser) { Start-Process $dash }
    exit 0
} else {
    Write-Output "[bridge] FAILED to start - check camera_bridge.py / phone still on Wi-Fi"
    exit 1
}
