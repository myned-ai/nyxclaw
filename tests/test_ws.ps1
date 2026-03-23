$token = Read-Host "Please enter your ZeroClaw token (e.g., zc_...)"
if ([string]::IsNullOrWhiteSpace($token)) {
    Write-Host "Token cannot be empty. Exiting." -ForegroundColor Red
    exit
}

$uri = "ws://localhost:42617/ws/avatar?token=$token"
$ws = New-Object System.Net.WebSockets.ClientWebSocket
$cancellationToken = [System.Threading.CancellationToken]::None

Try {
    Write-Host "Connecting to ZeroClaw..." -ForegroundColor Cyan
    $ws.ConnectAsync([uri]$uri, $cancellationToken).Wait()
    Write-Host "Connected successfully!" -ForegroundColor Green

    $jsonPayload = '{"type":"chat","messages":[{"role":"user","content":"Hello!"}]}'
    $messageBytes = [System.Text.Encoding]::UTF8.GetBytes($jsonPayload)
    $buffer = New-Object System.ArraySegment[byte] -ArgumentList @(,@($messageBytes))
    
    $ws.SendAsync($buffer, [System.Net.WebSockets.WebSocketMessageType]::Text, $true, $cancellationToken).Wait()
    Write-Host "Sent: $jsonPayload" -ForegroundColor Yellow

    for ($i = 0; $i -lt 3; $i++) {
        $receiveBuffer = New-Object byte[] 4096
        $receiveSegment = New-Object System.ArraySegment[byte] -ArgumentList @(,@($receiveBuffer))
        $receiveTask = $ws.ReceiveAsync($receiveSegment, $cancellationToken)
        $receiveTask.Wait()

        $response = [System.Text.Encoding]::UTF8.GetString($receiveBuffer, 0, $receiveTask.Result.Count)
        Write-Host "Received Chunk [$i]: $response" -ForegroundColor Magenta
        
        Start-Sleep -Milliseconds 200
    }
} Catch {
    Write-Host "Connection failed: $_" -ForegroundColor Red
} Finally {
    if ($ws.State -eq 'Open') {
        Write-Host "Aborting socket cleanly..." -ForegroundColor DarkGray
        $ws.Abort()
    }
    $ws.Dispose()
}
