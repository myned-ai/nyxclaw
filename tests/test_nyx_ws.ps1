$uri = "ws://localhost:8081/ws"
$ws = New-Object System.Net.WebSockets.ClientWebSocket
$cancellationToken = [System.Threading.CancellationToken]::None

Try {
    Write-Host "Connecting to NyxClaw Proxy at $uri..." -ForegroundColor Cyan
    $ws.ConnectAsync([uri]$uri, $cancellationToken).Wait()
    Write-Host "Connected successfully to NyxClaw!" -ForegroundColor Green

    # Now we can send the actual simulated user interaction right away
    $jsonPayload = '{"type":"chat","messages":[{"role":"user","content":"Are you connected through NyxClaw?"}]}'
    $messageBytes = [System.Text.Encoding]::UTF8.GetBytes($jsonPayload)
    $buffer = New-Object System.ArraySegment[byte] -ArgumentList @(,$messageBytes)
    
    $ws.SendAsync($buffer, [System.Net.WebSockets.WebSocketMessageType]::Text, $true, $cancellationToken).Wait()
    Write-Host "Sent user chat message: $jsonPayload" -ForegroundColor Yellow

    # Monitor the websocket for returning chunk streams
    for ($i = 0; $i -lt 5; $i++) {
        $receiveBuffer = New-Object byte[] 4096
        $receiveSegment = New-Object System.ArraySegment[byte] -ArgumentList @(,$receiveBuffer)
        $receiveTask = $ws.ReceiveAsync($receiveSegment, $cancellationToken)
        $receiveTask.Wait()

        if ($receiveTask.Result.Count -gt 0) {
            $response = [System.Text.Encoding]::UTF8.GetString($receiveBuffer, 0, $receiveTask.Result.Count)
            Write-Host "Received Chunk [$i]: $response" -ForegroundColor Magenta
        }
        
        Start-Sleep -Milliseconds 200
    }
} Catch {
    Write-Host "Connection failed: $($_.Exception.Message)" -ForegroundColor Red
    if ($_.Exception.InnerException) {
        Write-Host "Inner Exception: $($_.Exception.InnerException.Message)" -ForegroundColor Red
    }
} Finally {
    # If open, abruptly dispose without attempting an asynchronous graceful close which causes race conditions
    if ($ws.State -eq 'Open') {
        Write-Host "Dropping socket..." -ForegroundColor DarkGray
        $ws.Abort()
    }
    $ws.Dispose()
}