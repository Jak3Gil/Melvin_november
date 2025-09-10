# Melvin Unified Brain Windows Build Script
# This script builds the Melvin cognitive system with proper error handling

param(
    [switch]$Clean,
    [switch]$Verbose
)

$ErrorActionPreference = "Stop"

Write-Host "🧠 Melvin Unified Brain Windows Build Script" -ForegroundColor Cyan
Write-Host "=============================================" -ForegroundColor Cyan

# Set build directory
$BuildDir = "build"
$LogFile = "build.log"

# Clean build directory if requested
if ($Clean) {
    Write-Host "🧹 Cleaning build directory..." -ForegroundColor Yellow
    if (Test-Path $BuildDir) {
        Remove-Item -Recurse -Force $BuildDir
    }
}

# Create build directory
if (-not (Test-Path $BuildDir)) {
    New-Item -ItemType Directory -Path $BuildDir | Out-Null
    Write-Host "📁 Created build directory: $BuildDir" -ForegroundColor Green
}

# Change to build directory
Set-Location $BuildDir

try {
    Write-Host "🔧 Running CMake configuration..." -ForegroundColor Yellow
    
    # Try Visual Studio 2022 first
    $cmakeArgs = @("-S", "..", "-B", ".", "-G", "Visual Studio 17 2022")
    
    try {
        if ($Verbose) {
            cmake @cmakeArgs 2>&1 | Tee-Object -FilePath $LogFile
        } else {
            cmake @cmakeArgs 2>&1 | Tee-Object -FilePath $LogFile | Out-Null
        }
        Write-Host "✅ CMake configuration successful with Visual Studio 2022" -ForegroundColor Green
    }
    catch {
        Write-Host "⚠️  Visual Studio 2022 not found, trying MinGW..." -ForegroundColor Yellow
        
        # Fallback to MinGW Makefiles
        $cmakeArgs = @("-S", "..", "-B", ".", "-G", "MinGW Makefiles")
        
        if ($Verbose) {
            cmake @cmakeArgs 2>&1 | Tee-Object -FilePath $LogFile
        } else {
            cmake @cmakeArgs 2>&1 | Tee-Object -FilePath $LogFile | Out-Null
        }
        Write-Host "✅ CMake configuration successful with MinGW" -ForegroundColor Green
    }
    
    Write-Host "🔨 Building Melvin Unified Brain..." -ForegroundColor Yellow
    
    # Build the project
    if ($Verbose) {
        cmake --build . --config Release 2>&1 | Tee-Object -FilePath $LogFile -Append
    } else {
        cmake --build . --config Release 2>&1 | Tee-Object -FilePath $LogFile -Append | Out-Null
    }
    
    Write-Host "✅ Build completed successfully!" -ForegroundColor Green
    
    # Run tests
    Write-Host "🧪 Running startup tests..." -ForegroundColor Yellow
    
    if (Test-Path "Release\test_startup.exe") {
        $testResult = & ".\Release\test_startup.exe" 2>&1
        Write-Host "Test output:" -ForegroundColor Cyan
        Write-Host $testResult -ForegroundColor White
        
        if ($LASTEXITCODE -eq 0) {
            Write-Host "✅ All tests passed!" -ForegroundColor Green
        } else {
            Write-Host "❌ Tests failed with exit code: $LASTEXITCODE" -ForegroundColor Red
        }
    } else {
        Write-Host "⚠️  Test executable not found, skipping tests" -ForegroundColor Yellow
    }
    
    # Show executable info
    if (Test-Path "Release\melvin_unified.exe") {
        $exeInfo = Get-Item "Release\melvin_unified.exe"
        Write-Host "📦 Executable created: $($exeInfo.FullName)" -ForegroundColor Green
        Write-Host "📏 Size: $([math]::Round($exeInfo.Length / 1MB, 2)) MB" -ForegroundColor Cyan
        Write-Host "📅 Created: $($exeInfo.CreationTime)" -ForegroundColor Cyan
    } else {
        Write-Host "❌ Main executable not found!" -ForegroundColor Red
    }
    
    # Show log file tail
    Write-Host "📋 Build log summary:" -ForegroundColor Cyan
    if (Test-Path $LogFile) {
        $logContent = Get-Content $LogFile -Tail 10
        Write-Host $logContent -ForegroundColor Gray
    }
    
    # Check for debug log
    if (Test-Path "melvin_debug.log") {
        Write-Host "📋 Debug log summary:" -ForegroundColor Cyan
        $debugContent = Get-Content "melvin_debug.log" -Tail 5
        Write-Host $debugContent -ForegroundColor Gray
    }
    
    Write-Host "`n🎉 Melvin Unified Brain build completed successfully!" -ForegroundColor Green
    Write-Host "Run '.\Release\melvin_unified.exe' to start the cognitive system" -ForegroundColor Cyan
    Write-Host "Run '.\Release\melvin_unified.exe --diag' for diagnostics" -ForegroundColor Cyan
    
}
catch {
    Write-Host "❌ Build failed: $($_.Exception.Message)" -ForegroundColor Red
    Write-Host "Check $LogFile for detailed error information" -ForegroundColor Yellow
    
    # Show last few lines of log
    if (Test-Path $LogFile) {
        Write-Host "`nLast 10 lines of build log:" -ForegroundColor Red
        Get-Content $LogFile -Tail 10 | ForEach-Object { Write-Host $_ -ForegroundColor Gray }
    }
    
    exit 1
}
finally {
    # Return to original directory
    Set-Location ..
}