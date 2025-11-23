# Parbiomech Video Analyzer - EXE 빌드 스크립트

Write-Host "============================================" -ForegroundColor Cyan
Write-Host "  Parbiomech Video Analyzer - EXE 빌드" -ForegroundColor Cyan
Write-Host "============================================" -ForegroundColor Cyan
Write-Host ""

# 1. 필수 패키지 확인
Write-Host "[1/4] 필수 패키지 확인 중..." -ForegroundColor Yellow
$packages = @("PyQt5", "pyqtgraph", "pyinstaller", "mediapipe", "opencv-python", "pandas")

foreach ($pkg in $packages) {
    try {
        python -c "import $($pkg.Replace('-', '_').ToLower().Split('python')[0])" 2>$null
        Write-Host "  ✓ $pkg 설치됨" -ForegroundColor Green
    }
    catch {
        Write-Host "  ✗ $pkg 미설치 - 설치 중..." -ForegroundColor Red
        pip install $pkg
    }
}

Write-Host ""

# 2. 이전 빌드 제거
Write-Host "[2/4] 이전 빌드 파일 제거 중..." -ForegroundColor Yellow
if (Test-Path "dist") {
    Remove-Item -Recurse -Force "dist"
    Write-Host "  ✓ dist 폴더 삭제" -ForegroundColor Green
}
if (Test-Path "build") {
    Remove-Item -Recurse -Force "build"
    Write-Host "  ✓ build 폴더 삭제" -ForegroundColor Green
}

Write-Host ""

# 3. EXE 빌드
Write-Host "[3/4] EXE 파일 빌드 중... (수 분 소요)" -ForegroundColor Yellow
Write-Host "  이 작업은 시간이 걸릴 수 있습니다. 기다려주세요..." -ForegroundColor Gray

pyinstaller desktop_app.spec

if ($LASTEXITCODE -eq 0) {
    Write-Host "  ✓ 빌드 성공!" -ForegroundColor Green
} else {
    Write-Host "  ✗ 빌드 실패! 오류를 확인하세요." -ForegroundColor Red
    exit 1
}

Write-Host ""

# 4. 결과 확인
Write-Host "[4/4] 빌드 결과 확인" -ForegroundColor Yellow

$exePath = "dist\Parbiomech_Video_Analyzer.exe"

if (Test-Path $exePath) {
    $fileSize = (Get-Item $exePath).Length / 1MB
    Write-Host "  ✓ EXE 파일 생성 완료!" -ForegroundColor Green
    Write-Host "  위치: $exePath" -ForegroundColor Cyan
    Write-Host "  크기: $([math]::Round($fileSize, 2)) MB" -ForegroundColor Cyan
    
    Write-Host ""
    Write-Host "============================================" -ForegroundColor Green
    Write-Host "  빌드 완료!" -ForegroundColor Green
    Write-Host "============================================" -ForegroundColor Green
    Write-Host ""
    Write-Host "다음 단계:" -ForegroundColor Yellow
    Write-Host "  1. 'dist' 폴더로 이동" -ForegroundColor White
    Write-Host "  2. 'Parbiomech_Video_Analyzer.exe' 더블클릭" -ForegroundColor White
    Write-Host "  3. 비디오 파일 선택 후 분석 시작" -ForegroundColor White
    Write-Host ""
    
    # 실행 옵션 제공
    $run = Read-Host "지금 실행하시겠습니까? (Y/N)"
    if ($run -eq "Y" -or $run -eq "y") {
        Write-Host "EXE 실행 중..." -ForegroundColor Cyan
        Start-Process $exePath
    }
} else {
    Write-Host "  ✗ EXE 파일을 찾을 수 없습니다!" -ForegroundColor Red
    Write-Host "  빌드 로그를 확인하세요." -ForegroundColor Yellow
}
