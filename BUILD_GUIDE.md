# Parbiomech 비디오 분석 - EXE 빌드 가이드

## 📋 사전 준비

### 1. 필수 패키지 설치
```powershell
pip install PyQt5 pyqtgraph pyinstaller mediapipe opencv-python pandas numpy
```

### 2. 테스트 실행
데스크톱 앱이 정상 작동하는지 먼저 확인:
```powershell
python desktop_app.py
```

## 🔨 EXE 빌드 방법

### 방법 1: spec 파일 사용 (권장)
```powershell
pyinstaller desktop_app.spec
```

### 방법 2: 직접 명령어 사용
```powershell
pyinstaller --onefile --noconsole --name "Parbiomech_Video_Analyzer" desktop_app.py
```

## 📦 빌드 결과

- **위치**: `dist/Parbiomech_Video_Analyzer.exe`
- **크기**: 약 200-300MB (MediaPipe 포함)
- **실행**: 더블클릭하면 바로 실행됨

## ✅ 사용 방법

1. **비디오 선택**: "📁 비디오 선택" 버튼 클릭
2. **설정 조정**:
   - 신뢰도 임계값 조정 (0.1 ~ 1.0)
   - 타임포인트 입력 (예: 0.5, 1.0, 2.5)
3. **분석 시작**: "🔍 분석 시작" 버튼 클릭
4. **결과 확인**: 탭에서 분석 결과 확인
5. **CSV 다운로드**: (추가 예정)

## 🎯 주요 기능

### 현재 구현된 기능
- ✅ 비디오 업로드
- ✅ 실시간 진행 상태 표시
- ✅ MediaPipe 포즈 분석
- ✅ 스켈레톤 오버레이 비디오 생성
- ✅ 타임포인트 분석
- ✅ 결과 테이블 표시
- ✅ 궤적 차트 표시

### 추가 예정 기능
- ⏳ CSV 데이터 내보내기
- ⏳ 비디오 플레이어 내장
- ⏳ 더 많은 각도 계산 옵션
- ⏳ 실시간 비디오 미리보기

## 🔧 문제 해결

### MediaPipe 오류 발생 시
```python
# desktop_app.py 상단에 이미 추가되어 있음:
os.environ['MEDIAPIPE_RESOURCE_CACHE_DIR'] = tempfile.gettempdir()
```

### EXE 크기 줄이기
```powershell
# UPX 압축 사용 (이미 spec 파일에 포함)
pyinstaller --onefile --upx-dir=C:/upx desktop_app.spec
```

### 실행 오류 디버깅
콘솔창에서 오류 확인:
```powershell
# spec 파일의 console=False를 console=True로 변경
pyinstaller desktop_app.spec
```

## 📤 배포 방법

### 단일 파일 배포
1. `dist/Parbiomech_Video_Analyzer.exe` 파일 복사
2. 사용자에게 전달
3. 사용자는 더블클릭으로 바로 실행

### 주의사항
- ⚠️ 처음 실행 시 Windows Defender가 경고할 수 있음 (정상)
- ⚠️ 파일 크기가 크므로 압축하여 전달 권장
- ⚠️ 백신 프로그램이 차단할 수 있음 (예외 추가 필요)

## 🔐 코드 보호

### PyInstaller의 보안
- ✅ Python 코드가 바이트코드로 컴파일됨
- ✅ 바이너리 내부에 포함되어 숨겨짐
- ✅ 일반 사용자는 코드 확인 불가능
- ⚠️ 전문가는 디컴파일 가능 (하지만 매우 어려움)

### 추가 보호 (선택사항)
```powershell
# PyArmor로 추가 암호화
pip install pyarmor
pyarmor gen desktop_app.py
pyinstaller desktop_app_obfuscated.spec
```

## 📊 성능

### 예상 처리 시간
- 30초 비디오: 약 1-2분
- 1분 비디오: 약 2-4분
- 5분 비디오: 약 10-20분

### 시스템 요구사항
- **OS**: Windows 10/11 (64-bit)
- **RAM**: 최소 4GB, 권장 8GB
- **CPU**: 멀티코어 프로세서 권장
- **디스크**: 500MB 여유 공간

## 🆘 지원

문제가 발생하면:
1. 콘솔 모드로 재빌드 (`console=True`)
2. 오류 메시지 확인
3. MediaPipe/OpenCV 호환성 확인

---

**빌드 완료 후 테스트 필수!**
