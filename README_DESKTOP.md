# 🎥 Parbiomech 비디오 분석기

MediaPipe 기반 포즈 분석 데스크톱 애플리케이션

## 📌 개요

이 프로그램은 비디오에서 사람의 움직임을 분석하고 관절 각도를 측정하는 Windows 데스크톱 애플리케이션입니다.

### 주요 기능
- ✅ 비디오 파일 업로드 및 분석
- ✅ MediaPipe를 이용한 실시간 포즈 감지
- ✅ 스켈레톤 오버레이 비디오 생성
- ✅ 관절 각도 측정 및 추적
- ✅ 타임포인트별 상세 분석
- ✅ 차트 및 데이터 시각화
- ✅ **코드 보호** (EXE 내부에 암호화)

## 🚀 빠른 시작

### 방법 1: EXE 파일 사용 (권장)

1. **EXE 다운로드**
   - `dist/Parbiomech_Video_Analyzer.exe` 다운로드

2. **실행**
   - 파일 더블클릭
   - (Windows Defender 경고 시 "자세히" → "실행" 선택)

3. **사용**
   - 비디오 파일 선택
   - 설정 조정
   - 분석 시작!

### 방법 2: Python 소스코드 실행

#### 1. 필수 패키지 설치
```powershell
pip install PyQt5 pyqtgraph pyinstaller mediapipe opencv-python pandas numpy
```

#### 2. 실행
```powershell
python desktop_app.py
```

## 🔨 EXE 빌드 방법

### 자동 빌드 (권장)
```powershell
.\build_exe.ps1
```

### 수동 빌드
```powershell
# spec 파일 사용
pyinstaller desktop_app.spec

# 또는 직접 빌드
pyinstaller --onefile --noconsole --name "Parbiomech_Video_Analyzer" desktop_app.py
```

빌드된 EXE는 `dist/` 폴더에 생성됩니다.

## 📖 사용 가이드

### 1️⃣ 비디오 업로드
- "📁 비디오 선택" 버튼 클릭
- MP4, AVI, MOV, MKV 등 지원

### 2️⃣ 분석 설정
- **신뢰도 임계값**: 0.1 ~ 1.0 (기본: 0.5)
  - 낮을수록 더 많은 포즈 감지 (정확도 낮음)
  - 높을수록 확실한 포즈만 감지 (정확도 높음)

- **타임포인트**: 쉼표로 구분 (예: `0.5, 1.0, 2.5`)
  - 특정 시점의 상세 분석
  - 생략 시 전체 비디오만 분석

### 3️⃣ 분석 시작
- "🔍 분석 시작" 버튼 클릭
- 진행 상태 실시간 표시
- 완료까지 수 분 소요 가능

### 4️⃣ 결과 확인
- **타임포인트 분석**: 지정한 시점의 각도 데이터
- **궤적 차트**: 시간에 따른 움직임 그래프
- **비디오 정보**: 분석된 비디오 파일 경로

## 📊 출력 데이터

### 1. 분석 비디오
- 스켈레톤 오버레이가 포함된 MP4 파일
- 임시 폴더에 자동 저장

### 2. 타임포인트 데이터
- 시간, 프레임 번호
- 관절 각도 측정값

### 3. 추적 데이터
- 모든 프레임의 관절 좌표
- CSV 내보내기 (추가 예정)

## 🖥️ 시스템 요구사항

### 최소 사양
- **OS**: Windows 10/11 (64-bit)
- **RAM**: 4GB
- **CPU**: Intel i3 또는 동급
- **디스크**: 500MB 여유 공간

### 권장 사양
- **RAM**: 8GB 이상
- **CPU**: Intel i5 또는 동급 (멀티코어)
- **SSD**: 더 빠른 처리 속도

## 🔐 코드 보호

### EXE의 보안성
- ✅ Python 코드가 **바이트코드로 컴파일**됨
- ✅ 바이너리 내부에 **암호화되어 포함**
- ✅ 일반 사용자는 **코드 확인 불가능**
- ✅ 디컴파일 **매우 어려움**

### 추가 보호 옵션
PyArmor를 사용한 고급 암호화:
```powershell
pip install pyarmor
pyarmor gen desktop_app.py
```

## 📦 배포 방법

### 단일 파일 배포
1. `dist/Parbiomech_Video_Analyzer.exe` 파일만 전달
2. 사용자는 바로 실행 가능
3. 추가 설치 불필요

### 주의사항
- ⚠️ 파일 크기: 약 200-300MB
- ⚠️ 처음 실행 시 백신/방화벽 경고 가능
- ⚠️ Windows Defender 예외 추가 권장

## 🛠️ 문제 해결

### MediaPipe 오류
```
PermissionError: [Errno 13] Permission denied
```
→ 이미 해결됨 (임시 폴더 사용)

### 분석 속도 느림
- GPU 가속 비활성화됨 (호환성 우선)
- 더 빠른 CPU 사용 권장
- 비디오 해상도 낮추기

### EXE 실행 안됨
1. 콘솔 모드로 재빌드:
   ```powershell
   # desktop_app.spec에서 console=True로 변경
   pyinstaller desktop_app.spec
   ```
2. 오류 메시지 확인
3. 누락된 DLL 확인

## 📁 프로젝트 구조

```
Parbiomech_video/
├── desktop_app.py          # 메인 데스크톱 앱
├── desktop_app.spec        # PyInstaller 설정
├── build_exe.ps1          # 빌드 스크립트
├── BUILD_GUIDE.md         # 빌드 가이드
├── README_DESKTOP.md      # 이 파일
├── streamlit_app.py       # 웹앱 버전 (참고용)
└── dist/                  # 빌드된 EXE (생성됨)
    └── Parbiomech_Video_Analyzer.exe
```

## 🆚 웹앱 vs 데스크톱 앱

| 기능 | Streamlit 웹앱 | PyQt5 데스크톱 |
|------|---------------|---------------|
| 핸드폰 사용 | ✅ 가능 | ❌ 불가능 |
| 오프라인 | ❌ 서버 필요 | ✅ 가능 |
| 코드 보호 | ⚠️ 부분적 | ✅ 완전 |
| 배포 | URL 공유 | EXE 전달 |
| 설치 | 불필요 | 불필요 |
| 속도 | 서버 성능 | PC 성능 |

## 🔄 업데이트

### 계획된 기능
- [ ] CSV 데이터 내보내기
- [ ] 비디오 플레이어 내장
- [ ] 더 많은 각도 측정 옵션
- [ ] 여러 비디오 비교 분석
- [ ] 설정 저장/불러오기
- [ ] 사용자 정의 관절 정의

## 📞 지원

문제 발생 시:
1. `BUILD_GUIDE.md` 확인
2. 오류 로그 확인
3. GitHub Issues 등록

## 📄 라이선스

이 프로젝트는 개인/연구용으로 제작되었습니다.

## 🙏 감사의 글

- **MediaPipe**: Google의 포즈 감지 라이브러리
- **PyQt5**: 크로스 플랫폼 GUI 프레임워크
- **OpenCV**: 컴퓨터 비전 라이브러리

---

**Made with ❤️ for Biomechanics Research**
