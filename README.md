# 🎯 Keypoint Tracker - 동작 분석 도구

MediaPipe Pose를 활용한 비디오 동작 분석 Streamlit 애플리케이션입니다.

## ✨ 주요 기능

- 📹 비디오 파일 업로드 및 자동 포즈 분석
- 🦴 33개 신체 키포인트 자동 추적
- 📊 관절 각도 실시간 계산 (팔꿈치, 무릎, 고관절 등)
- 📈 인터랙티브 데이터 시각화 (Plotly)
- 🔄 로우패스 필터를 통한 데이터 스무딩
- 💾 분석 결과 CSV 다운로드

## 🚀 Streamlit Cloud 배포 방법

### 1. GitHub 리포지토리 준비
이미 GitHub에 코드가 있으므로 다음 단계로 진행하세요.

### 2. Streamlit Cloud 배포

1. [Streamlit Cloud](https://share.streamlit.io/)에 접속
2. GitHub 계정으로 로그인
3. "New app" 클릭
4. 리포지토리 정보 입력:
   - **Repository**: `parabiomech/Parbiomech_video`
   - **Branch**: `main`
   - **Main file path**: `streamlit_app.py`
5. "Deploy!" 클릭

### 3. 배포 완료!
몇 분 후 앱이 배포되고 공유 가능한 URL이 생성됩니다.

## 💻 로컬 실행 방법

### 필요 사항
- Python 3.8 이상

### 설치 및 실행

```bash
# 패키지 설치
pip install -r requirements.txt

# 앱 실행
streamlit run streamlit_app.py
```

브라우저에서 자동으로 `http://localhost:8501`이 열립니다.

## 📖 사용 방법

1. **비디오 업로드**: MP4, MOV, AVI, MKV 형식 지원
   - ⚠️ **중요**: H.264 (AVC) 또는 H.265 (HEVC) 코덱 필요
   - ❌ AV1 코덱은 지원되지 않음 (아래 변환 방법 참조)
2. **원본 영상 재생 및 시점 태그**:
   - 비디오 플레이어에서 원하는 시점에서 "시점 추가" 클릭
   - 슬라이더를 드래그하거나 재생 중 실시간으로 시점 추가 가능
3. **설정 조정** (사이드바):
   - 감지 신뢰도 임계값 조정
4. **분석 시작**: "🚀 분석 시작" 버튼 클릭
5. **결과 확인**:
   - 🎥 스켈레톤 오버레이 비디오 다운로드
   - 📊 시점별 각도 분석 (절대 각도 + 상대 각도)
   - 📍 궤적 분석 그래프
   - 💾 CSV 파일 다운로드

## 🎬 비디오 형식 요구사항

### ✅ 지원되는 코덱
- H.264 (AVC)
- H.265 (HEVC)

### ❌ 지원되지 않는 코덱
- AV1 (변환 필요)

### 🔄 AV1 → H.264 변환 방법

**방법 1: FFmpeg 사용 (권장)**
```bash
ffmpeg -i input.mp4 -c:v libx264 -crf 23 -preset medium -c:a aac output.mp4
```

**방법 2: 온라인 변환 도구**
- [CloudConvert](https://cloudconvert.com/mp4-converter)
- [FreeConvert](https://www.freeconvert.com/video-converter)
- [Online-Convert](https://www.online-convert.com/)

**FFmpeg 설치 방법:**
- Windows: [https://ffmpeg.org/download.html](https://ffmpeg.org/download.html)
- Mac: `brew install ffmpeg`
- Linux: `sudo apt install ffmpeg`

## 📊 분석되는 각도

### 절대 각도 (Absolute Angles) - 영상 기준 분절 각도
- 머리, 어깨, 골반, 몸통
- 좌/우 상완, 하완
- 좌/우 대퇴, 하퇴

### 상대 각도 (Relative Angles) - 관절 각도
- 왼쪽/오른쪽 어깨, 팔꿈치, 손목
- 왼쪽/오른쪽 엉덩이, 무릎, 발목

## 🔧 기술 스택

- **Streamlit**: 웹 애플리케이션 프레임워크
- **MediaPipe**: Google의 포즈 추정 라이브러리
- **OpenCV**: 비디오 처리
- **Plotly**: 인터랙티브 데이터 시각화
- **Pandas**: 데이터 처리 및 분석

## 📝 파일 구조

```
Parbiomech_video/
├── streamlit_app.py      # 메인 Streamlit 앱
├── requirements.txt      # Python 패키지 의존성
├── README.md            # 이 파일
├── index.html           # (이전 HTML 버전)
└── app.js              # (이전 JavaScript 버전)
```

## ⚠️ 주의사항

- **비디오 코덱**: H.264 또는 H.265 코덱만 지원됩니다. AV1 코덱은 변환이 필요합니다.
- 대용량 비디오 파일은 처리 시간이 오래 걸릴 수 있습니다
- Streamlit Cloud 무료 버전은 리소스 제한이 있습니다
- 최적의 성능을 위해 1-2분 이내의 비디오를 권장합니다
- 비디오 해상도가 높을수록 처리 시간이 증가합니다

## 📄 라이선스

MIT License

## 🤝 기여

이슈 및 풀 리퀘스트를 환영합니다!

---

**Made with ❤️ using Streamlit and MediaPipe**
