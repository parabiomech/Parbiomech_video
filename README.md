# 🎯 Parbiomech Video Analysis

**MediaPipe 기반 웹 포즈 분석 시스템**

## 🌐 라이브 데모

**👉 [https://parabiomech.github.io/Parbiomech_video/](https://parabiomech.github.io/Parbiomech_video/)**

브라우저에서 바로 사용 가능 - 설치 불필요!

## ✨ 주요 기능

- 📹 **비디오 업로드**: 드래그 앤 드롭 지원 (MP4, AVI, MOV, WebM)
- 🏷️ **시점 태그**: 분석할 특정 시점 선택 (스페이스바 단축키)
- 🎯 **포즈 감지**: MediaPipe.js로 33개 키포인트 실시간 추적
- 📊 **각도 분석**: 
  - 절대각도: 분절의 수평면 대비 기울기
  - 상대각도: 관절의 굴곡/신전 각도
- 💾 **결과 표시**: 스켈레톤 오버레이 이미지 + 각도 데이터 테이블

## 🚀 사용 방법

1. **비디오 업로드**: "📁 파일 선택" 버튼 클릭 또는 드래그 앤 드롭
2. **시점 태그**: 영상을 재생하며 분석할 시점에서:
   - "🏷️ 현재 시점 태그" 버튼 클릭
   - 또는 **스페이스바** 누르기
3. **분석 시작**: 여러 시점을 태그한 후 "🔍 분석 시작" 클릭
4. **결과 확인**: 각 시점의 포즈 이미지와 각도 데이터 확인

## 💡 기술 스택

- **Frontend**: HTML5, CSS3, Vanilla JavaScript
- **AI Model**: MediaPipe Pose (Google)
- **라이브러리**: MediaPipe.js (CDN)
- **호스팅**: GitHub Pages

## 🎨 특징

- ✅ **완전 무료**: 서버 비용 없음
- ✅ **설치 불필요**: 브라우저만 있으면 실행
- ✅ **Python 불필요**: 순수 JavaScript로 구현
- ✅ **크로스 플랫폼**: PC, Mac, 모바일 모두 지원
- ✅ **오프라인 가능**: CDN 로드 후 로컬 실행 가능

## 📱 지원 브라우저

- Chrome/Edge (권장)
- Firefox
- Safari
- Opera

## 📝 분석 항목

### 절대각도 (분절 기울기)
- 좌/우 상완, 하완
- 좌/우 대퇴, 하퇴

### 상대각도 (관절각도)
- 팔꿈치 (좌/우)
- 무릎 (좌/우)
- 엉덩이/고관절 (좌/우)

## 🔧 로컬 실행

```bash
# 간단한 HTTP 서버 시작
python -m http.server 8000

# 또는 그냥 파일 열기
open index.html  # Mac
start index.html # Windows
```

## 📄 라이센스

MIT License

## 👨‍💻 개발자

Parbiomech Team

---

**문의**: GitHub Issues에 등록해주세요!
