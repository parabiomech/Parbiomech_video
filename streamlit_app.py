import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
import tempfile
import os
import pandas as pd
import plotly.graph_objects as go

# MediaPipe 초기화
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

def check_video_codec(video_path):
    """비디오 코덱을 확인하고 지원 여부를 반환"""
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return False, "needs_conversion", "비디오를 열 수 없습니다."
        
        # 첫 프레임을 읽어서 확인
        ret, frame = cap.read()
        cap.release()
        
        if not ret:
            return False, "needs_conversion", "비디오 코덱이 지원되지 않습니다."
        
        return True, "ok", "지원되는 형식"
    except Exception as e:
        return False, "error", f"비디오 확인 오류: {str(e)}"

def convert_video_to_h264(input_path, output_path):
    """ffmpeg를 사용하여 비디오를 H.264 코덱으로 변환"""
    import subprocess
    
    try:
        # ffmpeg 명령어
        cmd = [
            'ffmpeg',
            '-i', input_path,
            '-c:v', 'libx264',
            '-crf', '23',
            '-preset', 'medium',
            '-c:a', 'aac',
            '-y',  # 덮어쓰기
            output_path
        ]
        
        # 프로세스 실행
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True
        )
        
        stdout, stderr = process.communicate()
        
        if process.returncode == 0:
            return True, "변환 성공"
        else:
            return False, f"변환 실패: {stderr}"
            
    except FileNotFoundError:
        return False, "ffmpeg가 설치되어 있지 않습니다."
    except Exception as e:
        return False, f"변환 중 오류: {str(e)}"

# 페이지 설정
st.set_page_config(
    page_title="Parbiomech Video Analysis",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="collapsed"
)

def calculate_segment_angle(point1, point2):
    """두 점을 잇는 분절의 수평면 대비 절대 각도 계산 (영상 기준, 도 단위)"""
    p1 = np.array(point1)
    p2 = np.array(point2)
    
    # 수평선(x축) 대비 각도
    radians = np.arctan2(p2[1] - p1[1], p2[0] - p1[0])
    angle = np.degrees(radians)
    
    # 0~360도 범위로 정규화
    if angle < 0:
        angle += 360
        
    return angle

def calculate_joint_angle(a, b, c):
    """세 점 사이의 관절 각도 계산 (근위-관절-원위, 도 단위)"""
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    
    if angle > 180.0:
        angle = 360-angle
        
    return angle

def calculate_all_angles(landmarks):
    """모든 절대각도와 상대각도 계산"""
    angles = {
        'absolute': {},  # 절대각도 (분절 기울기)
        'relative': {}   # 상대각도 (관절각도)
    }
    
    # 절대각도 계산 (영상 기준 분절 기울기)
    # 머리 각도
    if all(landmarks[i].visibility > 0.5 for i in [0, 1]):
        angles['absolute']['머리'] = calculate_segment_angle(
            [landmarks[1].x, landmarks[1].y],
            [landmarks[0].x, landmarks[0].y]
        )
    
    # 어깨 각도
    if all(landmarks[i].visibility > 0.5 for i in [11, 12]):
        angles['absolute']['어깨'] = calculate_segment_angle(
            [landmarks[11].x, landmarks[11].y],
            [landmarks[12].x, landmarks[12].y]
        )
    
    # 골반 각도
    if all(landmarks[i].visibility > 0.5 for i in [23, 24]):
        angles['absolute']['골반'] = calculate_segment_angle(
            [landmarks[23].x, landmarks[23].y],
            [landmarks[24].x, landmarks[24].y]
        )
    
    # 몸통 기울기
    if all(landmarks[i].visibility > 0.5 for i in [11, 12, 23, 24]):
        shoulder_center = [(landmarks[11].x + landmarks[12].x)/2, (landmarks[11].y + landmarks[12].y)/2]
        hip_center = [(landmarks[23].x + landmarks[24].x)/2, (landmarks[23].y + landmarks[24].y)/2]
        angles['absolute']['몸통'] = calculate_segment_angle(hip_center, shoulder_center)
    
    # 좌우 상완 각도
    if all(landmarks[i].visibility > 0.5 for i in [11, 13]):
        angles['absolute']['좌_상완'] = calculate_segment_angle(
            [landmarks[11].x, landmarks[11].y],
            [landmarks[13].x, landmarks[13].y]
        )
    if all(landmarks[i].visibility > 0.5 for i in [12, 14]):
        angles['absolute']['우_상완'] = calculate_segment_angle(
            [landmarks[12].x, landmarks[12].y],
            [landmarks[14].x, landmarks[14].y]
        )
    
    # 좌우 하완 각도
    if all(landmarks[i].visibility > 0.5 for i in [13, 15]):
        angles['absolute']['좌_하완'] = calculate_segment_angle(
            [landmarks[13].x, landmarks[13].y],
            [landmarks[15].x, landmarks[15].y]
        )
    if all(landmarks[i].visibility > 0.5 for i in [14, 16]):
        angles['absolute']['우_하완'] = calculate_segment_angle(
            [landmarks[14].x, landmarks[14].y],
            [landmarks[16].x, landmarks[16].y]
        )
    
    # 좌우 대퇴 각도
    if all(landmarks[i].visibility > 0.5 for i in [23, 25]):
        angles['absolute']['좌_대퇴'] = calculate_segment_angle(
            [landmarks[23].x, landmarks[23].y],
            [landmarks[25].x, landmarks[25].y]
        )
    if all(landmarks[i].visibility > 0.5 for i in [24, 26]):
        angles['absolute']['우_대퇴'] = calculate_segment_angle(
            [landmarks[24].x, landmarks[24].y],
            [landmarks[26].x, landmarks[26].y]
        )
    
    # 좌우 하퇴 각도
    if all(landmarks[i].visibility > 0.5 for i in [25, 27]):
        angles['absolute']['좌_하퇴'] = calculate_segment_angle(
            [landmarks[25].x, landmarks[25].y],
            [landmarks[27].x, landmarks[27].y]
        )
    if all(landmarks[i].visibility > 0.5 for i in [26, 28]):
        angles['absolute']['우_하퇴'] = calculate_segment_angle(
            [landmarks[26].x, landmarks[26].y],
            [landmarks[28].x, landmarks[28].y]
        )
    
    # 상대각도 계산 (관절각도)
    # 좌우 어깨
    if all(landmarks[i].visibility > 0.5 for i in [11, 13, 23]):
        angles['relative']['좌_어깨'] = calculate_joint_angle(
            [landmarks[23].x, landmarks[23].y],
            [landmarks[11].x, landmarks[11].y],
            [landmarks[13].x, landmarks[13].y]
        )
    if all(landmarks[i].visibility > 0.5 for i in [12, 14, 24]):
        angles['relative']['우_어깨'] = calculate_joint_angle(
            [landmarks[24].x, landmarks[24].y],
            [landmarks[12].x, landmarks[12].y],
            [landmarks[14].x, landmarks[14].y]
        )
    
    # 좌우 팔꿈치
    if all(landmarks[i].visibility > 0.5 for i in [11, 13, 15]):
        angles['relative']['좌_팔꿈치'] = calculate_joint_angle(
            [landmarks[11].x, landmarks[11].y],
            [landmarks[13].x, landmarks[13].y],
            [landmarks[15].x, landmarks[15].y]
        )
    if all(landmarks[i].visibility > 0.5 for i in [12, 14, 16]):
        angles['relative']['우_팔꿈치'] = calculate_joint_angle(
            [landmarks[12].x, landmarks[12].y],
            [landmarks[14].x, landmarks[14].y],
            [landmarks[16].x, landmarks[16].y]
        )
    
    # 좌우 손목
    if all(landmarks[i].visibility > 0.5 for i in [13, 15, 17]):
        angles['relative']['좌_손목'] = calculate_joint_angle(
            [landmarks[13].x, landmarks[13].y],
            [landmarks[15].x, landmarks[15].y],
            [landmarks[17].x, landmarks[17].y]
        )
    if all(landmarks[i].visibility > 0.5 for i in [14, 16, 18]):
        angles['relative']['우_손목'] = calculate_joint_angle(
            [landmarks[14].x, landmarks[14].y],
            [landmarks[16].x, landmarks[16].y],
            [landmarks[18].x, landmarks[18].y]
        )
    
    # 좌우 엉덩이(고관절)
    if all(landmarks[i].visibility > 0.5 for i in [11, 23, 25]):
        angles['relative']['좌_엉덩이'] = calculate_joint_angle(
            [landmarks[11].x, landmarks[11].y],
            [landmarks[23].x, landmarks[23].y],
            [landmarks[25].x, landmarks[25].y]
        )
    if all(landmarks[i].visibility > 0.5 for i in [12, 24, 26]):
        angles['relative']['우_엉덩이'] = calculate_joint_angle(
            [landmarks[12].x, landmarks[12].y],
            [landmarks[24].x, landmarks[24].y],
            [landmarks[26].x, landmarks[26].y]
        )
    
    # 좌우 무릎
    if all(landmarks[i].visibility > 0.5 for i in [23, 25, 27]):
        angles['relative']['좌_무릎'] = calculate_joint_angle(
            [landmarks[23].x, landmarks[23].y],
            [landmarks[25].x, landmarks[25].y],
            [landmarks[27].x, landmarks[27].y]
        )
    if all(landmarks[i].visibility > 0.5 for i in [24, 26, 28]):
        angles['relative']['우_무릎'] = calculate_joint_angle(
            [landmarks[24].x, landmarks[24].y],
            [landmarks[26].x, landmarks[26].y],
            [landmarks[28].x, landmarks[28].y]
        )
    
    # 좌우 발목
    if all(landmarks[i].visibility > 0.5 for i in [25, 27, 31]):
        angles['relative']['좌_발목'] = calculate_joint_angle(
            [landmarks[25].x, landmarks[25].y],
            [landmarks[27].x, landmarks[27].y],
            [landmarks[31].x, landmarks[31].y]
        )
    if all(landmarks[i].visibility > 0.5 for i in [26, 28, 32]):
        angles['relative']['우_발목'] = calculate_joint_angle(
            [landmarks[26].x, landmarks[26].y],
            [landmarks[28].x, landmarks[28].y],
            [landmarks[32].x, landmarks[32].y]
        )
    
    return angles

def analyze_frame_at_time(video_path, time_sec, pose_detector):
    """특정 시점의 프레임 분석"""
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_num = int(time_sec * fps)
    
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        return None, None, None
    
    # RGB로 변환
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # 포즈 감지
    results = pose_detector.process(image_rgb)
    
    # 포즈 그리기
    annotated_frame = frame.copy()
    angles = None
    
    if results.pose_landmarks:
        # 포즈 랜드마크 그리기
        mp_drawing.draw_landmarks(
            annotated_frame,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()
        )
        
        landmarks = results.pose_landmarks.landmark
        angles = calculate_all_angles(landmarks)
    
    # BGR to RGB for display
    annotated_frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
    
    return annotated_frame_rgb, angles, results.pose_landmarks is not None

def process_video(video_file, timepoints, confidence_threshold=0.5):
    """비디오를 처리하고 전체 분석 영상 + 시점별 분석"""
    # 임시 파일로 저장
    temp_path = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    temp_path.write(video_file.read())
    temp_path.close()
    
    # 비디오 정보 가져오기
    cap = cv2.VideoCapture(temp_path.name)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # 전체 영상 분석 및 스켈레톤 오버레이
    processed_frames = []
    tracking_data = []
    
    with mp_pose.Pose(
        min_detection_confidence=confidence_threshold,
        min_tracking_confidence=confidence_threshold,
        model_complexity=1
    ) as pose:
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        frame_count = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # RGB로 변환
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            
            # 포즈 감지
            results = pose.process(image)
            
            # 다시 쓰기 가능하게
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            
            # 포즈 랜드마크 그리기
            if results.pose_landmarks:
                mp_drawing.draw_landmarks(
                    image,
                    results.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()
                )
                
                # 랜드마크 데이터 저장
                landmarks = results.pose_landmarks.landmark
                frame_data = {
                    'frame': frame_count,
                    'time': frame_count / fps
                }
                
                # 각 랜드마크의 x, y 좌표 저장
                for idx, landmark in enumerate(landmarks):
                    frame_data[f'x_{idx}'] = landmark.x
                    frame_data[f'y_{idx}'] = landmark.y
                    frame_data[f'z_{idx}'] = landmark.z
                    frame_data[f'visibility_{idx}'] = landmark.visibility
                
                tracking_data.append(frame_data)
            
            # 처리된 프레임을 메모리에 저장
            processed_frames.append(image)
            
            frame_count += 1
            progress = int((frame_count / total_frames) * 50)  # 50%까지만 (전체 영상 처리)
            progress_bar.progress(progress)
            status_text.text(f'전체 영상 분석 중: {frame_count}/{total_frames} 프레임')
        
        cap.release()
        
        # 처리된 프레임들을 비디오 파일로 저장
        output_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        output_path = output_file.name
        output_file.close()
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        for frame in processed_frames:
            out.write(frame)
        
        out.release()
        
        # 시점별 분석 수행
        timepoint_results = []
        
        status_text.text('시점별 상세 분석 중...')
        
        for idx, time_point in enumerate(timepoints):
            progress = 50 + int(((idx + 1) / len(timepoints)) * 50)  # 50%~100%
            progress_bar.progress(progress)
            status_text.text(f'시점 {idx+1}/{len(timepoints)} 분석 중... ({time_point:.2f}초)')
            
            frame, angles, detected = analyze_frame_at_time(
                temp_path.name,
                time_point,
                pose
            )
            
            if frame is not None:
                timepoint_results.append({
                    'time': time_point,
                    'frame': frame,
                    'angles': angles,
                    'detected': detected
                })
        
        progress_bar.empty()
        status_text.empty()
    
    # 임시 파일 삭제
    try:
        os.unlink(temp_path.name)
    except:
        pass
    
    # 추적 데이터 DataFrame 생성
    df_tracking = pd.DataFrame(tracking_data)
    
    return timepoint_results, df_tracking, output_path, fps, width, height

# 메인 애플리케이션
st.title("🎯 Parbiomech Video Analysis")
st.markdown("**MediaPipe 기반 포즈 분석 시스템**")

# 비디오 형식 안내
st.info("""
ℹ️ **지원되는 비디오 형식**: H.264 (AVC) 또는 H.265 (HEVC) 코덱  
⚠️ **지원되지 않는 형식**: AV1 코덱 (변환 필요)
""")

# 사이드바에 설정 추가
st.sidebar.header("⚙️ 분석 설정")
confidence_threshold = st.sidebar.slider(
    "감지 신뢰도 임계값",
    min_value=0.0,
    max_value=1.0,
    value=0.5,
    step=0.05,
    help="포즈 감지의 최소 신뢰도를 설정합니다."
)

# 시점 관리를 위한 session state 초기화
if 'timepoints' not in st.session_state:
    st.session_state['timepoints'] = []

# 비디오 업로드
uploaded_file = st.file_uploader(
    "비디오 파일을 업로드하세요",
    type=['mp4', 'avi', 'mov', 'mkv'],
    help="H.264 또는 H.265 코덱을 사용하는 비디오를 선택하세요."
)

if uploaded_file is not None:
    # 비디오를 session state에 저장 (한 번만 처리)
    if 'uploaded_file_name' not in st.session_state or st.session_state.get('uploaded_file_name') != uploaded_file.name:
        
        with st.spinner("비디오를 불러오는 중..."):
            video_bytes = uploaded_file.read()
            
            # 임시 파일로 저장하여 정보 추출
            temp_check = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
            temp_check.write(video_bytes)
            temp_check.close()
            
            # 코덱 체크
            codec_supported, codec_status, codec_message = check_video_codec(temp_check.name)
            
            if not codec_supported and codec_status == "needs_conversion":
                st.warning("⚠️ 비디오 코덱 변환 중...")
                converted_path = tempfile.NamedTemporaryFile(delete=False, suffix='_h264.mp4').name
                conversion_success, conversion_message = convert_video_to_h264(temp_check.name, converted_path)
                
                if conversion_success:
                    st.success("✅ 변환 완료!")
                    os.unlink(temp_check.name)
                    temp_check.name = converted_path
                    # 변환된 파일 읽기
                    with open(converted_path, 'rb') as f:
                        video_bytes = f.read()
                else:
                    st.error(f"❌ 변환 실패: {conversion_message}")
                    os.unlink(temp_check.name)
                    st.stop()
            elif not codec_supported:
                st.error(f"⚠️ 비디오 형식 오류: {codec_message}")
                os.unlink(temp_check.name)
                st.stop()
            
            # 비디오 정보 추출
            cap = cv2.VideoCapture(temp_check.name)
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            total_time = total_frames / fps if fps > 0 else 0
            cap.release()
            
            # 임시 파일 삭제
            os.unlink(temp_check.name)
            
            # Session state에 저장
            st.session_state['original_video_bytes'] = video_bytes
            st.session_state['uploaded_file_name'] = uploaded_file.name
            st.session_state['timepoints'] = []
            st.session_state['fps'] = fps
            st.session_state['total_frames'] = total_frames
            st.session_state['total_time'] = total_time
            
            uploaded_file.seek(0)
    
    # Session state에서 정보 가져오기
    fps = st.session_state['fps']
    total_frames = st.session_state['total_frames']
    total_time = st.session_state['total_time']
    
    st.info(f"📹 비디오 정보: {total_time:.2f}초 ({total_frames} 프레임, {fps:.1f}fps)")
    
    # 영상 재생 중 시점 태그 기능
    st.markdown("### 📹 원본 영상")
    
    # 비디오를 HTML5 플레이어로 표시
    import base64
    video_base64 = base64.b64encode(st.session_state['original_video_bytes']).decode()
    
    # 현재 추가된 시점 표시용
    current_timepoints = ", ".join([f"{t:.2f}초" for t in st.session_state['timepoints']]) if st.session_state['timepoints'] else "없음"
    
    video_html = f"""
    <style>
        .video-container {{
            margin-bottom: 20px;
        }}
        .tag-panel {{
            background: #f0f2f6;
            padding: 20px;
            border-radius: 10px;
            margin-top: 15px;
        }}
        .tag-controls {{
            display: flex;
            gap: 15px;
            align-items: center;
            margin-bottom: 15px;
        }}
        .time-display {{
            flex: 1;
            font-size: 16px;
            font-weight: bold;
        }}
        .add-btn {{
            padding: 12px 24px;
            background: #FF4B4B;
            color: white;
            border: none;
            border-radius: 5px;
            cursor: pointer;
            font-weight: bold;
            font-size: 14px;
        }}
        .add-btn:hover {{
            background: #FF6B6B;
        }}
        .timepoint-list {{
            font-size: 14px;
            color: #666;
            margin-top: 10px;
        }}
    </style>
    
    <div class="video-container">
        <video id="mainVideo" width="100%" controls>
            <source src="data:video/mp4;base64,{video_base64}" type="video/mp4">
        </video>
    </div>
    
    <div class="tag-panel">
        <h4>⏱️ 시점 태그</h4>
        <div class="tag-controls">
            <div class="time-display">
                현재 시점: <span id="currentTime">0.00</span>초 / {total_time:.2f}초
            </div>
            <button class="add-btn" onclick="addCurrentTime()">➕ 현재 시점 추가</button>
        </div>
        <div class="timepoint-list">
            추가된 시점: <span id="timepointDisplay">{current_timepoints}</span>
        </div>
    </div>
    
    <script>
        const video = document.getElementById('mainVideo');
        const timeDisplay = document.getElementById('currentTime');
        const timepointDisplay = document.getElementById('timepointDisplay');
        
        // 시점 목록 (Python session state와 동기화)
        let timepoints = {st.session_state['timepoints']};
        
        // 비디오 시간 업데이트
        video.addEventListener('timeupdate', function() {{
            timeDisplay.textContent = video.currentTime.toFixed(2);
        }});
        
        // 시점 목록 업데이트
        function updateTimepointDisplay() {{
            if (timepoints.length > 0) {{
                timepointDisplay.textContent = timepoints.map(t => t.toFixed(2) + '초').join(', ');
            }} else {{
                timepointDisplay.textContent = '없음';
            }}
        }}
        
        // 현재 시점 추가
        function addCurrentTime() {{
            const currentTime = parseFloat(video.currentTime.toFixed(2));
            
            if (!timepoints.includes(currentTime)) {{
                timepoints.push(currentTime);
                timepoints.sort((a, b) => a - b);
                updateTimepointDisplay();
                
                // Streamlit에 데이터 전송
                window.parent.postMessage({{
                    type: 'streamlit:setComponentValue',
                    value: currentTime
                }}, '*');
                
                // 성공 메시지
                const tempMsg = document.createElement('div');
                tempMsg.style.cssText = 'position:fixed;top:20px;right:20px;background:#4CAF50;color:white;padding:15px 25px;border-radius:5px;z-index:9999;';
                tempMsg.textContent = `✓ ${{currentTime.toFixed(2)}}초 추가됨`;
                document.body.appendChild(tempMsg);
                setTimeout(() => tempMsg.remove(), 2000);
            }} else {{
                alert('이미 추가된 시점입니다.');
            }}
        }}
        
        // 키보드 단축키: 스페이스바로 시점 추가
        document.addEventListener('keydown', function(e) {{
            if (e.code === 'Space' && e.target.tagName !== 'INPUT') {{
                e.preventDefault();
                addCurrentTime();
            }}
        }});
    </script>
    """
    
    # HTML 컴포넌트 표시
    from streamlit.components.v1 import html
    added_time = html(video_html, height=600)
    
    # JavaScript에서 추가된 시점 처리
    if added_time is not None:
        try:
            time_value = float(added_time)
            if time_value > 0 and time_value not in st.session_state['timepoints']:
                st.session_state['timepoints'].append(time_value)
                st.session_state['timepoints'].sort()
                st.rerun()
        except (ValueError, TypeError):
            pass
    
    st.markdown("---")
    
    # 시점 관리 섹션
    st.markdown("### 📋 시점 관리")
    
    # 현재 시점 목록
    if st.session_state['timepoints']:
        st.success(f"✅ {len(st.session_state['timepoints'])}개의 시점이 추가되었습니다")
        
        # 시점 목록을 카드 형식으로 표시
        cols = st.columns(6)
        for idx, time_point in enumerate(st.session_state['timepoints']):
            with cols[idx % 6]:
                if st.button(f"🗑️ {time_point:.2f}초", key=f"del_{idx}", use_container_width=True):
                    st.session_state['timepoints'].remove(time_point)
                    st.rerun()
        
        # 전체 삭제 버튼
        if st.button("🗑️ 전체 삭제", use_container_width=False, type="secondary"):
            st.session_state['timepoints'] = []
            st.rerun()
    else:
        st.info("📹 영상을 보며 원하는 시점에서 '➕ 시점 추가' 버튼을 클릭하세요")
    
    # 수동 입력 옵션 (접기)
    with st.expander("⌨️ 수동으로 시점 입력하기", expanded=False):
        col1, col2 = st.columns([3, 1])
        
        with col1:
            manual_time = st.number_input(
                "시점 (초)",
                min_value=0.0,
                max_value=total_time,
                value=0.0,
                step=0.1,
                key="manual_time_input"
            )
        
        with col2:
            st.write("")  # 공간 확보
            st.write("")  # 공간 확보
            if st.button("추가", use_container_width=True, type="primary"):
                if manual_time not in st.session_state['timepoints']:
                    st.session_state['timepoints'].append(manual_time)
                    st.session_state['timepoints'].sort()
                    st.rerun()
                else:
                    st.warning("중복 시점")
    
    st.markdown("---")
    
    # 분석 버튼
    if st.session_state['timepoints']:
        if st.button("🔍 분석 시작", type="primary", use_container_width=True):
            # uploaded_file을 다시 읽기
            uploaded_file.seek(0)
            
            with st.spinner("분석 중... 잠시만 기다려주세요."):
                timepoint_results, df_tracking, output_video_path, fps_result, width, height = process_video(
                    uploaded_file,
                    st.session_state['timepoints'],
                    confidence_threshold
                )
                
                # 결과를 세션 상태에 저장
                st.session_state['timepoint_results'] = timepoint_results
                st.session_state['df_tracking'] = df_tracking
                st.session_state['output_video_path'] = output_video_path
                st.session_state['fps'] = fps_result
                st.session_state['video_info'] = f"{width}x{height} @ {fps_result:.1f}fps"
            
            st.success("✅ 분석 완료!")
    else:
        st.warning("⚠️ 먼저 분석할 시점을 추가해주세요.")
    
    # 결과 표시
    if 'timepoint_results' in st.session_state and st.session_state['timepoint_results']:
        st.markdown("---")
        st.header("📊 분석 결과")
        
        results = st.session_state['timepoint_results']
        
        # 다운로드 섹션
        st.subheader("💾 다운로드")
        
        col1, col2, col3 = st.columns(3)
        
        # 분석 영상 다운로드
        with col1:
            if 'output_video_path' in st.session_state and os.path.exists(st.session_state['output_video_path']):
                with open(st.session_state['output_video_path'], 'rb') as video_file:
                    video_bytes = video_file.read()
                
                st.download_button(
                    label="🎥 분석 영상 다운로드",
                    data=video_bytes,
                    file_name="skeleton_overlay_video.mp4",
                    mime="video/mp4",
                    use_container_width=True
                )
        
        # CSV 데이터 생성 및 다운로드
        with col2:
            download_data = []
            for result in results:
                if result['angles']:
                    row = {'시점(초)': result['time']}
                    
                    # 절대각도 추가
                    for name, value in result['angles']['absolute'].items():
                        row[f'절대각도_{name}'] = f"{value:.2f}"
                    
                    # 상대각도 추가
                    for name, value in result['angles']['relative'].items():
                        row[f'상대각도_{name}'] = f"{value:.2f}"
                    
                    download_data.append(row)
            
            if download_data:
                df_download = pd.DataFrame(download_data)
                csv = df_download.to_csv(index=False, encoding='utf-8-sig')
                
                st.download_button(
                    label="📥 각도 데이터 CSV",
                    data=csv,
                    file_name="angle_data.csv",
                    mime="text/csv",
                    use_container_width=True
                )
        
        # 궤적 데이터 다운로드
        with col3:
            if 'df_tracking' in st.session_state and not st.session_state['df_tracking'].empty:
                tracking_csv = st.session_state['df_tracking'].to_csv(index=False, encoding='utf-8-sig')
                st.download_button(
                    label="📍 궤적 데이터 CSV",
                    data=tracking_csv,
                    file_name="tracking_data.csv",
                    mime="text/csv",
                    use_container_width=True
                )
        
        st.info(f"📊 비디오 정보: {st.session_state['video_info']}")
        
        st.markdown("---")
        
        # 시점별 분석 결과
        st.subheader("📸 시점별 분석 결과")
        
        for idx, result in enumerate(results):
            with st.expander(f"🔍 시점 {idx + 1}: {result['time']:.2f}초", expanded=(idx == 0)):
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    # 포즈가 그려진 이미지
                    st.image(result['frame'], caption=f"{result['time']:.2f}초", use_container_width=True)
                
                with col2:
                    # 각도 정보
                    if result['detected'] and result['angles']:
                        # 절대각도
                        with st.expander("📐 절대각도 (분절 기울기)", expanded=False):
                            # 2열로 표시
                            abs_items = list(result['angles']['absolute'].items())
                            cols = st.columns(2)
                            for i, (joint, angle) in enumerate(abs_items):
                                with cols[i % 2]:
                                    st.markdown(f"**{joint}**: {angle:.1f}°")
                        
                        # 상대각도
                        with st.expander("🔢 상대각도 (관절각도)", expanded=False):
                            # 2열로 표시
                            rel_items = list(result['angles']['relative'].items())
                            cols = st.columns(2)
                            for i, (joint, angle) in enumerate(rel_items):
                                with cols[i % 2]:
                                    st.markdown(f"**{joint}**: {angle:.1f}°")
                    else:
                        st.warning("포즈를 감지하지 못했습니다.")
        
        # 시점간 각도 비교 그래프
        if len(results) > 1:
            st.subheader("📈 시점간 각도 비교")
            
            # 탭으로 절대각도와 상대각도 구분
            tab1, tab2 = st.tabs(["절대각도 (분절 기울기)", "상대각도 (관절각도)"])
            
            with tab1:
                # 절대각도 그래프
                all_abs_angles = set()
                for result in results:
                    if result['angles'] and 'absolute' in result['angles']:
                        all_abs_angles.update(result['angles']['absolute'].keys())
                
                if all_abs_angles:
                    selected_abs = st.multiselect(
                        "비교할 절대각도 선택",
                        list(all_abs_angles),
                        default=list(all_abs_angles)[:3] if len(all_abs_angles) >= 3 else list(all_abs_angles),
                        key="abs_angles"
                    )
                    
                    if selected_abs:
                        fig = go.Figure()
                        
                        for angle_name in selected_abs:
                            times = []
                            angles = []
                            
                            for result in results:
                                if result['angles'] and 'absolute' in result['angles'] and angle_name in result['angles']['absolute']:
                                    times.append(result['time'])
                                    angles.append(result['angles']['absolute'][angle_name])
                            
                            if times:
                                fig.add_trace(go.Scatter(
                                    x=times,
                                    y=angles,
                                    mode='lines+markers',
                                    name=angle_name,
                                    marker=dict(size=12),
                                    line=dict(width=3)
                                ))
                        
                        fig.update_layout(
                            title="시점별 절대각도 변화 (분절 기울기)",
                            xaxis_title="시간 (초)",
                            yaxis_title="각도 (도)",
                            height=500,
                            hovermode='x unified'
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
            
            with tab2:
                # 상대각도 그래프
                all_rel_angles = set()
                for result in results:
                    if result['angles'] and 'relative' in result['angles']:
                        all_rel_angles.update(result['angles']['relative'].keys())
                
                if all_rel_angles:
                    selected_rel = st.multiselect(
                        "비교할 상대각도 선택",
                        list(all_rel_angles),
                        default=list(all_rel_angles)[:3] if len(all_rel_angles) >= 3 else list(all_rel_angles),
                        key="rel_angles"
                    )
                    
                    if selected_rel:
                        fig = go.Figure()
                        
                        for angle_name in selected_rel:
                            times = []
                            angles = []
                            
                            for result in results:
                                if result['angles'] and 'relative' in result['angles'] and angle_name in result['angles']['relative']:
                                    times.append(result['time'])
                                    angles.append(result['angles']['relative'][angle_name])
                            
                            if times:
                                fig.add_trace(go.Scatter(
                                    x=times,
                                    y=angles,
                                    mode='lines+markers',
                                    name=angle_name,
                                    marker=dict(size=12),
                                    line=dict(width=3)
                                ))
                        
                        fig.update_layout(
                            title="시점별 상대각도 변화 (관절각도)",
                            xaxis_title="시간 (초)",
                            yaxis_title="각도 (도)",
                            height=500,
                            hovermode='x unified'
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
        
        # 궤적 분석 추가
        if 'df_tracking' in st.session_state and not st.session_state['df_tracking'].empty:
            st.markdown("---")
            st.subheader("📍 키포인트 궤적 분석")
            
            df_tracking = st.session_state['df_tracking']
            
            # 주요 키포인트 선택
            keypoint_names = {
                0: "코", 11: "왼쪽 어깨", 12: "오른쪽 어깨",
                13: "왼쪽 팔꿈치", 14: "오른쪽 팔꿈치",
                15: "왼쪽 손목", 16: "오른쪽 손목",
                23: "왼쪽 엉덩이", 24: "오른쪽 엉덩이",
                25: "왼쪽 무릎", 26: "오른쪽 무릎",
                27: "왼쪽 발목", 28: "오른쪽 발목"
            }
            
            selected_keypoints = st.multiselect(
                "궤적을 분석할 키포인트 선택",
                list(keypoint_names.keys()),
                default=[15, 16, 27, 28],  # 손목, 발목
                format_func=lambda x: keypoint_names[x],
                key="keypoint_select"
            )
            
            if selected_keypoints:
                # 궤적 그래프
                fig = go.Figure()
                
                for kp_idx in selected_keypoints:
                    x_col = f'x_{kp_idx}'
                    y_col = f'y_{kp_idx}'
                    
                    if x_col in df_tracking.columns and y_col in df_tracking.columns:
                        fig.add_trace(go.Scatter(
                            x=df_tracking[x_col],
                            y=df_tracking[y_col],
                            mode='markers',
                            name=keypoint_names[kp_idx],
                            marker=dict(
                                size=3,
                                color=df_tracking['frame'],
                                colorscale='Viridis',
                                showscale=True if kp_idx == selected_keypoints[0] else False,
                                colorbar=dict(title="프레임")
                            )
                        ))
                
                fig.update_layout(
                    title="키포인트 2D 궤적 (색상: 시간 진행)",
                    xaxis_title="X 좌표",
                    yaxis_title="Y 좌표",
                    height=600,
                    yaxis=dict(scaleanchor="x", scaleratio=1),
                    hovermode='closest'
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # 시계열 그래프
                st.markdown("### 시간에 따른 좌표 변화")
                
                for kp_idx in selected_keypoints:
                    x_col = f'x_{kp_idx}'
                    y_col = f'y_{kp_idx}'
                    
                    if x_col in df_tracking.columns and y_col in df_tracking.columns:
                        from plotly.subplots import make_subplots
                        
                        fig2 = make_subplots(
                            rows=2, cols=1,
                            subplot_titles=(f"{keypoint_names[kp_idx]} - X 좌표", f"{keypoint_names[kp_idx]} - Y 좌표")
                        )
                        
                        fig2.add_trace(
                            go.Scatter(x=df_tracking['time'], y=df_tracking[x_col], 
                                      mode='lines', name='X', line=dict(color='blue')),
                            row=1, col=1
                        )
                        
                        fig2.add_trace(
                            go.Scatter(x=df_tracking['time'], y=df_tracking[y_col], 
                                      mode='lines', name='Y', line=dict(color='red')),
                            row=2, col=1
                        )
                        
                        fig2.update_xaxes(title_text="시간 (초)", row=2, col=1)
                        fig2.update_yaxes(title_text="X", row=1, col=1)
                        fig2.update_yaxes(title_text="Y", row=2, col=1)
                        fig2.update_layout(height=500, showlegend=False, 
                                          title=f"{keypoint_names[kp_idx]} 시계열 변화")
                        
                        st.plotly_chart(fig2, use_container_width=True)

else:
    st.info("👆 비디오 파일을 업로드하여 분석을 시작하세요.")
