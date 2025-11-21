import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
import tempfile
import os
from pathlib import Path
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from PIL import Image
import io

# MediaPipe 초기화
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# 페이지 설정
st.set_page_config(
    page_title="Keypoint Tracker - 동작 분석 도구",
    page_icon="🎯",
    layout="wide"
)

# 스타일 설정
st.markdown("""
    <style>
    .main {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    .stApp {
        background: white;
        border-radius: 20px;
        padding: 20px;
    }
    </style>
""", unsafe_allow_html=True)

# 제목
st.title("🎯 Keypoint Tracker - 동작 분석 도구")
st.markdown("---")

# 포즈 랜드마크 정의
POSE_LANDMARKS = {
    0: "Nose", 11: "Left Shoulder", 12: "Right Shoulder",
    13: "Left Elbow", 14: "Right Elbow", 15: "Left Wrist", 16: "Right Wrist",
    23: "Left Hip", 24: "Right Hip", 25: "Left Knee", 26: "Right Knee",
    27: "Left Ankle", 28: "Right Ankle"
}

def calculate_angle(a, b, c):
    """세 점으로 각도 계산"""
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    
    if angle > 180.0:
        angle = 360-angle
        
    return angle

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
    angles = {}
    
    if results.pose_landmarks:
        # 포즈 랜드마크 그리기
        mp_drawing.draw_landmarks(
            annotated_frame,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()
        )
        
        landmarks = results.pose_landmarks.landmark
        
        # 주요 각도 계산
        # 왼쪽 팔꿈치
        if all(landmarks[i].visibility > 0.5 for i in [11, 13, 15]):
            shoulder = [landmarks[11].x, landmarks[11].y]
            elbow = [landmarks[13].x, landmarks[13].y]
            wrist = [landmarks[15].x, landmarks[15].y]
            angles['왼쪽 팔꿈치'] = calculate_angle(shoulder, elbow, wrist)
        
        # 오른쪽 팔꿈치
        if all(landmarks[i].visibility > 0.5 for i in [12, 14, 16]):
            shoulder = [landmarks[12].x, landmarks[12].y]
            elbow = [landmarks[14].x, landmarks[14].y]
            wrist = [landmarks[16].x, landmarks[16].y]
            angles['오른쪽 팔꿈치'] = calculate_angle(shoulder, elbow, wrist)
        
        # 왼쪽 무릎
        if all(landmarks[i].visibility > 0.5 for i in [23, 25, 27]):
            hip = [landmarks[23].x, landmarks[23].y]
            knee = [landmarks[25].x, landmarks[25].y]
            ankle = [landmarks[27].x, landmarks[27].y]
            angles['왼쪽 무릎'] = calculate_angle(hip, knee, ankle)
        
        # 오른쪽 무릎
        if all(landmarks[i].visibility > 0.5 for i in [24, 26, 28]):
            hip = [landmarks[24].x, landmarks[24].y]
            knee = [landmarks[26].x, landmarks[26].y]
            ankle = [landmarks[28].x, landmarks[28].y]
            angles['오른쪽 무릎'] = calculate_angle(hip, knee, ankle)
        
        # 왼쪽 고관절
        if all(landmarks[i].visibility > 0.5 for i in [11, 23, 25]):
            shoulder = [landmarks[11].x, landmarks[11].y]
            hip = [landmarks[23].x, landmarks[23].y]
            knee = [landmarks[25].x, landmarks[25].y]
            angles['왼쪽 고관절'] = calculate_angle(shoulder, hip, knee)
        
        # 오른쪽 고관절
        if all(landmarks[i].visibility > 0.5 for i in [12, 24, 26]):
            shoulder = [landmarks[12].x, landmarks[12].y]
            hip = [landmarks[24].x, landmarks[24].y]
            knee = [landmarks[26].x, landmarks[26].y]
            angles['오른쪽 고관절'] = calculate_angle(shoulder, hip, knee)
    
    # BGR to RGB for display
    annotated_frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
    
    return annotated_frame_rgb, angles, results.pose_landmarks is not None

def apply_lowpass_filter(data, strength=5):
    """로우패스 필터 적용"""
    if len(data) == 0:
        return data
    
    filtered = []
    for i in range(len(data)):
        start = max(0, i - strength)
        end = min(len(data), i + strength + 1)
        filtered.append(np.mean(data[start:end]))
    
    return filtered

def process_video(video_file, confidence_threshold=0.5, filter_strength=5):
    """비디오 처리 및 키포인트 추출"""
    # 임시 파일로 저장
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    tfile.write(video_file.read())
    tfile.close()
    
    # 비디오 캡처
    cap = cv2.VideoCapture(tfile.name)
    
    # 비디오 정보
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # 결과 저장을 위한 리스트
    tracking_data = []
    angle_data = []
    processed_frames = []  # 프레임을 메모리에 저장
    
    # 진행 상황 표시
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # MediaPipe Pose 초기화 (model_complexity=1로 변경하여 권한 문제 회피)
    with mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        min_detection_confidence=confidence_threshold,
        min_tracking_confidence=confidence_threshold
    ) as pose:
        
        frame_num = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # RGB로 변환
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # 포즈 감지
            results = pose.process(image_rgb)
            
            # 포즈 그리기
            annotated_frame = frame.copy()
            if results.pose_landmarks:
                # 포즈 랜드마크 그리기
                mp_drawing.draw_landmarks(
                    annotated_frame,
                    results.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()
                )
            
            # 프레임을 리스트에 저장
            processed_frames.append(annotated_frame)
            
            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark
                
                # 프레임별 키포인트 저장
                frame_data = {'frame': frame_num, 'time': frame_num / fps}
                
                for idx, name in POSE_LANDMARKS.items():
                    lm = landmarks[idx]
                    frame_data[f'{name}_x'] = lm.x
                    frame_data[f'{name}_y'] = lm.y
                    frame_data[f'{name}_z'] = lm.z
                    frame_data[f'{name}_visibility'] = lm.visibility
                
                tracking_data.append(frame_data)
                
                # 각도 계산
                angles = {}
                
                # 왼쪽 팔꿈치 각도
                if all(landmarks[i].visibility > confidence_threshold for i in [11, 13, 15]):
                    shoulder = [landmarks[11].x, landmarks[11].y]
                    elbow = [landmarks[13].x, landmarks[13].y]
                    wrist = [landmarks[15].x, landmarks[15].y]
                    angles['left_elbow'] = calculate_angle(shoulder, elbow, wrist)
                
                # 오른쪽 팔꿈치 각도
                if all(landmarks[i].visibility > confidence_threshold for i in [12, 14, 16]):
                    shoulder = [landmarks[12].x, landmarks[12].y]
                    elbow = [landmarks[14].x, landmarks[14].y]
                    wrist = [landmarks[16].x, landmarks[16].y]
                    angles['right_elbow'] = calculate_angle(shoulder, elbow, wrist)
                
                # 왼쪽 무릎 각도
                if all(landmarks[i].visibility > confidence_threshold for i in [23, 25, 27]):
                    hip = [landmarks[23].x, landmarks[23].y]
                    knee = [landmarks[25].x, landmarks[25].y]
                    ankle = [landmarks[27].x, landmarks[27].y]
                    angles['left_knee'] = calculate_angle(hip, knee, ankle)
                
                # 오른쪽 무릎 각도
                if all(landmarks[i].visibility > confidence_threshold for i in [24, 26, 28]):
                    hip = [landmarks[24].x, landmarks[24].y]
                    knee = [landmarks[26].x, landmarks[26].y]
                    ankle = [landmarks[28].x, landmarks[28].y]
                    angles['right_knee'] = calculate_angle(hip, knee, ankle)
                
                # 왼쪽 고관절 각도
                if all(landmarks[i].visibility > confidence_threshold for i in [11, 23, 25]):
                    shoulder = [landmarks[11].x, landmarks[11].y]
                    hip = [landmarks[23].x, landmarks[23].y]
                    knee = [landmarks[25].x, landmarks[25].y]
                    angles['left_hip'] = calculate_angle(shoulder, hip, knee)
                
                # 오른쪽 고관절 각도
                if all(landmarks[i].visibility > confidence_threshold for i in [12, 24, 26]):
                    shoulder = [landmarks[12].x, landmarks[12].y]
                    hip = [landmarks[24].x, landmarks[24].y]
                    knee = [landmarks[26].x, landmarks[26].y]
                    angles['right_hip'] = calculate_angle(shoulder, hip, knee)
                
                angles['frame'] = frame_num
                angles['time'] = frame_num / fps
                angle_data.append(angles)
            
            frame_num += 1
            progress = frame_num / total_frames
            progress_bar.progress(progress)
            status_text.text(f"처리 중: {frame_num}/{total_frames} 프레임")
    
    cap.release()
    os.unlink(tfile.name)
    
    # 프레임을 비디오 파일로 저장
    output_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_file.name, fourcc, fps, (width, height))
    
    for frame in processed_frames:
        out.write(frame)
    
    out.release()
    
    progress_bar.empty()
    status_text.empty()
    
    # DataFrame으로 변환
    df_tracking = pd.DataFrame(tracking_data)
    df_angles = pd.DataFrame(angle_data)
    
    # 필터 적용
    if filter_strength > 0 and len(df_angles) > 0:
        for col in df_angles.columns:
            if col not in ['frame', 'time']:
                df_angles[col] = apply_lowpass_filter(df_angles[col].values, filter_strength)
    
    return df_tracking, df_angles, fps, width, height, output_file.name

# 사이드바 설정
with st.sidebar:
    st.header("⚙️ 설정")
    
    confidence = st.slider(
        "감지 신뢰도 임계값",
        min_value=0.0,
        max_value=1.0,
        value=0.5,
        step=0.1,
        help="포즈 감지의 최소 신뢰도"
    )
    
    filter_strength = st.slider(
        "필터 강도",
        min_value=0,
        max_value=20,
        value=5,
        help="데이터 스무딩 정도 (0 = 필터 없음)"
    )
    
    st.markdown("---")
    st.markdown("""
    ### 📋 사용 방법
    1. 비디오 파일 업로드
    2. 처리 시작
    3. 결과 확인 및 다운로드
    
    ### 📊 분석 항목
    - 관절 각도 분석
    - 키포인트 궤적
    - 시간별 변화 그래프
    """)

# 메인 영역
uploaded_file = st.file_uploader(
    "비디오 파일을 업로드하세요",
    type=['mp4', 'mov', 'avi', 'mkv'],
    help="동작 분석을 위한 비디오 파일을 선택하세요"
)

if uploaded_file is not None:
    st.success(f"✅ 파일 업로드 완료: {uploaded_file.name}")
    
    # 원본 비디오 표시
    st.markdown("---")
    st.header("📹 원본 비디오")
    
    # 비디오 파일을 바이트로 읽기 (재사용을 위해 세션에 저장)
    if 'original_video_bytes' not in st.session_state or st.session_state.get('last_uploaded_file') != uploaded_file.name:
        video_bytes = uploaded_file.read()
        st.session_state['original_video_bytes'] = video_bytes
        st.session_state['last_uploaded_file'] = uploaded_file.name
        # 파일 포인터를 처음으로 되돌림
        uploaded_file.seek(0)
    
    # 원본 비디오 표시
    st.video(st.session_state['original_video_bytes'])
    
    st.markdown("---")
    
    col1, col2 = st.columns([1, 4])
    
    with col1:
        if st.button("🚀 분석 시작", type="primary", use_container_width=True):
            with st.spinner("비디오 처리 중..."):
                df_tracking, df_angles, fps, width, height, output_video_path = process_video(
                    uploaded_file,
                    confidence,
                    filter_strength
                )
                
                # 세션 상태에 저장
                st.session_state['df_tracking'] = df_tracking
                st.session_state['df_angles'] = df_angles
                st.session_state['fps'] = fps
                st.session_state['video_info'] = f"{width}x{height} @ {fps}fps"
                st.session_state['output_video_path'] = output_video_path
                
                st.success("✅ 분석 완료!")

# 결과 표시
if 'df_tracking' in st.session_state and 'df_angles' in st.session_state:
    df_tracking = st.session_state['df_tracking']
    df_angles = st.session_state['df_angles']
    
    st.markdown("---")
    st.header("📊 분석 결과")
    
    # 비디오 정보
    info_col1, info_col2, info_col3 = st.columns(3)
    with info_col1:
        st.metric("총 프레임", len(df_tracking))
    with info_col2:
        st.metric("비디오 정보", st.session_state['video_info'])
    with info_col3:
        st.metric("분석 시간", f"{len(df_tracking) / st.session_state['fps']:.2f}초")
    
    # 탭으로 결과 구분
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["🎥 분석 결과 비디오", "⏱️ 시점 분석", "📈 관절 각도", "📍 궤적 분석", "💾 다운로드"])
    
    with tab1:
        st.subheader("포즈 스켈레톤 감지 결과")
        
        if 'output_video_path' in st.session_state and os.path.exists(st.session_state['output_video_path']):
            video_path = st.session_state['output_video_path']
            
            try:
                with open(video_path, 'rb') as video_file:
                    video_bytes = video_file.read()
                
                if len(video_bytes) > 0:
                    # 비디오 표시
                    st.video(video_bytes)
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # 비디오 다운로드 버튼
                        st.download_button(
                            label="📥 분석 비디오 다운로드",
                            data=video_bytes,
                            file_name="pose_analysis_video.mp4",
                            mime="video/mp4",
                            use_container_width=True
                        )
                    
                    with col2:
                        st.info(f"📊 비디오 정보: {st.session_state['video_info']}")
                else:
                    st.error("비디오 파일이 비어있습니다. 다시 분석해주세요.")
            except Exception as e:
                st.error(f"비디오 로드 중 오류: {str(e)}")
                st.info("다시 분석을 시도해주세요.")
        else:
            st.info("비디오 분석을 실행하면 결과가 여기에 표시됩니다.")
            if 'output_video_path' in st.session_state:
                st.warning(f"파일 경로: {st.session_state.get('output_video_path', 'N/A')}")
                st.warning("파일이 존재하지 않습니다. 분석을 다시 실행해주세요.")
    
    with tab2:
        st.subheader("⏱️ 시점별 동작 분석")
        
        if 'original_video_bytes' in st.session_state and 'df_tracking' in st.session_state:
            # 비디오를 임시 파일로 저장
            temp_video = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
            temp_video.write(st.session_state['original_video_bytes'])
            temp_video.close()
            
            # 비디오 정보
            total_time = len(st.session_state['df_tracking']) / st.session_state['fps']
            
            st.info(f"📹 비디오 길이: {total_time:.2f}초")
            
            # 시점 입력 섹션
            st.markdown("### 시점 지정")
            
            col1, col2 = st.columns([3, 1])
            
            with col1:
                # 시점 추가 방법 선택
                method = st.radio(
                    "시점 지정 방법",
                    ["슬라이더로 선택", "직접 입력"],
                    horizontal=True
                )
            
            # 시점 리스트 초기화
            if 'timepoints' not in st.session_state:
                st.session_state['timepoints'] = []
            
            if method == "슬라이더로 선택":
                selected_time = st.slider(
                    "시점 선택 (초)",
                    min_value=0.0,
                    max_value=total_time,
                    value=0.0,
                    step=0.1
                )
            else:
                selected_time = st.number_input(
                    "시점 입력 (초)",
                    min_value=0.0,
                    max_value=total_time,
                    value=0.0,
                    step=0.1
                )
            
            col1, col2, col3 = st.columns([1, 1, 2])
            
            with col1:
                if st.button("➕ 시점 추가", use_container_width=True):
                    if selected_time not in st.session_state['timepoints']:
                        st.session_state['timepoints'].append(selected_time)
                        st.session_state['timepoints'].sort()
                        st.success(f"시점 {selected_time:.2f}초 추가됨")
                    else:
                        st.warning("이미 추가된 시점입니다.")
            
            with col2:
                if st.button("🗑️ 전체 삭제", use_container_width=True):
                    st.session_state['timepoints'] = []
                    st.success("모든 시점이 삭제되었습니다.")
            
            # 현재 시점 목록
            if st.session_state['timepoints']:
                st.markdown("### 📋 지정된 시점")
                
                # 시점 표시 및 개별 삭제
                cols = st.columns(min(len(st.session_state['timepoints']), 5))
                for idx, time_point in enumerate(st.session_state['timepoints']):
                    with cols[idx % 5]:
                        if st.button(f"❌ {time_point:.2f}초", key=f"del_{idx}"):
                            st.session_state['timepoints'].remove(time_point)
                            st.rerun()
                
                st.markdown("---")
                
                # 분석 시작
                if st.button("🔍 시점별 분석 시작", type="primary", use_container_width=True):
                    with st.spinner("시점별 분석 중..."):
                        # MediaPipe Pose 초기화
                        with mp_pose.Pose(
                            static_image_mode=True,
                            model_complexity=1,
                            min_detection_confidence=0.5
                        ) as pose:
                            timepoint_results = []
                            
                            for time_point in st.session_state['timepoints']:
                                frame, angles, detected = analyze_frame_at_time(
                                    temp_video.name,
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
                            
                            st.session_state['timepoint_results'] = timepoint_results
                    
                    st.success("✅ 시점별 분석 완료!")
                
                # 분석 결과 표시
                if 'timepoint_results' in st.session_state and st.session_state['timepoint_results']:
                    st.markdown("---")
                    st.markdown("### 📊 시점별 분석 결과")
                    
                    results = st.session_state['timepoint_results']
                    
                    # 시점별로 표시
                    for idx, result in enumerate(results):
                        st.markdown(f"#### 시점 {idx + 1}: {result['time']:.2f}초")
                        
                        col1, col2 = st.columns([2, 1])
                        
                        with col1:
                            # 포즈가 그려진 이미지
                            st.image(result['frame'], caption=f"{result['time']:.2f}초", use_container_width=True)
                        
                        with col2:
                            # 각도 정보
                            if result['detected'] and result['angles']:
                                st.markdown("**🔢 관절 각도**")
                                for joint, angle in result['angles'].items():
                                    st.metric(joint, f"{angle:.1f}°")
                            else:
                                st.warning("포즈를 감지하지 못했습니다.")
                        
                        st.markdown("---")
                    
                    # 시점간 각도 비교 그래프
                    if len(results) > 1:
                        st.markdown("### 📈 시점간 각도 비교")
                        
                        # 모든 관절 이름 수집
                        all_joints = set()
                        for result in results:
                            if result['angles']:
                                all_joints.update(result['angles'].keys())
                        
                        if all_joints:
                            selected_joints = st.multiselect(
                                "비교할 관절 선택",
                                list(all_joints),
                                default=list(all_joints)[:3] if len(all_joints) >= 3 else list(all_joints)
                            )
                            
                            if selected_joints:
                                fig = go.Figure()
                                
                                for joint in selected_joints:
                                    times = []
                                    angles = []
                                    
                                    for result in results:
                                        if joint in result['angles']:
                                            times.append(result['time'])
                                            angles.append(result['angles'][joint])
                                    
                                    if times:
                                        fig.add_trace(go.Scatter(
                                            x=times,
                                            y=angles,
                                            mode='lines+markers',
                                            name=joint,
                                            marker=dict(size=12),
                                            line=dict(width=3)
                                        ))
                                
                                fig.update_layout(
                                    title="시점별 관절 각도 변화",
                                    xaxis_title="시간 (초)",
                                    yaxis_title="각도 (도)",
                                    height=500,
                                    hovermode='x unified'
                                )
                                
                                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("👆 시점을 추가하여 분석을 시작하세요.")
            
            # 임시 파일 정리
            try:
                os.unlink(temp_video.name)
            except:
                pass
        else:
            st.info("비디오를 업로드하고 먼저 전체 분석을 실행해주세요.")
    
    with tab3:
        st.subheader("관절 각도 변화")
        
        if len(df_angles) > 0:
            # 각도 선택
            available_angles = [col for col in df_angles.columns if col not in ['frame', 'time']]
            selected_angles = st.multiselect(
                "표시할 관절 선택",
                available_angles,
                default=available_angles[:3] if len(available_angles) >= 3 else available_angles
            )
            
            if selected_angles:
                # Plotly 그래프 생성
                fig = go.Figure()
                
                for angle in selected_angles:
                    fig.add_trace(go.Scatter(
                        x=df_angles['time'],
                        y=df_angles[angle],
                        mode='lines',
                        name=angle.replace('_', ' ').title(),
                        line=dict(width=2)
                    ))
                
                fig.update_layout(
                    title="관절 각도 변화",
                    xaxis_title="시간 (초)",
                    yaxis_title="각도 (도)",
                    hovermode='x unified',
                    height=500
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # 통계 정보
                st.subheader("📊 통계 정보")
                stats_df = df_angles[selected_angles].describe()
                st.dataframe(stats_df, use_container_width=True)
        else:
            st.warning("각도 데이터가 없습니다.")
    
    with tab4:
        st.subheader("키포인트 궤적 분석")
        
        # 키포인트 컬럼 추출
        keypoint_cols = [col for col in df_tracking.columns if col not in ['frame', 'time']]
        
        if keypoint_cols:
            # 키포인트 선택
            col1, col2 = st.columns([1, 3])
            
            with col1:
                keypoints = list(set([col.rsplit('_', 1)[0] for col in keypoint_cols if '_x' in col or '_y' in col]))
                selected_keypoint = st.selectbox("키포인트 선택", keypoints)
            
            if selected_keypoint:
                x_col = f"{selected_keypoint}_x"
                y_col = f"{selected_keypoint}_y"
                
                if x_col in df_tracking.columns and y_col in df_tracking.columns:
                    # 궤적 그래프
                    fig = go.Figure()
                    
                    fig.add_trace(go.Scatter(
                        x=df_tracking[x_col],
                        y=df_tracking[y_col],
                        mode='lines+markers',
                        name=selected_keypoint.replace('_', ' ').title(),
                        marker=dict(
                            size=4,
                            color=df_tracking['frame'],
                            colorscale='Viridis',
                            showscale=True,
                            colorbar=dict(title="프레임")
                        ),
                        line=dict(width=2),
                        hovertemplate='<b>프레임: %{marker.color}</b><br>X: %{x:.3f}<br>Y: %{y:.3f}<extra></extra>'
                    ))
                    
                    fig.update_layout(
                        title=f"{selected_keypoint.replace('_', ' ').title()} 궤적",
                        xaxis_title="X 좌표",
                        yaxis_title="Y 좌표",
                        yaxis=dict(scaleanchor="x", scaleratio=1, autorange="reversed"),
                        height=500
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # 시간별 좌표 변화
                    st.subheader("시간별 좌표 변화")
                    
                    fig2 = make_subplots(
                        rows=2, cols=1,
                        subplot_titles=('X 좌표 변화', 'Y 좌표 변화'),
                        vertical_spacing=0.1
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
                    fig2.update_yaxes(title_text="X 좌표", row=1, col=1)
                    fig2.update_yaxes(title_text="Y 좌표", row=2, col=1)
                    fig2.update_layout(height=600, showlegend=False)
                    
                    st.plotly_chart(fig2, use_container_width=True)
                    
                    # 키포인트 데이터 테이블
                    st.subheader("📊 키포인트 데이터")
                    st.dataframe(df_tracking, use_container_width=True, height=300)
        else:
            st.warning("키포인트 데이터가 없습니다.")
    
    with tab5:
        st.subheader("데이터 다운로드")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # 키포인트 데이터 다운로드
            csv_tracking = df_tracking.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 키포인트 데이터 다운로드 (CSV)",
                data=csv_tracking,
                file_name="keypoint_tracking_data.csv",
                mime="text/csv",
                use_container_width=True
            )
        
        with col2:
            # 각도 데이터 다운로드
            if len(df_angles) > 0:
                csv_angles = df_angles.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 각도 데이터 다운로드 (CSV)",
                    data=csv_angles,
                    file_name="angle_data.csv",
                    mime="text/csv",
                    use_container_width=True
                )
            else:
                st.info("각도 데이터가 없습니다.")

else:
    st.info("👆 비디오 파일을 업로드하고 분석을 시작하세요.")

# 푸터
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p>Keypoint Tracker v1.0 | Powered by MediaPipe & Streamlit</p>
</div>
""", unsafe_allow_html=True)
