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

# 페이지 설정
st.set_page_config(
    page_title="Parbiomech Video Analysis",
    page_icon="🎯",
    layout="wide"
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
    """비디오를 처리하고 지정된 시점들을 분석"""
    # 임시 파일로 저장
    temp_path = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    temp_path.write(video_file.read())
    temp_path.close()
    
    # 비디오 정보 가져오기
    cap = cv2.VideoCapture(temp_path.name)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    
    # 시점별 분석 수행
    timepoint_results = []
    
    with mp_pose.Pose(
        static_image_mode=True,
        model_complexity=1,
        min_detection_confidence=confidence_threshold
    ) as pose:
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for idx, time_point in enumerate(timepoints):
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
            
            progress = int(((idx + 1) / len(timepoints)) * 100)
            progress_bar.progress(progress)
        
        progress_bar.empty()
        status_text.empty()
    
    # 임시 파일 삭제
    try:
        os.unlink(temp_path.name)
    except:
        pass
    
    return timepoint_results, fps, width, height

# 메인 애플리케이션
st.title("🎯 Parbiomech Video Analysis")
st.markdown("**MediaPipe 기반 포즈 분석 시스템**")

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
    help="분석할 비디오를 선택하세요"
)

if uploaded_file is not None:
    # 원본 비디오 표시
    st.subheader("📹 원본 영상")
    
    # 비디오를 session state에 저장
    if 'original_video_bytes' not in st.session_state or st.session_state.get('uploaded_file_name') != uploaded_file.name:
        video_bytes = uploaded_file.read()
        st.session_state['original_video_bytes'] = video_bytes
        st.session_state['uploaded_file_name'] = uploaded_file.name
        st.session_state['timepoints'] = []  # 새 비디오 업로드 시 시점 초기화
        uploaded_file.seek(0)  # 파일 포인터를 처음으로 되돌림
    
    # 저장된 비디오 표시
    st.video(st.session_state['original_video_bytes'])
    
    st.markdown("---")
    
    # 시점 태그 섹션
    st.subheader("⏱️ 시점 태그")
    
    # 비디오 길이 계산
    temp_video = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    temp_video.write(st.session_state['original_video_bytes'])
    temp_video.close()
    
    cap = cv2.VideoCapture(temp_video.name)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    total_time = total_frames / fps if fps > 0 else 0
    cap.release()
    
    # 임시 파일 삭제
    try:
        os.unlink(temp_video.name)
    except:
        pass
    
    st.info(f"📹 비디오 길이: {total_time:.2f}초 ({total_frames} 프레임)")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        # 시점 추가 방법 선택
        method = st.radio(
            "시점 지정 방법",
            ["슬라이더로 선택", "직접 입력"],
            horizontal=True
        )
    
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
    else:
        st.info("👆 시점을 추가하여 분석할 구간을 지정하세요.")
    
    st.markdown("---")
    
    # 분석 버튼
    if st.session_state['timepoints']:
        if st.button("🔍 분석 시작", type="primary", use_container_width=True):
            with st.spinner("시점별 분석 중... 잠시만 기다려주세요."):
                timepoint_results, fps, width, height = process_video(
                    uploaded_file,
                    st.session_state['timepoints'],
                    confidence_threshold
                )
                
                # 결과를 세션 상태에 저장
                st.session_state['timepoint_results'] = timepoint_results
                st.session_state['fps'] = fps
                st.session_state['video_info'] = f"{width}x{height} @ {fps:.1f}fps"
            
            st.success("✅ 분석 완료!")
    else:
        st.warning("⚠️ 먼저 분석할 시점을 추가해주세요.")
    
    # 결과 표시
    if 'timepoint_results' in st.session_state and st.session_state['timepoint_results']:
        st.markdown("---")
        st.header("📊 분석 결과")
        
        results = st.session_state['timepoint_results']
        
        # 다운로드 섹션
        st.subheader("💾 데이터 다운로드")
        
        # CSV 데이터 생성
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
            
            col1, col2 = st.columns(2)
            with col1:
                st.download_button(
                    label="📥 분석 데이터 CSV 다운로드",
                    data=csv,
                    file_name="pose_analysis_data.csv",
                    mime="text/csv",
                    use_container_width=True
                )
            with col2:
                st.info(f"📊 비디오 정보: {st.session_state['video_info']}")
        
        st.markdown("---")
        
        # 시점별 분석 결과
        st.subheader("📸 시점별 분석 결과")
        
        for idx, result in enumerate(results):
            st.markdown(f"### 시점 {idx + 1}: {result['time']:.2f}초")
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                # 포즈가 그려진 이미지
                st.image(result['frame'], caption=f"{result['time']:.2f}초", use_container_width=True)
            
            with col2:
                # 각도 정보
                if result['detected'] and result['angles']:
                    st.markdown("**📐 절대각도 (분절 기울기)**")
                    for joint, angle in result['angles']['absolute'].items():
                        st.metric(joint, f"{angle:.1f}°")
                    
                    st.markdown("**🔢 상대각도 (관절각도)**")
                    for joint, angle in result['angles']['relative'].items():
                        st.metric(joint, f"{angle:.1f}°")
                else:
                    st.warning("포즈를 감지하지 못했습니다.")
            
            st.markdown("---")
        
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

else:
    st.info("👆 비디오 파일을 업로드하여 분석을 시작하세요.")
