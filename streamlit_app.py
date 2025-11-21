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

def detect_events(df_angles, threshold_change=30):
    """각도 변화를 기반으로 주요 시점 검출"""
    events = []
    
    if len(df_angles) == 0:
        return pd.DataFrame(events)
    
    angle_cols = [col for col in df_angles.columns if col not in ['frame', 'time']]
    
    for col in angle_cols:
        if col in df_angles.columns:
            values = df_angles[col].dropna().values
            
            if len(values) < 2:
                continue
            
            # 각도 변화율 계산
            changes = np.diff(values)
            
            # 최대/최소 각도 지점 찾기
            max_idx = np.argmax(values)
            min_idx = np.argmin(values)
            
            # 급격한 변화 지점 찾기
            significant_changes = np.where(np.abs(changes) > threshold_change)[0]
            
            # 최대 각도 이벤트
            if max_idx < len(df_angles):
                events.append({
                    'time': df_angles.iloc[max_idx]['time'],
                    'frame': df_angles.iloc[max_idx]['frame'],
                    'joint': col,
                    'event_type': '최대 굴곡',
                    'angle': values[max_idx]
                })
            
            # 최소 각도 이벤트
            if min_idx < len(df_angles):
                events.append({
                    'time': df_angles.iloc[min_idx]['time'],
                    'frame': df_angles.iloc[min_idx]['frame'],
                    'joint': col,
                    'event_type': '최대 신전',
                    'angle': values[min_idx]
                })
            
            # 급격한 변화 이벤트
            for change_idx in significant_changes[:3]:  # 최대 3개만
                if change_idx < len(df_angles):
                    events.append({
                        'time': df_angles.iloc[change_idx]['time'],
                        'frame': df_angles.iloc[change_idx]['frame'],
                        'joint': col,
                        'event_type': '급격한 변화',
                        'angle': values[change_idx]
                    })
    
    events_df = pd.DataFrame(events)
    if len(events_df) > 0:
        events_df = events_df.sort_values('time').reset_index(drop=True)
    
    return events_df

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
    
    # 출력 비디오 설정
    output_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_file.name, fourcc, fps, (width, height))
    
    # 결과 저장을 위한 리스트
    tracking_data = []
    angle_data = []
    
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
            
            # 프레임을 출력 비디오에 저장
            out.write(annotated_frame)
            
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
    out.release()
    os.unlink(tfile.name)
    
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
    
    event_threshold = st.slider(
        "이벤트 검출 민감도",
        min_value=10,
        max_value=50,
        value=30,
        help="각도 변화 임계값 (낮을수록 더 많은 이벤트 검출)"
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
                
                # 이벤트 검출
                events_df = detect_events(df_angles)
                st.session_state['events_df'] = events_df
                
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
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["🎥 포즈 비디오", "📈 관절 각도", "⚡ 주요 시점", "📍 키포인트 데이터", "💾 다운로드"])
    
    with tab1:
        st.subheader("포즈 감지 결과 비디오")
        
        if 'output_video_path' in st.session_state:
            # 비디오 파일 읽기
            video_path = st.session_state['output_video_path']
            
            if os.path.exists(video_path):
                with open(video_path, 'rb') as video_file:
                    video_bytes = video_file.read()
                
                # 비디오 표시
                st.video(video_bytes)
                
                # 다운로드 버튼
                st.download_button(
                    label="📥 포즈 비디오 다운로드",
                    data=video_bytes,
                    file_name="pose_analysis_video.mp4",
                    mime="video/mp4",
                    use_container_width=True
                )
            else:
                st.warning("비디오 파일을 찾을 수 없습니다.")
        else:
            st.info("비디오 분석 결과가 없습니다.")
    
    with tab2:
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
                
                # 이벤트 마커 추가
                if 'events_df' in st.session_state and len(st.session_state['events_df']) > 0:
                    events_df = st.session_state['events_df']
                    for angle in selected_angles:
                        angle_events = events_df[events_df['joint'] == angle]
                        if len(angle_events) > 0:
                            fig.add_trace(go.Scatter(
                                x=angle_events['time'],
                                y=angle_events['angle'],
                                mode='markers',
                                name=f"{angle.replace('_', ' ').title()} 이벤트",
                                marker=dict(size=10, symbol='star'),
                                text=angle_events['event_type'],
                                hovertemplate='<b>%{text}</b><br>시간: %{x:.2f}초<br>각도: %{y:.1f}°<extra></extra>'
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
    
    with tab3:
        st.subheader("⚡ 주요 시점 검출")
        
        if 'events_df' in st.session_state and len(st.session_state['events_df']) > 0:
            events_df = st.session_state['events_df']
            
            # 이벤트 타입별 필터
            event_types = ['전체'] + list(events_df['event_type'].unique())
            selected_event_type = st.selectbox("이벤트 유형 선택", event_types)
            
            # 관절별 필터
            joints = ['전체'] + list(events_df['joint'].unique())
            selected_joint = st.selectbox("관절 선택", joints)
            
            # 필터링
            filtered_events = events_df.copy()
            if selected_event_type != '전체':
                filtered_events = filtered_events[filtered_events['event_type'] == selected_event_type]
            if selected_joint != '전체':
                filtered_events = filtered_events[filtered_events['joint'] == selected_joint]
            
            # 이벤트 표시
            st.dataframe(
                filtered_events.style.format({
                    'time': '{:.2f}초',
                    'frame': '{:.0f}',
                    'angle': '{:.1f}°'
                }),
                use_container_width=True,
                height=400
            )
            
            # 통계
            st.subheader("📊 이벤트 통계")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("총 이벤트 수", len(events_df))
            with col2:
                st.metric("검출된 관절 수", events_df['joint'].nunique())
            with col3:
                st.metric("이벤트 유형 수", events_df['event_type'].nunique())
            
            # 타임라인 시각화
            st.subheader("🕐 이벤트 타임라인")
            
            fig = go.Figure()
            
            for joint in events_df['joint'].unique():
                joint_events = events_df[events_df['joint'] == joint]
                fig.add_trace(go.Scatter(
                    x=joint_events['time'],
                    y=[joint] * len(joint_events),
                    mode='markers+text',
                    name=joint.replace('_', ' ').title(),
                    marker=dict(size=15, symbol='diamond'),
                    text=joint_events['event_type'],
                    textposition='top center',
                    hovertemplate='<b>%{text}</b><br>시간: %{x:.2f}초<br>각도: ' + 
                                  joint_events['angle'].astype(str) + '°<extra></extra>'
                ))
            
            fig.update_layout(
                title="관절별 이벤트 타임라인",
                xaxis_title="시간 (초)",
                yaxis_title="관절",
                height=400,
                showlegend=True
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
        else:
            st.info("검출된 주요 시점이 없습니다.")
    
    with tab4:
        st.subheader("키포인트 추적 데이터")
        
        # 키포인트 선택
        keypoint_cols = [col for col in df_tracking.columns if col not in ['frame', 'time']]
        
        if keypoint_cols:
            st.dataframe(df_tracking, use_container_width=True, height=400)
            
            # 특정 키포인트의 궤적 시각화
            st.subheader("키포인트 궤적")
            
            keypoints = list(set([col.rsplit('_', 1)[0] for col in keypoint_cols if '_x' in col or '_y' in col]))
            selected_keypoint = st.selectbox("키포인트 선택", keypoints)
            
            if selected_keypoint:
                x_col = f"{selected_keypoint}_x"
                y_col = f"{selected_keypoint}_y"
                
                if x_col in df_tracking.columns and y_col in df_tracking.columns:
                    fig = go.Figure()
                    
                    fig.add_trace(go.Scatter(
                        x=df_tracking[x_col],
                        y=df_tracking[y_col],
                        mode='lines+markers',
                        name=selected_keypoint,
                        marker=dict(size=4),
                        line=dict(width=2)
                    ))
                    
                    fig.update_layout(
                        title=f"{selected_keypoint} 궤적",
                        xaxis_title="X 좌표",
                        yaxis_title="Y 좌표",
                        yaxis=dict(scaleanchor="x", scaleratio=1, autorange="reversed"),
                        height=500
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("키포인트 데이터가 없습니다.")
    
    with tab5:
        st.subheader("데이터 다운로드")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # 키포인트 데이터 다운로드
            csv_tracking = df_tracking.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 키포인트 데이터 (CSV)",
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
                    label="📥 각도 데이터 (CSV)",
                    data=csv_angles,
                    file_name="angle_data.csv",
                    mime="text/csv",
                    use_container_width=True
                )
            else:
                st.info("각도 데이터가 없습니다.")
        
        with col3:
            # 이벤트 데이터 다운로드
            if 'events_df' in st.session_state and len(st.session_state['events_df']) > 0:
                csv_events = st.session_state['events_df'].to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 이벤트 데이터 (CSV)",
                    data=csv_events,
                    file_name="event_data.csv",
                    mime="text/csv",
                    use_container_width=True
                )
            else:
                st.info("이벤트 데이터가 없습니다.")

else:
    st.info("👆 비디오 파일을 업로드하고 분석을 시작하세요.")

# 푸터
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p>Keypoint Tracker v1.0 | Powered by MediaPipe & Streamlit</p>
</div>
""", unsafe_allow_html=True)
