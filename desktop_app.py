"""
Parbiomech 비디오 분석 데스크톱 애플리케이션
PyQt5 기반 GUI 프로그램
"""

import sys
import os
import tempfile
from pathlib import Path

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QFileDialog, QProgressBar, QTableWidget,
    QTableWidgetItem, QTabWidget, QSlider, QSpinBox, QGroupBox,
    QMessageBox, QTextEdit, QSplitter
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer
from PyQt5.QtGui import QFont, QPixmap, QImage
import cv2
import numpy as np
import mediapipe as mp
import pandas as pd
import pyqtgraph as pg

# MediaPipe 설정
os.environ['MEDIAPIPE_RESOURCE_CACHE_DIR'] = tempfile.gettempdir()
os.environ['MEDIAPIPE_DISABLE_GPU'] = '1'

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles


class VideoAnalysisThread(QThread):
    """비디오 분석을 백그라운드 스레드에서 실행"""
    progress = pyqtSignal(int)
    status = pyqtSignal(str)
    finished = pyqtSignal(dict)
    error = pyqtSignal(str)
    
    def __init__(self, video_path, timepoints, confidence):
        super().__init__()
        self.video_path = video_path
        self.timepoints = timepoints
        self.confidence = confidence
        
    def run(self):
        try:
            result = self.process_video()
            self.finished.emit(result)
        except Exception as e:
            self.error.emit(str(e))
    
    def process_video(self):
        """비디오 분석 메인 함수"""
        self.status.emit("비디오 로드 중...")
        
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            raise Exception("비디오를 열 수 없습니다.")
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # 출력 비디오 준비
        output_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        output_path = output_file.name
        output_file.close()
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        tracking_data = []
        timepoint_results = {}
        
        self.status.emit("포즈 분석 중...")
        
        with mp_pose.Pose(
            min_detection_confidence=self.confidence,
            min_tracking_confidence=self.confidence,
            model_complexity=0
        ) as pose:
            
            frame_count = 0
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                # RGB 변환
                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image.flags.writeable = False
                
                # 포즈 감지
                results = pose.process(image)
                
                # 다시 BGR로
                image.flags.writeable = True
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                
                # 스켈레톤 그리기
                if results.pose_landmarks:
                    mp_drawing.draw_landmarks(
                        image,
                        results.pose_landmarks,
                        mp_pose.POSE_CONNECTIONS,
                        landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()
                    )
                    
                    # 랜드마크 데이터 저장
                    landmarks = results.pose_landmarks.landmark
                    tracking_data.append({
                        'frame': frame_count,
                        'time': frame_count / fps,
                        'nose_x': landmarks[0].x,
                        'nose_y': landmarks[0].y,
                        'left_shoulder_x': landmarks[11].x,
                        'left_shoulder_y': landmarks[11].y,
                        'right_shoulder_x': landmarks[12].x,
                        'right_shoulder_y': landmarks[12].y,
                        'left_elbow_x': landmarks[13].x,
                        'left_elbow_y': landmarks[13].y,
                        'right_elbow_x': landmarks[14].x,
                        'right_elbow_y': landmarks[14].y,
                        'left_wrist_x': landmarks[15].x,
                        'left_wrist_y': landmarks[15].y,
                        'right_wrist_x': landmarks[16].x,
                        'right_wrist_y': landmarks[16].y,
                        'left_hip_x': landmarks[23].x,
                        'left_hip_y': landmarks[23].y,
                        'right_hip_x': landmarks[24].x,
                        'right_hip_y': landmarks[24].y,
                        'left_knee_x': landmarks[25].x,
                        'left_knee_y': landmarks[25].y,
                        'right_knee_x': landmarks[26].x,
                        'right_knee_y': landmarks[26].y,
                        'left_ankle_x': landmarks[27].x,
                        'left_ankle_y': landmarks[27].y,
                        'right_ankle_x': landmarks[28].x,
                        'right_ankle_y': landmarks[28].y,
                    })
                
                # 비디오에 직접 쓰기
                out.write(image)
                
                frame_count += 1
                
                # 진행률 업데이트
                progress_pct = int((frame_count / total_frames) * 100)
                self.progress.emit(progress_pct)
                
                if frame_count % 30 == 0:
                    self.status.emit(f"처리 중: {frame_count}/{total_frames} 프레임")
        
        cap.release()
        out.release()
        
        # 타임포인트 분석
        self.status.emit("타임포인트 분석 중...")
        df_tracking = pd.DataFrame(tracking_data)
        
        for tp in self.timepoints:
            target_frame = int(tp * fps)
            if target_frame < len(tracking_data):
                data = tracking_data[target_frame]
                
                # 각도 계산 (예시: 왼쪽 팔꿈치)
                shoulder = np.array([data['left_shoulder_x'], data['left_shoulder_y']])
                elbow = np.array([data['left_elbow_x'], data['left_elbow_y']])
                wrist = np.array([data['left_wrist_x'], data['left_wrist_y']])
                
                angle = self.calculate_angle(shoulder, elbow, wrist)
                
                timepoint_results[tp] = {
                    'left_elbow_angle': angle,
                    'frame': target_frame
                }
        
        self.status.emit("분석 완료!")
        self.progress.emit(100)
        
        return {
            'output_video': output_path,
            'tracking_data': df_tracking,
            'timepoint_results': timepoint_results,
            'fps': fps
        }
    
    def calculate_angle(self, point1, point2, point3):
        """세 점 사이의 각도 계산"""
        vector1 = point1 - point2
        vector2 = point3 - point2
        
        dot_product = np.dot(vector1, vector2)
        magnitude1 = np.linalg.norm(vector1)
        magnitude2 = np.linalg.norm(vector2)
        
        if magnitude1 == 0 or magnitude2 == 0:
            return 0
        
        cos_angle = dot_product / (magnitude1 * magnitude2)
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        angle = np.arccos(cos_angle)
        
        return np.degrees(angle)


class MainWindow(QMainWindow):
    """메인 애플리케이션 윈도우"""
    
    def __init__(self):
        super().__init__()
        self.video_path = None
        self.analysis_result = None
        self.timepoints = []
        
        self.init_ui()
    
    def init_ui(self):
        """UI 초기화"""
        self.setWindowTitle("Parbiomech 비디오 분석")
        self.setGeometry(100, 100, 1200, 800)
        
        # 중앙 위젯
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # 메인 레이아웃
        layout = QVBoxLayout()
        central_widget.setLayout(layout)
        
        # 타이틀
        title = QLabel("🎥 Parbiomech 비디오 분석 프로그램")
        title.setFont(QFont("Arial", 20, QFont.Bold))
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)
        
        # 비디오 업로드 섹션
        upload_group = QGroupBox("1. 비디오 업로드")
        upload_layout = QHBoxLayout()
        upload_group.setLayout(upload_layout)
        
        self.file_label = QLabel("선택된 파일 없음")
        upload_layout.addWidget(self.file_label)
        
        self.upload_btn = QPushButton("📁 비디오 선택")
        self.upload_btn.clicked.connect(self.select_video)
        upload_layout.addWidget(self.upload_btn)
        
        layout.addWidget(upload_group)
        
        # 설정 섹션
        settings_group = QGroupBox("2. 분석 설정")
        settings_layout = QVBoxLayout()
        settings_group.setLayout(settings_layout)
        
        # 신뢰도 설정
        confidence_layout = QHBoxLayout()
        confidence_layout.addWidget(QLabel("신뢰도 임계값:"))
        
        self.confidence_slider = QSlider(Qt.Horizontal)
        self.confidence_slider.setMinimum(1)
        self.confidence_slider.setMaximum(10)
        self.confidence_slider.setValue(5)
        self.confidence_slider.valueChanged.connect(self.update_confidence_label)
        confidence_layout.addWidget(self.confidence_slider)
        
        self.confidence_label = QLabel("0.5")
        confidence_layout.addWidget(self.confidence_label)
        
        settings_layout.addLayout(confidence_layout)
        
        # 타임포인트 설정
        timepoint_layout = QHBoxLayout()
        timepoint_layout.addWidget(QLabel("타임포인트 (초):"))
        
        self.timepoint_input = QTextEdit()
        self.timepoint_input.setMaximumHeight(60)
        self.timepoint_input.setPlaceholderText("예: 0.5, 1.0, 2.5")
        timepoint_layout.addWidget(self.timepoint_input)
        
        settings_layout.addLayout(timepoint_layout)
        
        layout.addWidget(settings_group)
        
        # 분석 버튼
        self.analyze_btn = QPushButton("🔍 분석 시작")
        self.analyze_btn.setFont(QFont("Arial", 14, QFont.Bold))
        self.analyze_btn.setMinimumHeight(50)
        self.analyze_btn.clicked.connect(self.start_analysis)
        self.analyze_btn.setEnabled(False)
        layout.addWidget(self.analyze_btn)
        
        # 진행 상태
        self.status_label = QLabel("")
        layout.addWidget(self.status_label)
        
        self.progress_bar = QProgressBar()
        layout.addWidget(self.progress_bar)
        
        # 결과 탭
        self.result_tabs = QTabWidget()
        layout.addWidget(self.result_tabs)
        
        # 스타일 적용
        self.setStyleSheet("""
            QGroupBox {
                font-weight: bold;
                border: 2px solid #cccccc;
                border-radius: 5px;
                margin-top: 10px;
                padding-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px 0 5px;
            }
            QPushButton {
                background-color: #4CAF50;
                color: white;
                border: none;
                padding: 10px;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
            QPushButton:disabled {
                background-color: #cccccc;
            }
        """)
    
    def update_confidence_label(self, value):
        """신뢰도 라벨 업데이트"""
        self.confidence_label.setText(f"{value / 10:.1f}")
    
    def select_video(self):
        """비디오 파일 선택"""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "비디오 파일 선택",
            "",
            "Video Files (*.mp4 *.avi *.mov *.mkv);;All Files (*)"
        )
        
        if file_path:
            self.video_path = file_path
            self.file_label.setText(f"선택됨: {Path(file_path).name}")
            self.analyze_btn.setEnabled(True)
    
    def start_analysis(self):
        """비디오 분석 시작"""
        # 타임포인트 파싱
        timepoint_text = self.timepoint_input.toPlainText().strip()
        if timepoint_text:
            try:
                self.timepoints = [float(x.strip()) for x in timepoint_text.split(',')]
            except:
                QMessageBox.warning(self, "입력 오류", "타임포인트 형식이 올바르지 않습니다.")
                return
        else:
            self.timepoints = []
        
        confidence = self.confidence_slider.value() / 10.0
        
        # UI 비활성화
        self.analyze_btn.setEnabled(False)
        self.upload_btn.setEnabled(False)
        
        # 분석 스레드 시작
        self.analysis_thread = VideoAnalysisThread(
            self.video_path,
            self.timepoints,
            confidence
        )
        self.analysis_thread.progress.connect(self.update_progress)
        self.analysis_thread.status.connect(self.update_status)
        self.analysis_thread.finished.connect(self.analysis_complete)
        self.analysis_thread.error.connect(self.analysis_error)
        self.analysis_thread.start()
    
    def update_progress(self, value):
        """진행률 업데이트"""
        self.progress_bar.setValue(value)
    
    def update_status(self, message):
        """상태 메시지 업데이트"""
        self.status_label.setText(message)
    
    def analysis_complete(self, result):
        """분석 완료 처리"""
        self.analysis_result = result
        
        # UI 재활성화
        self.analyze_btn.setEnabled(True)
        self.upload_btn.setEnabled(True)
        
        # 결과 표시
        self.display_results(result)
        
        QMessageBox.information(self, "완료", "비디오 분석이 완료되었습니다!")
    
    def analysis_error(self, error_msg):
        """분석 오류 처리"""
        self.analyze_btn.setEnabled(True)
        self.upload_btn.setEnabled(True)
        
        QMessageBox.critical(self, "오류", f"분석 중 오류 발생:\n{error_msg}")
    
    def display_results(self, result):
        """분석 결과 표시"""
        # 기존 탭 제거
        self.result_tabs.clear()
        
        # 타임포인트 결과 탭
        if result['timepoint_results']:
            timepoint_widget = QTableWidget()
            timepoint_widget.setColumnCount(3)
            timepoint_widget.setHorizontalHeaderLabels(['시간 (초)', '프레임', '왼쪽 팔꿈치 각도'])
            timepoint_widget.setRowCount(len(result['timepoint_results']))
            
            for i, (time, data) in enumerate(result['timepoint_results'].items()):
                timepoint_widget.setItem(i, 0, QTableWidgetItem(f"{time:.2f}"))
                timepoint_widget.setItem(i, 1, QTableWidgetItem(str(data['frame'])))
                timepoint_widget.setItem(i, 2, QTableWidgetItem(f"{data['left_elbow_angle']:.1f}°"))
            
            self.result_tabs.addTab(timepoint_widget, "타임포인트 분석")
        
        # 추적 데이터 차트
        if not result['tracking_data'].empty:
            chart_widget = pg.PlotWidget()
            chart_widget.setBackground('w')
            chart_widget.setLabel('left', '각도 (도)')
            chart_widget.setLabel('bottom', '시간 (초)')
            chart_widget.setTitle('궤적 분석')
            
            # 예시: nose Y 좌표 플롯
            times = result['tracking_data']['time'].values
            nose_y = result['tracking_data']['nose_y'].values
            chart_widget.plot(times, nose_y, pen='b', name='Nose Y')
            
            self.result_tabs.addTab(chart_widget, "궤적 차트")
        
        # 비디오 경로 표시
        video_info = QLabel(f"분석된 비디오: {result['output_video']}\n\n"
                           f"이 파일을 미디어 플레이어로 열어보세요.")
        video_info.setWordWrap(True)
        self.result_tabs.addTab(video_info, "비디오 정보")


def main():
    """메인 실행 함수"""
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
