// 전역 변수
let videoElement, canvasElement, canvasCtx;
let poseSkeletonCanvas, poseSkeletonCtx;
let sagittalAngleCanvas, sagittalAngleCtx;
let frontalAngleCanvas, frontalAngleCtx;
let pose, camera;
let isPlaying = false;
let trackingData = [];
let filteredData = [];
let angleData = [];
let markers = [];
let chart, angleChart, trajectoryChart;
let selectedKeypoints = [0, 15, 16]; // 기본: 코, 왼손목, 오른손목
let filterStrength = 5;
let currentAngleView = 'local'; // 'local' or 'relative'
let currentPoseLandmarks = null; // 현재 포즈 랜드마크
let selectedAngles = ['kneeFlex', 'hipFlex', 'elbowFlex']; // 기본 선택된 각도
let angleMode = 'interior'; // 'interior' or 'exterior'
let showAngleSign = false; // +/- 부호 표시 여부

// MediaPipe Pose Landmarks
const POSE_LANDMARKS = {
    0: "Nose", 1: "Left Eye Inner", 2: "Left Eye", 3: "Left Eye Outer",
    4: "Right Eye Inner", 5: "Right Eye", 6: "Right Eye Outer",
    7: "Left Ear", 8: "Right Ear", 9: "Mouth Left", 10: "Mouth Right",
    11: "Left Shoulder", 12: "Right Shoulder", 13: "Left Elbow", 14: "Right Elbow",
    15: "Left Wrist", 16: "Right Wrist", 17: "Left Pinky", 18: "Right Pinky",
    19: "Left Index", 20: "Right Index", 21: "Left Thumb", 22: "Right Thumb",
    23: "Left Hip", 24: "Right Hip", 25: "Left Knee", 26: "Right Knee",
    27: "Left Ankle", 28: "Right Ankle", 29: "Left Heel", 30: "Right Heel",
    31: "Left Foot Index", 32: "Right Foot Index"
};

// 초기화
document.addEventListener('DOMContentLoaded', function() {
    videoElement = document.getElementById('videoElement');
    canvasElement = document.getElementById('canvasElement');
    canvasCtx = canvasElement.getContext('2d');
    
    poseSkeletonCanvas = document.getElementById('poseSkeletonCanvas');
    poseSkeletonCtx = poseSkeletonCanvas.getContext('2d');
    
    sagittalAngleCanvas = document.getElementById('sagittalAngleCanvas');
    sagittalAngleCtx = sagittalAngleCanvas.getContext('2d');
    
    frontalAngleCanvas = document.getElementById('frontalAngleCanvas');
    frontalAngleCtx = frontalAngleCanvas.getContext('2d');
    
    // 각도 설명 캔버스 클릭 이벤트
    sagittalAngleCanvas.addEventListener('click', handleSagittalCanvasClick);
    frontalAngleCanvas.addEventListener('click', handleFrontalCanvasClick);
    
    // 이벤트 리스너
    document.getElementById('videoInput').addEventListener('change', handleVideoUpload);
    document.getElementById('playPauseBtn').addEventListener('click', togglePlayPause);
    document.getElementById('stopBtn').addEventListener('click', stopVideo);
    document.getElementById('addMarkerBtn').addEventListener('click', addManualMarker);
    document.getElementById('autoMarkerBtn').addEventListener('click', addAutoMarkers);
    document.getElementById('exportCSVBtn').addEventListener('click', exportCSV);
    document.getElementById('exportJSONBtn').addEventListener('click', exportJSON);
    document.getElementById('keypointSelect').addEventListener('change', updateSelectedKeypoints);
    document.getElementById('filterStrength').addEventListener('input', function(e) {
        filterStrength = parseInt(e.target.value);
        document.getElementById('filterStrengthValue').textContent = filterStrength;
        applyFilterAndUpdate();
    });
    document.getElementById('videoProgress').addEventListener('input', handleProgressChange);
    
    // 비디오 시간 업데이트 리스너
    videoElement.addEventListener('timeupdate', updateVideoProgress);
    
    // 각돀 선택 체크박스 리스너
    document.querySelectorAll('.angle-checkbox').forEach(checkbox => {
        checkbox.addEventListener('change', updateSelectedAngles);
    });
    
    // 각도 모드 리스너
    const angleModeRadios = document.querySelectorAll('input[name="angleMode"]');
    if (angleModeRadios.length > 0) {
        angleModeRadios.forEach(radio => {
            radio.addEventListener('change', function() {
                angleMode = this.value;
                applyFilterAndUpdate();
            });
        });
    }
    
    const angleSignToggle = document.getElementById('angleSignToggle');
    if (angleSignToggle) {
        angleSignToggle.addEventListener('change', function() {
            showAngleSign = this.checked;
            applyFilterAndUpdate();
        });
    }
    
    // MediaPipe Pose 초기화
    initMediaPipe();
    
    // 차트 초기화
    initChart();
    initAngleChart();
    initTrajectoryChart();
});

// MediaPipe 초기화
function initMediaPipe() {
    if (typeof Pose === 'undefined') {
        console.error('MediaPipe Pose 라이브러리가 로드되지 않았습니다.');
        alert('MediaPipe 라이브러리 로딩 실패. 페이지를 새로고침 해주세요.');
        return;
    }
    
    try {
        pose = new Pose({
            locateFile: (file) => {
                return `https://cdn.jsdelivr.net/npm/@mediapipe/pose/${file}`;
            }
        });
        
        pose.setOptions({
            modelComplexity: 1,
            smoothLandmarks: true,
            enableSegmentation: false,
            smoothSegmentation: false,
            minDetectionConfidence: 0.5,
            minTrackingConfidence: 0.5
        });
        
        pose.onResults(onPoseResults);
        console.log('MediaPipe Pose 초기화 완료');
    } catch (error) {
        console.error('MediaPipe 초기화 오류:', error);
        alert('MediaPipe 초기화 실패: ' + error.message);
    }
}

// 비디오 업로드 처리
function handleVideoUpload(event) {
    const file = event.target.files[0];
    if (file) {
        console.log('비디오 파일 선택됨:', file.name, file.type, file.size);
        const url = URL.createObjectURL(file);
        videoElement.src = url;
        videoElement.load();
        
        videoElement.onloadedmetadata = function() {
            console.log('비디오 메타데이터 로드 완료:', {
                width: videoElement.videoWidth,
                height: videoElement.videoHeight,
                duration: videoElement.duration
            });
            canvasElement.width = videoElement.videoWidth;
            canvasElement.height = videoElement.videoHeight;
            
            // 버튼 활성화
            enableControls();
            
            // 데이터 초기화
            trackingData = [];
            markers = [];
            updateMarkerList();
            updateStats();
        };
        
        videoElement.onerror = function(e) {
            console.error('비디오 로드 에러:', e);
            alert('비디오를 로드하는 중 오류가 발생했습니다. 다른 비디오 파일을 시도해보세요.');
        };
    } else {
        console.log('파일이 선택되지 않음');
    }
}

// 선택된 키포인트 업데이트
function updateSelectedKeypoints() {
    const select = document.getElementById('keypointSelect');
    selectedKeypoints = Array.from(select.selectedOptions).map(opt => parseInt(opt.value));
    updateChart();
}

// 선택된 각도 업데이트
function updateSelectedAngles() {
    selectedAngles = Array.from(document.querySelectorAll('.angle-checkbox:checked'))
        .map(cb => cb.value);
    updateAngleChart();
    drawPoseSkeleton();
    drawAngleExplanation();
}

// Pose 결과 처리
function onPoseResults(results) {
    if (!results.poseLandmarks) return;
    
    // 현재 포즈 저장
    currentPoseLandmarks = results.poseLandmarks;
    
    // 캔버스에 그리기
    canvasCtx.save();
    canvasCtx.clearRect(0, 0, canvasElement.width, canvasElement.height);
    canvasCtx.drawImage(results.image, 0, 0, canvasElement.width, canvasElement.height);
    
    // 연결선 그리기
    drawConnectors(canvasCtx, results.poseLandmarks, POSE_CONNECTIONS, {color: '#00FF00', lineWidth: 2});
    
    // 랜드마크 그리기
    drawLandmarks(canvasCtx, results.poseLandmarks, {color: '#FF0000', lineWidth: 1, radius: 5});
    
    // 선택된 키포인트 강조
    selectedKeypoints.forEach(idx => {
        if (results.poseLandmarks[idx]) {
            const landmark = results.poseLandmarks[idx];
            canvasCtx.beginPath();
            canvasCtx.arc(
                landmark.x * canvasElement.width,
                landmark.y * canvasElement.height,
                8, 0, 2 * Math.PI
            );
            canvasCtx.fillStyle = '#00FFFF';
            canvasCtx.fill();
        }
    });
    
    canvasCtx.restore();
    
    // 데이터 저장
    if (isPlaying) {
        const frameData = {
            timestamp: videoElement.currentTime,
            frame: trackingData.length,
            landmarks: results.poseLandmarks.map((lm, idx) => ({
                id: idx,
                name: POSE_LANDMARKS[idx],
                x: lm.x,
                y: lm.y,
                z: lm.z,
                visibility: lm.visibility
            }))
        };
        trackingData.push(frameData);
        
        // 필터링 및 각도 계산
        applyFilterAndUpdate();
        updateStats();
    }
    
    // 포즈 스켈레톤 그리기
    drawPoseSkeleton();
    drawAngleExplanation();
    updateAngleTextDisplay();
}

// MediaPipe 연결선
const POSE_CONNECTIONS = [
    [0, 1], [1, 2], [2, 3], [3, 7], [0, 4], [4, 5], [5, 6], [6, 8],
    [9, 10], [11, 12], [11, 13], [13, 15], [15, 17], [15, 19], [15, 21],
    [17, 19], [12, 14], [14, 16], [16, 18], [16, 20], [16, 22], [18, 20],
    [11, 23], [12, 24], [23, 24], [23, 25], [25, 27], [27, 29], [27, 31],
    [29, 31], [24, 26], [26, 28], [28, 30], [28, 32], [30, 32]
];

// 연결선 그리기 함수
function drawConnectors(ctx, landmarks, connections, style) {
    connections.forEach(([start, end]) => {
        const from = landmarks[start];
        const to = landmarks[end];
        if (from && to) {
            ctx.beginPath();
            ctx.moveTo(from.x * canvasElement.width, from.y * canvasElement.height);
            ctx.lineTo(to.x * canvasElement.width, to.y * canvasElement.height);
            ctx.strokeStyle = style.color;
            ctx.lineWidth = style.lineWidth;
            ctx.stroke();
        }
    });
}

// 랜드마크 그리기 함수
function drawLandmarks(ctx, landmarks, style) {
    landmarks.forEach(landmark => {
        ctx.beginPath();
        ctx.arc(
            landmark.x * canvasElement.width,
            landmark.y * canvasElement.height,
            style.radius, 0, 2 * Math.PI
        );
        ctx.fillStyle = style.color;
        ctx.fill();
    });
}

// 비디오 처리 루프
async function processFrame() {
    if (isPlaying && !videoElement.paused && !videoElement.ended) {
        if (pose && typeof pose.send === 'function') {
            try {
                await pose.send({image: videoElement});
            } catch (error) {
                console.error('프레임 처리 오류:', error);
            }
        } else {
            console.error('Pose 객체가 초기화되지 않았습니다.');
            return;
        }
        requestAnimationFrame(processFrame);
    }
}

// 비디오 컨트롤
function togglePlayPause() {
    if (videoElement.paused) {
        playVideo();
    } else {
        pauseVideo();
    }
}

function playVideo() {
    videoElement.play();
    isPlaying = true;
    document.getElementById('playPauseBtn').innerHTML = '⏸️ 일시정지';
    processFrame();
}

function pauseVideo() {
    videoElement.pause();
    isPlaying = false;
    document.getElementById('playPauseBtn').innerHTML = '▶️ 재생';
}

function stopVideo() {
    videoElement.pause();
    videoElement.currentTime = 0;
    isPlaying = false;
    trackingData = [];
    filteredData = [];
    angleData = [];
    updateChart();
    updateAngleChart();
    updateStats();
    document.getElementById('playPauseBtn').innerHTML = '▶️ 재생';
}

// 비디오 프로그레스 업데이트
function updateVideoProgress() {
    if (!videoElement.duration) return;
    
    const progress = (videoElement.currentTime / videoElement.duration) * 100;
    document.getElementById('videoProgress').value = progress;
    
    // 시간 표시 업데이트
    const currentTime = formatTime(videoElement.currentTime);
    const duration = formatTime(videoElement.duration);
    document.getElementById('videoTimeDisplay').textContent = `${currentTime} / ${duration}`;
}

// 프로그레스 바 변경 처리
function handleProgressChange(e) {
    if (!videoElement.duration) return;
    
    const time = (e.target.value / 100) * videoElement.duration;
    videoElement.currentTime = time;
}

// 시간 포맷 함수 (mm:ss)
function formatTime(seconds) {
    if (isNaN(seconds)) return '0:00';
    
    const mins = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    return `${mins}:${secs.toString().padStart(2, '0')}`;
}

// 차트 초기화
function initChart() {
    const ctx = document.getElementById('chartCanvas').getContext('2d');
    chart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: [],
            datasets: []
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                title: {
                    display: true,
                    text: '키포인트 Y좌표 추적 (필터링 적용)'
                },
                legend: {
                    display: true,
                    position: 'top'
                },
                annotation: {
                    annotations: {}
                }
            },
            scales: {
                x: {
                    title: {
                        display: true,
                        text: '시간 (초)'
                    }
                },
                y: {
                    title: {
                        display: true,
                        text: 'Y 좌표 (정규화)'
                    },
                    reverse: true // Y축 반전 (위쪽이 0)
                }
            }
        }
    });
}

// 각돀 차트 초기화
function initAngleChart() {
    const ctx = document.getElementById('angleChartCanvas').getContext('2d');
    angleChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: [],
            datasets: []
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                title: {
                    display: true,
                    text: '관절 각도 변화 (로컬)'
                },
                legend: {
                    display: true,
                    position: 'top'
                }
            },
            scales: {
                x: {
                    title: {
                        display: true,
                        text: '시간 (초)'
                    }
                },
                y: {
                    title: {
                        display: true,
                        text: '각도 (도)'
                    },
                    min: 0,
                    max: 180
                }
            }
        }
    });
}

// XY 궤적 차트 초기화
function initTrajectoryChart() {
    const ctx = document.getElementById('trajectoryChartCanvas').getContext('2d');
    trajectoryChart = new Chart(ctx, {
        type: 'scatter',
        data: {
            datasets: []
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                title: {
                    display: true,
                    text: '관절점 이동 궤적 (X-Y 좌표)'
                },
                legend: {
                    display: true,
                    position: 'top'
                }
            },
            scales: {
                x: {
                    title: {
                        display: true,
                        text: 'X 좌표 (정규화)'
                    },
                    min: 0,
                    max: 1
                },
                y: {
                    title: {
                        display: true,
                        text: 'Y 좌표 (정규화)'
                    },
                    min: 0,
                    max: 1,
                    reverse: true
                }
            }
        }
    });
}

// 차트 업데이트
function updateChart() {
    const dataToUse = filteredData.length > 0 ? filteredData : trackingData;
    
    if (dataToUse.length === 0) {
        chart.data.labels = [];
        chart.data.datasets = [];
        chart.update();
        return;
    }
    
    // 시간 라벨
    const labels = dataToUse.map(d => d.timestamp.toFixed(2));
    
    // 선택된 키포인트별 데이터셋
    const colors = ['#FF6384', '#36A2EB', '#FFCE56', '#4BC0C0', '#9966FF', '#FF9F40'];
    const datasets = selectedKeypoints.map((keypointIdx, idx) => {
        const data = dataToUse.map(frame => {
            const landmark = frame.landmarks[keypointIdx];
            return landmark ? landmark.y : null;
        });
        
        return {
            label: POSE_LANDMARKS[keypointIdx],
            data: data,
            borderColor: colors[idx % colors.length],
            backgroundColor: colors[idx % colors.length] + '33',
            borderWidth: 2,
            pointRadius: 1,
            tension: 0.4
        };
    });
    
    chart.data.labels = labels;
    chart.data.datasets = datasets;
    chart.update();
    
    // XY 궤적 차트 업데이트
    updateTrajectoryChart();
}

// XY 궤적 차트 업데이트
function updateTrajectoryChart() {
    const dataToUse = filteredData.length > 0 ? filteredData : trackingData;
    
    if (dataToUse.length === 0) {
        trajectoryChart.data.datasets = [];
        trajectoryChart.update();
        return;
    }
    
    const colors = ['#FF6384', '#36A2EB', '#FFCE56', '#4BC0C0', '#9966FF', '#FF9F40'];
    const datasets = selectedKeypoints.map((keypointIdx, idx) => {
        const points = dataToUse.map(frame => {
            const landmark = frame.landmarks[keypointIdx];
            return landmark ? { x: landmark.x, y: landmark.y } : null;
        }).filter(p => p !== null);
        
        return {
            label: POSE_LANDMARKS[keypointIdx],
            data: points,
            backgroundColor: colors[idx % colors.length],
            borderColor: colors[idx % colors.length],
            pointRadius: 2,
            showLine: true,
            borderWidth: 1,
            tension: 0.4
        };
    });
    
    trajectoryChart.data.datasets = datasets;
    trajectoryChart.update();
}

// 이동 평균 필터
function movingAverageFilter(data, windowSize) {
    if (data.length === 0) return [];
    
    const filtered = JSON.parse(JSON.stringify(data)); // Deep copy
    
    filtered.forEach((frame, frameIdx) => {
        frame.landmarks.forEach((landmark, lmIdx) => {
            let sumX = 0, sumY = 0, sumZ = 0;
            let count = 0;
            
            const halfWindow = Math.floor(windowSize / 2);
            const startIdx = Math.max(0, frameIdx - halfWindow);
            const endIdx = Math.min(data.length - 1, frameIdx + halfWindow);
            
            for (let i = startIdx; i <= endIdx; i++) {
                const lm = data[i].landmarks[lmIdx];
                if (lm) {
                    sumX += lm.x;
                    sumY += lm.y;
                    sumZ += lm.z;
                    count++;
                }
            }
            
            if (count > 0) {
                landmark.x = sumX / count;
                landmark.y = sumY / count;
                landmark.z = sumZ / count;
            }
        });
    });
    
    return filtered;
}

// 필터링 및 업데이트
function applyFilterAndUpdate() {
    if (trackingData.length === 0) return;
    
    filteredData = movingAverageFilter(trackingData, filterStrength);
    calculateAngles();
    updateChart();
    updateAngleChart();
}

// 각도 계산 (생체역학 정의에 따름)
function calculateAngles() {
    angleData = filteredData.map(frame => {
        const lm = frame.landmarks;
        
        return {
            timestamp: frame.timestamp,
            frame: frame.frame,
            
            // === Joint Angles (관절 각도) ===
            // Ankle Dorsiflexion (발목 배굴): heel → ankle → knee
            leftAnkleDorsi: calculateJointAngleBiomech(lm[29], lm[27], lm[25], -90), // heel-ankle-knee
            rightAnkleDorsi: calculateJointAngleBiomech(lm[30], lm[28], lm[26], -90),
            
            // Knee Flexion (무릎 굴곡): hip → knee → ankle
            leftKneeFlex: calculateJointAngleBiomech(lm[23], lm[25], lm[27], 0),
            rightKneeFlex: calculateJointAngleBiomech(lm[24], lm[26], lm[28], 0),
            
            // Hip Flexion (엉덩이 굴곡): knee → hip → shoulder
            leftHipFlex: calculateJointAngleBiomech(lm[25], lm[23], lm[11], 0),
            rightHipFlex: calculateJointAngleBiomech(lm[26], lm[24], lm[12], 0),
            
            // Shoulder Flexion (어깨 굴곡): hip → shoulder → elbow
            leftShoulderFlex: calculateJointAngleBiomech(lm[23], lm[11], lm[13], 180),
            rightShoulderFlex: calculateJointAngleBiomech(lm[24], lm[12], lm[14], 180),
            
            // Elbow Flexion (팔꿈치 굴곡): wrist → elbow → shoulder
            leftElbowFlex: calculateJointAngleBiomech(lm[15], lm[13], lm[11], 0),
            rightElbowFlex: calculateJointAngleBiomech(lm[16], lm[14], lm[12], 0),
            
            // === Segment Angles (분절 각도) ===
            // Foot: heel → big toe
            leftFootSeg: calculateSegmentAngle(lm[29], lm[31]),
            rightFootSeg: calculateSegmentAngle(lm[30], lm[32]),
            
            // Shank: ankle → knee
            leftShankSeg: calculateSegmentAngle(lm[27], lm[25]),
            rightShankSeg: calculateSegmentAngle(lm[28], lm[26]),
            
            // Thigh: hip → knee  
            leftThighSeg: calculateSegmentAngle(lm[23], lm[25]),
            rightThighSeg: calculateSegmentAngle(lm[24], lm[26]),
            
            // Trunk: midpoint(hips) → midpoint(shoulders)
            trunkSeg: calculateTrunkSegmentAngle(lm[23], lm[24], lm[11], lm[12]),
            
            // Arm: shoulder → elbow
            leftArmSeg: calculateSegmentAngle(lm[11], lm[13]),
            rightArmSeg: calculateSegmentAngle(lm[12], lm[14]),
            
            // Forearm: elbow → wrist
            leftForearmSeg: calculateSegmentAngle(lm[13], lm[15]),
            rightForearmSeg: calculateSegmentAngle(lm[14], lm[16])
        };
    });
}

// 3점 관절 각도 계산 (생체역학 정의)
// referenceAngle: 정렬 상태일 때의 기준 각도
function calculateJointAngleBiomech(point1, point2, point3, referenceAngle = 0) {
    if (!point1 || !point2 || !point3) return null;
    
    // 벡터 계산
    const vector1 = {
        x: point1.x - point2.x,
        y: point1.y - point2.y,
        z: point1.z - point2.z
    };
    
    const vector2 = {
        x: point3.x - point2.x,
        y: point3.y - point2.y,
        z: point3.z - point2.z
    };
    
    // 내적
    const dotProduct = vector1.x * vector2.x + vector1.y * vector2.y + vector1.z * vector2.z;
    
    // 크기
    const magnitude1 = Math.sqrt(vector1.x ** 2 + vector1.y ** 2 + vector1.z ** 2);
    const magnitude2 = Math.sqrt(vector2.x ** 2 + vector2.y ** 2 + vector2.z ** 2);
    
    if (magnitude1 === 0 || magnitude2 === 0) return null;
    
    // 각도 계산 (radian to degree)
    const cosAngle = dotProduct / (magnitude1 * magnitude2);
    let angle = Math.acos(Math.max(-1, Math.min(1, cosAngle))) * (180 / Math.PI);
    
    // 기준 각도 적용
    if (referenceAngle === 180) {
        angle = 180 - angle; // 어깨는 180도가 정렬 상태
    } else if (referenceAngle === -90) {
        angle = angle - 90; // 발목은 -90도가 정렬 상태
    }
    // referenceAngle === 0인 경우는 그대로 사용
    
    return angle;
}

// 분절 각도 계산 (수평선 기준, 반시계방향)
function calculateSegmentAngle(startPoint, endPoint) {
    if (!startPoint || !endPoint) return null;
    
    const dx = endPoint.x - startPoint.x;
    const dy = endPoint.y - startPoint.y;
    
    // atan2로 각도 계산 (반시계방향, -180 ~ 180)
    let angle = Math.atan2(dy, dx) * (180 / Math.PI);
    
    return angle;
}

// 체간 분절 각도 계산
function calculateTrunkSegmentAngle(leftHip, rightHip, leftShoulder, rightShoulder) {
    if (!leftHip || !rightHip || !leftShoulder || !rightShoulder) return null;
    
    // 중점 계산
    const hipMid = {
        x: (leftHip.x + rightHip.x) / 2,
        y: (leftHip.y + rightHip.y) / 2,
        z: (leftHip.z + rightHip.z) / 2
    };
    
    const shoulderMid = {
        x: (leftShoulder.x + rightShoulder.x) / 2,
        y: (leftShoulder.y + rightShoulder.y) / 2,
        z: (leftShoulder.z + rightShoulder.z) / 2
    };
    
    return calculateSegmentAngle(hipMid, shoulderMid);
}

// 각도 차트 업데이트
function updateAngleChart() {
    if (angleData.length === 0) {
        angleChart.data.labels = [];
        angleChart.data.datasets = [];
        angleChart.update();
        return;
    }
    
    const labels = angleData.map(d => d.timestamp.toFixed(2));
    let datasets = [];
    
    // 새로운 각도 맵핑 (생체역학 정의)
    const angleMap = {
        // 관절 각도
        'ankleDorsi': [
            { label: '왼쪽 발목 배굴', data: angleData.map(d => d.leftAnkleDorsi), color: '#10b981' },
            { label: '오른쪽 발목 배굴', data: angleData.map(d => d.rightAnkleDorsi), color: '#34d399' }
        ],
        'kneeFlex': [
            { label: '왼쪽 무릎 굴곡', data: angleData.map(d => d.leftKneeFlex), color: '#FFCE56' },
            { label: '오른쪽 무릎 굴곡', data: angleData.map(d => d.rightKneeFlex), color: '#4BC0C0' }
        ],
        'hipFlex': [
            { label: '왼쪽 엉덩이 굴곡', data: angleData.map(d => d.leftHipFlex), color: '#9966FF' },
            { label: '오른쪽 엉덩이 굴곡', data: angleData.map(d => d.rightHipFlex), color: '#FF9F40' }
        ],
        'shoulderFlex': [
            { label: '왼쪽 어깨 굴곡', data: angleData.map(d => d.leftShoulderFlex), color: '#8B5CF6' },
            { label: '오른쪽 어깨 굴곡', data: angleData.map(d => d.rightShoulderFlex), color: '#EC4899' }
        ],
        'elbowFlex': [
            { label: '왼쪽 팔꿈치 굴곡', data: angleData.map(d => d.leftElbowFlex), color: '#FF6384' },
            { label: '오른쪽 팔꿈치 굴곡', data: angleData.map(d => d.rightElbowFlex), color: '#36A2EB' }
        ],
        
        // 분절 각도
        'footSeg': [
            { label: '왼쪽 발 분절', data: angleData.map(d => d.leftFootSeg), color: '#06b6d4' },
            { label: '오른쪽 발 분절', data: angleData.map(d => d.rightFootSeg), color: '#22d3ee' }
        ],
        'shankSeg': [
            { label: '왼쪽 하퇴 분절', data: angleData.map(d => d.leftShankSeg), color: '#f59e0b' },
            { label: '오른쪽 하퇴 분절', data: angleData.map(d => d.rightShankSeg), color: '#fbbf24' }
        ],
        'thighSeg': [
            { label: '왼쪽 대퇴 분절', data: angleData.map(d => d.leftThighSeg), color: '#ef4444' },
            { label: '오른쪽 대퇴 분절', data: angleData.map(d => d.rightThighSeg), color: '#f87171' }
        ],
        'trunkSeg': [
            { label: '체간 분절', data: angleData.map(d => d.trunkSeg), color: '#6366f1' }
        ],
        'armSeg': [
            { label: '왼쪽 상완 분절', data: angleData.map(d => d.leftArmSeg), color: '#8b5cf6' },
            { label: '오른쪽 상완 분절', data: angleData.map(d => d.rightArmSeg), color: '#a78bfa' }
        ],
        'forearmSeg': [
            { label: '왼쪽 전완 분절', data: angleData.map(d => d.leftForearmSeg), color: '#ec4899' },
            { label: '오른쪽 전완 분절', data: angleData.map(d => d.rightForearmSeg), color: '#f472b6' }
        ]
    };
    
    selectedAngles.forEach(angle => {
        if (angleMap[angle]) {
            angleMap[angle].forEach(item => {
                datasets.push({
                    label: item.label,
                    data: item.data,
                    borderColor: item.color,
                    backgroundColor: item.color + '33',
                    borderWidth: 2,
                    pointRadius: 1,
                    tension: 0.4
                });
            });
        }
    });
    
    angleChart.options.plugins.title.text = '각도 변화 (Joint & Segment Angles)';
    
    // 마커 표시 추가
    const annotations = {};
    markers.forEach((marker, idx) => {
        annotations[`marker${idx}`] = {
            type: 'line',
            xMin: marker.time,
            xMax: marker.time,
            borderColor: 'rgba(255, 99, 132, 0.5)',
            borderWidth: 2,
            borderDash: [5, 5],
            label: {
                display: true,
                content: marker.label,
                position: 'start',
                backgroundColor: 'rgba(255, 99, 132, 0.8)',
                color: '#fff',
                font: {
                    size: 10
                }
            }
        };
    });
    
    if (!angleChart.options.plugins.annotation) {
        angleChart.options.plugins.annotation = { annotations: {} };
    }
    angleChart.options.plugins.annotation.annotations = annotations;
    
    angleChart.data.labels = labels;
    angleChart.data.datasets = datasets;
    angleChart.update();
}

// 각도 통계 업데이트 (더 이상 사용되지 않음 - updateAngleTextDisplay로 대체됨)
function updateAngleStats() {
    // 이 함수는 이전 버전과의 호환성을 위해 유지되지만 더 이상 작동하지 않습니다
    // updateAngleTextDisplay() 함수가 대신 사용됩니다
    return;
}

// 각도 뷰 전환
function switchAngleView(view) {
    currentAngleView = view;
    
    // 버튼 스타일 업데이트
    document.getElementById('btnLocalAngle').classList.toggle('active', view === 'local');
    document.getElementById('btnRelativeAngle').classList.toggle('active', view === 'relative');
    
    updateAngleChart();
    drawPoseSkeleton();
    drawAngleExplanation();
}

// 포즈 스켈레톤 그리기
function drawPoseSkeleton() {
    if (!currentPoseLandmarks) {
        // 포즈가 없으면 빈 캔버스
        poseSkeletonCtx.clearRect(0, 0, poseSkeletonCanvas.width, poseSkeletonCanvas.height);
        poseSkeletonCtx.fillStyle = '#f9fafb';
        poseSkeletonCtx.fillRect(0, 0, poseSkeletonCanvas.width, poseSkeletonCanvas.height);
        poseSkeletonCtx.fillStyle = '#9ca3af';
        poseSkeletonCtx.font = '16px sans-serif';
        poseSkeletonCtx.textAlign = 'center';
        poseSkeletonCtx.fillText('영상을 재생해주세요', poseSkeletonCanvas.width / 2, poseSkeletonCanvas.height / 2);
        return;
    }
    
    const width = poseSkeletonCanvas.width;
    const height = poseSkeletonCanvas.height;
    
    // 캔버스 초기화
    poseSkeletonCtx.clearRect(0, 0, width, height);
    poseSkeletonCtx.fillStyle = '#f9fafb';
    poseSkeletonCtx.fillRect(0, 0, width, height);
    
    // 랜드마크를 캔버스 좌표로 변환 (여백 최소화하여 확대)
    const padding = 10;
    const drawWidth = width - padding * 2;
    const drawHeight = height - padding * 2;
    
    const landmarks = currentPoseLandmarks.map(lm => ({
        x: lm.x * drawWidth + padding,
        y: lm.y * drawHeight + padding,
        z: lm.z,
        visibility: lm.visibility
    }));
    
    // 연결선 그리기 (얇게)
    poseSkeletonCtx.strokeStyle = '#8B5CF6';
    poseSkeletonCtx.lineWidth = 2;
    POSE_CONNECTIONS.forEach(([start, end]) => {
        const from = landmarks[start];
        const to = landmarks[end];
        if (from && to && from.visibility > 0.5 && to.visibility > 0.5) {
            poseSkeletonCtx.beginPath();
            poseSkeletonCtx.moveTo(from.x, from.y);
            poseSkeletonCtx.lineTo(to.x, to.y);
            poseSkeletonCtx.stroke();
        }
    });
    
    // 관절점 그리기 (작게)
    landmarks.forEach((lm, idx) => {
        if (lm.visibility > 0.5) {
            poseSkeletonCtx.beginPath();
            poseSkeletonCtx.arc(lm.x, lm.y, 3, 0, 2 * Math.PI);
            poseSkeletonCtx.fillStyle = '#EC4899';
            poseSkeletonCtx.fill();
        }
    });
    
    // 각도 시각화
    if (angleData.length > 0) {
        const lastAngle = angleData[angleData.length - 1];
        
        if (currentAngleView === 'relative') {
            // 상대 관절 각도 시각화
            if (selectedAngles.includes('elbow')) {
                drawAngleArc(landmarks, 11, 13, 15, lastAngle.leftElbow, '왼팔 팔꿈치');
                drawAngleArc(landmarks, 12, 14, 16, lastAngle.rightElbow, '오른팔 팔꿈치');
            }
            if (selectedAngles.includes('knee')) {
                drawAngleArc(landmarks, 23, 25, 27, lastAngle.leftKnee, '왼쪽 무릎');
                drawAngleArc(landmarks, 24, 26, 28, lastAngle.rightKnee, '오른쪽 무릎');
            }
            if (selectedAngles.includes('hip')) {
                drawAngleArc(landmarks, 11, 23, 25, lastAngle.leftHip, '왼쪽 엉덩이');
                drawAngleArc(landmarks, 12, 24, 26, lastAngle.rightHip, '오른쪽 엉덩이');
            }
            if (selectedAngles.includes('shoulder')) {
                drawAngleArc(landmarks, 23, 11, 13, lastAngle.leftShoulder, '왼쪽 어깨');
                drawAngleArc(landmarks, 24, 12, 14, lastAngle.rightShoulder, '오른쪽 어깨');
            }
        } else {
            // 로컬 각도 시각화 (2점 + 수평선)
            if (selectedAngles.includes('elbow')) {
                drawLocalAngleArc(landmarks, 13, 15, lastAngle.leftElbowLocal, '왼팔 팔꿈치');
                drawLocalAngleArc(landmarks, 14, 16, lastAngle.rightElbowLocal, '오른팔 팔꿈치');
            }
            if (selectedAngles.includes('knee')) {
                drawLocalAngleArc(landmarks, 25, 27, lastAngle.leftKneeLocal, '왼쪽 무릎');
                drawLocalAngleArc(landmarks, 26, 28, lastAngle.rightKneeLocal, '오른쪽 무릎');
            }
            if (selectedAngles.includes('shoulder')) {
                drawLocalAngleArc(landmarks, 11, 13, lastAngle.leftShoulderLocal, '왼쪽 어깨');
                drawLocalAngleArc(landmarks, 12, 14, lastAngle.rightShoulderLocal, '오른쪽 어깨');
            }
        }
    }
}

// 3점 각돀 호 그리기 (상대 관절 각돀)
function drawAngleArc(landmarks, idx1, idx2, idx3, angle, label) {
    if (!angle || !landmarks[idx1] || !landmarks[idx2] || !landmarks[idx3]) return;
    if (landmarks[idx1].visibility < 0.5 || landmarks[idx2].visibility < 0.5 || landmarks[idx3].visibility < 0.5) return;
    
    const center = landmarks[idx2];
    const point1 = landmarks[idx1];
    const point3 = landmarks[idx3];
    
    // 각도 계산
    const angle1 = Math.atan2(point1.y - center.y, point1.x - center.x);
    const angle3 = Math.atan2(point3.y - center.y, point3.x - center.x);
    
    // 호 그리기 (크기 축소: 40 → 25)
    const radius = 25;
    poseSkeletonCtx.beginPath();
    poseSkeletonCtx.arc(center.x, center.y, radius, angle1, angle3, angle3 < angle1);
    poseSkeletonCtx.strokeStyle = '#f59e0b';
    poseSkeletonCtx.lineWidth = 2;
    poseSkeletonCtx.stroke();
    
    // 각도 텍스트는 표시하지 않음 (배경 패널에만 표시)
}

// 2점 각돀 호 그리기 (로컬 각돀)
function drawLocalAngleArc(landmarks, idx1, idx2, angle, label) {
    if (!angle || !landmarks[idx1] || !landmarks[idx2]) return;
    if (landmarks[idx1].visibility < 0.5 || landmarks[idx2].visibility < 0.5) return;
    
    const start = landmarks[idx1];
    const end = landmarks[idx2];
    
    // 수평선 그리기 (참고선)
    poseSkeletonCtx.beginPath();
    poseSkeletonCtx.moveTo(start.x - 50, start.y);
    poseSkeletonCtx.lineTo(start.x + 50, start.y);
    poseSkeletonCtx.strokeStyle = '#d1d5db';
    poseSkeletonCtx.lineWidth = 1;
    poseSkeletonCtx.setLineDash([5, 5]);
    poseSkeletonCtx.stroke();
    poseSkeletonCtx.setLineDash([]);
    
    // 각도 계산
    const lineAngle = Math.atan2(end.y - start.y, end.x - start.x);
    
    // 호 그리기 (0도에서 선분 각도까지)
    const radius = 30;
    poseSkeletonCtx.beginPath();
    poseSkeletonCtx.arc(start.x, start.y, radius, 0, lineAngle);
    poseSkeletonCtx.strokeStyle = '#10b981';
    poseSkeletonCtx.lineWidth = 3;
    poseSkeletonCtx.stroke();
    
    // 각돀 텍스트
    const textAngle = lineAngle / 2;
    const textX = start.x + Math.cos(textAngle) * (radius + 20);
    const textY = start.y + Math.sin(textAngle) * (radius + 20);
    
    poseSkeletonCtx.fillStyle = '#000';
    poseSkeletonCtx.font = 'bold 14px sans-serif';
    poseSkeletonCtx.textAlign = 'center';
    poseSkeletonCtx.fillText(angle.toFixed(1) + '°', textX, textY);
}

// 통계 업데이트
function updateStats() {
    document.getElementById('frameCount').textContent = trackingData.length;
    document.getElementById('timeDisplay').textContent = 
        trackingData.length > 0 ? trackingData[trackingData.length - 1].timestamp.toFixed(1) + 's' : '0.0s';
    document.getElementById('markerCount').textContent = markers.length;
}

// 각도 설명 캔버스 그리기 (분리된 버전)
function drawAngleExplanation() {
    drawSagittalAngleCanvas();
    drawFrontalAngleCanvas();
}

// 시상면 각도 캔버스 그리기
function drawSagittalAngleCanvas() {
    if (!sagittalAngleCtx) return;
    
    const width = sagittalAngleCanvas.width;
    const height = sagittalAngleCanvas.height;
    
    // 캔버스 초기화
    sagittalAngleCtx.clearRect(0, 0, width, height);
    
    // 배경
    sagittalAngleCtx.fillStyle = '#f9fafb';
    sagittalAngleCtx.fillRect(0, 0, width, height);
    
    // 시상면 스켈레톤과 모든 각도 그리기
    drawSagittalView(sagittalAngleCtx, 50, 50, width - 100, height - 80);
}

// 정면 각도 캔버스 그리기
function drawFrontalAngleCanvas() {
    if (!frontalAngleCtx) return;
    
    const width = frontalAngleCanvas.width;
    const height = frontalAngleCanvas.height;
    
    // 캔버스 초기화
    frontalAngleCtx.clearRect(0, 0, width, height);
    
    // 배경
    frontalAngleCtx.fillStyle = '#f9fafb';
    frontalAngleCtx.fillRect(0, 0, width, height);
    
    // 정면 스켈레톤과 모든 각도 그리기
    drawFrontalView(frontalAngleCtx, 50, 50, width - 100, height - 80);
}

// 시상면 스켈레톤 그리기 (옆모습)
function drawSagittalView(ctx, startX, startY, viewWidth, viewHeight) {
    ctx.save();
    
    // 타이틀
    ctx.fillStyle = '#667eea';
    ctx.font = 'bold 13px sans-serif';
    ctx.textAlign = 'center';
    ctx.fillText('시상면 (Sagittal)', startX + viewWidth / 2, startY - 10);
    
    const centerX = startX + viewWidth / 2;
    const scale = viewHeight / 500;
    
    // 간단한 측면 스켈레톤
    const skeleton = {
        head: [centerX, startY + 30 * scale],
        shoulder: [centerX, startY + 80 * scale],
        hip: [centerX, startY + 200 * scale],
        knee: [centerX + 10 * scale, startY + 320 * scale],
        ankle: [centerX + 5 * scale, startY + 440 * scale],
        heel: [centerX, startY + 460 * scale],
        toe: [centerX + 30 * scale, startY + 465 * scale],
        elbow: [centerX - 60 * scale, startY + 150 * scale],
        wrist: [centerX - 80 * scale, startY + 220 * scale]
    };
    
    // 연결선
    ctx.strokeStyle = '#9ca3af';
    ctx.lineWidth = 2;
    const connections = [
        ['head', 'shoulder'], ['shoulder', 'hip'], 
        ['hip', 'knee'], ['knee', 'ankle'],
        ['ankle', 'heel'], ['heel', 'toe'],
        ['shoulder', 'elbow'], ['elbow', 'wrist']
    ];
    
    connections.forEach(([start, end]) => {
        ctx.beginPath();
        ctx.moveTo(skeleton[start][0], skeleton[start][1]);
        ctx.lineTo(skeleton[end][0], skeleton[end][1]);
        ctx.stroke();
    });
    
    // 관절점
    ctx.fillStyle = '#6b7280';
    Object.values(skeleton).forEach(point => {
        ctx.beginPath();
        ctx.arc(point[0], point[1], 3, 0, 2 * Math.PI);
        ctx.fill();
    });
    
    // 각도 표시 (모든 각도 표시, 선택된 것은 강조)
    ctx.font = '10px sans-serif';
    
    // 발목 배굴
    const isAnkleSelected = selectedAngles.includes('ankleDorsi');
    drawSmallAngleArc(ctx, skeleton.heel, skeleton.ankle, skeleton.knee, 
        isAnkleSelected ? '#10b981' : '#d1d5db', '발목', isAnkleSelected);
    
    // 무릎 굴곡
    const isKneeSelected = selectedAngles.includes('kneeFlex');
    drawSmallAngleArc(ctx, skeleton.hip, skeleton.knee, skeleton.ankle, 
        isKneeSelected ? '#FFCE56' : '#d1d5db', '무릎', isKneeSelected);
    
    // 엉덩이 굴곡
    const isHipSelected = selectedAngles.includes('hipFlex');
    drawSmallAngleArc(ctx, skeleton.knee, skeleton.hip, skeleton.shoulder, 
        isHipSelected ? '#9966FF' : '#d1d5db', '엉덩이', isHipSelected);
    
    // 팔꿈치 굴곡
    const isElbowSelected = selectedAngles.includes('elbowFlex');
    drawSmallAngleArc(ctx, skeleton.wrist, skeleton.elbow, skeleton.shoulder, 
        isElbowSelected ? '#FF6384' : '#d1d5db', '팔꿈치', isElbowSelected);
    
    // 분절 각도 - 수평 참조선
    ctx.strokeStyle = '#d1d5db';
    ctx.lineWidth = 1;
    ctx.setLineDash([3, 3]);
    ctx.beginPath();
    ctx.moveTo(startX, skeleton.ankle[1]);
    ctx.lineTo(startX + viewWidth, skeleton.ankle[1]);
    ctx.stroke();
    ctx.setLineDash([]);
    
    // 발 분절
    const isFootSelected = selectedAngles.includes('footSeg');
    if (isFootSelected || !selectedAngles.includes('footSeg')) {
        drawSegmentAngleArc(ctx, skeleton.heel, skeleton.toe, 
            isFootSelected ? '#36A2EB' : '#d1d5db', '발', isFootSelected);
    }
    
    // 하퇴 분절
    const isShankSelected = selectedAngles.includes('shankSeg');
    drawSegmentAngleArc(ctx, skeleton.ankle, skeleton.knee, 
        isShankSelected ? '#f59e0b' : '#d1d5db', '하퇴', isShankSelected);
    
    // 대퇴 분절
    const isThighSelected = selectedAngles.includes('thighSeg');
    drawSegmentAngleArc(ctx, skeleton.knee, skeleton.hip, 
        isThighSelected ? '#ef4444' : '#d1d5db', '대퇴', isThighSelected);
    
    ctx.restore();
}

// 정면 스켈레톤 그리기
function drawFrontalView(ctx, startX, startY, viewWidth, viewHeight) {
    ctx.save();
    
    // 타이틀
    ctx.fillStyle = '#667eea';
    ctx.font = 'bold 13px sans-serif';
    ctx.textAlign = 'center';
    ctx.fillText('정면 (Frontal)', startX + viewWidth / 2, startY - 10);
    
    const centerX = startX + viewWidth / 2;
    const scale = viewHeight / 500;
    
    // 정면 스켈레톤
    const skeleton = {
        head: [centerX, startY + 30 * scale],
        leftShoulder: [centerX - 35 * scale, startY + 80 * scale],
        rightShoulder: [centerX + 35 * scale, startY + 80 * scale],
        leftElbow: [centerX - 50 * scale, startY + 160 * scale],
        rightElbow: [centerX + 50 * scale, startY + 160 * scale],
        leftWrist: [centerX - 60 * scale, startY + 230 * scale],
        rightWrist: [centerX + 60 * scale, startY + 230 * scale],
        leftHip: [centerX - 20 * scale, startY + 200 * scale],
        rightHip: [centerX + 20 * scale, startY + 200 * scale],
        leftKnee: [centerX - 25 * scale, startY + 320 * scale],
        rightKnee: [centerX + 25 * scale, startY + 320 * scale],
        leftAnkle: [centerX - 22 * scale, startY + 440 * scale],
        rightAnkle: [centerX + 22 * scale, startY + 440 * scale]
    };
    
    // 연결선
    ctx.strokeStyle = '#9ca3af';
    ctx.lineWidth = 2;
    const connections = [
        ['head', 'leftShoulder'], ['head', 'rightShoulder'],
        ['leftShoulder', 'rightShoulder'],
        ['leftShoulder', 'leftElbow'], ['leftElbow', 'leftWrist'],
        ['rightShoulder', 'rightElbow'], ['rightElbow', 'rightWrist'],
        ['leftShoulder', 'leftHip'], ['rightShoulder', 'rightHip'],
        ['leftHip', 'rightHip'],
        ['leftHip', 'leftKnee'], ['leftKnee', 'leftAnkle'],
        ['rightHip', 'rightKnee'], ['rightKnee', 'rightAnkle']
    ];
    
    connections.forEach(([start, end]) => {
        ctx.beginPath();
        ctx.moveTo(skeleton[start][0], skeleton[start][1]);
        ctx.lineTo(skeleton[end][0], skeleton[end][1]);
        ctx.stroke();
    });
    
    // 관절점
    ctx.fillStyle = '#6b7280';
    Object.values(skeleton).forEach(point => {
        ctx.beginPath();
        ctx.arc(point[0], point[1], 3, 0, 2 * Math.PI);
        ctx.fill();
    });
    
    // 분절 각도 표시 (모든 각도, 선택 시 강조)
    ctx.font = '10px sans-serif';
    
    // 체간 분절
    const isTrunkSelected = selectedAngles.includes('trunkSeg');
    const hipMid = [(skeleton.leftHip[0] + skeleton.rightHip[0]) / 2, 
                   (skeleton.leftHip[1] + skeleton.rightHip[1]) / 2];
    const shoulderMid = [(skeleton.leftShoulder[0] + skeleton.rightShoulder[0]) / 2,
                        (skeleton.leftShoulder[1] + skeleton.rightShoulder[1]) / 2];
    drawSegmentAngleArc(ctx, hipMid, shoulderMid, 
        isTrunkSelected ? '#6366f1' : '#d1d5db', '체간', isTrunkSelected);
    
    // 어깨 굴곡
    const isShoulderSelected = selectedAngles.includes('shoulderFlex');
    drawSmallAngleArc(ctx, hipMid, skeleton.leftShoulder, skeleton.leftElbow,
        isShoulderSelected ? '#4BC0C0' : '#d1d5db', 'L어깨', isShoulderSelected);
    
    // 상완 분절
    const isArmSelected = selectedAngles.includes('armSeg');
    drawSegmentAngleArc(ctx, skeleton.leftShoulder, skeleton.leftElbow, 
        isArmSelected ? '#8b5cf6' : '#d1d5db', 'L상완', isArmSelected);
    
    // 전완 분절
    const isForearmSelected = selectedAngles.includes('forearmSeg');
    drawSegmentAngleArc(ctx, skeleton.leftElbow, skeleton.leftWrist, 
        isForearmSelected ? '#ec4899' : '#d1d5db', 'L전완', isForearmSelected);
    
    ctx.restore();
}

// 작은 각도 호 그리기 (설명용)
function drawSmallAngleArc(ctx, p1, p2, p3, color, label, isSelected = true) {
    const angle1 = Math.atan2(p1[1] - p2[1], p1[0] - p2[0]);
    const angle3 = Math.atan2(p3[1] - p2[1], p3[0] - p2[0]);
    
    ctx.strokeStyle = color;
    ctx.lineWidth = isSelected ? 2 : 1;
    ctx.beginPath();
    ctx.arc(p2[0], p2[1], 15, angle1, angle3, angle3 < angle1);
    ctx.stroke();
    
    if (isSelected) {
        ctx.fillStyle = color;
        ctx.font = 'bold 9px sans-serif';
        ctx.textAlign = 'center';
        const midAngle = (angle1 + angle3) / 2;
        const labelX = p2[0] + Math.cos(midAngle) * 28;
        const labelY = p2[1] + Math.sin(midAngle) * 28;
        ctx.fillText(label, labelX, labelY);
    }
}

// 분절 각도 호 그리기
function drawSegmentAngleArc(ctx, start, end, color, label, isSelected = true) {
    const segAngle = Math.atan2(end[1] - start[1], end[0] - start[0]);
    
    ctx.strokeStyle = color;
    ctx.lineWidth = isSelected ? 2 : 1;
    ctx.beginPath();
    ctx.arc(start[0], start[1], 20, 0, segAngle);
    ctx.stroke();
    
    if (isSelected) {
        ctx.fillStyle = color;
        ctx.font = 'bold 9px sans-serif';
        ctx.textAlign = 'center';
        const midAngle = segAngle / 2;
        const labelX = start[0] + Math.cos(midAngle) * 32;
        const labelY = start[1] + Math.sin(midAngle) * 32;
        ctx.fillText(label, labelX, labelY);
    }
}

// 각도 텍스트 표시 업데이트 (모든 각도 나열, 클릭으로 활성화)
function updateAngleTextDisplay() {
    const panel = document.getElementById('angleTextDisplay');
    if (!panel) return;
    
    const currentAngles = angleData.length > 0 ? angleData[angleData.length - 1] : {};
    
    const angleDefinitions = [
        { key: 'ankleDorsi', label: '발목 배굴', leftKey: 'leftAnkleDorsi', rightKey: 'rightAnkleDorsi', bgColor: '#ecfdf5', activeColor: '#10b981' },
        { key: 'kneeFlex', label: '무릎 굴곡', leftKey: 'leftKneeFlex', rightKey: 'rightKneeFlex', bgColor: '#fef3c7', activeColor: '#FFCE56' },
        { key: 'hipFlex', label: '엉덩이 굴곡', leftKey: 'leftHipFlex', rightKey: 'rightHipFlex', bgColor: '#f3e8ff', activeColor: '#9966FF' },
        { key: 'shoulderFlex', label: '어깨 굴곡', leftKey: 'leftShoulderFlex', rightKey: 'rightShoulderFlex', bgColor: '#dbeafe', activeColor: '#4BC0C0' },
        { key: 'elbowFlex', label: '팔꿈치 굴곡', leftKey: 'leftElbowFlex', rightKey: 'rightElbowFlex', bgColor: '#fee2e2', activeColor: '#FF6384' },
        { key: 'footSeg', label: '발 분절', leftKey: 'leftFootSeg', rightKey: 'rightFootSeg', bgColor: '#e0e7ff', activeColor: '#36A2EB' },
        { key: 'shankSeg', label: '하퇴 분절', leftKey: 'leftShankSeg', rightKey: 'rightShankSeg', bgColor: '#fef9c3', activeColor: '#f59e0b' },
        { key: 'thighSeg', label: '대퇴 분절', leftKey: 'leftThighSeg', rightKey: 'rightThighSeg', bgColor: '#fecaca', activeColor: '#ef4444' },
        { key: 'trunkSeg', label: '체간 분절', leftKey: null, rightKey: 'trunkSeg', bgColor: '#e9d5ff', activeColor: '#6366f1' },
        { key: 'armSeg', label: '상완 분절', leftKey: 'leftArmSeg', rightKey: 'rightArmSeg', bgColor: '#ddd6fe', activeColor: '#8b5cf6' },
        { key: 'forearmSeg', label: '전완 분절', leftKey: 'leftForearmSeg', rightKey: 'rightForearmSeg', bgColor: '#fbcfe8', activeColor: '#ec4899' }
    ];
    
    let html = '<h3 style="font-size: 14px; margin-bottom: 10px; color: #667eea;">💡 각도 값 (클릭하여 선택)</h3>';
    html += '<div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 8px;">';
    
    angleDefinitions.forEach(def => {
        const isSelected = selectedAngles.includes(def.key);
        const opacity = isSelected ? '1' : '0.4';
        const borderStyle = isSelected ? `2px solid ${def.activeColor}` : '1px solid #e5e7eb';
        const cursor = 'pointer';
        
        if (def.leftKey && def.rightKey && def.key !== 'trunkSeg') {
            // 양쪽 각도
            html += `
                <div class="angle-display-item" data-angle="${def.key}" 
                     style="padding: 8px; background: ${def.bgColor}; border-radius: 5px; opacity: ${opacity}; 
                            border: ${borderStyle}; cursor: ${cursor}; transition: all 0.2s;">
                    <strong>L ${def.label}:</strong> ${formatAngle(currentAngles[def.leftKey])}°
                </div>
                <div class="angle-display-item" data-angle="${def.key}" 
                     style="padding: 8px; background: ${def.bgColor}; border-radius: 5px; opacity: ${opacity}; 
                            border: ${borderStyle}; cursor: ${cursor}; transition: all 0.2s;">
                    <strong>R ${def.label}:</strong> ${formatAngle(currentAngles[def.rightKey])}°
                </div>
            `;
        } else if (def.key === 'trunkSeg') {
            // 단일 각도 (체간)
            html += `
                <div class="angle-display-item" data-angle="${def.key}" 
                     style="padding: 8px; background: ${def.bgColor}; border-radius: 5px; opacity: ${opacity}; 
                            border: ${borderStyle}; cursor: ${cursor}; transition: all 0.2s; grid-column: span 2;">
                    <strong>${def.label}:</strong> ${formatAngle(currentAngles[def.rightKey])}°
                </div>
            `;
        }
    });
    
    html += '</div>';
    panel.innerHTML = html;
    
    // 클릭 이벤트 추가
    document.querySelectorAll('.angle-display-item').forEach(item => {
        item.addEventListener('click', function() {
            const angleName = this.getAttribute('data-angle');
            toggleAngleSelection(angleName);
        });
    });
}

// 각도 포맷 헬퍼 함수
function formatAngle(angle) {
    if (angle == null) return 'N/A';
    return angle.toFixed(1);
}

// 수동 마커 추가
function addManualMarker() {
    if (trackingData.length === 0) return;
    
    const currentTime = videoElement.currentTime;
    const currentFrame = trackingData.length - 1;
    
    // 현재 프레임의 각도 데이터 찾기
    const currentAngleData = angleData.length > 0 ? angleData[angleData.length - 1] : null;
    
    // 비디오 프레임 캡처
    const videoFrame = captureVideoFrame();
    
    const marker = {
        time: currentTime,
        frame: currentFrame,
        type: 'manual',
        label: `수동 마커 ${markers.length + 1}`,
        landmarks: currentPoseLandmarks ? JSON.parse(JSON.stringify(currentPoseLandmarks)) : null,
        angles: currentAngleData ? JSON.parse(JSON.stringify(currentAngleData)) : null,
        videoFrame: videoFrame
    };
    
    markers.push(marker);
    updateMarkerList();
    updateStats();
    updatePoseComparison();
}

// 자동 마커 추가 (Min/Max)
function addAutoMarkers() {
    if (trackingData.length === 0 || selectedKeypoints.length === 0) return;
    
    selectedKeypoints.forEach(keypointIdx => {
        const yValues = trackingData.map(d => d.landmarks[keypointIdx]?.y).filter(v => v != null);
        
        if (yValues.length === 0) return;
        
        // 최소값 (가장 위)
        const minY = Math.min(...yValues);
        const minIdx = yValues.indexOf(minY);
        markers.push({
            time: trackingData[minIdx].timestamp,
            frame: minIdx,
            type: 'auto-min',
            label: `${POSE_LANDMARKS[keypointIdx]} - 최고점`,
            value: minY
        });
        
        // 최대값 (가장 아래)
        const maxY = Math.max(...yValues);
        const maxIdx = yValues.indexOf(maxY);
        markers.push({
            time: trackingData[maxIdx].timestamp,
            frame: maxIdx,
            type: 'auto-max',
            label: `${POSE_LANDMARKS[keypointIdx]} - 최저점`,
            value: maxY
        });
    });
    
    updateMarkerList();
    updateStats();
}

// 마커 리스트 업데이트
function updateMarkerList() {
    const listDiv = document.getElementById('markerList');
    listDiv.innerHTML = '';
    
    markers.sort((a, b) => a.time - b.time);
    
    markers.forEach((marker, idx) => {
        const item = document.createElement('div');
        item.className = 'marker-item';
        item.innerHTML = `
            <span><strong>${marker.time.toFixed(2)}s</strong> - ${marker.label}</span>
            <button class="btn btn-danger" onclick="removeMarker(${idx})">삭제</button>
        `;
        listDiv.appendChild(item);
    });
}

// 마커 삭제
function removeMarker(index) {
    markers.splice(index, 1);
    updateMarkerList();
    updateStats();
}

// CSV 내보내기
function exportCSV() {
    if (trackingData.length === 0) {
        alert('추적 데이터가 없습니다!');
        return;
    }
    
    let csv = 'Frame,Time(s)';
    
    // 헤더 - 선택된 키포인트
    selectedKeypoints.forEach(idx => {
        const name = POSE_LANDMARKS[idx];
        csv += `,${name}_X,${name}_Y,${name}_Z,${name}_Visibility`;
    });
    csv += '\n';
    
    // 데이터 행
    trackingData.forEach(frame => {
        csv += `${frame.frame},${frame.timestamp.toFixed(3)}`;
        
        selectedKeypoints.forEach(idx => {
            const landmark = frame.landmarks[idx];
            if (landmark) {
                csv += `,${landmark.x.toFixed(4)},${landmark.y.toFixed(4)},${landmark.z.toFixed(4)},${landmark.visibility.toFixed(4)}`;
            } else {
                csv += ',,,';
            }
        });
        
        csv += '\n';
    });
    
    // 마커 정보 추가
    if (markers.length > 0) {
        csv += '\n\nMarkers\n';
        csv += 'Time(s),Frame,Type,Label\n';
        markers.forEach(marker => {
            csv += `${marker.time.toFixed(3)},${marker.frame},${marker.type},"${marker.label}"\n`;
        });
    }
    
    downloadFile(csv, 'keypoint_data.csv', 'text/csv');
}

// JSON 내보내기
function exportJSON() {
    if (trackingData.length === 0) {
        alert('추적 데이터가 없습니다!');
        return;
    }
    
    const exportData = {
        metadata: {
            totalFrames: trackingData.length,
            duration: trackingData[trackingData.length - 1].timestamp,
            selectedKeypoints: selectedKeypoints.map(idx => ({
                id: idx,
                name: POSE_LANDMARKS[idx]
            })),
            exportDate: new Date().toISOString()
        },
        trackingData: trackingData,
        markers: markers
    };
    
    const json = JSON.stringify(exportData, null, 2);
    downloadFile(json, 'keypoint_data.json', 'application/json');
}

// 파일 다운로드 헬퍼
function downloadFile(content, filename, mimeType) {
    const blob = new Blob([content], { type: mimeType });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
}

// 컨트롤 활성화
function enableControls() {
    document.getElementById('playPauseBtn').disabled = false;
    document.getElementById('stopBtn').disabled = false;
    document.getElementById('addMarkerBtn').disabled = false;
    document.getElementById('autoMarkerBtn').disabled = false;
    document.getElementById('exportCSVBtn').disabled = false;
    document.getElementById('exportJSONBtn').disabled = false;
}

// 선택된 각도 업데이트
function updateSelectedAngles() {
    selectedAngles = Array.from(document.querySelectorAll('.angle-checkbox:checked'))
        .map(cb => cb.value);
    updateAngleChart();
    drawPoseSkeleton();
}

// 시점별 포즈 비교 업데이트
function updatePoseComparison() {
    const container = document.getElementById('poseComparisonContainer');
    container.innerHTML = '';
    
    if (markers.length === 0) {
        container.innerHTML = '<div style="text-align: center; color: #9ca3af; padding: 40px;">마커를 추가하면 시점별 비교가 여기에 표시됩니다</div>';
        return;
    }
    
    // 마커를 시간 순서대로 정렬
    const sortedMarkers = [...markers].sort((a, b) => a.time - b.time);
    
    sortedMarkers.forEach((marker, idx) => {
        if (!marker.landmarks || !marker.angles) return;
        
        const card = document.createElement('div');
        card.className = 'comparison-card';
        
        const title = document.createElement('h4');
        title.textContent = `${marker.label} (${formatTime(marker.time)})`;
        card.appendChild(title);
        
        // 포즈 이미지 캔버스 생성
        const imageCanvas = document.createElement('canvas');
        imageCanvas.width = 250;
        imageCanvas.height = 350;
        imageCanvas.style.width = '100%';
        card.appendChild(imageCanvas);
        
        // 포즈 이미지 그리기
        drawComparisonPose(imageCanvas, marker.landmarks, marker.angles, marker.videoFrame);
        
        // 스켈레톤 캔버스 생성
        const skeletonCanvas = document.createElement('canvas');
        skeletonCanvas.width = 250;
        skeletonCanvas.height = 350;
        skeletonCanvas.style.width = '100%';
        skeletonCanvas.style.marginTop = '10px';
        skeletonCanvas.style.border = '2px solid #e5e7eb';
        skeletonCanvas.style.borderRadius = '10px';
        skeletonCanvas.style.background = '#f9fafb';
        card.appendChild(skeletonCanvas);
        
        // 스켈레톤만 그리기
        drawComparisonSkeleton(skeletonCanvas, marker.landmarks, marker.angles);
        
        // 각도 정보 박스
        const angleInfo = document.createElement('div');
        angleInfo.style.cssText = 'margin-top: 10px; padding: 10px; background: #f3f4f6; border-radius: 8px; font-size: 11px;';
        
        const angles = marker.angles;
        let angleHTML = '<div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 5px;">';
        
        // 선택된 각도만 표시
        if (selectedAngles.includes('ankleDorsi')) {
            if (angles.leftAnkleDorsi != null) angleHTML += `<div><strong>L 발목:</strong> ${angles.leftAnkleDorsi.toFixed(1)}°</div>`;
            if (angles.rightAnkleDorsi != null) angleHTML += `<div><strong>R 발목:</strong> ${angles.rightAnkleDorsi.toFixed(1)}°</div>`;
        }
        if (selectedAngles.includes('kneeFlex')) {
            if (angles.leftKneeFlex != null) angleHTML += `<div><strong>L 무릎:</strong> ${angles.leftKneeFlex.toFixed(1)}°</div>`;
            if (angles.rightKneeFlex != null) angleHTML += `<div><strong>R 무릎:</strong> ${angles.rightKneeFlex.toFixed(1)}°</div>`;
        }
        if (selectedAngles.includes('hipFlex')) {
            if (angles.leftHipFlex != null) angleHTML += `<div><strong>L 엉덩이:</strong> ${angles.leftHipFlex.toFixed(1)}°</div>`;
            if (angles.rightHipFlex != null) angleHTML += `<div><strong>R 엉덩이:</strong> ${angles.rightHipFlex.toFixed(1)}°</div>`;
        }
        if (selectedAngles.includes('shoulderFlex')) {
            if (angles.leftShoulderFlex != null) angleHTML += `<div><strong>L 어깨:</strong> ${angles.leftShoulderFlex.toFixed(1)}°</div>`;
            if (angles.rightShoulderFlex != null) angleHTML += `<div><strong>R 어깨:</strong> ${angles.rightShoulderFlex.toFixed(1)}°</div>`;
        }
        if (selectedAngles.includes('elbowFlex')) {
            if (angles.leftElbowFlex != null) angleHTML += `<div><strong>L 팔꿈치:</strong> ${angles.leftElbowFlex.toFixed(1)}°</div>`;
            if (angles.rightElbowFlex != null) angleHTML += `<div><strong>R 팔꿈치:</strong> ${angles.rightElbowFlex.toFixed(1)}°</div>`;
        }
        if (selectedAngles.includes('footSeg')) {
            if (angles.leftFootSeg != null) angleHTML += `<div><strong>L 발 분절:</strong> ${angles.leftFootSeg.toFixed(1)}°</div>`;
            if (angles.rightFootSeg != null) angleHTML += `<div><strong>R 발 분절:</strong> ${angles.rightFootSeg.toFixed(1)}°</div>`;
        }
        if (selectedAngles.includes('shankSeg')) {
            if (angles.leftShankSeg != null) angleHTML += `<div><strong>L 하퇴:</strong> ${angles.leftShankSeg.toFixed(1)}°</div>`;
            if (angles.rightShankSeg != null) angleHTML += `<div><strong>R 하퇴:</strong> ${angles.rightShankSeg.toFixed(1)}°</div>`;
        }
        if (selectedAngles.includes('thighSeg')) {
            if (angles.leftThighSeg != null) angleHTML += `<div><strong>L 대퇴:</strong> ${angles.leftThighSeg.toFixed(1)}°</div>`;
            if (angles.rightThighSeg != null) angleHTML += `<div><strong>R 대퇴:</strong> ${angles.rightThighSeg.toFixed(1)}°</div>`;
        }
        if (selectedAngles.includes('trunkSeg')) {
            if (angles.trunkSeg != null) angleHTML += `<div style="grid-column: span 2;"><strong>체간:</strong> ${angles.trunkSeg.toFixed(1)}°</div>`;
        }
        if (selectedAngles.includes('armSeg')) {
            if (angles.leftArmSeg != null) angleHTML += `<div><strong>L 상완:</strong> ${angles.leftArmSeg.toFixed(1)}°</div>`;
            if (angles.rightArmSeg != null) angleHTML += `<div><strong>R 상완:</strong> ${angles.rightArmSeg.toFixed(1)}°</div>`;
        }
        if (selectedAngles.includes('forearmSeg')) {
            if (angles.leftForearmSeg != null) angleHTML += `<div><strong>L 전완:</strong> ${angles.leftForearmSeg.toFixed(1)}°</div>`;
            if (angles.rightForearmSeg != null) angleHTML += `<div><strong>R 전완:</strong> ${angles.rightForearmSeg.toFixed(1)}°</div>`;
        }
        
        angleHTML += '</div>';
        angleInfo.innerHTML = angleHTML;
        card.appendChild(angleInfo);
        
        container.appendChild(card);
    });
}

// 비교 스켈레톤만 그리기 (각도 표시 포함)
function drawComparisonSkeleton(canvas, poseLandmarks, angles) {
    const ctx = canvas.getContext('2d');
    const width = canvas.width;
    const height = canvas.height;
    
    // 배경
    ctx.fillStyle = '#f9fafb';
    ctx.fillRect(0, 0, width, height);
    
    // 랜드마크 변환
    const padding = 20;
    const drawWidth = width - padding * 2;
    const drawHeight = height - padding * 2;
    
    const landmarks = poseLandmarks.map(lm => ({
        x: lm.x * drawWidth + padding,
        y: lm.y * drawHeight + padding,
        z: lm.z,
        visibility: lm.visibility
    }));
    
    // 연결선
    ctx.strokeStyle = '#8B5CF6';
    ctx.lineWidth = 2;
    POSE_CONNECTIONS.forEach(([start, end]) => {
        const from = landmarks[start];
        const to = landmarks[end];
        if (from && to && from.visibility > 0.5 && to.visibility > 0.5) {
            ctx.beginPath();
            ctx.moveTo(from.x, from.y);
            ctx.lineTo(to.x, to.y);
            ctx.stroke();
        }
    });
    
    // 관절점
    landmarks.forEach((lm, idx) => {
        if (lm.visibility > 0.5) {
            ctx.beginPath();
            ctx.arc(lm.x, lm.y, 4, 0, 2 * Math.PI);
            ctx.fillStyle = '#EC4899';
            ctx.fill();
            ctx.strokeStyle = '#FFFFFF';
            ctx.lineWidth = 2;
            ctx.stroke();
        }
    });
    
    // 선택된 각도 호 표시
    if (selectedAngles.includes('kneeFlex')) {
        drawComparisonAngleArc(ctx, landmarks, 23, 25, 27, angles.leftKneeFlex);
        drawComparisonAngleArc(ctx, landmarks, 24, 26, 28, angles.rightKneeFlex);
    }
    if (selectedAngles.includes('hipFlex')) {
        drawComparisonAngleArc(ctx, landmarks, 25, 23, 11, angles.leftHipFlex);
        drawComparisonAngleArc(ctx, landmarks, 26, 24, 12, angles.rightHipFlex);
    }
    if (selectedAngles.includes('elbowFlex')) {
        drawComparisonAngleArc(ctx, landmarks, 15, 13, 11, angles.leftElbowFlex);
        drawComparisonAngleArc(ctx, landmarks, 16, 14, 12, angles.rightElbowFlex);
    }
    if (selectedAngles.includes('ankleDorsi')) {
        drawComparisonAngleArc(ctx, landmarks, 29, 27, 25, angles.leftAnkleDorsi);
        drawComparisonAngleArc(ctx, landmarks, 30, 28, 26, angles.rightAnkleDorsi);
    }
    if (selectedAngles.includes('shoulderFlex')) {
        drawComparisonAngleArc(ctx, landmarks, 23, 11, 13, angles.leftShoulderFlex);
        drawComparisonAngleArc(ctx, landmarks, 24, 12, 14, angles.rightShoulderFlex);
    }
}

// 비교 포즈 그리기
function drawComparisonPose(canvas, poseLandmarks, angles, videoFrame) {
    const ctx = canvas.getContext('2d');
    const width = canvas.width;
    const height = canvas.height;
    
    // 배경 초기화
    ctx.clearRect(0, 0, width, height);
    
    if (videoFrame) {
        // 비디오 프레임 이미지 로드
        const img = new Image();
        img.onload = function() {
            console.log('이미지 로드 성공');
            // 이미지를 캔버스에 맞게 그리기
            ctx.drawImage(img, 0, 0, width, height);
            
            // 스켈레톤 오버레이
            drawSkeletonOverlay(ctx, poseLandmarks, angles, width, height);
        };
        img.onerror = function(e) {
            console.error('이미지 로드 실패:', e);
            // 오류 시 기본 배경으로 대체
            ctx.fillStyle = '#f9fafb';
            ctx.fillRect(0, 0, width, height);
            drawSkeletonOverlay(ctx, poseLandmarks, angles, width, height);
        };
        console.log('이미지 로드 시작:', videoFrame.substring(0, 50) + '...');
        img.src = videoFrame;
    } else {
        console.log('비디오 프레임 없음 - 기본 배경 사용');
        // 비디오 이미지가 없으면 기본 배경
        ctx.fillStyle = '#f9fafb';
        ctx.fillRect(0, 0, width, height);
        drawSkeletonOverlay(ctx, poseLandmarks, angles, width, height);
    }
}

// 스켈레톤 오버레이 그리기
function drawSkeletonOverlay(ctx, poseLandmarks, angles, width, height) {
    // 랜드마크 변환 (비디오 프레임과 정확히 일치하도록 padding 0)
    const padding = 0;
    const drawWidth = width - padding * 2;
    const drawHeight = height - padding * 2;
    
    const landmarks = poseLandmarks.map(lm => ({
        x: lm.x * drawWidth + padding,
        y: lm.y * drawHeight + padding,
        z: lm.z,
        visibility: lm.visibility
    }));
    
    // 연결선 (얇고 트랜디하게)
    ctx.strokeStyle = '#8B5CF6'; // Purple
    ctx.lineWidth = 1.5;
    ctx.shadowColor = 'rgba(139, 92, 246, 0.3)';
    ctx.shadowBlur = 2;
    POSE_CONNECTIONS.forEach(([start, end]) => {
        const from = landmarks[start];
        const to = landmarks[end];
        if (from && to && from.visibility > 0.5 && to.visibility > 0.5) {
            ctx.beginPath();
            ctx.moveTo(from.x, from.y);
            ctx.lineTo(to.x, to.y);
            ctx.stroke();
        }
    });
    
    // 관절점 (작고 선명하게)
    ctx.shadowBlur = 3;
    landmarks.forEach((lm, idx) => {
        if (lm.visibility > 0.5) {
            ctx.beginPath();
            ctx.arc(lm.x, lm.y, 3, 0, 2 * Math.PI);
            ctx.fillStyle = '#EC4899'; // Pink
            ctx.fill();
            ctx.strokeStyle = '#FFFFFF';
            ctx.lineWidth = 2;
            ctx.stroke();
        }
    });
    
    ctx.shadowBlur = 0;
    
    // 각돀 표시
    if (currentAngleView === 'relative') {
        if (selectedAngles.includes('elbow')) {
            drawComparisonAngleArc(ctx, landmarks, 11, 13, 15, angles.leftElbow);
            drawComparisonAngleArc(ctx, landmarks, 12, 14, 16, angles.rightElbow);
        }
        if (selectedAngles.includes('knee')) {
            drawComparisonAngleArc(ctx, landmarks, 23, 25, 27, angles.leftKnee);
            drawComparisonAngleArc(ctx, landmarks, 24, 26, 28, angles.rightKnee);
        }
        if (selectedAngles.includes('hip')) {
            drawComparisonAngleArc(ctx, landmarks, 11, 23, 25, angles.leftHip);
            drawComparisonAngleArc(ctx, landmarks, 12, 24, 26, angles.rightHip);
        }
    }
}

// 비교 각도 호 그리기 (간략화)
function drawComparisonAngleArc(ctx, landmarks, idx1, idx2, idx3, angle) {
    if (!angle || !landmarks[idx1] || !landmarks[idx2] || !landmarks[idx3]) return;
    if (landmarks[idx1].visibility < 0.5 || landmarks[idx2].visibility < 0.5 || landmarks[idx3].visibility < 0.5) return;
    
    const center = landmarks[idx2];
    const point1 = landmarks[idx1];
    const point3 = landmarks[idx3];
    
    const angle1 = Math.atan2(point1.y - center.y, point1.x - center.x);
    const angle3 = Math.atan2(point3.y - center.y, point3.x - center.x);
    
    const radius = 25;
    ctx.beginPath();
    ctx.arc(center.x, center.y, radius, angle1, angle3, angle3 < angle1);
    ctx.strokeStyle = '#f59e0b';
    ctx.lineWidth = 2;
    ctx.stroke();
}

// 비디오 프레임 캡처
function captureVideoFrame() {
    try {
        if (!videoElement || !videoElement.videoWidth || !videoElement.videoHeight) {
            console.log('비디오가 준비되지 않았습니다');
            return null;
        }
        
        const captureCanvas = document.createElement('canvas');
        captureCanvas.width = videoElement.videoWidth;
        captureCanvas.height = videoElement.videoHeight;
        const ctx = captureCanvas.getContext('2d');
        ctx.drawImage(videoElement, 0, 0, captureCanvas.width, captureCanvas.height);
        const dataURL = captureCanvas.toDataURL('image/jpeg', 0.8);
        console.log('비디오 프레임 캡처 완료:', dataURL.substring(0, 50) + '...');
        return dataURL;
    } catch (error) {
        console.error('비디오 캡처 오류:', error);
        return null;
    }
}

// 각도 선택 토글 함수
function toggleAngleSelection(angleName) {
    const index = selectedAngles.indexOf(angleName);
    if (index > -1) {
        selectedAngles.splice(index, 1);
    } else {
        selectedAngles.push(angleName);
    }
    
    // 체크박스 상태도 업데이트
    const checkbox = document.querySelector(`.angle-checkbox[value="${angleName}"]`);
    if (checkbox) {
        checkbox.checked = selectedAngles.includes(angleName);
    }
    
    // 즉시 화면 업데이트
    drawAngleExplanation();
    updateAngleChart();
    updateAngleTextDisplay();
    if (currentPoseLandmarks) {
        drawPoseSkeleton(currentPoseLandmarks);
    }
}

// 시상면 캔버스 클릭 핸들러
function handleSagittalCanvasClick(event) {
    const rect = sagittalAngleCanvas.getBoundingClientRect();
    const scaleX = sagittalAngleCanvas.width / rect.width;
    const scaleY = sagittalAngleCanvas.height / rect.height;
    
    const x = (event.clientX - rect.left) * scaleX;
    const y = (event.clientY - rect.top) * scaleY;
    
    const width = sagittalAngleCanvas.width;
    const height = sagittalAngleCanvas.height;
    const viewHeight = height - 80;
    const startY = 50;
    const centerX = width / 2;
    const scale = viewHeight / 500;
    
    // 시상면 각도 영역 정의
    const sagittalAngles = {
        ankleDorsi: { center: [centerX + 5 * scale, startY + 440 * scale], radius: 30 },
        kneeFlex: { center: [centerX + 10 * scale, startY + 320 * scale], radius: 30 },
        hipFlex: { center: [centerX, startY + 200 * scale], radius: 30 },
        elbowFlex: { center: [centerX - 60 * scale, startY + 150 * scale], radius: 30 },
        footSeg: { center: [centerX + 15 * scale, startY + 462 * scale], radius: 35 },
        shankSeg: { center: [centerX + 7.5 * scale, startY + 380 * scale], radius: 35 },
        thighSeg: { center: [centerX + 5 * scale, startY + 260 * scale], radius: 35 }
    };
    
    // 클릭한 위치에서 가장 가까운 각도 찾기
    let clickedAngle = null;
    let minDistance = Infinity;
    
    for (const [angleName, angleData] of Object.entries(sagittalAngles)) {
        const dx = x - angleData.center[0];
        const dy = y - angleData.center[1];
        const distance = Math.sqrt(dx * dx + dy * dy);
        
        if (distance < angleData.radius && distance < minDistance) {
            minDistance = distance;
            clickedAngle = angleName;
        }
    }
    
    if (clickedAngle) {
        toggleAngleSelection(clickedAngle);
    }
}

// 정면 캔버스 클릭 핸들러
function handleFrontalCanvasClick(event) {
    const rect = frontalAngleCanvas.getBoundingClientRect();
    const scaleX = frontalAngleCanvas.width / rect.width;
    const scaleY = frontalAngleCanvas.height / rect.height;
    
    const x = (event.clientX - rect.left) * scaleX;
    const y = (event.clientY - rect.top) * scaleY;
    
    const width = frontalAngleCanvas.width;
    const height = frontalAngleCanvas.height;
    const viewHeight = height - 80;
    const startY = 50;
    const centerX = width / 2;
    const scale = viewHeight / 500;
    
    // 정면 각도 영역 정의
    const frontalAngles = {
        trunkSeg: { center: [centerX, startY + 140 * scale], radius: 40 },
        shoulderFlex: { center: [centerX - 35 * scale, startY + 80 * scale], radius: 30 },
        armSeg: { center: [centerX - 42 * scale, startY + 120 * scale], radius: 35 },
        forearmSeg: { center: [centerX - 55 * scale, startY + 195 * scale], radius: 35 }
    };
    
    // 클릭한 위치에서 가장 가까운 각도 찾기
    let clickedAngle = null;
    let minDistance = Infinity;
    
    for (const [angleName, angleData] of Object.entries(frontalAngles)) {
        const dx = x - angleData.center[0];
        const dy = y - angleData.center[1];
        const distance = Math.sqrt(dx * dx + dy * dy);
        
        if (distance < angleData.radius && distance < minDistance) {
            minDistance = distance;
            clickedAngle = angleName;
        }
    }
    
    if (clickedAngle) {
        toggleAngleSelection(clickedAngle);
    }
}
