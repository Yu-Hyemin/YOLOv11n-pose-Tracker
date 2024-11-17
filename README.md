# YOLOv11n-pose-Tracker
YOLOv11 모델로 특정 객체 Pose감지 실습

## 🖥️ 개발환경
* <img src="https://img.shields.io/badge/Google Colab-F9AB00?style=for-the-badge&logo=Google Colab&logoColor=white">
<br>

## ➡️ 흐름
1. YOLO Pose 모델을 사용하여 Key-point값 추출
2. YOLO Pose + Tracker 기능 추가
   ① 고유 ID값에 따라 색상을 주어 시각화
   ② 동영상에서 객체 추적하는 파라미터 추가
   ③ results.py 파일을 통해 단순한 코드로 시각화
   ④ ③의 코드를 참고하여 원하는 모양으로 시각화코드 수정
3. 특정 사람만 지정하여 Key-point값 추적
