from ultralytics import YOLO

# 1. 모델 로드 (예: YOLOv8n, n은 nano 버전으로 가장 빠르고 작음)
# 사전 학습된 가중치 사용 권장
model = YOLO('yolov8n.pt')

# 2. 모델 학습
# data: 위에서 생성한 data.yaml 파일 경로
# epochs: 학습 반복 횟수
# imgsz: 학습 시 이미지 크기 (기본값 640)
# batch: 배치 크기 (GPU 메모리에 따라 조절)
results = model.train(
    data='path/to/your/data.yaml',
    epochs=50,
    imgsz=640,
    batch=16,
    name='my_custom_yolov8_run' # 결과 저장 폴더 이름
)

# 학습이 완료되면 결과 파일(학습된 가중치, 성능 그래프 등)은
# `runs/detect/my_custom_yolov8_run` 폴더에 저장됩니다.