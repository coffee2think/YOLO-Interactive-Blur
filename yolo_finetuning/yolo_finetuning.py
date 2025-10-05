# yolo_finetuning.py

# 프로세스
# 1. yolo가 인식할 수 있도록 데이터셋의 라벨 데이터(JSON)을 TXT로 변환
# 2. 모델 학습

from ultralytics import YOLO
import os

# 1. 초기 가중치 파일 설정
# YOLOv8n 가중치를 로드하여 전이 학습의 시작점으로 사용합니다.
INITIAL_WEIGHTS = 'yolov8n.pt'

# 2. 데이터 설정 파일 경로
# AI 허브 데이터셋 정보만 담고 있는 yaml 파일을 지정합니다.
DATA_CONFIG = 'data_ai_hub.yaml'

# 3. 테스트 관련 경로
TEST_IMAGE_PATH = 'data/inputs/test_img.jpg'
OUTPUT_DIR = 'data/outputs'

# 4. 학습 하이퍼파라미터 설정 (테스트를 위해 epochs는 낮게 설정)
NUM_EPOCHS = 10
BATCH_SIZE = 16  # GPU/CPU 메모리에 따라 조절하세요.
IMAGE_SIZE = 640  # 학습 속도를 위해 640으로 설정

def run_detection_test(weights_path):
    """
    학습된 가중치를 사용하여 테스트 이미지를 탐지하고 결과를 저장합니다.
    """
    print("\n--- 🔎 학습된 모델로 테스트 이미지 탐지 시작 ---")

    # 1. 출력 디렉토리 생성
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 2. 학습된 가중치 로드
    test_model = YOLO(weights_path)

    # 3. 탐지 실행 및 결과 저장
    # predict() 메서드는 탐지 결과를 자동으로 'runs/detect/predictX' 폴더에 저장합니다.
    # save=True, project=OUTPUT_DIR, name='results'를 사용하여 원하는 경로에 저장합니다.
    results = test_model.predict(
        source=TEST_IMAGE_PATH,
        save=True,  # 탐지된 이미지를 저장
        imgsz=IMAGE_SIZE,  # 학습 시 사용한 이미지 크기와 동일하게 설정
        conf=0.25,  # 탐지 신뢰도 임계값 (필요에 따라 조절)
        iou=0.7,  # IoU 임계값
        project=OUTPUT_DIR,  # 결과가 저장될 상위 폴더
        name='detection_results',  # 결과가 저장될 하위 폴더 이름
        exist_ok=True  # 폴더가 이미 있으면 덮어씁니다.
    )

    # 4. 최종 저장 경로 안내
    final_save_path = os.path.join(OUTPUT_DIR, 'detection_results', os.path.basename(TEST_IMAGE_PATH))
    print(f"--- ✅ 탐지 완료! 결과는 다음 경로에 저장되었습니다: {os.path.abspath(final_save_path)}")

def run_ai_hub_training():
    """AI 허브 데이터셋만 사용하여 YOLOv8 모델을 파인튜닝합니다."""
    print("--- 🚀 AI Hub 데이터셋 전용 파인튜닝 시작 (49개 클래스만 학습) ---")

    try:
        # 모델 인스턴스화: yolov8n.pt 로드 및 50개 클래스에 맞게 출력층 재설정
        model = YOLO(INITIAL_WEIGHTS)

        print(f"모델: {INITIAL_WEIGHTS}")
        print(f"데이터셋 설정: {DATA_CONFIG}")
        print(f"에폭: {NUM_EPOCHS}, 배치 크기: {BATCH_SIZE}")

        # 학습 실행
        results = model.train(
            # data=DATA_CONFIG,  # 데이터셋 설정 파일
            epochs=NUM_EPOCHS,  # 학습 횟수
            batch=BATCH_SIZE,  # 배치 사이즈
            imgsz=IMAGE_SIZE,  # 입력 이미지 크기
            name='aihub_train_second_test'  # 학습 결과 저장 폴더 이름
        )

        print("--- ✅ 모델 학습 완료 ---")
        # 최종 가중치 파일 경로를 출력합니다.
        final_weights_path = f"{results.save_dir}/weights/best.pt"
        print(f"최종 가중치 파일: {final_weights_path}")

        # ⚠️ 학습 완료 후 가중치 파일 경로를 찾습니다.
        # Ultralytics는 'runs/detect/{name}/weights/best.pt'에 저장합니다.
        weights_dir = os.path.join(results.save_dir, 'weights')
        final_weights_path = os.path.join(weights_dir, 'best.pt')

        # 5. 학습된 가중치로 탐지 테스트 실행
        if os.path.exists(final_weights_path) and os.path.exists(TEST_IMAGE_PATH):
            run_detection_test(final_weights_path)
        else:
            print("경고: 학습된 가중치 파일 또는 테스트 이미지를 찾을 수 없어 탐지 테스트를 건너뜁니다.")


    except Exception as e:
        print(f"모델 학습 중 오류 발생: {e}")

if __name__ == '__main__':
    run_ai_hub_training()