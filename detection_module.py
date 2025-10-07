import cv2
import numpy as np
import sys
from pathlib import Path
from ultralytics import YOLO
from typing import Optional, Tuple, Dict, Any, List
import random

# ----------------------------------------------------------------------
# 1. DetectionModule 클래스 정의
# ----------------------------------------------------------------------

class DetectionModule:
    """
    학습된 YOLOv8 Segmentation 모델을 사용하여 탐지를 수행하고,
    결과를 YOLO TXT 파일 및 시각화 이미지로 특정 경로에 저장하는 모듈.
    """
    # 모델 로드를 한 번만 수행하기 위한 클래스 변수
    MODEL: Optional[YOLO] = None

    def __init__(self, model_path: str | Path, data_yaml_path: str | Path):
        """
        DetectionModule 초기화. 모델 및 데이터셋 정보를 로드합니다.

        Args:
            model_path (str | Path): 학습된 가중치 파일 (예: models/best.pt).
            data_yaml_path (str | Path): 클래스 정보가 포함된 data.yaml 경로.
        """
        self.model_path = Path(model_path)
        self.data_yaml_path = Path(data_yaml_path)

        if not self.model_path.is_file():
            print(f"[ERROR] 모델 파일 경로를 찾을 수 없습니다: {self.model_path}", file=sys.stderr)
            raise FileNotFoundError(f"Model file not found at {self.model_path}")

        if DetectionModule.MODEL is None:
            print(f"YOLO 모델 로드 중: {self.model_path.name} ...")
            try:
                # 모델 로드 및 CPU 강제
                DetectionModule.MODEL = YOLO(str(self.model_path))
                print("모델 로드 완료.")
            except Exception as e:
                print(f"[FATAL] 모델 로드 중 오류 발생: {e}", file=sys.stderr)
                raise RuntimeError(f"Failed to load YOLO model: {e}")

        self.model = DetectionModule.MODEL

    def detect_and_save(self, image_path: str | Path, output_base_dir: str | Path) -> bool:
        """
        객체 탐지를 수행하고, 결과를 YOLO TXT와 시각화 이미지로 저장합니다.

        Args:
            image_path: 탐지할 이미지 파일 경로.
            output_base_dir: 결과를 저장할 기본 디렉터리 (예: data/outputs/detections).

        Returns:
            bool: 탐지 및 저장이 성공했는지 여부.
        """
        if self.model is None:
            return False

        image_path = Path(image_path)
        output_base_dir = Path(output_base_dir)

        if not image_path.is_file():
            print(f"[ERROR] 이미지 파일을 찾을 수 없습니다: {image_path}", file=sys.stderr)
            return False

        # 1. YOLO 예측 실행 (TXT 라벨 및 기본 시각화 파일 저장)
        # project와 name을 사용하여 임시 출력 폴더를 설정
        temp_output_name = f"detection_run_{image_path.stem}"

        # YOLOv8의 predict 기능을 사용하여 TXT와 이미지를 자동으로 저장합니다.
        results = self.model.predict(
            source=image_path,
            # data.yaml 경로를 명시적으로 전달 (클래스 이름 로딩에 도움)
            data=str(self.data_yaml_path),
            device='cpu',
            save=True,  # 기본 시각화 이미지 저장
            save_txt=True,  # YOLO TXT 라벨 파일 저장
            save_conf=True,  # TXT 파일에 신뢰도 포함 (세그멘테이션은 기본적으로 신뢰도 포함)
            project=str(output_base_dir),
            name=temp_output_name,
            verbose=False,
            # 이미지당 하나의 배치이므로 batch=1 설정 불필요
        )

        # YOLO가 저장한 실제 경로를 파악합니다.
        if not results or not hasattr(results[0], 'save_dir'):
            print("[ERROR] YOLO 예측 결과 객체에서 저장 경로를 찾을 수 없습니다.", file=sys.stderr)
            return False

        save_dir = Path(results[0].save_dir)
        label_dir = save_dir / "labels"

        if not save_dir.is_dir():
            print("[ERROR] YOLO 예측 결과 폴더 생성 실패.", file=sys.stderr)
            return False

        # 2. TXT 파일 경로 파악
        txt_file_name = f"{image_path.stem}.txt"
        temp_txt_path = label_dir / txt_file_name

        if not temp_txt_path.is_file():
            print(f"[INFO] 탐지된 객체가 없거나 TXT 파일 저장에 실패했습니다. (경로: {temp_txt_path})", file=sys.stderr)
            # 탐지된 객체가 없더라도 기본 시각화 이미지는 있을 수 있으므로 실패로 처리하지 않음
            return True

        print(f"\n✅ 탐지 및 저장 완료.")
        print(f"  TXT 라벨: {temp_txt_path}")

        return True


# ----------------------------------------------------------------------
# 2. 모듈 테스트 (선택적)
# ----------------------------------------------------------------------
if __name__ == '__main__':
    # 🚨 경로 설정 (실제 경로로 수정 필요)

    # 1. 학습된 모델 및 data.yaml 경로
    MODEL_PATH = Path("models/best.pt")
    DATA_YAML_PATH = Path("data.yaml")

    # 2. 테스트 이미지 경로
    TEST_IMAGE_PATH = Path("data/inputs/test_img.jpg")

    # 3. 결과 저장 경로
    OUTPUT_BASE_DIR = Path("data/outputs/detection_results")

    # 더미 파일 생성 (경로만 맞   추기 위해)
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    DATA_YAML_PATH.touch(exist_ok=True)
    if not MODEL_PATH.is_file():
        print(f"[WARN] {MODEL_PATH.name} 더미 파일을 생성합니다. 실제 가중치 파일을 넣어주세요.")
        MODEL_PATH.write_text("Dummy model file")

    # TEST_IMAGE_PATH의 부모 디렉토리가 없으면 오류나므로, 존재하지 않으면 더미 이미지 생성
    if not TEST_IMAGE_PATH.is_file():
        print(f"[WARN] {TEST_IMAGE_PATH.name} 파일을 찾을 수 없습니다. 임시 더미 이미지를 사용합니다.")
        TEST_IMAGE_PATH.parent.mkdir(parents=True, exist_ok=True)
        dummy_img = np.zeros((640, 640, 3), dtype=np.uint8)
        cv2.putText(dummy_img, "DUMMY IMAGE", (100, 320), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)
        cv2.imwrite(str(TEST_IMAGE_PATH), dummy_img)

    try:
        # 모듈 인스턴스 생성
        detector = DetectionModule(MODEL_PATH, DATA_YAML_PATH)

        # 탐지 및 저장 실행
        detector.detect_and_save(TEST_IMAGE_PATH, OUTPUT_BASE_DIR)

    except (FileNotFoundError, RuntimeError) as e:
        print(f"\n[FATAL] 모듈 실행 실패: {e}")
        print("경로 설정을 확인하고 실제 best.pt 및 data.yaml 파일을 준비해주세요.")