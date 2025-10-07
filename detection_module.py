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

    def _parse_yolo_txt_for_object_id(self, txt_path: Path) -> List[Tuple[int, List[str]]]:
        """
        저장된 YOLO TXT 파일을 읽어 object_id(index)를 부여한다.

        Returns:
            [(object_id, [class_id, norm_x1, norm_y1, ...]), ...]
        """
        if not txt_path.is_file():
            return []

        lines = txt_path.read_text(encoding='utf-8').strip().split('\n')

        parsed_data = []
        for i, line in enumerate(lines):
            if line.strip():
                # TXT 라인: class_id x1 y1 x2 y2 ...
                parts = line.split()
                if len(parts) >= 3 and len(parts) % 2 == 1:
                    # object_id = i (0부터 시작)
                    # class_id와 정규화 좌표를 문자열 리스트로 저장
                    parsed_data.append((i, parts))
        return parsed_data

    def _visualize_with_object_id(self, image_path: Path, output_image_path: Path,
                                  parsed_data: List[Tuple[int, List[str]]]):
        """
        YOLO TXT 데이터와 object_id를 사용하여 이미지에 세그멘테이션과 라벨을 그린다.
        (visualize_segmentation.py의 로직 재활용)
        """
        # model.names를 사용하여 클래스 이름을 가져오도록 합니다.
        class_names_map = self.model.names

        image = cv2.imread(str(image_path))
        if image is None:
            print(f"[ERROR] 이미지 로드 실패: {image_path.name}", file=sys.stderr)
            return

        height, width = image.shape[:2]
        overlay = image.copy()
        ALPHA = 0.5

        # 색상 설정 (모델이 로드한 클래스 개수만큼 랜덤 색상 생성)
        COLORS = [tuple(random.randint(0, 255) for _ in range(3)) for _ in range(len(class_names_map))]

        for object_id, parts in parsed_data:
            try:
                class_id = int(parts[0])
                normalized_coords = [float(p) for p in parts[1:]]

                class_name = class_names_map.get(class_id, f"Unknown_{class_id}")
                color_bgr = COLORS[class_id % len(COLORS)]

                # 픽셀 좌표로 변환 및 다각형 재구성
                pixel_coords = []
                x_coords = []
                y_coords = []
                for i in range(0, len(normalized_coords), 2):
                    x = int(normalized_coords[i] * width)
                    y = int(normalized_coords[i + 1] * height)
                    pixel_coords.append([x, y])
                    x_coords.append(x)
                    y_coords.append(y)

                if not pixel_coords: continue
                polygon = np.array([pixel_coords], dtype=np.int32)

                # 바운딩 박스 계산 (라벨 위치)
                x_min, y_min = min(x_coords), min(y_coords)

                # a. Segmentation 마스크 채우기
                cv2.fillPoly(overlay, polygon, color=color_bgr)

                # b. 라벨 텍스트 생성 (object_id 포함)
                label_text = f"#{object_id}. {class_name}"

                # c. 라벨 위치 및 배경 그리기
                label_size = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                text_w, text_h = label_size[0]
                text_x = x_min
                text_y = y_min - 10 if y_min > text_h + 10 else y_min + text_h + 10

                cv2.rectangle(overlay, (text_x, text_y - text_h - 5), (text_x + text_w, text_y + 5), color_bgr, -1)
                cv2.putText(overlay, label_text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            except Exception as e:
                print(f"[ERROR] 시각화 중 오류 발생: {e}", file=sys.stderr)
                continue

        # 블렌딩 및 저장
        final_image = cv2.addWeighted(image, 1 - ALPHA, overlay, ALPHA, 0)
        cv2.imwrite(str(output_image_path), final_image)
        print(f"✅ 시각화 이미지 저장: {output_image_path.name}")

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

        # 3. TXT 파일 읽기 및 object_id 부여
        parsed_data_with_id = self._parse_yolo_txt_for_object_id(temp_txt_path)

        # 4. object_id가 포함된 커스텀 시각화 이미지 저장
        # YOLO 기본 저장 이미지와 이름이 겹치지 않게 'custom_' 접두사 사용
        output_image_path = save_dir / f"custom_{image_path.name}"
        self._visualize_with_object_id(image_path, output_image_path, parsed_data_with_id)

        print(f"\n✅ 탐지 및 저장 완료.")
        print(f"  TXT 라벨: {temp_txt_path}")
        print(f"  시각화: {output_image_path}")

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