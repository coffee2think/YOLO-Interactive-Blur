"""
YOLO 세그멘테이션 TXT 결과로부터 객체를 선택적으로 블러 처리하는 모듈.
세그멘테이션 마스크를 기반으로 정교하게 익명화(블러)를 적용합니다.
"""

# --------------------------------------------------------------------------------
# [실행 방법]
# 터미널에서 다음 명령어를 사용하여 실행합니다.
# 이 스크립트는 YOLOv8 세그멘테이션 TXT 파일, 원본 이미지를 필수 인자로 요구합니다.
#
# <Windows/Linux/macOS (PowerShell/Bash/Zsh)>
# python selective_blurrer.py \
#   --source-image data/inputs/input.jpg \
#   --source-label data/outputs/detection_results/detection_run_input/labels/input.txt \
#   --model-path models/best.pt \
#   --output-dir data/outputs/detection_results \
#   --blur-kernel 31
#
# * 명령어의 '\' (Windows에서는 '^')는 줄바꿈 기호이며, 한 줄로 연결하여 입력해도 됩니다.
# * --source-image : 블러 처리할 원본 이미지 파일의 경로입니다. 이 이미지를 로드하여 블러를 적용합니다.
# * --source-label : YOLO 탐지 결과가 담긴 TXT 레이블 파일의 경로입니다.
#                    이 파일에서 객체의 다각형 좌표와 신뢰도를 읽어옵니다.
# * --model-path: 클래스 이름(class name)을 가져오기 위한 가중치 파일(.pt)의 경로 인자입니다. (선택 사항, 기본값: models/best.pt)
# * --output-dir : 블러 처리된 최종 결과 이미지를 저장할 디렉토리 경로입니다. (선택 사항, 기본값: data/outputs/detection_result)
# * --blur-kernel: 블러 강도(홀수 정수)를 지정합니다 (선택 사항, 기본값: 31).
# --------------------------------------------------------------------------------

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import NamedTuple, Iterable

import cv2
import numpy as np
from ultralytics import YOLO

# 마스크 결과를 저장할 기본 폴더
DEFAULT_OUTPUT_DIR = Path("data/outputs/detection_results")
DEFAULT_BLUR_KERNEL = 31  # 기본 블러 강도: 31x31 커널 사용 (홀수 양의 정수)
DEFAULT_MODEL_PATH = Path("models/best.pt")

class SegDetection(NamedTuple):
    """단일 세그멘테이션 감지 정보를 담는 구조체."""
    class_id: int
    confidence: float
    polygon_norm: list[float]  # 정규화된 [x1, y1, x2, y2, ...] 리스트


def parse_args() -> argparse.Namespace:
    """CLI 인자를 정의하고 파싱한다."""
    parser = argparse.ArgumentParser(
        description="Apply selective Gaussian blur based on YOLO segmentation masks.",
    )
    parser.add_argument(
        "--source-image",
        type=Path,
        required=True,
        help="Input image path to be processed.",
    )
    parser.add_argument(
        "--source-label",
        type=Path,
        required=True,
        help="YOLO segmentation TXT label file path.",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=DEFAULT_MODEL_PATH,
        help="Path to the YOLO model file (e.g., best.pt) to load class names.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for saving the anonymized image.",
    )
    parser.add_argument(
        "--blur-kernel",
        type=int,
        default=DEFAULT_BLUR_KERNEL,
        help="Gaussian blur kernel size (odd positive integer). Default is 21.",
    )
    return parser.parse_args()


def validate_kernel(kernel: int) -> None:
    """블러 커널 조건을 검증한다."""
    if kernel <= 0 or kernel % 2 == 0:
        raise ValueError(f"--blur-kernel must be an odd positive integer, got {kernel}.")


def load_image(path: Path) -> np.ndarray:
    """이미지를 로드하고 실패 시 예외를 발생시킨다."""
    image = cv2.imread(str(path))
    if image is None:
        raise FileNotFoundError(f"Unable to load image: {path}")
    return image

def load_class_names(model_path: Path) -> dict[int, str]:
    """YOLO 모델 파일에서 클래스 ID-이름 매핑을 로드한다."""
    if not model_path.is_file():
        raise FileNotFoundError(f"YOLO model file not found: {model_path}")
    try:
        model = YOLO(str(model_path))
        return model.names
    except Exception as e:
        print(f"[ERROR] Failed to load YOLO model or class names from {model_path}: {e}", file=sys.stderr)
        raise RuntimeError("Model loading failed.") from e


def parse_seg_labels(label_path: Path) -> list[SegDetection]:
    """YOLO 세그멘테이션 TXT 파일을 파싱한다."""
    detections: list[SegDetection] = []
    if not label_path.is_file():
        return detections

    for line in label_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            parts = line.strip().split()

            # YOLO Seg TXT Format: class_id x1 y1 x2 y2 ... conf
            # 최소 길이 검사: class_id (1) + conf (1) + 최소 3쌍의 좌표 (6) = 8
            if len(parts) < 8:
                print(f"[WARN] Skipping line due to insufficient parts: {line[:50]}...", file=sys.stderr)
                continue

            class_id = int(parts[0])
            confidence = float(parts[-1])
            polygon_norm = [float(p) for p in parts[1:-1]]

            if len(polygon_norm) < 6 or len(polygon_norm) % 2 != 0:
                print(f"[WARN] Skipping malformed polygon in line: {line[:50]}...", file=sys.stderr)
                continue

            detections.append(SegDetection(class_id, confidence, polygon_norm))
        except (ValueError, IndexError) as err:
            print(f"[WARN] Skipping malformed line in {label_path}: {err}", file=sys.stderr)

    # 신뢰도 내림차순으로 정렬
    detections.sort(key=lambda d: d.confidence, reverse=True)
    return detections


def get_user_selection(detections: list[SegDetection], class_names: dict[int, str]) -> list[int]:
    """감지 목록을 출력하고 사용자로부터 블러 처리할 항목을 입력받는다."""
    print("\n--- Detected Objects (Sorted by Confidence Descending) ---")
    if not detections:
        print("No valid detections found to process.")
        return []

    for i, det in enumerate(detections):
        class_name = class_names.get(det.class_id, f"Unknown ({det.class_id})")

        print(
            f"[{i + 1:2d}] {class_name:15s}"
            f" | Conf: {det.confidence:.4f}"
            # f" | Points: {len(det.polygon_norm) // 2}"
        )

    print("\nEnter item numbers to blur (e.g., 1,3,5) or press Enter to skip all:")
    raw_input = input("> ").strip()
    if not raw_input:
        return []

    try:
        selected_indices = []
        for part in raw_input.split(','):
            item_num = int(part.strip())
            if 1 <= item_num <= len(detections):
                selected_indices.append(item_num - 1)
            else:
                print(f"[WARN] Item number {item_num} is out of range. Skipping.", file=sys.stderr)
        return list(set(selected_indices))  # 중복 제거
    except ValueError:
        print("[ERROR] Invalid input format. Please enter comma-separated numbers.", file=sys.stderr)
        return []


def create_mask_and_apply_blur(
        image: np.ndarray,
        detections: list[SegDetection],
        selected_indices: list[int],
        kernel_size: int
) -> np.ndarray:
    """선택된 감지 객체에 대해 마스크를 생성하고 블러를 적용한다."""
    height, width = image.shape[:2]
    anonymized_image = image.copy()

    # 블러 처리된 이미지 전체 영역을 미리 계산 (반복 계산 방지)
    blurred_full = cv2.GaussianBlur(anonymized_image, (kernel_size, kernel_size), sigmaX=0)

    # 마스크를 적용할 다각형 목록
    polygons_to_mask = []
    for i in selected_indices:
        det = detections[i]
        # 정규화된 좌표를 픽셀 좌표로 변환
        pixel_coords = []
        for x, y in zip(det.polygon_norm[::2], det.polygon_norm[1::2]):
            # 좌표를 이미지 경계 내로 클램프
            px = int(np.clip(x * width, 0, width - 1))
            py = int(np.clip(y * height, 0, height - 1))
            pixel_coords.append((px, py))

        # NumPy 배열 형태로 변환 (cv2.fillPoly 요구 사항)
        polygons_to_mask.append(np.array(pixel_coords, dtype=np.int32).reshape((-1, 1, 2)))

    if not polygons_to_mask:
        return anonymized_image  # 선택된 객체 없음

    # 1. 블러 마스크 생성 (전체 이미지 크기의 단일 채널 마스크)
    mask = np.zeros((height, width), dtype=np.uint8)
    # 선택된 모든 다각형 영역을 흰색(255)으로 채움
    cv2.fillPoly(mask, polygons_to_mask, 255)

    # 2. 마스크된 블러 영역 추출 (원본 이미지 크기)
    # 마스크가 1채널이므로 3채널로 확장 (cv2.bitwise_and를 위해 필요)
    mask_3channel = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

    # 마스크된 블러 영역 = 블러된 이미지 & 마스크
    masked_blurred = cv2.bitwise_and(blurred_full, mask_3channel)

    # 3. 마스크된 원본 이미지 추출 (블러를 적용하지 않을 영역)
    # NOT 연산으로 마스크 반전 (블러 영역 제외)
    mask_inv = cv2.bitwise_not(mask)
    mask_inv_3channel = cv2.cvtColor(mask_inv, cv2.COLOR_GRAY2BGR)

    # 마스크된 원본 영역 = 원본 이미지 & 반전 마스크
    masked_original = cv2.bitwise_and(anonymized_image, mask_inv_3channel)

    # 4. 두 영역을 합쳐 최종 이미지 생성 (블러 + 원본)
    final_image = cv2.add(masked_blurred, masked_original)

    return final_image


def main() -> int:
    """스크립트 진입점: 선택적 익명화 파이프라인을 실행한다."""
    args = parse_args()

    try:
        validate_kernel(args.blur_kernel)
        image = load_image(args.source_image)
        detections = parse_seg_labels(args.source_label)
        class_names = load_class_names(args.model_path)
    except (FileNotFoundError, ValueError, RuntimeError) as err:
        print(f"[ERROR] {err}", file=sys.stderr)
        return 1

    if not detections:
        print(f"[INFO] No valid detections found in {args.source_label}. Exiting.", file=sys.stderr)
        return 0

    # 1. 사용자로부터 블러 처리할 객체 선택
    selected_indices = get_user_selection(detections, class_names)

    if not selected_indices:
        print("[INFO] No objects selected for blur. Exiting.", file=sys.stderr)
        return 0

    selected_item_numbers = sorted([i + 1 for i in selected_indices])
    index_string = "_".join(map(str, selected_item_numbers))

    print(f"\nApplying Gaussian blur (Kernel: {args.blur_kernel}) to {len(selected_indices)} selected objects...")

    # 2. 블러 처리 적용
    anonymized_image = create_mask_and_apply_blur(
        image,
        detections,
        selected_indices,
        args.blur_kernel
    )

    # 3. 결과 저장
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = args.source_image.stem
    detection_dir_name = f"detection_run_{stem}"
    detection_dir = output_dir / detection_dir_name
    output_path = detection_dir / f"{stem}_anonymized_{args.blur_kernel}_i{index_string}.png"

    if not cv2.imwrite(str(output_path), anonymized_image):
        print(f"[ERROR] Failed to write anonymized image: {output_path}", file=sys.stderr)
        return 1

    print("--- Anonymization Complete ---")
    print(f"  Input Image: {args.source_image}")
    print(f"  Output Image: {output_path}")
    print(f"  Objects Blurred: {len(selected_indices)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
