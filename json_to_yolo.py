import json
import argparse
from pathlib import Path
from typing import Dict, Any, List

import sys

# COCO 클래스 개수를 상수로 정의 (YOLO TXT ID를 COCO 이후로 매핑하기 위해)
COCO_CLASS_COUNT = 80

# ====================================================
# 터미널 사용법
# python json_to_yolo.py --json-dir "path/to/labels/json/dir"
# ====================================================

def parse_args() -> argparse.Namespace:
    """CLI 인자를 정의하고 파싱한다."""
    parser = argparse.ArgumentParser(
        description="Convert COCO-style Segmentation JSON to YOLOv8 Segmentation TXT format.",
    )
    # parser.add_argument(
    #     "--json-file",
    #     type=Path,
    #     required=True,
    #     help="Path to the source JSON label file.",
    # )
    parser.add_argument(
        "--json-dir",
        type=Path,
        required=True,
        help="Path to the directory containing source JSON label files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory to save the YOLO TXT label files. Defaults to --json-dir.",
    )
    return parser.parse_args()


def normalize_segmentation(segmentation: List[int | float], width: int, height: int) -> List[str]:
    """
    픽셀 단위의 segmentation 좌표 리스트를 이미지 크기로 정규화한다.
    YOLO TXT 형식에 맞게 문자열 리스트로 반환한다.
    """
    normalized_coords: List[str] = []

    # segmentation 리스트는 [x1, y1, x2, y2, ..., xn, yn] 형태로 짝수 길이여야 함
    if len(segmentation) % 2 != 0:
        print("경고: Segmentation 좌표가 홀수입니다. 건너뜁니다.")
        return []

    for i in range(0, len(segmentation), 2):
        x = segmentation[i]
        y = segmentation[i + 1]

        # X 좌표 정규화 (0.0 ~ 1.0)
        norm_x = x / width

        # Y 좌표 정규화 (0.0 ~ 1.0)
        norm_y = y / height

        # 소수점 6자리까지 반올림하여 문자열로 저장 (정밀도 유지)
        normalized_coords.append(f"{norm_x:.6f}")
        normalized_coords.append(f"{norm_y:.6f}")

    return normalized_coords


def map_category_id_to_index(categories: List[Dict[str, Any]], annotation_id: int) -> int | None:
    """
    어노테이션의 category_id를 YOLO의 80부터 시작하는 클래스 인덱스로 변환한다.
    """

    # 모든 category_id를 문자열로 통일하여 맵핑을 만듭니다.
    # 제공된 categories 리스트 순서대로 80, 81, 82... 인덱스를 얻습니다.
    # cabinet(80), drawers(81), ..., DEFWALL(128)
    # custom_id_map = {str(cat.get("id")): index for index, cat in enumerate(categories)}
    # target_id_str = str(annotation_id)
    # custom_index = custom_id_map.get(target_id_str)
    #
    # if custom_index is not None:
    #     # 사용자 정의 인덱스(0-48)에 COCO 클래스 개수(80)를 더하여
    #     # 최종 YOLO ID (80-128)를 계산합니다.
    #     yolo_class_id = custom_index + COCO_CLASS_COUNT
    #     return yolo_class_id
    #
    # return None
    return annotation_id + COCO_CLASS_COUNT - 1

def process_json_to_yolo_txt(json_path: Path, output_dir: Path) -> bool:
    """단일 JSON 파일을 읽고 YOLO TXT 형식으로 변환하여 저장한다."""
    print(f"JSON 파일 로드 중: {json_path}")

    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"[ERROR] JSON 파일을 찾을 수 없습니다: {json_path}", file=sys.stderr)
        return False
    except json.JSONDecodeError:
        print(f"[ERROR] JSON 파일 형식이 올바르지 않습니다: {json_path}", file=sys.stderr)
        return False

    # JSON에서 필요한 정보 추출
    image_info = data.get("image", {})
    annotations = data.get("annotations", [])
    categories = data.get("categories", [])

    image_width = image_info.get("width")
    image_height = image_info.get("height")
    file_name = image_info.get("file_name")

    if not all([image_width, image_height, file_name]):
        print("[ERROR] JSON에 'image' 정보(width, height, file_name)가 부족합니다.", file=sys.stderr)
        return False

    # 출력 디렉터리 생성
    output_dir.mkdir(parents=True, exist_ok=True)

    # TXT 파일명 결정 (원본 이미지 파일명에서 확장자만 .txt로 변경)
    yolo_label_name = Path(file_name).stem + ".txt"
    output_path = output_dir / yolo_label_name

    yolo_lines: List[str] = []

    for annotation in annotations:
        category_id = annotation.get("category_id")
        if category_id >= 47:
            continue

        segmentation = annotation.get("segmentation")  # 픽셀 단위 좌표 리스트 [x1, y1, x2, y2, ...]

        if not all([category_id, segmentation]):
            print(f"[WARN] 어노테이션 ID {annotation.get('id')}에 category_id 또는 segmentation 데이터가 부족하여 건너뜁니다.")
            continue

        # 1. 클래스 ID를 YOLO 인덱스(0부터 시작)로 변환
        class_index = map_category_id_to_index(categories, category_id)
        if class_index is None:
            print(f"[WARN] category_id {category_id}에 해당하는 클래스 인덱스를 찾을 수 없습니다. 건너뜁니다.")
            continue

        # 2. 픽셀 좌표를 정규화 좌표로 변환
        normalized_coords = normalize_segmentation(segmentation, image_width, image_height)
        if not normalized_coords:
            continue

        # 3. YOLO TXT 형식 라인 생성 (클래스 ID + 정규화 좌표들)
        # 예: 49 0.739583 0.421296 0.739583 0.422222 ...
        yolo_line = f"{class_index} {' '.join(normalized_coords)}"
        yolo_lines.append(yolo_line)

    if not yolo_lines:
        print(f"[WARN] {file_name}에 유효한 segmentation 어노테이션이 없어 TXT 파일 생성을 건너뜁니다.")
        return False

    # 4. TXT 파일 저장
    output_path.write_text('\n'.join(yolo_lines), encoding="utf-8")

    print(f"\n✅ 성공적으로 변환 및 저장되었습니다.")
    print(f"  원본 JSON: {json_path.name}")
    print(f"  YOLO TXT: {output_path.resolve()}")
    print(f"  변환된 어노테이션 수: {len(yolo_lines)}")

    # 참고: YOLO 학습에 필요한 data.yaml 파일 생성을 위해 categories 정보를 기록합니다.
    print("\n💡 YOLO data.yaml 파일 생성을 위해 아래 categories 정보를 사용하세요:")
    class_names = [cat.get("name") for cat in categories]
    print(f"  names: {class_names}")

    return True


def main() -> int:
    """스크립트 진입점: 배치 처리를 수행한다."""
    args = parse_args()

    json_dir: Path = args.json_dir
    if not json_dir.is_dir():
        print(f"[ERROR] JSON 디렉터리를 찾을 수 없습니다: {json_dir}", file=sys.stderr)
        return 1

    # 출력 디렉터리가 지정되지 않았다면, 입력 디렉터리(json_dir)로 설정합니다.
    output_dir: Path = args.output_dir if args.output_dir else json_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"JSON 파일 검색 중: {json_dir}")
    print(f"TXT 파일 저장 경로: {output_dir.resolve()}")

    json_files = list(json_dir.glob("*.json"))
    if not json_files:
        print("[WARN] 지정된 디렉터리에서 JSON 파일을 찾을 수 없습니다. 경로를 확인하세요.", file=sys.stderr)
        return 0

    total_processed = 0
    total_success = 0

    for json_file in json_files:
        if process_json_to_yolo_txt(json_file, output_dir):
            total_success += 1
        total_processed += 1

    print("\n--- 처리 결과 ---")
    print(f"총 JSON 파일 수: {total_processed}")
    print(f"성공적으로 처리된 파일 수: {total_success}")

    return 0


if __name__ == "__main__":
    sys.exit(main())