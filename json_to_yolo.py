import json
import os
from ultralytics.data.converter import convert_

def convert_json_to_yolo_txt(json_data, output_dir="yolo_labels"):
    """
    COCO-like JSON 데이터에서 segmentation 정보를 기반으로 바운딩 박스를 계산하여
    YOLO 형식의 .txt 파일로 변환하고 저장합니다.
    """

    # 이미지 정보 추출
    image_info = json_data.get("image", {})
    image_width = image_info.get("width")
    image_height = image_info.get("height")
    file_name_base = os.path.splitext(image_info.get("file_name", "output"))[0]

    # 클래스 ID와 이름 매핑 (YOLO는 0부터 시작하는 순차적인 클래스 인덱스를 사용해야 함)
    categories = json_data.get("categories", [])

    # 클래스 ID (문자열)를 0부터 시작하는 순차적인 YOLO 클래스 인덱스로 매핑
    # 예: "1" -> 0, "6" -> 1, "7" -> 2 ...
    id_to_yolo_idx = {
        str(cat["id"]): i
        for i, cat in enumerate(categories)
    }
    # 클래스 이름 리스트 (dataset.yaml 파일에 사용)
    class_names = [cat["name"] for cat in categories]

    if not image_width or not image_height:
        print("ERROR: Image width or height is missing in the JSON data.")
        return

    # 출력 폴더 생성
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{file_name_base}.txt")

    yolo_lines = []

    # 1. Annotations 순회 및 바운딩 박스 계산
    for ann in json_data.get("annotations", []):
        category_id = str(ann.get("category_id"))  # JSON의 category_id (문자열)
        segmentation = ann.get("segmentation", [])  # [x1, y1, x2, y2, ...]

        if not segmentation:
            continue

        # YOLO 클래스 인덱스 (0, 1, 2, ...)
        yolo_class_index = id_to_yolo_idx.get(category_id)

        if yolo_class_index is None:
            print(f"WARNING: Unknown category_id {category_id} skipped.")
            continue

        # 2. Segmentation 좌표에서 바운딩 박스 (x_min, y_min, x_max, y_max) 계산
        # segmentation 리스트는 [x1, y1, x2, y2, ...] 형태이므로 짝수 인덱스는 x, 홀수 인덱스는 y
        x_coords = segmentation[0::2]
        y_coords = segmentation[1::2]

        x_min = min(x_coords)
        y_min = min(y_coords)
        x_max = max(x_coords)
        y_max = max(y_coords)

        # 3. YOLO 형식 (Normalized x_center, y_center, width, height)으로 변환

        # 픽셀 단위 계산
        bbox_width = x_max - x_min
        bbox_height = y_max - y_min
        x_center = x_min + bbox_width / 2
        y_center = y_min + bbox_height / 2

        # 정규화 (0.0 ~ 1.0)
        x_center_norm = x_center / image_width
        y_center_norm = y_center / image_height
        width_norm = bbox_width / image_width
        height_norm = bbox_height / image_height

        # YOLO .txt 형식: <class-index> <x_center> <y_center> <width> <height>
        # 소수점 6자리까지 사용 (일반적인 정확도 기준)
        yolo_line = f"{yolo_class_index} {x_center_norm:.6f} {y_center_norm:.6f} {width_norm:.6f} {height_norm:.6f}"
        yolo_lines.append(yolo_line)

    # 4. .txt 파일로 저장
    if yolo_lines:
        with open(output_path, 'w') as f:
            f.write("\n".join(yolo_lines))
        print(f"SUCCESS: {len(yolo_lines)} annotations converted and saved to {output_path}")
    else:
        print(f"WARNING: No valid annotations found for {file_name_base}. No output file generated.")

    return class_names


# --- 변환 함수 실행 예시 ---

# 제공된 JSON 데이터 (딕셔너리)
json_data = {
    "info": {
        "description": "가상 실내 공간 3D 합성 데이터",
        "version": "1.0.0",
        "year": 2023
    },
    "categories": [
        {"id": "1", "name": "cabinet", "type": "thing"},
        {"id": "6", "name": "chair", "type": "thing"},
        {"id": "7", "name": "desk", "type": "thing"},
        {"id": "25", "name": "aircondition", "type": "thing"},
        {"id": "41", "name": "ceilinglight", "type": "stuff"},
        {"id": "46", "name": "window", "type": "stuff"},
        {"id": "47", "name": "DEFCEIL", "type": "stuff"},
        {"id": "48", "name": "DEFFLOOR", "type": "stuff"},
        {"id": "49", "name": "DEFWALL", "type": "stuff"}
    ],
    "image": {
        "id": 12435520,
        "width": 1920,
        "height": 1080,
        "file_name": "etc_education_l_001_normal_0.jpg",
        "date_captured": "2023-10-27",
        "format": "jpg",
        "caption": "2인용 책상과 의자 2개가 한세트로 여러 세트가 방 안 가득 배치되어 있다."
    },
    "annotations": [
        {
            "id": "EMAzRhsRLZartMuEeAPLr",
            "category_id": 49,
            "segmentation": [1420, 455, 1420, 456, 1419, 457, 1419, 459, 1418, 460, 1418, 461, 1429, 461, 1430, 460,
                             1439, 460, 1438, 459, 1435, 459, 1434, 458, 1430, 458, 1429, 457, 1426, 457, 1425, 456,
                             1421, 456],
            "color": [185, 227, 222]
        },
        # 여기에 다른 annotations 데이터가 있다고 가정
    ]
}

# 변환 실행 (실제 JSON 파일을 로드하는 코드로 대체해야 함)
# json_file_path = "path/to/your/annotation.json"
# with open(json_file_path, 'r', encoding='utf-8') as f:
#     json_data = json.load(f)

class_names = convert_json_to_yolo_txt(json_data, output_dir="yolo_labels_output")

# --- dataset.yaml 파일 예시 ---
if class_names:
    print("\n✅ 다음은 YOLO 학습을 위한 dataset.yaml 파일의 'names' 섹션입니다:")
    print("names:")
    for i, name in enumerate(class_names):
        print(f"  {i}: {name}")