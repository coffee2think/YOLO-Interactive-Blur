import cv2
import numpy as np
import sys
from pathlib import Path
from PIL import Image
from ultralytics import YOLO


# Chapter 3의 JSON 스키마를 따르는 데이터 구조 정의
# 웹 연동 시 이 구조를 JSON으로 직렬화하여 사용합니다.

class DetectionModule:
    """
    이미지 분석 실습의 데이터 파이프라인을 통합한 객체 탐지 모듈.
    - YOLOv8n 모델 사용 (model=yolov8n.pt)
    - CPU 모드 강제 실행
    - 출력: 시각화된 PIL Image + Ch3 JSON 스키마 형태의 탐지 데이터
    """

    # YOLO 모델 인스턴스는 한 번만 로드하도록 클래스 변수로 관리 (웹 서버 확장성 고려)
    MODEL = None

    def __init__(self, model_name: str = '../yolov8n.pt'):
        """
        DetectionModule 초기화. 모델은 최초 1회만 로드합니다.

        Args:
            model_name (str): 사용할 YOLO 모델 파일 경로 (예: 'yolov8n.pt' 또는 'yolov8n-seg.pt').
        """
        if DetectionModule.MODEL is None:
            print(f"YOLO 모델 로드 중: {model_name} (CPU 강제)...")
            try:
                # GPU 리소스 제한을 고려하여 device='cpu'를 명시적으로 지정
                DetectionModule.MODEL = YOLO(model_name)
                print("모델 로드 완료 (CPU 모드).")
            except Exception as e:
                print(f"모델 로드 중 오류 발생: {e}", file=sys.stderr)
                DetectionModule.MODEL = None

        self.model = DetectionModule.MODEL

        # Segmentation 모델 사용 여부 확인 (블러 확장 방안 대비)
        self.is_seg_model = '-seg' in model_name

    def detect_objects(self, image_path: str | Path):
        """
        이미지 파일 경로에서 객체를 탐지하고 결과를 구조화하여 반환합니다.

        Args:
            image_path (str | Path): 탐지할 이미지 파일의 경로. (예: 'data/inputs/image.jpg')

        Returns:
            tuple: (visualized_image_pil, structured_data)
                   - visualized_image_pil (PIL.Image or None): 탐지 결과가 시각화된 PIL 이미지 객체.
                   - structured_data (dict or None): Chapter 3 JSON 스키마를 따르는 Python 딕셔너리.
        """
        if self.model is None:
            print("오류: YOLO 모델이 로드되지 않았습니다.", file=sys.stderr)
            return None, None

        image_path = Path(image_path)
        if not image_path.is_file():
            print(f"오류: 이미지 파일 경로를 찾을 수 없습니다: {image_path}", file=sys.stderr)
            return None, None

        # 1. 객체 탐지 실행 (CPU 강제)
        # result 객체는 감지 이미지, 바운딩 박스, 클래스 ID, 신뢰도 등을 포함합니다.
        results = self.model(
            source=image_path,
            device='cpu',
            verbose=False,
            # YOLO CLI 실습 플로우를 따라 TXT 라벨 저장을 흉내내고 데이터 파싱 (save_txt=True)
            save_txt=False,  # 실제 파일 저장은 이 모듈에서 직접 처리하지 않음
            save_conf=True
        )

        if not results:
            print("탐지 결과가 없습니다.")
            return None, None

        result = results[0]  # 단일 이미지이므로 첫 번째 결과만 사용

        # 2. 이미지 정보 로드 (Ch3 JSON 스키마의 width/height 확보)
        # result.orig_img를 사용해 원본 이미지의 크기를 확보합니다.
        height, width = result.orig_img.shape[:2]

        # 3. 탐지 결과 파싱 및 JSON 스키마 구조화 (export_detections.py/refine_detections.py의 목적 통합)
        detections_list = []

        # classes 이름 매핑 정보를 가져옵니다. (refine_detections.py 참고)
        class_names_map = self.model.names

        # result.boxes.data는 [x1, y1, x2, y2, conf, cls] 형태의 텐서를 포함합니다.
        for i, box in enumerate(result.boxes):
            # 픽셀 좌표 (x1, y1, x2, y2)
            xyxy_pixels = [int(val) for val in box.xyxy[0].tolist()]
            x1, y1, x2, y2 = xyxy_pixels

            # 정규화 좌표 (cx, cy, w, h) - YOLO TXT 포맷과 유사
            # box.xywhn[0].tolist()는 정규화된 cx, cy, w, h를 반환
            cx_norm, cy_norm, w_norm, h_norm = [round(val, 4) for val in box.xywhn[0].tolist()]

            confidence = round(box.conf[0].item(), 4)
            class_id = int(box.cls[0].item())
            class_name = class_names_map.get(class_id, 'unknown')

            # Segmentation 데이터 추출 (확장 방안)
            segmentation_mask_data = None
            if self.is_seg_model and result.masks and i < len(result.masks.data):
                # 블러 처리를 위해 마스크 텐서를 NumPy 배열로 변환
                # (H, W) 형태의 이진 마스크 (0 또는 1)
                mask_tensor = result.masks.data[i]
                segmentation_mask_data = mask_tensor.cpu().numpy().tolist()  # 리스트 형태로 직렬화 준비

            detection_record = {
                # Chapter 3 JSON 스키마 필드
                "object_id": i,
                "class_id": class_id,
                "class_name": class_name,
                "confidence": confidence,
                "bbox": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},  # 픽셀 좌표
                "bbox_norm": {"cx": cx_norm, "cy": cy_norm, "w": w_norm, "h": h_norm},  # 정규화 좌표
                # 블러 처리 확장을 위한 추가 필드
                "segmentation_mask_data": segmentation_mask_data if segmentation_mask_data else None
            }
            detections_list.append(detection_record)

        # 4. 시각화된 이미지 생성
        # result.plot()을 사용하여 바운딩 박스, 라벨이 그려진 이미지 (NumPy BGR)를 얻습니다.
        annotated_frame_bgr = result.plot()

        # 탐지된 객체에 번호를 시각적으로 추가
        for det in detections_list:
            obj_id = det['object_id']
            x1, y1 = det['bbox']['x1'], det['bbox']['y1']
            label = f"#{obj_id}"
            
            # Bbox 내부 좌상단에 번호를 추가합니다.
            text_origin = (x1 + 5, y1 + 30)
            cv2.putText(annotated_frame_bgr, label, text_origin, 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2, cv2.LINE_AA)

        annotated_frame_rgb = cv2.cvtColor(annotated_frame_bgr, cv2.COLOR_BGR2RGB)
        visualized_image_pil = Image.fromarray(annotated_frame_rgb)

        # 5. 최종 구조화된 데이터 생성 (Ch3 JSON 스키마를 따름)
        structured_data = {
            "image": image_path.name,
            "width": width,
            "height": height,
            "detections": detections_list,
            # 메타 정보는 refine_detections.py의 형식을 따를 수 있으나, 여기서는 생략
            # "meta": {"num_detections": len(detections_list), ...}
        }

        return visualized_image_pil, structured_data


# ----------------------------------------------------------------------
# 모듈 테스트 (Ch1/Ch3의 스크립트 실행 플로우 참고)
if __name__ == '__main__':
    # 실습 폴더 구조 준수: data/inputs
    ROOT_DIR = Path.cwd()
    INPUT_DIR = ROOT_DIR / "data" / "inputs"
    OUTPUT_DIR = ROOT_DIR / "data" / "outputs"

    # 디렉터리 생성 (실습 플로우 참고)
    INPUT_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 🚨 테스트용 이미지 파일 경로 (실습 README의 'input.webp'를 가정)
    test_image_filename = 'input.jpg'
    test_image_path = INPUT_DIR / test_image_filename

    if not test_image_path.is_file():
        # 더미 이미지 생성 또는 사용자에게 요청
        print(f"'{test_image_path}' 파일이 존재하지 않습니다. 실제 이미지를 넣어주세요.")
        # 임시로 검은색 더미 이미지 생성
        dummy_img = np.zeros((640, 640, 3), dtype=np.uint8)
        cv2.putText(dummy_img, "PLACEHOLDER", (100, 320), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)
        cv2.imwrite(str(test_image_path), dummy_img)
        print("더미 이미지를 생성했습니다. 탐지 결과는 의미 없을 수 있습니다.")

    # 1. 모듈 인스턴스 생성 (yolov8n.pt 사용)
    # Segmentation 모델 테스트를 원하시면 'yolov8n-seg.pt'로 변경하세요.
    detector = DetectionModule(model_name='yolov8n.pt')

    # 2. 객체 탐지 수행
    print(f"\n'{test_image_path.name}' 이미지 분석 시작...")
    visualized_img_pil, structured_detections = detector.detect_objects(test_image_path)

    # 3. 결과 출력 및 저장

    # 3-1. 시각화 이미지 저장 (Ch2/Ch3 플로우 참고)
    output_image_path = OUTPUT_DIR / f"annotated_{test_image_filename.split('.')[0]}.jpg"
    if visualized_img_pil:
        visualized_img_pil.save(output_image_path)
        print(f"\n✅ 시각화된 이미지가 '{output_image_path}'에 저장되었습니다.")

    # 3-2. 정제된 감지 JSON 데이터 출력 및 저장 (Ch3 플로우 참고)
    import json

    output_json_path = OUTPUT_DIR / "detections_refined.json"

    if structured_detections:
        # JSON 포맷으로 직렬화하여 저장
        with open(output_json_path, 'w', encoding='utf-8') as f:
            json.dump(structured_detections, f, indent=2, ensure_ascii=False)
        print(f"\n✅ 정제된 탐지 데이터가 '{output_json_path}'에 저장되었습니다.")

        # 콘솔 출력 (detection_summary.py의 요약 기능 대용)
        print("\n=== 정제된 탐지 결과 요약 (Structured Data) ===")
        print(
            f"이미지: {structured_detections.get('image')}, 크기: {structured_detections.get('width')}x{structured_detections.get('height')}")
        for i, det in enumerate(structured_detections['detections']):
            seg_status = " (Segmentation O)" if det.get('segmentation_mask_data') else ""
            print(
                f"  - #{i}: [{det['class_id']}] {det['class_name']}: Conf={det['confidence']:.4f}, "
                f"Pixel Box={det['bbox']}"
                f"{seg_status}"
            )
    else:
        print("\n❌ 탐지된 객체가 없거나 데이터 구조화에 실패했습니다.")
