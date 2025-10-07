# ===============================================
# ✅ 터미널에서 직접 학습할 시 명령어
# -----------------------------------------------
# yolo task=segment mode=train ^
#   model=models/yolov8n-seg.pt ^
#   data=data.yaml ^
#   epochs=300 ^
#   imgsz=640 ^
#   batch=16 ^
#   name=custom_seg_final ^
#   device=0
# -----------------------------------------------
# ⚠️ 주의사항
# - GPU가 없을 경우 device=cpu로 변경
# - torch가 CPU 버전인지, CUDA(GPU) 버전인지 확인(예: torch-2.6.0+cpu, torch-2.8.0+cu126)
# ===============================================

import sys
from ultralytics import YOLO
from pathlib import Path

def train_yolo():
    """
    YOLOv8 세그멘테이션 모델을 학습하고, 학습 중 발생하는 주요 에러 처리
    """
    try:
        # 1. 모델 로드 (예: YOLOv8n, n은 nano 버전으로 가장 빠르고 작음)
        # 학습에 사용할 pt(pytorch) 파일 로드
        MODEL_PATH = Path("models/yolov8n-seg.pt")
        print(f"💡 모델 로드 중: {MODEL_PATH.name}")
        # model = YOLO('models/yolov8n.pt') # 바운딩박스 구조로 학습할 시
        model = YOLO(str(MODEL_PATH)) # 세그멘테이션 구조로 학습할 시

        # 2. 모델 학습 시작
        # data: data.yaml 파일 경로
        # epochs: 학습 반복 횟수
        # imgsz: 학습 시 이미지 크기 (기본값 640)
        # batch: 배치 크기 (GPU 메모리에 따라 조절. 8 또는 16 권장)
        print("🚀 모델 학습 시작...")
        results = model.train(
            data='data.yaml',
            epochs=300,
            imgsz=640,
            batch=16,
            name='custom_seg_final', # 결과 저장 폴더 이름
            device=0 # 장치 지정. CPU 사용시 cpu, GPU 사용시 cuda 혹은 GPU 인덱스(0부터 시작) 입력
        )

        # 3. 학습 완료 메시지 출력
        print("\n✅ 학습 완료!")
        print(f"결과는 기본 경로인 runs/segment/{results.name} 폴더에 저장되었습니다.")

    except RuntimeError as e:
        # GPU 메모리 부족 등 런타임 에러 처리
        if "CUDA out of memory" in str(e):
            print("\n❌ 에러 발생: GPU 메모리 부족 (CUDA Out of Memory)")
            print("  -> 해결 방법: 'batch' 인자 값을 16보다 작은 값 (예: 8 또는 4)으로 줄이거나, 'imgsz' 값을 줄여보세요.")
        else:
            print(f"\n❌ 치명적인 런타임 에러 발생: {e}")
            print("  -> 해결 방법: PyTorch와 CUDA 환경 설정 및 GPU 연결 상태를 확인해보세요.")
        sys.exit(1)  # 에러 발생 시 프로그램 종료

    except FileNotFoundError as e:
        # 파일 경로 관련 에러 처리 (data.yaml, model file 등)
        print(f"\n❌ 에러 발생: 필요한 파일을 찾을 수 없음 (FileNotFoundError)")
        print(f"  -> 원인: {e}")
        print(f"  -> 해결 방법: '{MODEL_PATH.name}' 파일과 'data.yaml' 파일의 경로를 확인하세요.")
        sys.exit(1)

    except Exception as e:
        # 그 외 예상치 못한 에러 처리
        print(f"\n❌ 예상치 못한 에러 발생: {e}")
        print("  -> 해결 방법: 에러 메시지를 검색하거나, 학습 파라미터를 다시 확인해보세요.")
        sys.exit(1)

if __name__ == '__main__':
    train_yolo()