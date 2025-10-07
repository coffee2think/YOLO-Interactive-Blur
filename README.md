# YOLO-Interactive-Blur: 객체 탐지 및 선택적 블러 처리 시스템

## 💡 프로젝트 개요

**YOLO-Interactive-Blur**는 이미지 내 객체에 대한 **선택적 프라이버시 필터링** 기능을 구현하는 프로젝트입니다.

이 프로젝트의 목적은 객체 탐지부터 사용자 선택 기반 블러 처리에 이르는 핵심 로직의 기능 구현(PoC, MVP)에 중점을 둡니다.

## 🔄 핵심 기능 및 프로세스

1. **이미지 입력**: 분석할 이미지 파일을 입력합니다.
2. **객체 탐지**: 입력된 이미지에서 파인튜닝한 `yolov8n` 모델을 사용하여 주요 객체를 탐지합니다.
3. **객체 지정**: 탐지된 객체 목록 중, 블러 처리가 필요한 객체를 지정합니다.
4. **선택적 블러링**: 지정된 객체의 영역에만 블러 필터를 적용하여 최종 결과 이미지를 출력합니다.

## 🛠️ 기술 스택 (Tech Stack)

| **구분**          | **기술**               | **설명**                                                                |
|-----------------|----------------------|-----------------------------------------------------------------------|
| **객체 탐지 모델**    | YOLOv8n Segmentation | 빠르고 경량화된 모델로 이미지 내 객체의 픽셀 단위 영역(세그멘테이션)을 탐지하고 추출하는 데 활용됩니다.           |
| **딥러닝 백엔드**     | PyTorch              | YOLOv8 모델의 학습 및 추론을 위한 기반 딥러닝 프레임워크입니다.                               |
| **AI/ML 라이브러리** | Ultralytics          | YOLOv8 모델의 로드, 학습, 탐지 실행 및 결과 관리를 담당하는 공식 라이브러리입니다.                   |
| **이미지 처리**      | OpenCV (`cv2`)       | 이미지 로드/저장, 세그멘테이션 마스크 생성, 가우시안 블러 필터 적용 등 핵심 익명화 로직 구현에 사용됩니다.        |
| **수치 연산**       | NumPy (`np`)         | 이미지 데이터를 다차원 배열로 처리하고, 좌표 변환 및 수학적 연산을 효율적으로 수행합니다.                   |
| **웹 프레임워크**     | Flask                | 사용자 이미지 업로드, 탐지 결과 표시, 객체 선택 및 최종 블러 이미지 출력을 담당하는 경량 웹 서비스 구현에 사용됩니다. |

## 📁 파일 및 폴더 구조 (Project Structure)
```markdown
YOLO-Interative-Blur/
├── data/
│   ├── inputs/             # 탐지/익명화할 원본 이미지 파일
│   └── outputs/            # 탐지 결과, TXT 라벨, 최종 익명화 이미지 저장 폴더
├── models/
│   ├── yolov8n-seg.pt      # YOLOv8 초기 가중치 파일 (학습용)
│   └── best.pt             # (★) 학습이 완료된 최종 가중치 파일 (탐지/서비스용)
├── runs/                   # YOLOv8 학습(`model_train.py`) 및 탐지 결과가 자동 저장되는 폴더
├── web/
│   ├── app.py              # (★) Flask 기반 웹 서비스 메인 스크립트
│   ├── templates/          # Flask 템플릿 파일 (HTML). 웹 페이지의 구조와 UI 정의
│   └── static/             # 웹에서 접근 가능한 정적 리소스 폴더 (CSS, JS, 이미지 등)
│       └── images/         # 웹 서비스에서 업로드된 이미지와 처리된 결과 이미지를 임시로 저장하여 웹에 표시하는 폴더
├── *.py
│   ├── move_files.py       # 데이터셋 구축을 위해 하위 폴더의 파일을 하나의 폴더로 평탄화하는 유틸리티 스크립트
│   ├── json_to_yolo.py     # 데이터셋 전처리 스크립트 (COCO/JSON 형식의 라벨을 YOLOv8 TXT 세그멘테이션 형식으로 변환)
│   ├── model_train.py      # 모델 학습용 스크립트
│   ├── detection_module.py # YOLO 모델 로드 및 탐지 실행 모듈
│   └── selective_blurrer.py# 마스크 생성 및 블러 적용 핵심 로직 모듈
├── data.yaml               # YOLO 학습/탐지에 사용되는 데이터셋 설정 파일 (클래스 정의 포함)
└── README.md
```

## 🚀 시작하기 (Getting Started)
### 1. 가상 환경 생성
```bash
python -m venv venv
```

### 2. 가상 환경 활성화
```bash
# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### 3. 패키지 설치
#### 기본 필수 패키지 설치
```bash
pip install ultralytics opencv-python numpy flask
```

#### (선택 사항) PyTorch 설치
YOLOv8은 모델 학습 시 PyTorch를 백엔드로 사용합니다.
- GPU를 사용할 경우 (권장):
```bash
# `PyTorch 공식 홈페이지 > Get Started` 로 이동하여
# `Run this Command`에 적혀있는 설치 명령어를 확인하십시오.
# 예시:
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu126
```
- CPU만 사용할 경우:
```bash
pip install torch torchvision
```

## 🌐 웹 서비스 실행
준비된 모델(`models/best.pt`)이 있다면, 웹 서비스를 구동하여 사용할 수 있습니다.
```bash
python web/app.py
# 구동 후 `Running on` 뒤에 나오는 주소(예: http://127.0.0.1:5000)로 접속합니다.
```

## 🛠️ 데이터셋 준비 및 모델 학습 (Data Preparation & Training)
웹 서비스에 사용되는 `models/best.pt` 파일을 직접 학습하거나 커스텀 데이터셋을 사용하고 싶다면 아래 단계를 따릅니다.

### 1. 데이터셋 다운로드 및 구성
이 프로젝트의 학습은 AI 허브에서 제공하는 `가상 실내 공간 3D 합성 데이터`를 사용했습니다.

#### 데이터셋(`가상 실내 공간 3D 합성 데이터`) 특징
- **특징**: 실제 촬영 환경이 아닌, 3D 환경에서 렌더링된 고품질의 합성 데이터입니다.
- **장점**: 실제 데이터 수집의 어려움과 프라이버시 문제를 해소하며, 다양한 각도와 조명 조건에서의 객체 데이터를 확보하여 모델의 일반화 능력을 향상시키는 데 기여합니다.
- **활용**: 주로 가구(Furniture), 특정 실내 물품(Objects) 등의 탐지 정확도를 높이는 데 중점을 두고 있습니다.

#### 다운로드 및 준비 절차
1. **AI 허브 접속**: [AI Hub](https://www.aihub.or.kr/)에 접속하여 "가상 실내 공간 3D 합성 데이터"를 검색 후 다운로드 받습니다.
2. **데이터 구성**: 다운로드 받은 데이터셋 파일을 프로젝트의 학습 구조에 맞게 구성합니다. 이 과정에서 필요한 경우 `move_files.py` 스크립트를 사용하여 이미지 파일과 라벨 파일들을 하나의 디렉터리로 평탄화할 수 있습니다.

### 2. 데이터 전처리 (JSON -> YOLO TXT)
프로젝트에서 사용하는 COCO-style JSON 세그멘테이션 라벨을 YOLOv8 학습 형식인 TXT 파일로 변환합니다.
변환된 라벨 파일은 원본 JSON과 동일한 디렉터리에 생성됩니다.

```bash
# JSON 라벨이 포함된 디렉터리를 지정하여 실행
python json_to_yolo.py --json-dir "path/to/your/labels/json/dir" 

# 참고: --output-dir 옵션을 사용하여 변환된 TXT 파일의 저장 경로를 별도로 지정할 수도 있습니다.
```

### 3. `data.yaml` 파일 준비
학습 이미지 경로, 클래스 개수(nc), 클래스 목록(names) 등이 정의된 `data.yaml` 파일을 프로젝트 루트 경로에 준비합니다.

### 4. 모델 학습
`data.yaml`과 베이스 모델(`models/yolov8n-seg.pt`)이 준비되었다면 학습을 시작합니다.

```bash
python model_train.py

# 또는 YOLO CLI 명령어로 직접 학습 파라미터를 지정하여 실행
# (Windows의 경우 Command Prompt / PowerShell 환경에 따라 \ 대신 ^ 또는 한 줄로 입력)
# yolo task=segment mode=train \
#   model=models/yolov8n-seg.pt \
#   data=data.yaml \
#   epochs=300 \
#   imgsz=640 \
#   batch=16 \
#   name=custom_seg_final \
#   device=0
```
⭐ **중요:** 학습이 완료되면, 생성된 가중치 파일 `runs/segment/custom_seg_final/weights/best.pt`를 `models/best.pt` 경로로 복사하여 웹 서비스가 사용하도록 설정해야 합니다.
