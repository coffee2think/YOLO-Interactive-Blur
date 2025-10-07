# web/app.py

import os
import sys
import shutil
from pathlib import Path
from flask import Flask, render_template, request, redirect, url_for, flash, session
from werkzeug.utils import secure_filename

# 상위 폴더(YOLO-Interative-Blur)의 모듈을 임포트하기 위해 sys.path 추가
# Flask 앱이 /web 폴더에서 실행되므로, 상위 경로를 추가해야 합니다.
sys.path.append(str(Path(__file__).parent.parent))

# 기존 모듈 임포트
try:
    from detection_module import DetectionModule
    from selective_blurrer import parse_seg_labels, create_mask_and_apply_blur, SegDetection, load_image, \
        validate_kernel
except ImportError as e:
    print(f"[FATAL] 모듈 임포트 오류: {e}")
    print("detection_module.py와 selective_blurrer.py가 상위 폴더에 있는지 확인하세요.")
    sys.exit(1)

# ----------------------------------------------------------------------
# 1. 초기 설정 및 전역 변수
# ----------------------------------------------------------------------

app = Flask(__name__)
# 세션 관리를 위해 SECRET_KEY 설정 (Flash 메시지, 임시 데이터 저장 등에 필요)
app.secret_key = 'cf665958b0a4c34cb2b53f57a7bcfafead05f6f27560d97b102d8da91348dabe4cc86725fd32011ce03996a35426f99ff550f4c1871e41d0e18dcd02c0ecdeee'

# 'YOLO-Interative-Blur' 프로젝트 루트 폴더를 기준으로 경로 설정
PROJECT_ROOT = Path(__file__).parent.parent
MODEL_PATH = PROJECT_ROOT / "models/best.pt"
DATA_YAML_PATH = PROJECT_ROOT / "data.yaml"

# 웹에서 접근 가능한 정적(Static) 폴더 내부 경로
STATIC_DIR = Path(app.static_folder)
if STATIC_DIR.name != 'static':
    STATIC_DIR = PROJECT_ROOT / 'web' / 'static'

UPLOAD_FOLDER = STATIC_DIR / 'images'
UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)  # 폴더 생성
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB 제한

# ----------------------------------------------------------------------
# 2. 전역 탐지 모듈 초기화 (한 번만 로드)
# ----------------------------------------------------------------------
try:
    # DetectionModule 인스턴스 생성 (모델 로딩을 한 번만 수행)
    detector = DetectionModule(MODEL_PATH, DATA_YAML_PATH)
    CLASS_NAMES = detector.model.names
except (FileNotFoundError, RuntimeError) as e:
    # 모델 로딩 실패 시 앱 실행 중단
    print(f"[FATAL] YOLO 모델 초기화 실패: {e}", file=sys.stderr)
    print(f"경로 확인: {MODEL_PATH} 및 {DATA_YAML_PATH}")
    sys.exit(1)


# ----------------------------------------------------------------------
# 3. Flask 라우팅
# ----------------------------------------------------------------------

@app.route('/', methods=['GET'])
def index():
    """Step 1: 이미지 업로드 폼."""
    return render_template('index.html')


@app.route('/detect', methods=['POST'])
def detect():
    """Step 2-1: 이미지 업로드 및 객체 탐지 수행."""
    if 'file' not in request.files:
        flash('파일이 없습니다.')
        return redirect(request.url)

    file = request.files['file']

    if file.filename == '':
        flash('선택된 파일이 없습니다.')
        return redirect(request.url)

    if file:
        filename = secure_filename(file.filename)
        # 업로드 폴더에 이미지 저장
        uploaded_path = app.config['UPLOAD_FOLDER'] / filename
        file.save(uploaded_path)

        # 1. 탐지 실행 및 결과 저장 (YOLO가 output_dir/detection_run_*/ 에 저장)
        output_base_dir = PROJECT_ROOT / "data" / "outputs" / "detection_results"
        detector.detect_and_save(uploaded_path, output_base_dir)

        # 2. 결과 경로 파악
        stem = uploaded_path.stem
        detection_run_dir_name = f"detection_run_{stem}"
        detection_dir = output_base_dir / detection_run_dir_name

        # TXT 라벨 파일 경로
        txt_file_path = detection_dir / "labels" / f"{stem}.txt"

        # 커스텀 시각화 이미지 경로 (웹에 표시할 이미지)
        # 이 이미지를 /web/static/images 폴더로 복사하여 웹에서 접근 가능하게 만듭니다.
        temp_visual_path = detection_dir / f"{uploaded_path.name}"
        visual_filename = f"{stem}_detected.jpg"  # 웹 표시용 파일명
        final_visual_path = app.config['UPLOAD_FOLDER'] / visual_filename

        # 파일이 존재하면 복사 (탐지 실패 시 TXT가 없을 수는 있음)
        if temp_visual_path.is_file():
            shutil.copy(temp_visual_path, final_visual_path)

        # 3. TXT 파싱 및 객체 목록 준비
        detections = parse_seg_labels(txt_file_path)

        # 4. 세션에 최소한의 정보만 저장
        session['source_image_path'] = str(uploaded_path)
        session['label_file_path'] = str(txt_file_path)
        session['kernel_size'] = 31  # 기본값 설정

        return render_template(
            'detection_result.html',
            visual_image_url=url_for('static', filename=f'images/{visual_filename}'),
            detections=detections,
            class_names=CLASS_NAMES,
            kernel_size=session['kernel_size']
        )


@app.route('/blur', methods=['POST'])
def blur():
    """Step 2-2: 선택된 객체에 블러 적용."""
    # 세션에서 필요한 데이터 로드
    source_image_path_str = session.get('source_image_path')
    label_file_path_str = session.get('label_file_path')
    kernel_size = int(request.form.get('blur_kernel', session.get('kernel_size', 31)))

    if not source_image_path_str or not label_file_path_str:
        flash('탐지 데이터가 세션에 없습니다. 처음부터 다시 시작해주세요.')
        return redirect(url_for('index'))

    # 💡 TXT 파일 경로를 사용하여 객체 목록을 다시 파싱합니다
    source_image_path = Path(source_image_path_str)
    label_file_path = Path(label_file_path_str)

    # TXT 파일을 읽어 detections 객체 복원
    detections = parse_seg_labels(label_file_path)

    # 선택된 객체 인덱스 가져오기 (체크박스 이름: 'select_0', 'select_1', ...)
    selected_indices = []
    for key, value in request.form.items():
        if key.startswith('select_'):
            try:
                # 'select_i'에서 i는 detections 리스트의 0-based 인덱스
                index = int(key.split('_')[1])
                selected_indices.append(index)
            except (ValueError, IndexError):
                pass

    if not selected_indices:
        flash('블러 처리할 객체가 선택되지 않았습니다. 원본 이미지를 그대로 표시합니다.')
        # 선택이 없으면 원본 이미지를 최종 결과로 전달
        final_image_filename = Path(source_image_path).name
        return render_template(
            'final_result.html',
            final_image_url=url_for('static', filename=f'images/{final_image_filename}')
        )

    try:
        validate_kernel(kernel_size)
    except ValueError as e:
        flash(f'블러 커널 오류: {e}')
        return redirect(url_for('index'))

    # 1. 원본 이미지 로드
    image = load_image(source_image_path)

    # 2. 블러 처리 적용
    anonymized_image = create_mask_and_apply_blur(
        image,
        detections,
        selected_indices,
        kernel_size
    )

    # 3. 최종 결과 이미지 저장 (웹에서 접근 가능한 static/images 폴더에 저장)
    stem = Path(source_image_path).stem
    selected_item_numbers = sorted([i + 1 for i in selected_indices])
    index_string = "_".join(map(str, selected_item_numbers))
    final_image_filename = f"{stem}_anonymized_{kernel_size}_i{index_string}.jpg"
    output_path = app.config['UPLOAD_FOLDER'] / final_image_filename

    if not cv2.imwrite(str(output_path), anonymized_image):
        flash('최종 블러 이미지 저장 실패.')
        return redirect(url_for('index'))

    flash(f"✅ 블러 처리 완료! {len(selected_indices)}개 객체에 블러 적용됨.")

    return render_template(
        'final_result.html',
        final_image_url=url_for('static', filename=f'images/{final_image_filename}')
    )


if __name__ == '__main__':
    import cv2

    app.run(debug=True)