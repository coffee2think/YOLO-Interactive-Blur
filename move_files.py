import os
import shutil

# --- 설정 (이 두 경로만 수정하면 됩니다) ---
# 1. 파일 탐색을 시작할 최상위 경로
# source_dir = r"D:\4. AI\039.가상 실내 공간 3D 합성 데이터\3.개방데이터\1.데이터\Training\images"
# source_dir = r"D:\4. AI\039.가상 실내 공간 3D 합성 데이터\3.개방데이터\1.데이터\Training\labels"
# source_dir = r"D:\4. AI\039.가상 실내 공간 3D 합성 데이터\3.개방데이터\1.데이터\Validation\images"
source_dir = r"D:\4. AI\039.가상 실내 공간 3D 합성 데이터\3.개방데이터\1.데이터\Validation\labels"


# 2. 모든 파일을 이동시킬 최종 목표 경로
# 여기서는 images 폴더 바로 아래로 옮기는 것을 목표로 합니다.
# target_dir = r"D:\4. AI\039.가상 실내 공간 3D 합성 데이터\3.개방데이터\1.데이터\Training\images"
# target_dir = r"D:\4. AI\039.가상 실내 공간 3D 합성 데이터\3.개방데이터\1.데이터\Training\labels"
# target_dir = r"D:\4. AI\039.가상 실내 공간 3D 합성 데이터\3.개방데이터\1.데이터\Validation\images"
target_dir = r"D:\4. AI\039.가상 실내 공간 3D 합성 데이터\3.개방데이터\1.데이터\Validation\labels"
# ---------------------------------------------

# 파일 확장자 목록 (필요에 따라 추가/수정 가능)
# allowed_extensions = ('.jpg', '.png', '.jpeg', '.txt')
allowed_extensions = ('.json')


# 파일 이동 함수 정의
def flatten_directory(src_root, dst_root):
    print(f"탐색 시작: {src_root}")
    moved_count = 0

    # os.walk를 사용하여 src_root 아래의 모든 파일과 폴더를 탐색
    for root, dirs, files in os.walk(src_root):
        for file in files:
            # 파일 이름과 확장자 분리
            filename, file_ext = os.path.splitext(file)

            # 허용된 확장자인지 확인
            if file_ext.lower() in allowed_extensions:
                source_path = os.path.join(root, file)

                # 파일 이름 충돌 방지를 위해 폴더 이름을 파일 이름에 추가 (선택 사항)
                # 예: etc_education_l_002_normal_0.jpg -> etc_education_l_002_normal_0_00.etc_education_l_002_normal_0.jpg
                # new_filename = os.path.basename(root) + "_" + file

                # 원본 파일 이름을 그대로 사용
                new_filename = file

                destination_path = os.path.join(dst_root, new_filename)

                # 파일 이동 (shutil.move)
                # 파일 이름이 중복되는 경우 오류가 발생할 수 있으니 주의해야 합니다.
                try:
                    shutil.move(source_path, destination_path)
                    moved_count += 1
                except shutil.Error as e:
                    # 파일 이름 충돌 시 대처 (예: 파일 이름 변경 후 다시 시도)
                    print(f"오류: {source_path}를 이동할 수 없습니다. (충돌 또는 기타 오류: {e})")
                    continue

    print(f"\n총 {moved_count}개의 파일이 {dst_root}로 이동되었습니다.")


if __name__ == "__main__":
    # 안전을 위해 source_dir과 target_dir이 같은 경우 이동을 시도하지 않거나,
    # target_dir이 source_dir의 하위 폴더인 경우만 이동하도록 로직을 추가할 수 있지만,
    # 여기서는 단순 이동을 위해 두 경로가 같다고 가정하고 실행합니다.
    flatten_directory(source_dir, target_dir)