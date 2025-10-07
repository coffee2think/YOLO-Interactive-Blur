import shutil
import argparse
import sys
from pathlib import Path
from typing import Tuple, List

# ----------------------------------------------------------------------
# 1. 파일 이동 함수 정의
# ----------------------------------------------------------------------

def flatten_directory(src_root: Path, dst_root: Path, allowed_extensions: Tuple[str, ...]) -> None:
    """
    지정된 확장자를 가진 파일을 src_root의 모든 하위 디렉터리에서 찾아 dst_root로 이동시킵니다.
    """
    print(f"============================================================")
    print(f"📂 탐색 시작 경로: {src_root}")
    print(f"🎯 이동 목표 경로: {dst_root}")
    print(f"🔍 대상 확장자: {', '.join(allowed_extensions)}")
    print(f"============================================================")

    # 목표 디렉터리가 없으면 생성
    dst_root.mkdir(parents=True, exist_ok=True)

    moved_count = 0

    # Path.glob()을 사용하여 src_root의 모든 하위 파일 탐색
    # '**/*'는 재귀적으로 모든 하위 디렉터리를 포함
    for file_path in src_root.glob('**/*'):
        # 파일인지 확인
        if file_path.is_file():
            # 확장자 추출 및 소문자 변환
            file_ext = file_path.suffix.lower()

            # 허용된 확장자인지 확인
            if file_ext in allowed_extensions:
                # 새 목표 경로 생성
                destination_path = dst_root / file_path.name

                # 파일 이동 (shutil.move)
                try:
                    # 목표 경로에 동일한 파일명이 이미 있는지 확인 (충돌 방지 로직)
                    if destination_path.exists():
                        print(f"[WARN] 파일명 충돌 발생. 스킵: {file_path.name}")
                        continue

                    # shutil.move는 Path 객체를 지원합니다.
                    shutil.move(str(file_path), str(destination_path))
                    moved_count += 1
                except shutil.Error as e:
                    print(f"[ERROR] 파일 이동 중 오류 발생: {file_path} -> {e}", file=sys.stderr)
                    continue

    print(f"\n✅ 작업 완료: 총 {moved_count}개의 파일이 {dst_root}로 이동되었습니다.")

# ----------------------------------------------------------------------
# 2. CLI 인자 파싱 및 메인 실행부
# ----------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    """CLI 인자를 정의하고 파싱한다."""
    parser = argparse.ArgumentParser(
        description="Recursively searches a directory for specified files and moves them to a single target directory (flattening the structure).",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "--source-dir",
        type=Path,  # Path 객체로 바로 받습니다.
        required=True,
        help="The root directory to start searching for files."
    )
    parser.add_argument(
        "--target-dir",
        type=Path,  # Path 객체로 바로 받습니다.
        required=True,
        help="The destination directory where all found files will be moved."
    )
    parser.add_argument(
        "--extensions",
        type=str,
        default=".json",
        help=(
            "Comma-separated list of file extensions to move (e.g., '.jpg,.png,.txt').\n"
            "Default is '.json'."
        )
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # 쉼표로 구분된 확장자 문자열을 소문자 튜플로 변환
    try:
        ext_list: List[str] = [
            ext.strip().lower()
            for ext in args.extensions.split(',')
            if ext.strip()
        ]

        # 확장자에 '.'이 없으면 추가
        ext_list = [ext if ext.startswith('.') else '.' + ext for ext in ext_list]

        if not ext_list:
            print(f"[FATAL] 유효한 확장자가 지정되지 않았습니다: {args.extensions}", file=sys.stderr)
            sys.exit(1)

        allowed_ext_tuple: Tuple[str, ...] = tuple(ext_list)

    except Exception as e:
        print(f"[FATAL] 확장자 파싱 오류 또는 알 수 없는 오류 발생: {e}", file=sys.stderr)
        sys.exit(1)

    # 경로 유효성 검사 (원본 경로만)
    if not args.source_dir.is_dir():
        print(f"[FATAL] 원본 디렉터리를 찾을 수 없습니다: {args.source_dir}", file=sys.stderr)
        sys.exit(1)

    # 실행
    flatten_directory(args.source_dir, args.target_dir, allowed_ext_tuple)
