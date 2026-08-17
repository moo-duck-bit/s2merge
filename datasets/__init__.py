# datasets module

# huggingface datasets 라이브러리를 re-export (sentence_transformers 호환성)
# sys.path를 조작해서 site-packages의 datasets를 먼저 찾도록 함
import sys
import os

# 현재 디렉토리를 임시로 제거
_current_dir = os.path.dirname(os.path.abspath(__file__))
_parent_dir = os.path.dirname(_current_dir)
_paths_to_remove = [p for p in sys.path if p in (_current_dir, _parent_dir)]
for p in _paths_to_remove:
    sys.path.remove(p)

try:
    # 이제 site-packages의 datasets를 임포트
    from datasets import *  # noqa
    from datasets import Dataset, load_dataset, DatasetDict, __version__  # noqa
except ImportError:
    pass
finally:
    # sys.path 복원
    for p in reversed(_paths_to_remove):
        sys.path.insert(0, p)
