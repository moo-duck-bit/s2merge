"""
통합 실험 스크립트
데이터셋별 YAML 설정 파일을 사용하여 Router 프레임워크 실험 실행

## 핵심: 팀원 PC에서 데이터셋 경로 지정 방법
이 스크립트는 `--config` 파일명으로 데이터셋 타입을 자동 감지한 뒤,
데이터셋을 **기본 경로(default)** 에서 읽습니다.
팀원마다 데이터셋 위치가 다르면 **`--data_path` 한 줄로 경로만 바꿔서** 실행하면 됩니다.

### 1) 기본 경로(default) 규칙 (레포 기준 상대경로)
- mmlu: `data/mmlu/test/`  (폴더, 내부에 `*.csv`)
- humaneval: `humaneval_dataset.json` (파일, JSON Lines)
- aqua: `data/aqua/train.json` (파일, JSON)
- gsm8k: `data/gsm8k/test.jsonl` (파일, JSONL)
- math_cot / math_nocot: `data/math/test.jsonl` (파일, JSONL)

### 2) `--data_path`로 덮어쓰기 (팀원 PC 경로 사용)
- mmlu는 **폴더 경로**를 넣어야 합니다. (그 폴더 안에 `*.csv` 존재)
  예) `--data_path "/abs/path/to/mmlu/test"`
- 그 외(humaneval/aqua/gsm8k/math_*)는 **파일 경로**를 넣어야 합니다.
  예) `--data_path "/abs/path/to/gsm8k/test.jsonl"`

### 3) 실행 예시
    python run_experiment.py --config config/mmlu_config.yaml --num_samples 10
    python run_experiment.py --config config/mmlu_config.yaml --num_samples -1   # 전체 데이터
    python run_experiment.py --config config/mmlu_config.yaml --data_path "/abs/path/to/mmlu/test" --num_samples 10
    python run_experiment.py --config config/humaneval_config.yaml --data_path "/abs/path/to/humaneval_dataset.json" --num_samples 5
    python run_experiment.py --config config/aqua_config.yaml --data_path "/abs/path/to/aqua/train.json" --num_samples 20
"""

import argparse
import json
import os
import pandas as pd
from typing import Dict, List, Any, Optional
from pathlib import Path
import yaml

from src.router import Router
from src.llm_client import create_llm_client


def detect_dataset_type(config_path: str) -> str:
    """
    Config 파일 이름으로 데이터셋 타입 자동 감지

    Args:
        config_path: Config 파일 경로

    Returns:
        데이터셋 타입: 'mmlu', 'humaneval', 'aqua', 'gsm8k', 'math_cot', 'math_nocot'
    """
    config_name = Path(config_path).stem.lower()

    if 'mmlu' in config_name:
        return 'mmlu'
    elif 'humaneval' in config_name:
        return 'humaneval'
    elif 'aqua' in config_name:
        return 'aqua'
    elif 'gsm8k' in config_name or 'gsm' in config_name:
        return 'gsm8k'
    elif 'math_cot' in config_name or 'mathcot' in config_name:
        return 'math_cot'
    elif 'math_nocot' in config_name or 'mathnocot' in config_name:
        return 'math_nocot'
    else:
        # 기본값: config 파일에서 추론 시도
        return 'unknown'


def load_mmlu_data(data_path: str = "data/mmlu/test", num_samples: Optional[int] = None) -> List[Dict]:
    """MMLU 데이터셋 로드

    MMLU CSV 형식: 헤더 없음, question,option1,option2,option3,option4,answer
    """
    import glob

    csv_files = glob.glob(f"{data_path}/*.csv")
    if not csv_files:
        raise FileNotFoundError(f"MMLU 데이터를 찾을 수 없습니다: {data_path}")

    all_data = []
    for csv_file in csv_files:
        # MMLU CSV 형식: 헤더 없음, question,option1,option2,option3,option4,answer
        df = pd.read_csv(csv_file, header=None, names=['question', 'option1', 'option2', 'option3', 'option4', 'answer'])

        for _, row in df.iterrows():
            # question이 제대로 있는지 확인
            question = str(row['question']).strip()
            if not question or question.lower() == 'question':
                print(f"경고: {csv_file}의 행에서 question이 비정상적입니다: {question}")
                continue

            # 옵션들을 리스트로 구성
            options = [
                str(row['option1']).strip(),
                str(row['option2']).strip(),
                str(row['option3']).strip(),
                str(row['option4']).strip()
            ]

            all_data.append({
                'question': question,
                'options': options,
                'correct': str(row['answer']).strip(),
                'subject': Path(csv_file).stem.replace('_test', '').replace('_val', '')
            })

    if num_samples is not None and num_samples > 0:
        all_data = all_data[:num_samples]

    return all_data


def load_humaneval_data(data_path: str = "humaneval_dataset.json", num_samples: Optional[int] = None) -> List[Dict]:
    """HumanEval 데이터셋 로드 (JSON Lines)"""
    samples = []
    with open(data_path, 'r') as f:
        for line in f:
            if line.strip():
                samples.append(json.loads(line))

    if num_samples is not None and num_samples > 0:
        samples = samples[:num_samples]

    return samples


def load_aqua_data(data_path: str = "data/aqua/train.json", num_samples: Optional[int] = None) -> List[Dict]:
    """AQUA 데이터셋 로드 (JSON)"""
    with open(data_path, 'r') as f:
        data = json.load(f)

    if num_samples is not None and num_samples > 0:
        data = data[:num_samples]

    return data


def load_gsm8k_data(data_path: str = "data/gsm8k/test.jsonl", num_samples: Optional[int] = None) -> List[Dict]:
    """GSM8K 데이터셋 로드 (JSONL)"""
    samples = []
    with open(data_path, 'r') as f:
        for line in f:
            if line.strip():
                samples.append(json.loads(line))

    if num_samples is not None and num_samples > 0:
        samples = samples[:num_samples]

    return samples


def load_math_data(data_path: str = "data/math/test.jsonl", num_samples: Optional[int] = None) -> List[Dict]:
    """Math 데이터셋 로드 (JSONL, GSM8K와 동일 형식)"""
    return load_gsm8k_data(data_path, num_samples)


def extract_question(sample: Dict, dataset_type: str) -> str:
    """샘플에서 질문 추출"""
    if dataset_type == 'mmlu':
        return sample['question']
    elif dataset_type == 'humaneval':
        return sample['prompt']
    elif dataset_type == 'aqua':
        return sample['question']
    elif dataset_type in ['gsm8k', 'math_cot', 'math_nocot']:
        return sample.get('question', sample.get('problem', ''))
    else:
        # 기본: 'question' 키 시도
        return sample.get('question', str(sample))


def run_experiment(
    config_path: str,
    dataset_type: str,
    num_samples: int = 10,
    data_path: Optional[str] = None,
    use_llm_summary: bool = True,
    save_results: bool = True
) -> Dict[str, Any]:
    """
    Router 실험 실행

    Args:
        config_path: Config YAML 파일 경로
        dataset_type: 데이터셋 타입
        num_samples: 실험할 샘플 수
        data_path: 데이터 파일 경로 (None이면 기본 경로 사용)
        use_llm_summary: LLM summary 사용 여부
        save_results: 결과 저장 여부

    Returns:
        실험 결과 딕셔너리
    """
    print("="*80)
    print(f"Router 실험 시작")
    print("="*80)
    print(f"Config: {config_path}")
    print(f"Dataset: {dataset_type}")
    print(f"Samples: {'ALL' if (num_samples is None or num_samples <= 0) else num_samples}")
    print()

    # Router 초기화
    print("Router 초기화 중...")
    router = Router(config_path)
    print("Router 초기화 완료!\n")

    # 데이터셋 로드
    print(f"데이터셋 로드 중...")

    # 기본 데이터 경로 설정
    if data_path is None:
        if dataset_type == 'mmlu':
            data_path = "data/mmlu/test"
        elif dataset_type == 'humaneval':
            data_path = "humaneval_dataset.json"
        elif dataset_type == 'aqua':
            data_path = "data/aqua/train.json"
        elif dataset_type == 'gsm8k':
            data_path = "data/gsm8k/test.jsonl"
        elif dataset_type in ['math_cot', 'math_nocot']:
            data_path = "data/math/test.jsonl"
        else:
            raise ValueError(f"알 수 없는 데이터셋 타입: {dataset_type}")

    print(f"  - data_path: {data_path}")

    # 데이터 로드
    if dataset_type == 'mmlu':
        samples = load_mmlu_data(data_path, num_samples)
    elif dataset_type == 'humaneval':
        samples = load_humaneval_data(data_path, num_samples)
    elif dataset_type == 'aqua':
        samples = load_aqua_data(data_path, num_samples)
    elif dataset_type == 'gsm8k':
        samples = load_gsm8k_data(data_path, num_samples)
    elif dataset_type in ['math_cot', 'math_nocot']:
        samples = load_math_data(data_path, num_samples)
    else:
        raise ValueError(f"지원하지 않는 데이터셋 타입: {dataset_type}")

    print(f"{len(samples)}개 샘플 로드 완료\n")

    # 실험 실행
    print("Router 실험 실행 중...\n")
    results = []

    for i, sample in enumerate(samples):
        question = extract_question(sample, dataset_type)

        # 질문이 제대로 추출되었는지 확인
        if not question or question == 'question' or len(str(question).strip()) < 10:
            print(f"  경고: 샘플 {i}의 질문이 비정상적입니다: {question}")
            # 샘플 전체를 출력해서 디버깅
            print(f"  샘플 내용: {sample}")
            continue

        print(f"[{i+1}/{len(samples)}] 처리 중...")
        print(f"  질문: {question[:100]}..." if len(question) > 100 else f"  질문: {question}")

        try:
            result = router.route(question, use_llm_summary=use_llm_summary)

            # 최종 에이전트 집합을 명확하게 구성
            final_agents = []
            for role in result['selected_roles']:
                prob = result['role_probabilities'].get(role, 0.0)
                final_agents.append({
                    'agent': role,
                    'probability': float(prob)
                })

            # 확률 순으로 정렬
            final_agents.sort(key=lambda x: x['probability'], reverse=True)

            result_entry = {
                'sample_idx': i,
                'question': str(question)[:500] + "..." if len(str(question)) > 500 else str(question),
                'final_agents': final_agents,  # 최종 에이전트 집합 (명확하게)
                'num_agents': result['num_agents'],
                'selected_blocks': result['selected_blocks'],
                'selected_roles': result['selected_roles'],  # 호환성을 위해 유지
                'num_blocks': result['num_blocks'],
                'block_uncertainty': result['uncertainty']['block_uncertainty'],
                'role_uncertainty': result['uncertainty']['role_uncertainty'],
                'block_probabilities': result['block_probabilities'],
                'role_probabilities': result['role_probabilities']
            }

            # 원본 샘플 정보 추가
            if dataset_type == 'humaneval':
                result_entry['task_id'] = sample.get('task_id', '')
            elif dataset_type == 'aqua':
                result_entry['correct'] = sample.get('correct', '')

            results.append(result_entry)

            print(f"  Blocks: {result['num_blocks']}개")
            print(f"     {result['selected_blocks']}")
            print(f"  최종 선택된 Agents: {result['num_agents']}개")
            print(f"     {result['selected_roles']}")
            print(f"  Uncertainty: Block={result['uncertainty']['block_uncertainty']:.3f}, Role={result['uncertainty']['role_uncertainty']:.3f}")

        except Exception as e:
            print(f"  오류: {e}")
            results.append({
                'sample_idx': i,
                'error': str(e)
            })

        print()

    # 통계 계산
    successful_results = [r for r in results if 'error' not in r]

    if successful_results:
        avg_blocks = sum(r['num_blocks'] for r in successful_results) / len(successful_results)
        avg_agents = sum(r['num_agents'] for r in successful_results) / len(successful_results)
        avg_block_uncertainty = sum(r['block_uncertainty'] for r in successful_results) / len(successful_results)
        avg_role_uncertainty = sum(r['role_uncertainty'] for r in successful_results) / len(successful_results)

        summary = {
            'config': config_path,
            'dataset_type': dataset_type,
            'total_samples': len(samples),
            'successful_samples': len(successful_results),
            'failed_samples': len(results) - len(successful_results),
            'avg_blocks': avg_blocks,
            'avg_agents': avg_agents,
            'avg_block_uncertainty': avg_block_uncertainty,
            'avg_role_uncertainty': avg_role_uncertainty,
            'results': results
        }
    else:
        summary = {
            'config': config_path,
            'dataset_type': dataset_type,
            'total_samples': len(samples),
            'successful_samples': 0,
            'failed_samples': len(results),
            'error': '모든 샘플 처리 실패',
            'results': results
        }

    # 결과 저장
    if save_results:
        os.makedirs('outputs', exist_ok=True)
        output_file = f"outputs/experiment_{dataset_type}_{num_samples}samples.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        print(f"결과 저장: {output_file}\n")

    # 요약 출력
    print("="*80)
    print("실험 결과 요약")
    print("="*80)
    if successful_results:
        print(f"성공: {len(successful_results)}/{len(samples)}")
        print(f"평균 Blocks: {avg_blocks:.2f}개")
        print(f"평균 Agents: {avg_agents:.2f}개")
        print(f"평균 Block Uncertainty: {avg_block_uncertainty:.3f}")
        print(f"평균 Role Uncertainty: {avg_role_uncertainty:.3f}")
        print()
        print("샘플별 질문 및 최종 선택된 Agents:")
        for i, result in enumerate(successful_results, 1):
            print(f"\n  [{i}] 질문:")
            print(f"      {result['question'][:150]}..." if len(result['question']) > 150 else f"      {result['question']}")
            print(f"      최종 선택된 Agents ({result['num_agents']}개):")
            # final_agents가 있으면 사용, 없으면 selected_roles 사용
            if 'final_agents' in result:
                for agent_info in result['final_agents']:
                    print(f"        - {agent_info['agent']} (확률: {agent_info['probability']:.3f})")
            else:
                for role in result['selected_roles']:
                    prob = result['role_probabilities'].get(role, 0.0)
                    print(f"        - {role} (확률: {prob:.3f})")
    else:
        print("모든 샘플 처리 실패")
    print("="*80)

    return summary


def main():
    parser = argparse.ArgumentParser(description='Router 프레임워크 통합 실험 스크립트')
    parser.add_argument('--config', type=str, required=True,
                        help='Config YAML 파일 경로 (예: config/mmlu_config.yaml)')
    parser.add_argument('--num_samples', type=int, default=-1,
                        help='실험할 샘플 수 (기본값: -1, 즉 전체). 특정 개수만 돌리려면 양수를 넣으세요 (예: 10).')
    parser.add_argument('--data_path', type=str, default=None,
                        help=(
                            "데이터 경로를 직접 지정해 기본 경로를 덮어씁니다. "
                            "mmlu는 폴더 경로(내부에 *.csv), 그 외는 파일 경로를 넣으세요. "
                            "예) --data_path data/mmlu/test  또는  --data_path /abs/path/to/gsm8k/test.jsonl"
                        ))
    parser.add_argument('--no_llm', action='store_true',
                        help='LLM summary 사용 안 함 (embedding만 사용)')
    parser.add_argument('--dataset_type', type=str, default=None,
                        help='데이터셋 타입 강제 지정 (mmlu, humaneval, aqua, gsm8k, math_cot, math_nocot)')
    parser.add_argument('--no_save', action='store_true',
                        help='결과 저장 안 함')

    args = parser.parse_args()

    # 데이터셋 타입 감지
    if args.dataset_type:
        dataset_type = args.dataset_type
    else:
        dataset_type = detect_dataset_type(args.config)
        if dataset_type == 'unknown':
            print("데이터셋 타입을 자동 감지할 수 없습니다. --dataset_type 옵션을 사용하세요.")
            return

    # 실험 실행
    summary = run_experiment(
        config_path=args.config,
        dataset_type=dataset_type,
        num_samples=args.num_samples,
        data_path=args.data_path,
        use_llm_summary=not args.no_llm,
        save_results=not args.no_save
    )

    print("\n실험 완료!")


if __name__ == "__main__":
    main()
