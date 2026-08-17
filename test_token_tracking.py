#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
토큰 추적 시스템 테스트 스크립트

T_sunk와 T_inference가 올바르게 측정되는지 검증합니다.
"""

import sys
import os

# Add project root to path
project_root = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, project_root)

from AgentDropout.utils.globals import (
    PromptTokens, CompletionTokens, Cost,
    SunkPromptTokens, SunkCompletionTokens, SunkCost
)


def test_singleton_pattern():
    """싱글톤 패턴 테스트"""
    print("=" * 80)
    print("Test 1: Singleton Pattern")
    print("=" * 80)

    # 여러 번 인스턴스를 생성해도 같은 객체여야 함
    p1 = PromptTokens.instance()
    p2 = PromptTokens.instance()

    assert p1 is p2, "PromptTokens is not a singleton!"
    print("PromptTokens singleton pattern works correctly")

    s1 = SunkPromptTokens.instance()
    s2 = SunkPromptTokens.instance()

    assert s1 is s2, "SunkPromptTokens is not a singleton!"
    print("SunkPromptTokens singleton pattern works correctly")


def test_token_counting():
    """토큰 카운팅 테스트"""
    print("\n" + "=" * 80)
    print("Test 2: Token Counting")
    print("=" * 80)

    # 초기화
    PromptTokens.instance().reset()
    CompletionTokens.instance().reset()
    SunkPromptTokens.instance().reset()
    SunkCompletionTokens.instance().reset()

    print("Initial state:")
    print(f"  PromptTokens: {PromptTokens.instance().value}")
    print(f"  CompletionTokens: {CompletionTokens.instance().value}")
    print(f"  SunkPromptTokens: {SunkPromptTokens.instance().value}")
    print(f"  SunkCompletionTokens: {SunkCompletionTokens.instance().value}")

    # Router 단계 시뮬레이션 (T_sunk)
    print("\nSimulating Router stage (T_sunk)...")
    router_prompt = 1000
    router_completion = 500

    PromptTokens.instance().add(router_prompt)
    CompletionTokens.instance().add(router_completion)
    SunkPromptTokens.instance().add(router_prompt)
    SunkCompletionTokens.instance().add(router_completion)

    print(f"  Added Router tokens: {router_prompt} prompt + {router_completion} completion")
    print(f"  Total: {PromptTokens.instance().value + CompletionTokens.instance().value}")
    print(f"  T_sunk: {SunkPromptTokens.instance().value + SunkCompletionTokens.instance().value}")

    # Training 단계 시뮬레이션 (T_inference)
    print("\nSimulating Training stage (T_inference)...")
    train_prompt = 5000
    train_completion = 2000

    PromptTokens.instance().add(train_prompt)
    CompletionTokens.instance().add(train_completion)
    # Training은 Sunk에 추가하지 않음

    print(f"  Added Training tokens: {train_prompt} prompt + {train_completion} completion")
    print(f"  Total: {PromptTokens.instance().value + CompletionTokens.instance().value}")
    print(f"  T_sunk: {SunkPromptTokens.instance().value + SunkCompletionTokens.instance().value}")

    # Evaluation 단계 시뮬레이션 (T_inference)
    print("\nSimulating Evaluation stage (T_inference)...")
    eval_prompt = 10000
    eval_completion = 4000

    PromptTokens.instance().add(eval_prompt)
    CompletionTokens.instance().add(eval_completion)
    # Evaluation도 Sunk에 추가하지 않음

    print(f"  Added Evaluation tokens: {eval_prompt} prompt + {eval_completion} completion")
    print(f"  Total: {PromptTokens.instance().value + CompletionTokens.instance().value}")
    print(f"  T_sunk: {SunkPromptTokens.instance().value + SunkCompletionTokens.instance().value}")

    # 검증
    print("\n" + "=" * 80)
    print("Final Results:")
    print("=" * 80)

    total_tokens = PromptTokens.instance().value + CompletionTokens.instance().value
    t_sunk = SunkPromptTokens.instance().value + SunkCompletionTokens.instance().value
    t_inference = total_tokens - t_sunk

    expected_total = router_prompt + router_completion + train_prompt + train_completion + eval_prompt + eval_completion
    expected_t_sunk = router_prompt + router_completion
    expected_t_inference = train_prompt + train_completion + eval_prompt + eval_completion

    print(f"Total Tokens: {total_tokens:,.0f}")
    print(f"   Expected: {expected_total:,.0f}")
    assert abs(total_tokens - expected_total) < 0.1, f"Total tokens mismatch! {total_tokens} != {expected_total}"
    print("   Correct")

    print(f"\nT_sunk (Router): {t_sunk:,.0f}")
    print(f"   Expected: {expected_t_sunk:,.0f}")
    assert abs(t_sunk - expected_t_sunk) < 0.1, f"T_sunk mismatch! {t_sunk} != {expected_t_sunk}"
    print("   Correct")

    print(f"\nT_inference (Training + Evaluation): {t_inference:,.0f}")
    print(f"   Expected: {expected_t_inference:,.0f}")
    assert abs(t_inference - expected_t_inference) < 0.1, f"T_inference mismatch! {t_inference} != {expected_t_inference}"
    print("   Correct")

    print("\nAll token counting tests passed!")


def test_conversation_log():
    """대화 로그 테스트"""
    print("\n" + "=" * 80)
    print("Test 3: Conversation Logging")
    print("=" * 80)

    from AgentDropout.utils.log import agent_conversation_log
    from pathlib import Path
    import tempfile

    # 임시 디렉토리에 로그 작성
    with tempfile.TemporaryDirectory() as tmpdir:
        log_path = Path(tmpdir) / "test_conversations.txt"

        # 대화 로그 기록
        agent_conversation_log(
            agent_name="TestAgent",
            round_num=1,
            input_text="What is 2+2?",
            output_text="The answer is 4.",
            log_file_path=log_path
        )

        # 로그 파일 확인
        assert log_path.exists(), "Log file was not created!"

        content = log_path.read_text(encoding='utf-8')
        assert "TestAgent" in content, "Agent name not in log!"
        assert "Round: 1" in content, "Round number not in log!"
        assert "What is 2+2?" in content, "Input not in log!"
        assert "The answer is 4." in content, "Output not in log!"

        print(f"Conversation log created successfully")
        print(f"   Log path: {log_path}")
        print(f"   Content preview:")
        print(content[:200] + "...")


def main():
    """메인 테스트 함수"""
    print("\n" + "=" * 80)
    print("Token Tracking System Test Suite")
    print("=" * 80)

    try:
        test_singleton_pattern()
        test_token_counting()
        test_conversation_log()

        print("\n" + "=" * 80)
        print("ALL TESTS PASSED!")
        print("=" * 80)
        print("\n토큰 추적 시스템이 올바르게 작동합니다.")
        print("실제 실험을 실행하여 T_sunk와 T_inference를 측정할 수 있습니다.\n")

        return 0

    except Exception as e:
        print("\n" + "=" * 80)
        print("TEST FAILED!")
        print("=" * 80)
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
