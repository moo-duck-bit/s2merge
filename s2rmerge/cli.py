"""Command line entry point: one run of one benchmark.

    python -m s2rmerge gsm8k --llm Meta-Llama-3.1-8B-Instruct

Pass ``--no-router`` to reproduce the w/o Router ablation, ``--topology`` for
the robustness study, and ``--no-fusion`` to replace prompt-level fusion with
plain concatenation.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path

from s2rmerge.benchmarks import available
from s2rmerge.paths import CONFIG_DIR
from s2rmerge.pipeline import Pipeline, Settings
from s2rmerge.topology import TOPOLOGIES

# MultiArith and SVAMP share GSM8K's role pool, so they share its router config.
CONFIG_FOR_BENCHMARK = {
    "gsm8k": "gsm8k",
    "multiarith": "gsm8k",
    "svamp": "gsm8k",
    "aqua": "aqua",
    "mmlu": "mmlu",
    "humaneval": "humaneval",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="s2rmerge",
        description="Run S2R-Merge on one benchmark.",
    )
    parser.add_argument("benchmark", choices=available())
    parser.add_argument(
        "--llm",
        default="gpt-4o-mini",
        help="model name; anything containing 'llama' or 'qwen' goes to the local "
        "vLLM server, everything else to the OpenAI API",
    )
    parser.add_argument(
        "--router-config",
        type=Path,
        default=None,
        help="router YAML; defaults to the one matching the benchmark",
    )
    parser.add_argument(
        "--no-router",
        action="store_true",
        help="skip routing and run the full agent pool (the w/o Router ablation)",
    )
    parser.add_argument("--topology", choices=TOPOLOGIES, default="FullConnected")
    parser.add_argument(
        "--merge-iterations",
        type=int,
        default=None,
        help="number of agents absorbed, one per iteration; defaults to the depth "
        "reported for this benchmark",
    )
    parser.add_argument(
        "--optimization-steps",
        type=int,
        default=5,
        help="policy-gradient steps per optimisation phase",
    )
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument(
        "--edge-dropout",
        type=float,
        default=0.1,
        help="fraction of edges dropped after merging (beta)",
    )
    parser.add_argument(
        "--rounds",
        type=int,
        default=None,
        help="communication rounds per query; defaults to the benchmark's setting",
    )
    parser.add_argument("--train-samples", type=int, default=None)
    parser.add_argument("--test-samples", type=int, default=None)
    parser.add_argument(
        "--max-agent-groups",
        type=int,
        default=None,
        help="cap on distinct routed agent sets; rarer sets are folded into the "
        "closest kept one",
    )
    parser.add_argument(
        "--no-fusion",
        action="store_true",
        help="concatenate prompts on merge instead of fusing them with an LLM",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--quiet", action="store_true")
    return parser


def settings_from_args(args: argparse.Namespace) -> Settings:
    router_config = None
    if not args.no_router:
        router_config = args.router_config or CONFIG_DIR / f"{CONFIG_FOR_BENCHMARK[args.benchmark]}.yaml"
        if not router_config.exists():
            raise SystemExit(f"router config not found: {router_config}")

    return Settings(
        benchmark=args.benchmark,
        llm_name=args.llm,
        router_config=router_config,
        topology=args.topology,
        merge_iterations=args.merge_iterations,
        optimization_steps=args.optimization_steps,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        edge_dropout=args.edge_dropout,
        rounds=args.rounds,
        train_samples=args.train_samples,
        test_samples=args.test_samples,
        max_agent_groups=args.max_agent_groups,
        use_fusion=not args.no_fusion,
        seed=args.seed,
    )


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.WARNING if args.quiet else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(message)s",
        datefmt="%H:%M:%S",
    )

    pipeline = Pipeline(settings_from_args(args))
    report = asyncio.run(pipeline.run())

    report_path = pipeline.output_dir / "report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report.as_dict(), indent=2), encoding="utf-8")

    print()
    print(f"{report.benchmark}: {report.metric_name} = {report.score * 100:.2f}% "
          f"over {report.num_evaluated} queries in {report.seconds / 60:.1f} min")
    print()
    for group in report.groups:
        print(f"  {group.final_nodes} agents after merging, "
              f"sparsity {group.final_sparsity:.2f}, "
              f"{group.num_test} queries, {group.accuracy * 100:.2f}%")
        for merge in group.merges:
            print(f"    {merge}")
    print()
    print(pipeline_usage_table(report))
    print()
    print(f"report written to {report_path}")
    return 0


def pipeline_usage_table(report) -> str:
    header = f"{'stage':<14}{'prompt':>14}{'completion':>14}{'cost':>12}"
    rows = [header]
    for stage, usage in report.usage.items():
        rows.append(
            f"{stage:<14}{usage['prompt_tokens']:>14,.0f}"
            f"{usage['completion_tokens']:>14,.0f}{usage['cost']:>12.4f}"
        )
    return "\n".join(rows)


if __name__ == "__main__":
    sys.exit(main())
