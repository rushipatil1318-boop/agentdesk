"""Run the scenario evaluation.

    python scripts/run_eval.py                 # deterministic stand-in planner (offline, reproducible)
    python scripts/run_eval.py --llm openai    # real model via an OpenAI-compatible endpoint (e.g. Ollama)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agentdesk.evaluation import evaluate_retrieval, render_report, run_scenarios, summarize  # noqa: E402
from agentdesk.llm import NaivePlanner, OpenAICompatibleLLM  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--llm", choices=["naive", "openai"], default="naive")
    parser.add_argument("--out", default="artifacts")
    args = parser.parse_args()
    factory = NaivePlanner if args.llm == "naive" else OpenAICompatibleLLM.from_env
    guard_on = run_scenarios(True, factory)
    guard_off = run_scenarios(False, factory)
    retrieval = evaluate_retrieval()
    report = render_report(guard_on, guard_off) + (
        f"\n\nPolicy retrieval ({retrieval['queries']} labelled queries): top-1 {retrieval['top1']:.0%}, "
        f"Recall@3 {retrieval['recall_at_3']:.0%}, MRR {retrieval['mrr']:.2f}")
    out = Path(args.out)
    out.mkdir(exist_ok=True)
    (out / f"eval_{args.llm}.md").write_text(report, encoding="utf-8")
    (out / f"eval_{args.llm}.json").write_text(
        json.dumps({"retrieval": retrieval, "guard_on": summarize(guard_on), "guard_off": summarize(guard_off),
                    "runs_guard_on": guard_on, "runs_guard_off": guard_off}, indent=2), encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
