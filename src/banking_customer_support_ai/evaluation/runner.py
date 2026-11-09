"""Evaluation runner: classification accuracy, routing success, reply QA, optional LLM-judge scores.

    uv run banking-eval                      # keyword mock, no LLM calls
    uv run banking-eval --backend llm        # real agents, ~1.3 LLM calls per scenario
    uv run banking-eval --backend llm --judge --output eval.json   # +1 judge call per scenario

Free-tier friendly (e.g. Gemini's 20 requests/day per model):
    --judge-model gemini-3.8-flash   judge on a different model, so it uses that model's own quota
    --resume eval.json               reuse scenarios that already succeeded; only re-run the rest

The LLM backend runs against a throwaway in-memory SQLite DB, so evaluation never touches PostgreSQL.
"""

import argparse
import json
import sys
import uuid
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path

from banking_customer_support_ai.evaluation.judge import ResponseJudge
from banking_customer_support_ai.evaluation.qa import QACheck, qa_checks
from banking_customer_support_ai.evaluation.scenarios import SCENARIOS, Scenario
from banking_customer_support_ai.models.schemas import Ticket
from banking_customer_support_ai.services.backend import SupportBackend

EVAL_CUSTOMER = "Vignesh"
_QUOTA_MARKERS = ("insufficient_quota", "RESOURCE_EXHAUSTED")


@dataclass
class ScenarioResult:
    message: str
    expected: str
    predicted: str | None
    agent_path: list[str]
    response: str
    checks: list[QACheck]
    judge: dict | None = None
    judge_error: str | None = None

    @property
    def passed(self) -> bool:
        return all(c.passed for c in self.checks)

    @property
    def agent_ok(self) -> bool:
        return all(c.passed for c in self.checks if c.name == "agent_success")

    @classmethod
    def from_dict(cls, data: dict) -> "ScenarioResult":
        return cls(**{**data, "checks": [QACheck(**c) for c in data["checks"]]})


@dataclass
class EvaluationReport:
    backend: str
    results: list[ScenarioResult] = field(default_factory=list)
    stopped_early: str | None = None

    def rate(self, check: str) -> float | None:
        values = [c.passed for r in self.results for c in r.checks if c.name == check]
        return sum(values) / len(values) if values else None

    def summary(self) -> dict:
        per_class: dict[str, list[bool]] = defaultdict(list)
        for r in self.results:
            per_class[r.expected].append(r.predicted == r.expected)
        check_names = sorted({c.name for r in self.results for c in r.checks})
        judged = [r.judge for r in self.results if r.judge]
        return {
            "backend": self.backend,
            "scenarios": len(self.results),
            "fully_passed": sum(r.passed for r in self.results),
            "stopped_early": self.stopped_early,
            "check_pass_rates": {name: self.rate(name) for name in check_names},
            "classification_accuracy_by_class": {k: sum(v) / len(v) for k, v in per_class.items()},
            "confusion": dict(Counter(f"{r.expected} -> {r.predicted}" for r in self.results)),
            "judged": len(judged),
            "judge_averages": {
                k: sum(j[k] for j in judged) / len(judged) for k in ("empathy", "clarity", "accuracy")
            }
            if judged
            else None,
        }


def _lookup(backend: SupportBackend):
    def lookup(ticket_id: str) -> Ticket | None:
        return next((t for t in backend.list_tickets() if t.ticket_id == ticket_id), None)

    return lookup


def _quota_hit(error: str | None) -> bool:
    return bool(error) and any(m in error for m in _QUOTA_MARKERS)


def run_evaluation(
    backend: SupportBackend,
    scenarios: list[Scenario] = SCENARIOS,
    judge: ResponseJudge | None = None,
    previous: dict[str, ScenarioResult] | None = None,
) -> EvaluationReport:
    """Run scenarios; reuse `previous` results (keyed by message) whose agents and judge already succeeded."""
    session_id = f"eval-{uuid.uuid4().hex[:6]}"
    report = EvaluationReport(backend=backend.name)
    previous = previous or {}
    for i, sc in enumerate(scenarios, 1):
        prior = previous.get(sc.message)
        if prior and prior.agent_ok and (judge is None or prior.judge):
            report.results.append(prior)
            print(f"[{i}/{len(scenarios)}] reused   {sc.message!r}")
            continue

        if prior and prior.agent_ok:
            scored = prior  # agents fine last time; only the judge score is missing
        else:
            result = backend.handle_message(sc.message, EVAL_CUSTOMER, session_id)
            scored = ScenarioResult(
                message=sc.message,
                expected=sc.expected.value,
                predicted=result.classification.value if result.classification else None,
                agent_path=result.agent_path,
                response=result.response,
                checks=qa_checks(result, sc.expected, _lookup(backend)),
            )
            if _quota_hit(result.error):
                report.stopped_early = result.error
        if judge is not None and scored.agent_ok and report.stopped_early is None:
            try:
                scored.judge = judge.score(sc.expected.label, sc.message, scored.response).model_dump()
                scored.judge_error = None
            except Exception as exc:
                scored.judge_error = f"{type(exc).__name__}: {exc}"
                if _quota_hit(scored.judge_error):
                    report.stopped_early = scored.judge_error
        report.results.append(scored)
        status = "ok" if scored.passed else "FAILED"
        print(f"[{i}/{len(scenarios)}] {status:<8} {sc.message!r}")

        if report.stopped_early:
            print("LLM quota exhausted; stopping early. Re-run later with --resume to continue.")
            # Keep the unrun scenarios' earlier results so --resume never loses work.
            report.results += [previous[s.message] for s in scenarios[i:] if s.message in previous]
            break
    return report


def _pct(value: float | None) -> str:
    return "—" if value is None else f"{value:.0%}"


def print_report(report: EvaluationReport) -> None:
    s = report.summary()
    print(f"\nBackend: {s['backend']}")
    print(f"Scenarios fully passed: {s['fully_passed']}/{s['scenarios']}\n")
    print("Check pass rates:")
    for name, rate in s["check_pass_rates"].items():
        print(f"  {name:<18} {_pct(rate)}")
    print("\nClassification accuracy by class:")
    for name, rate in s["classification_accuracy_by_class"].items():
        print(f"  {name:<18} {_pct(rate)}")
    if s["judge_averages"]:
        print(f"\nLLM judge averages (1-5, {s['judged']} replies judged):")
        for k, v in s["judge_averages"].items():
            print(f"  {k:<18} {v:.2f}")
    failures = [r for r in report.results if not r.passed or r.judge_error]
    if failures:
        print("\nFailures:")
        for r in failures:
            for c in r.checks:
                if not c.passed:
                    print(f"  [{c.name}] {r.message!r}: {c.detail[:200]}")
            if r.judge_error:
                print(f"  [judge] {r.message!r}: {r.judge_error[:200]}")


def _load_previous(path: str) -> dict[str, ScenarioResult]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return {r["message"]: ScenarioResult.from_dict(r) for r in data.get("results", [])}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--backend", choices=["mock", "llm"], default="mock")
    parser.add_argument("--judge", action="store_true", help="LLM-as-judge scoring (+1 LLM call per scenario)")
    parser.add_argument("--judge-model", help="model for the judge (default: same as the agents)")
    parser.add_argument("--limit", type=int, help="run only the first N scenarios")
    parser.add_argument("--output", help="write the full report as JSON to this path")
    parser.add_argument("--resume", help="earlier --output JSON; reuse its successful scenarios")
    args = parser.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")  # Windows cp1252 consoles can't print "→"

    scenarios = SCENARIOS[: args.limit] if args.limit else SCENARIOS
    previous = _load_previous(args.resume) if args.resume and Path(args.resume).exists() else {}
    if args.judge and args.backend == "mock":
        parser.error("--judge needs --backend llm")

    judge = None
    if args.backend == "mock":
        from banking_customer_support_ai.services.mock_backend import MockBackend

        backend: SupportBackend = MockBackend()
    else:
        from banking_customer_support_ai.config.settings import get_settings
        from banking_customer_support_ai.services.langgraph_backend import LangGraphBackend, make_llm

        settings = get_settings()
        backend = LangGraphBackend.from_settings(settings, database_url="sqlite://")
        todo = [s for s in scenarios if not (s.message in previous and previous[s.message].agent_ok)]
        calls = sum(2 if s.expected.value == "positive_feedback" else 1 for s in todo)
        print(f"Agents: {settings.llm_model}, {len(todo)} scenarios to run, ~{calls} LLM calls.")
        if args.judge:
            judge_settings = settings
            if args.judge_model:
                field_name = "gemini_model" if settings.llm_provider == "gemini" else "openai_model"
                judge_settings = settings.model_copy(update={field_name: args.judge_model})
            judge = ResponseJudge(make_llm(judge_settings))
            to_judge = sum(1 for s in scenarios if not (s.message in previous and previous[s.message].judge))
            print(f"Judge: {judge_settings.llm_model}, up to {to_judge} LLM calls.")

    report = run_evaluation(backend, scenarios, judge, previous)
    print_report(report)
    out = args.output or args.resume
    if out:
        with open(out, "w", encoding="utf-8") as f:
            json.dump({"summary": report.summary(), "results": [asdict(r) for r in report.results]}, f, indent=2)
        print(f"\nWrote {out}")
    return 0 if all(r.passed for r in report.results) and not report.stopped_early else 1


if __name__ == "__main__":
    sys.exit(main())
