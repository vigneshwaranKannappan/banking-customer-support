"""Spec Part 2 §7: evaluation pipeline runs end to end (fake LLM) and catches regressions."""

from banking_customer_support_ai.evaluation.runner import main, run_evaluation
from banking_customer_support_ai.evaluation.scenarios import SCENARIOS
from banking_customer_support_ai.models.schemas import Classification
from banking_customer_support_ai.services.langgraph_backend import LangGraphBackend
from banking_customer_support_ai.services.mock_backend import MockBackend

from .conftest import FakeLLM


def test_all_scenarios_pass_with_correct_classifier(backend):
    report = run_evaluation(backend)
    summary = report.summary()
    assert summary["fully_passed"] == len(SCENARIOS)
    assert summary["check_pass_rates"]["routing"] == 1.0
    assert set(summary["check_pass_rates"]) >= {"classification", "routing", "reply_format", "ticket_persisted", "status_reply"}


def test_misclassification_fails_classification_and_routing(repo):
    be = LangGraphBackend(FakeLLM(classify=lambda m: Classification.QUERY, reply=""), repo)
    summary = run_evaluation(be).summary()
    assert summary["check_pass_rates"]["routing"] < 1.0
    assert summary["classification_accuracy_by_class"]["query"] == 1.0
    assert summary["classification_accuracy_by_class"]["positive_feedback"] == 0.0


def test_mock_backend_meets_same_contract():
    assert run_evaluation(MockBackend()).summary()["fully_passed"] == len(SCENARIOS)


class _FlakyJudge:
    def __init__(self, error: str | None = None) -> None:
        self.error, self.calls = error, 0

    def score(self, category, message, reply):
        from banking_customer_support_ai.evaluation.judge import JudgeScore

        self.calls += 1
        if self.error:
            raise RuntimeError(self.error)
        return JudgeScore(empathy=4, clarity=5, accuracy=5, rationale="fake")


def test_resume_reuses_successful_scenarios_without_llm_calls(repo):
    first = run_evaluation(LangGraphBackend(FakeLLM(fail_classifier=True), repo), judge=_FlakyJudge())
    assert first.summary()["fully_passed"] == 0

    llm = FakeLLM(reply=lambda n: f"Thank you for your kind words, {n}! We're delighted to assist you.")
    second = run_evaluation(LangGraphBackend(llm, repo), judge=_FlakyJudge(), previous={r.message: r for r in first.results})
    assert second.summary()["fully_passed"] == len(SCENARIOS)
    calls_after_second = len(llm.calls)

    third_judge = _FlakyJudge()
    third = run_evaluation(LangGraphBackend(llm, repo), judge=third_judge, previous={r.message: r for r in second.results})
    assert len(llm.calls) == calls_after_second and third_judge.calls == 0
    assert third.summary()["judged"] == len(SCENARIOS)


def test_judge_error_is_recorded_not_fatal(backend):
    report = run_evaluation(backend, judge=_FlakyJudge("503 UNAVAILABLE"))
    assert report.summary()["fully_passed"] == len(SCENARIOS)
    assert all(r.judge is None and "503" in r.judge_error for r in report.results)


def test_quota_error_stops_early(backend):
    judge = _FlakyJudge("429 RESOURCE_EXHAUSTED")
    report = run_evaluation(backend, judge=judge)
    assert judge.calls == 1 and len(report.results) == 1
    assert "RESOURCE_EXHAUSTED" in report.stopped_early


def test_cli_mock_backend(tmp_path, capsys):
    out = tmp_path / "eval.json"
    assert main(["--backend", "mock", "--output", str(out)]) == 0
    assert "Scenarios fully passed" in capsys.readouterr().out
    assert out.exists()
