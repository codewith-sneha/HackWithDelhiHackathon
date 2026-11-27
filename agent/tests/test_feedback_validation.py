"""Targeted checks for quiz feedback validation, reasoning suppression and template fallback.

No network, no API keys, no Qdrant server: only the LLM client is stubbed, so this exercises
the real agent/quiz.py feedback path and the real provider wrappers in agent/app.py.

Covers the five required cases: valid feedback, reasoning leakage, truncated responses,
malformed JSON and LLM failure -- plus the regression guarantee that chat/RAG requests are
byte-for-byte unchanged (no reasoning_format key unless the quiz asks for it).

Run:  .venv312/bin/python agent/tests/test_feedback_validation.py
"""

from __future__ import annotations

import json
import os
import pathlib
import sys
import tempfile
import types

ROOT = pathlib.Path(__file__).resolve().parents[2]      # agent/tests/<this file> -> repo root
sys.path.insert(0, str(ROOT))

os.environ.setdefault("QDRANT_PATH", tempfile.mkdtemp(prefix="quiz_fb_test_"))
os.environ.setdefault("QUIZ_SECRET", "feedback-test-secret")
os.environ.pop("GEMINI_API_KEY", None)
os.environ.pop("GROQ_API_KEY", None)

import agent.app as core                                       # noqa: E402
import agent.quiz as quiz_mod                                  # noqa: E402
from agent.quiz import (                                       # noqa: E402
    SYSTEM_COACH,
    FeedbackIn,
    QuestionResult,
    _ai_summary,
    _clean_feedback,
    _template_summary,
)

PASS, FAIL = [], []

GOOD = (
    "You nailed linear search and explained it clearly. Revisit binary search: your answer ignored "
    "the sorted-array requirement. Next step: redo questions 2 and 3 and replay the linked moments."
)

# Verbatim shape of the text that reached production learners.
REASONING = (
    "**Refining and Checking Constraints:** Role/Tone: Encouraging study coach. The learner scored 3 "
    "of 5 and needs a plan. Objectives: mention strong topics, name the weak ones, add one next step."
)


def check(name: str, ok: bool, detail: str = "") -> None:
    (PASS if ok else FAIL).append(name)
    print(f"  {'PASS' if ok else 'FAIL'}  {name}" + (f"  [{detail}]" if detail else ""))


# ----------------------------------------------------------------- fakes
class FakeLLM:
    """Records call kwargs and returns a canned reply (or raises)."""

    def __init__(self, reply: str = "", raises: Exception | None = None):
        self.reply, self.raises, self.calls = reply, raises, []

    def __call__(self, system, prompt, json_mode=False, max_tokens=700, hide_reasoning=False):
        self.calls.append({"system": system, "json_mode": json_mode,
                           "max_tokens": max_tokens, "hide_reasoning": hide_reasoning})
        if self.raises is not None:
            raise self.raises
        return self.reply


class LegacyLLM:
    """A wrapper that predates the hint, to prove _call_coach degrades gracefully."""

    def __init__(self, reply: str):
        self.reply, self.calls = reply, []

    def __call__(self, system, prompt, json_mode=False, max_tokens=700):
        self.calls.append({"json_mode": json_mode, "max_tokens": max_tokens})
        return self.reply


class FakeAgent:
    def __init__(self, llm):
        self.llm = llm


def graded_results() -> list[QuestionResult]:
    """Two graded questions: one right, one wrong, so ai_feedback is actually requested."""
    def q(i, topic, correct_index, chosen):
        return QuestionResult(
            id=i, question=f"Q{i}?", options=["A", "B", "C", "D"], topic=topic, origin="video",
            chosen=chosen, correct_index=correct_index, correct=chosen == correct_index,
            explanation="because the passage says so.", review=[],
        )
    return [q(1, "linear search", 1, 1), q(2, "binary search", 0, 3)]


def seal_token(n: int = 2) -> str:
    qs = [{"question": f"Q{i}?", "options": ["A", "B", "C", "D"], "answer_index": 0,
           "explanation": "because", "topic": "alpha" if i == 1 else "beta",
           "origin": "video", "review": []} for i in range(1, n + 1)]
    return quiz_mod._seal({"v": 1, "q": qs})


def call_feedback(reply: str = "", raises: Exception | None = None):
    """Drive the real /quiz/feedback handler with a stubbed agent."""
    original = core.get_agent
    core.get_agent = lambda: FakeAgent(FakeLLM(reply, raises))
    try:
        # one right, one wrong => score < total, which is when the coach is actually consulted
        return quiz_mod.quiz_feedback(FeedbackIn(quiz_token=seal_token(), answers=[0, 1],
                                                 ai_feedback=True))
    finally:
        core.get_agent = original


# ----------------------------------------------------------------- 1. valid feedback
def test_valid_feedback_is_used():
    llm = FakeLLM(json.dumps({"summary": GOOD}))
    out = _ai_summary(FakeAgent(llm), 1, 2, graded_results(), ["binary search"])
    check("valid JSON feedback is accepted", out == GOOD, repr(out)[:70])
    check("coach call asks for JSON", llm.calls[0]["json_mode"] is True)
    check("coach call requests hidden reasoning", llm.calls[0]["hide_reasoning"] is True)
    check("coach token budget unchanged (600)", llm.calls[0]["max_tokens"] == 600)
    check("system prompt asks for a JSON summary envelope", '"summary"' in SYSTEM_COACH)


def test_legacy_wrapper_without_hint():
    llm = LegacyLLM(json.dumps({"summary": GOOD}))
    out = _ai_summary(FakeAgent(llm), 1, 2, graded_results(), ["binary search"])
    check("wrapper without the hint still works", out == GOOD, repr(out)[:70])
    check("no hint forced onto a legacy wrapper", "hide_reasoning" not in llm.calls[0])


# ----------------------------------------------------------------- 2. reasoning leakage
def test_reasoning_inside_json_is_rejected():
    out = _ai_summary(FakeAgent(FakeLLM(json.dumps({"summary": REASONING}))), 3, 5,
                      graded_results(), ["binary search"])
    check("reasoning text inside JSON is rejected", out is None, repr(out)[:70])


def test_raw_reasoning_prose_is_rejected():
    """Exactly what the Groq path produced: reasoning instead of an answer, no JSON at all."""
    out = _ai_summary(FakeAgent(FakeLLM(REASONING)), 3, 5, graded_results(), ["binary search"])
    check("raw reasoning prose is rejected", out is None, repr(out)[:70])


def test_harmony_markers_are_rejected():
    leak = '<|channel|>analysis<|message|>We need to write the coaching text.<|end|>'
    check("harmony markers rejected by validator", _clean_feedback(leak) is None)


# ----------------------------------------------------------------- 3. truncated responses
def test_truncated_feedback_is_rejected():
    cut = json.dumps({"summary": "You did well on linear search and binary search but you should "
                                 "revisit the common mistakes"})
    check("truncated sentence (no terminator) rejected",
          _ai_summary(FakeAgent(FakeLLM(cut)), 1, 2, graded_results(), ["binary search"]) is None)
    check("validator rejects a cut-off string directly",
          _clean_feedback("You did well on most topics and should keep practising the") is None)


def test_unterminated_json_is_rejected():
    broke = '{"summary": "You did well on linear search. Keep going'
    check("unterminated JSON body rejected",
          _ai_summary(FakeAgent(FakeLLM(broke)), 1, 2, graded_results(), ["binary search"]) is None)


# ----------------------------------------------------------------- 4. malformed JSON
def test_malformed_json_is_rejected():
    for label, reply in [
        ("plain prose", "You did well on linear search. Keep going, you are close."),
        ("wrong key", json.dumps({"advice": GOOD})),
        ("wrong type", json.dumps({"summary": 42})),
        ("empty summary", json.dumps({"summary": "   "})),
        ("empty reply", ""),
    ]:
        out = _ai_summary(FakeAgent(FakeLLM(reply)), 1, 2, graded_results(), ["binary search"])
        check(f"malformed reply rejected: {label}", out is None, repr(out)[:60])


def test_markdown_and_lists_are_rejected():
    summary = ("**Well done!** Revisit these: - binary search - sorted arrays - complexity. "
               "Then replay the lecture.")
    check("markdown/list rejected by the validator", _clean_feedback(summary) is None)
    check("markdown inside the JSON summary rejected end to end",
          _ai_summary(FakeAgent(FakeLLM(json.dumps({"summary": summary}))), 1, 2,
                      graded_results(), ["binary search"]) is None)


def test_too_many_sentences_rejected():
    long = " ".join(["One."] * 6)
    check("more than five sentences rejected", _clean_feedback(long) is None)


def test_subject_word_does_not_false_positive():
    """A lecture that is genuinely about constraints must not look like reasoning."""
    text = ("You handled foreign-key constraints well. Revisit cardinality, which you missed twice. "
            "Next step: rework question 3 against the linked page.")
    check("a constraint-themed lecture is not mistaken for reasoning",
          _clean_feedback(text) == text)


# ----------------------------------------------------------------- 5. LLM failure
def test_llm_failure_falls_back():
    out = _ai_summary(FakeAgent(FakeLLM(raises=RuntimeError("groq exploded"))), 1, 2,
                      graded_results(), ["binary search"])
    check("LLM exception swallowed, returns None", out is None)


def test_failed_feedback_uses_template_endpoint():
    result = call_feedback(raises=RuntimeError("groq exploded"))
    check("feedback endpoint survives LLM failure", result.summary ==
          _template_summary(1, 2, result.focus_areas), repr(result.summary)[:70])
    check("scoring preserved on failure", (result.score, result.total, result.percent) == (1, 2, 50))
    check("results preserved on failure", len(result.results) == 2)


def test_leaked_feedback_uses_template_endpoint():
    """The production symptom, end to end: no reasoning text may reach the response."""
    result = call_feedback(reply=REASONING)
    check("leaked reasoning never reaches the learner", "Refining" not in result.summary
          and "Role/Tone" not in result.summary, repr(result.summary)[:70])
    check("template used instead", result.summary == _template_summary(1, 2, result.focus_areas))


def test_valid_feedback_endpoint_uses_ai():
    result = call_feedback(reply=json.dumps({"summary": GOOD}))
    check("clean AI feedback is shown", result.summary == GOOD, repr(result.summary)[:70])
    check("summary stays a single plain block", "**" not in result.summary and "\n" not in result.summary)


# ----------------------------------------------------------------- provider compatibility
class _Completions:
    def __init__(self, sink):
        self.sink = sink

    def create(self, **kwargs):
        self.sink.append(kwargs)
        return types.SimpleNamespace(
            choices=[types.SimpleNamespace(message=types.SimpleNamespace(content="ok"),
                                           finish_reason="stop")])


class _FakeGroqClient:
    def __init__(self, sink):
        self.chat = types.SimpleNamespace(completions=_Completions(sink))


def test_groq_reasoning_suppression_is_quiz_only():
    sink = []
    llm = core.GroqLLM("openai/gpt-oss-120b")
    llm._client = _FakeGroqClient(sink)

    out = llm("sys", "prompt", json_mode=True, max_tokens=600, hide_reasoning=True)
    check("groq returns message.content", out == "ok")
    check("quiz call sets reasoning_format=hidden", sink[0].get("reasoning_format") == "hidden")
    check("quiz call keeps JSON mode", sink[0]["response_format"] == {"type": "json_object"})
    check("quiz call keeps reasoning_effort=low", sink[0]["reasoning_effort"] == "low")

    sink.clear()
    llm("sys", "prompt", json_mode=False, max_tokens=700)
    check("chat/RAG call sends no reasoning_format", "reasoning_format" not in sink[0], str(sink[0]))
    check("chat/RAG call is otherwise unchanged",
          sink[0]["model"] == "openai/gpt-oss-120b" and sink[0]["temperature"] == 0.2)


class _GemModels:
    def __init__(self, sink):
        self.sink = sink

    def generate_content(self, **kwargs):
        self.sink.append(kwargs)
        return types.SimpleNamespace(text="hello")


def test_gemini_accepts_the_hint():
    sink = []
    llm = core.Gemini("gemini-3.8-flash")
    llm._client = types.SimpleNamespace(models=_GemModels(sink))
    out = llm("sys", "prompt", json_mode=True, max_tokens=600, hide_reasoning=True)
    check("gemini accepts hide_reasoning and returns text", out == "hello")
    check("gemini request carries no groq-only parameter", "reasoning_format" not in str(sink[0]))


def test_fallback_forwards_the_hint():
    seen = {}

    def primary(system, prompt, json_mode=False, max_tokens=700, hide_reasoning=False):
        seen["primary"] = hide_reasoning
        raise RuntimeError("gemini unavailable")

    def secondary(system, prompt, json_mode=False, max_tokens=700, hide_reasoning=False):
        seen["secondary"] = hide_reasoning
        return "ok"

    out = core.FallbackLLM(primary, secondary)("s", "p", json_mode=True, max_tokens=100,
                                              hide_reasoning=True)
    check("fallback forwards the hint to both providers",
          out == "ok" and seen.get("primary") is True and seen.get("secondary") is True)

    seen.clear()
    core.FallbackLLM(primary, secondary)("s", "p")
    check("fallback defaults the hint to False (chat/RAG path)",
          seen.get("primary") is False and seen.get("secondary") is False)


TESTS = [
    test_valid_feedback_is_used,
    test_legacy_wrapper_without_hint,
    test_reasoning_inside_json_is_rejected,
    test_raw_reasoning_prose_is_rejected,
    test_harmony_markers_are_rejected,
    test_truncated_feedback_is_rejected,
    test_unterminated_json_is_rejected,
    test_malformed_json_is_rejected,
    test_markdown_and_lists_are_rejected,
    test_too_many_sentences_rejected,
    test_subject_word_does_not_false_positive,
    test_llm_failure_falls_back,
    test_failed_feedback_uses_template_endpoint,
    test_leaked_feedback_uses_template_endpoint,
    test_valid_feedback_endpoint_uses_ai,
    test_groq_reasoning_suppression_is_quiz_only,
    test_gemini_accepts_the_hint,
    test_fallback_forwards_the_hint,
]


if __name__ == "__main__":
    print("\nQuiz feedback validation")
    print("=" * 62)
    for fn in TESTS:
        print(f"\n{fn.__name__}")
        fn()
    print(f"\n{'=' * 62}\nPASSED {len(PASS)}   FAILED {len(FAIL)}")
    for name in FAIL:
        print("  failed:", name)
    print("=" * 62)
    sys.exit(1 if FAIL else 0)
