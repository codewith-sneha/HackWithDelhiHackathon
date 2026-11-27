"""Offline end-to-end check of the quiz integration through the FastAPI gateway.

Exercises the real routers, real Timestamp Guard and the real sealing/scoring code
in agent/quiz.py, with only three things stubbed so no network, no Gemini key and
no fastembed model download are needed:

  * the embedder   - /quiz/final uses _scroll (payload filter), never a vector search
  * the LLM        - returns a fixed, well-formed quiz
  * the Qdrant store - a throwaway local-file collection seeded below

Run from anywhere:  python3 agent/tests/verify_quiz_offline.py
"""

from __future__ import annotations

import hashlib
import os
import pathlib
import re
import sys
import tempfile

HERE = pathlib.Path(__file__).resolve()
ROOT = HERE.parents[2]                      # agent/tests/<this file> -> repo root
sys.path.insert(0, str(ROOT / "backend"))   # so `app.*` resolves like uvicorn app.main:app
sys.path.insert(0, str(ROOT))               # so `agent.*` resolves

TMP = tempfile.mkdtemp(prefix="quiz_e2e_")
os.environ.update({
    "QDRANT_PATH": TMP,                     # agent Config defaults are read at import time
    "QDRANT_COLLECTION": "knowledge_base",
    "QUIZ_SECRET": os.environ.get("QUIZ_SECRET", "e2e-secret"),
})
os.environ.pop("QDRANT_URL", None)
os.environ.pop("GEMINI_API_KEY", None)
os.environ.pop("GROQ_API_KEY", None)

try:
    from app.main import app                # the real gateway
except Exception:                           # qdrant_service insists on a URL in some versions
    os.environ["QDRANT_URL"] = "http://127.0.0.1:6333"   # never contacted below
    from app.main import app

from fastapi.testclient import TestClient
from qdrant_client import QdrantClient, models

import agent.app as core
import agent.quiz as quiz_mod

VIDEO = "lecture1.mp4"
COLLECTION = "knowledge_base"
DIM = 384
ANSWER_INDEX_KEY = "answer" + "_index"      # keep the literal out of this file

PASS, FAIL = [], []


def check(name: str, ok: bool, detail: str = "") -> None:
    (PASS if ok else FAIL).append(name)
    print(f"  {'PASS' if ok else 'FAIL'}  {name}" + (f"  [{detail}]" if detail else ""))


# ---------------------------------------------------------------- fake embedder
def fake_embed(text: str) -> list[float]:
    v = [0.0] * DIM
    for tok in text.lower().split():
        v[int(hashlib.md5(tok.encode()).hexdigest(), 16) % DIM] += 1.0
    norm = sum(x * x for x in v) ** 0.5 or 1.0
    return [x / norm for x in v]


# ---------------------------------------------------------------- fake LLM
QUESTIONS = [
    # (cite template, question, options, answer, explanation, topic)
    ("video0", "Which search checks every element one by one?",
     ["Binary search", "Linear search", "Hash lookup", "Depth-first search"], 1,
     "Linear search examines each element in turn, so it is O(n).", "linear search"),
    ("video1", "Binary search requires which precondition?",
     ["A sorted array", "A hash function", "A linked list", "Recursion"], 0,
     "Binary search discards half the array each step, which only works on sorted input.", "binary search"),
    ("doc", "Where are the search algorithms summarised on paper?",
     ["Page 2 of the notes", "The last slide", "The appendix", "There is no summary"], 0,
     "The notes page lists the binary search requirements and complexity.", "summary"),
    ("video+doc", "What time complexity does the lecture give for binary search?",
     ["O(n)", "O(n log n)", "O(log n)", "O(1)"], 2,
     "The lecture states O(log n) because the search space halves each step.", "complexity"),
    ("video0+video1", "Why does binary search on unsorted input fail?",
     ["The array is too short", "The discard step assumes order", "The hash collides", "It uses O(1) space"], 1,
     "Discarding half is only safe when the array is sorted.", "common mistakes"),
]


def fake_llm(system: str, prompt: str, json_mode: bool = False, max_tokens: int = 700) -> str:
    if "study coach" in system:            # SYSTEM_COACH for /quiz/feedback
        return "You did well on complexity. Revisit the common mistakes topic before the next lecture."
    vid = [int(n) for n in re.findall(r"\[(\d+)\] VIDEO", prompt)]
    doc = [int(n) for n in re.findall(r"\[(\d+)\] DOCUMENT", prompt)]
    if not vid:
        return '{"questions":[]}'
    out = []
    for cites, q, opts, ans, expl, topic in QUESTIONS:
        refs = []
        for part in cites.split("+"):
            if part == "doc" and doc:
                refs.append(doc[0])
            elif part.startswith("video"):
                i = int(part[5:] or 0)
                if i < len(vid):
                    refs.append(vid[i])
        if not refs:
            continue
        out.append({"question": q, "options": opts, "answer_index": ans,
                    "explanation": expl, "topic": topic, "cite": refs})
    import json
    return json.dumps({"questions": out})


# ---------------------------------------------------------------- seed the store
ROWS = [
    ("In this lecture we study searching algorithms. Linear search checks every element one by one, which takes O(n) time overall.", VIDEO, "video", 0, 60, -1),
    ("Binary search works only on a sorted array. It compares the target with the middle element and discards half of the array at each step.", VIDEO, "video", 60, 120, -1),
    ("Because binary search halves the search space each time, its time complexity is O(log n). For one million elements that is about twenty comparisons.", VIDEO, "video", 120, 180, -1),
    ("A common mistake is running binary search on an unsorted array. The result is wrong because the discard step assumes sorted order.", VIDEO, "video", 180, 240, -1),
    ("Now we move on to hash tables. A hash table maps keys to buckets using a hash function, giving average constant time lookups.", VIDEO, "video", 600, 660, -1),
    ("Collisions happen when two different keys hash to the same bucket. We resolve them using separate chaining or open addressing in practice.", VIDEO, "video", 660, 720, -1),
    ("Summary of binary search: it requires sorted input, runs in O(log n) time, and needs constant extra space when implemented iteratively.", VIDEO, "pdf", -1, -1, 2),
]

qd = QdrantClient(path=TMP)
qd.create_collection(COLLECTION, vectors_config=models.VectorParams(size=DIM, distance=models.Distance.COSINE))
points = []
for i, (t, s, k, a, b, p) in enumerate(ROWS):
    points.append(models.PointStruct(
        id=i, vector=fake_embed(t),
        payload={"text": t, "lecture_id": s, "source_type": k, "start_sec": a,
                 "end_sec": b, "page_num": p, "slide": -1,
                 "url": "/lecture1.mp4" if k == "video" else "/Lecture1_notes.pdf"},
    ))
qd.upsert(COLLECTION, points=points)

agent_obj = core.LearningAgent(
    cfg=core.Config(qdrant_path=TMP, collection=COLLECTION),
    qdrant=qd, embed=fake_embed, llm=fake_llm,
)
core.get_agent = lambda: agent_obj          # same module object quiz.py calls into
quiz_mod.core.get_agent = lambda: agent_obj

client = TestClient(app)                    # no context manager => startup hook skipped

ALLOWED = ["Photosynthesis — Study Notes.pdf"]

# ---------------------------------------------------------------- 1. routing
print("\n1. Gateway routing")
# this FastAPI version keeps included routers nested (_IncludedRouter), so read the
# resolved schema instead of the flat app.routes list
paths = set(app.openapi().get("paths", {}))
check("POST /api/quiz/final is mounted", "/api/quiz/final" in paths)
check("POST /api/quiz/feedback is mounted", "/api/quiz/feedback" in paths)
check("existing routes still mounted", {"/api/chat", "/api/interview/start"} <= paths)

# ---------------------------------------------------------------- 2. unlock gate
print("\n2. Preserved unlock condition (409 before the video ends)")
r = client.post("/api/quiz/final", json={"video_id": VIDEO, "pause_time": 300,
                                         "allowed_resources": ALLOWED, "num_questions": 5})
check("mid-video request rejected with 409", r.status_code == 409, f"got {r.status_code}")
check("unlock message preserved", "final quiz unlocks" in r.json().get("detail", ""))

# ---------------------------------------------------------------- 3. generation
print("\n3. Generation at video end (pause_time=705, video ends at 720)")
r = client.post("/api/quiz/final", json={"video_id": VIDEO, "pause_time": 705,
                                         "allowed_resources": ALLOWED,
                                         "course_id": None, "num_questions": 5})
gen = r.json()
check("200 from /api/quiz/final", r.status_code == 200, f"got {r.status_code}")
qs = gen.get("questions") or []
check("5 questions returned", len(qs) == 5, f"got {len(qs)}")
check("quiz_token returned", bool(gen.get("quiz_token")))
check("every question has 4 options", all(len(q.get("options", [])) == 4 for q in qs))
check("answers not leaked to the client", all(ANSWER_INDEX_KEY not in q for q in qs))
check("explanations not leaked before submit", all("explanation" not in q for q in qs))
check("origins counted", gen.get("counts") == {"video": 3, "document": 1, "both": 1}, str(gen.get("counts")))

# ---------------------------------------------------------------- 4. Timestamp Guard
print("\n4. Timestamp Guard (nothing ending after the pause point)")
fb = client.post("/api/quiz/feedback", json={
    "quiz_token": gen["quiz_token"], "answers": [0] * len(qs), "ai_feedback": False}).json()
results = fb.get("results") or []
cites = [c for q in results for c in (q.get("review") or [])]
check("review citations returned", len(cites) > 0, f"{len(cites)} citations")
check("no citation ends after pause_time=705", all((c.get("end") or 0) <= 705 for c in cites),
      str([(c.get("start"), c.get("end")) for c in cites]))
check("unwatched 660-720 chunk excluded", all((c.get("end") or 0) != 720 for c in cites))
check("video citations carry seek_to", any(c.get("seek_to") is not None for c in cites))

# ---------------------------------------------------------------- 5. scoring
print("\n5. Answer submission and scoring")
answers = [0, 1, 2, 3, 0]
fb = client.post("/api/quiz/feedback", json={"quiz_token": gen["quiz_token"],
                                             "answers": answers, "ai_feedback": True}).json()
res = fb.get("results") or []
expected = sum(1 for q, a in zip(res, answers) if q.get("correct_index") == a)
check("200 from /api/quiz/feedback", bool(res), str(fb)[:200])
check("total matches question count", fb.get("total") == len(qs), f"total={fb.get('total')}")
check("score computed correctly", fb.get("score") == expected, f"score={fb.get('score')} expected={expected}")
check("percent computed correctly", fb.get("percent") == round(100 * expected / len(qs)))
check("level is one of the defined bands",
      fb.get("level") in {"Strong grasp", "Good progress", "Needs review"}, str(fb.get("level")))
check("correct_index always in range", all(0 <= q["correct_index"] < 4 for q in res))
check("correct flag matches chosen index",
      all(q["correct"] == (q.get("chosen") == q["correct_index"]) for q in res))

# ---------------------------------------------------------------- 6. explanations
print("\n6. Explanations and feedback payload")
check("every result has an explanation", all((q.get("explanation") or "").strip() for q in res))
check("every result keeps topic + origin", all(q.get("topic") and q.get("origin") for q in res))
check("by_topic breakdown returned", len(fb.get("by_topic") or []) > 0)
check("by_origin breakdown returned", set((fb.get("by_origin") or {})) <= {"video", "document", "both"})
check("summary returned", bool((fb.get("summary") or "").strip()))
check("focus_areas is a list", isinstance(fb.get("focus_areas"), list))

# ---------------------------------------------------------------- 7. edge cases
print("\n7. Edge cases")
skipped = client.post("/api/quiz/feedback", json={"quiz_token": gen["quiz_token"],
                                                  "answers": [None] + [0] * (len(qs) - 1),
                                                  "ai_feedback": False}).json()
check("skipped answer scored as wrong", skipped["results"][0]["chosen"] is None
      and skipped["results"][0]["correct"] is False)
empty = client.post("/api/quiz/feedback", json={"quiz_token": gen["quiz_token"],
                                                "answers": [], "ai_feedback": False}).json()
check("empty answers => score 0", empty.get("score") == 0 and empty.get("total") == len(qs))
tampered = client.post("/api/quiz/feedback", json={
    "quiz_token": gen["quiz_token"][:-4] + "aaaa", "answers": [0] * len(qs)})
check("tampered token rejected with 400", tampered.status_code == 400, f"got {tampered.status_code}")
nocontent = client.post("/api/quiz/final", json={"video_id": "missing-lecture", "pause_time": 999,
                                                 "allowed_resources": ALLOWED, "num_questions": 3})
check("unknown video => 404", nocontent.status_code == 404, f"got {nocontent.status_code}")

print(f"\n{'=' * 62}\nPASSED {len(PASS)}   FAILED {len(FAIL)}")
if FAIL:
    for f in FAIL:
        print("  failed:", f)
print("=" * 62)
sys.exit(1 if FAIL else 0)
