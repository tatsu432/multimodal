## My recommended evaluation design

Build **three eval sets**, not one.

### 1. Small deterministic “unit-test” video set

This should be your first eval set.

Use 10–30 short scripted videos that you create yourself with a phone/Tapo camera. Example scenes:

> Put a red bottle on the desk.
> Move it to the shelf.
> Open a laptop.
> Show a notebook
> Leave the room.
> Come back wearing a black jacket.

For each video, create a gold event timeline:

```json
{
  "video_id": "desk_001",
  "events": [
    {
      "start_sec": 12.0,
      "end_sec": 18.0,
      "description": "red bottle is placed on desk",
      "objects": ["red bottle", "desk"],
      "location_label": "home desk"
    }
  ],
  "live_questions": [
    {
      "ask_at_sec": 15.0,
      "question": "What object did I just put on the desk?",
      "gold_answer": "red bottle",
      "gold_evidence_window": [12.0, 18.0]
    }
  ],
  "memory_questions": [
    {
      "ask_after_sec": 300.0,
      "question": "Where did I put the red bottle?",
      "gold_answer": "on the desk",
      "gold_evidence_window": [12.0, 18.0]
    }
  ]
}
```

This is boring but extremely valuable. You need deterministic tests before public benchmarks because your system has many possible failure points: frame sampling, timestamping, DB writing, retrieval filters, embeddings, prompt behavior, and final answering.

### 2. Public benchmark adaptation set

Then use public datasets to make the evaluation less toy-like.

For **streaming live QA**, good references are StreamingBench, SVBench, and RTV-Bench. StreamingBench is especially relevant because it asks questions at different time points to simulate continuous streaming and includes 900 videos and 4,500 human-curated QA pairs across real-time, omni-source, and contextual understanding tasks. ([StreamingBench][1]) SVBench is also relevant because it focuses on temporal multi-turn dialogues over streaming videos and reports 49,979 QA pairs from 1,353 streaming videos. ([arXiv][2]) RTV-Bench is useful because it explicitly evaluates continuous perception, understanding, and reasoning through multi-timestamp QA over dynamic video streams. ([arXiv][3])

For **long-term memory**, Ego4D Episodic Memory is the closest conceptual match. Its motivation is almost exactly your problem: egocentric wearable video as memory, where the model must localize where an answer can be seen in past video. It includes natural-language queries like “What did I put in the drawer?”, visual queries, and moments queries, with about 74K queries across 800 hours of video. ([EGO4D][4])

For **long-video reasoning**, LongVideoBench and MLVU are useful but less directly “wearable memory.” LongVideoBench has videos up to an hour long and 6,678 human-annotated multiple-choice questions that require retrieving and reasoning over relevant video details. ([arXiv][5]) MLVU covers videos from 3 minutes to 2 hours and evaluates nine long-video understanding task types. ([GitHub][6])

### 3. Real demo regression set

Finally, record your own sessions:

```text
10 min desk work
10 min walking outside
10 min kitchen/object manipulation
10 min conversation-like environment
10 min screen/whiteboard/text-heavy scene
```

This set should be small but stable. Every time you change retrieval, prompts, frame sampling, DB schema, or memory summarization, run this same set. This becomes your **product regression benchmark**.

## The most important principle: evaluate stages separately

For your system, I would track these stages:

```text
1. Stream replay quality
2. Live VLM answer quality
3. Memory write quality
4. Retrieval quality
5. Evidence-pack quality
6. Final answer quality
7. System latency/cost/stability
```

End-to-end accuracy alone is not enough. Suppose the final answer is wrong. Was it because:

```text
camera frame was stale?
sampled frames missed the event?
VLM described the event incorrectly?
memory was not written?
embedding search failed?
SQL time filter removed the correct memory?
answer generator ignored the evidence?
LLM hallucinated?
```

Without component metrics, you cannot know.

## Dataset construction

### Live QA examples

For live QA, each question should have an **ask time**.

```json
{
  "task_type": "live_current",
  "video_id": "kitchen_003",
  "ask_at_sec": 42.0,
  "question": "What am I holding?",
  "gold_answer": "a white mug",
  "gold_evidence_window": [39.0, 43.0],
  "answer_type": "short_text",
  "scoring": "llm_judge_with_gold"
}
```

Question categories:

```text
current object:       What am I holding?
current scene:        Where am I?
text/OCR:             What word is written on the notebook?
spatial:              What is to the left of the laptop?
recent change:        What changed in the last 10 seconds?
temporal order:       Did I open the box before picking up the bottle?
unanswerable:         What color is the object behind the camera?
```

The **unanswerable** category is important. Without it, your system will learn to always guess.

### Long-term memory QA examples

For long-term memory QA, each query should have a gold answer and a gold evidence window.

```json
{
  "task_type": "ltm_recall",
  "video_id": "desk_001",
  "query_time_sec": 900.0,
  "question": "Where did I leave the red bottle?",
  "gold_answer": "on the shelf",
  "gold_evidence_windows": [[220.0, 235.0]],
  "required_memory_type": "promoted_event_or_passive_observation",
  "answer_type": "short_text"
}
```

Question categories:

```text
object location:      Where did I leave my keys?
past state:           Was the window open earlier?
past activity:        What was I doing before lunch?
time localization:    When did I put the bottle on the shelf?
place-based recall:   What did I see near the station?
summary:              What happened during the last 30 minutes?
comparison:           Did the desk look cleaner before or after I came back?
negative recall:      Did I ever open the drawer?
```

The best gold label is not only the answer. It is:

```text
gold answer
gold evidence time window
gold supporting objects/events
acceptable aliases
whether the answer is answerable
```

For example:

```json
{
  "gold_answer": "on the shelf",
  "acceptable_answers": ["shelf", "on the bookshelf", "on the rack"],
  "unacceptable_answers": ["desk", "floor", "unknown"],
  "gold_evidence_windows": [[220.0, 235.0]]
}
```

## Metrics to track

### Live QA metrics

Track these:

```text
answer_accuracy
hallucination_rate
unanswerable_accuracy
p50_latency
p95_latency
frame_age_at_query
num_frames_used
capture_drop_rate
```

The underrated one is **frame_age_at_query**.

If the user asks at time `t = 42s`, but your latest frame is from `t = 36s`, the model may answer correctly for the wrong moment. So log:

```text
query_time_sec
latest_frame_time_sec
frame_age = query_time_sec - latest_frame_time_sec
```

For streaming systems, stale frames are a real failure mode.

### Retrieval metrics for long-term memory

This is the most important part.

For every memory QA query, evaluate whether the retriever found the correct evidence.

```text
Recall@K
MRR
evidence_window_IoU
temporal_distance_to_gold
retrieval_precision
```

Example:

```text
Gold evidence window: [220s, 235s]
Retrieved memories:
  1. [800s, 810s] wrong
  2. [224s, 230s] correct
  3. [100s, 105s] wrong

Recall@3 = 1
MRR = 1/2
```

For long-term memory, retrieval quality is more important than final answer quality at first. If retrieval fails, the answer generator has no chance.

### Final answer metrics

Use a mixture:

```text
exact_match / regex for simple answers
multiple-choice accuracy if dataset provides choices
LLM judge for open-ended answers
faithfulness-to-evidence score
unsupported-claim count
abstention quality
```

For LLM judging, do **not** ask “is this answer good?” That is too vague. Use a strict rubric:

```text
Given:
- question
- gold answer
- gold evidence
- system answer
- retrieved evidence

Score:
2 = correct and supported by evidence
1 = partially correct or vague but not harmful
0 = incorrect, contradicted, or unsupported

Also mark:
- hallucinated_object: true/false
- hallucinated_time: true/false
- hallucinated_location: true/false
- should_have_abstained: true/false
```

Use LLM judging for speed, but manually inspect a small sample every time. Otherwise, your eval can become fake-objective.

VidHalluc is a useful reminder that video-language models can hallucinate specifically on action, temporal sequence, and scene transition, so your rubric should explicitly include those failure types. ([arXiv][7])

## The evaluation pipeline I would build

The pipeline should use **logical simulated time**, not real waiting time. You do not need to wait hours or days to test long-term memory.

```text
video file
  ↓
ReplayController
  - emits frames according to simulated timestamps
  - never exposes future frames
  - can run at 1x, 5x, or as-fast-as-possible
  ↓
same frame sampler / ring buffer as production
  ↓
scheduled live QA events
  ↓
passive observer / active memory writer
  ↓
SQLite + Chroma
  ↓
daily summary job, if needed
  ↓
scheduled LTM queries
  ↓
scoring
  ↓
eval_runs.sqlite / JSON report
```

The key rule:

> At simulated time `t`, the system can only access frames and memories whose timestamps are `<= t`.

That single rule prevents future leakage.

## Minimal implementation plan

### Phase 1: eval manifest format

Create something like:

```text
eval/
  datasets/
    toy_videos/
      desk_001.mp4
      desk_001.json
  src/
    replay.py
    run_live_eval.py
    run_ltm_eval.py
    scorers.py
    report.py
  outputs/
    eval_runs.sqlite
```

Each manifest should contain:

```json
{
  "video_id": "desk_001",
  "video_path": "datasets/toy_videos/desk_001.mp4",
  "duration_sec": 600,
  "events": [],
  "live_questions": [],
  "memory_questions": []
}
```

### Phase 2: live QA eval

Run only this first:

```bash
uv run python -m eval.run_live_eval \
  --manifest eval/datasets/toy_videos/desk_001.json \
  --model gpt-4o-mini \
  --frame-sampling 1fps \
  --num-frames 4
```

Output:

```text
live_accuracy: 0.72
unanswerable_accuracy: 0.60
hallucination_rate: 0.18
p50_latency_ms: 2100
p95_latency_ms: 4800
mean_frame_age_sec: 0.7
```

### Phase 3: retrieval-only LTM eval

Before judging final answers, evaluate retrieval alone.

For each memory query:

```text
question: Where did I put the red bottle?
gold window: [220, 235]
retrieved windows:
  promoted_events: [...]
  passive_observations: [...]
  daily_summaries: [...]
```

Metrics:

```text
event_recall@1
event_recall@5
passive_recall@10
temporal_IoU
retrieval_latency_ms
```

This will tell you whether SQLite filters, Chroma, metadata filtering, and query planning are working.

### Phase 4: end-to-end LTM QA eval

Only after retrieval is measurable, evaluate the final answer.

```text
question
gold answer
retrieved evidence
system answer
judge score
```

Track:

```text
answer_score_avg
fully_correct_rate
unsupported_claim_rate
wrong_location_rate
wrong_time_rate
abstention_accuracy
```

## What I would not do

I would **not** rely only on “latest Video VLM models” to generate ground truth. That is useful for bootstrapping, but dangerous as final truth. Use this hierarchy:

```text
Best: human-labeled scripted videos
Good: public benchmark labels with temporal annotations
Okay: VLM-generated labels verified by you
Bad: VLM-generated labels used blindly as ground truth
```

For your internship MVP, a small verified dataset beats a huge noisy one. A 50-question dataset where every answer and evidence window is correct is more useful than a 5,000-question dataset with unknown label quality.

## Recommended benchmark mix

For your exact project, I’d choose:

```text
MVP:
- 20 self-recorded scripted clips
- 100 live QA questions
- 100 LTM QA questions
- exact/evidence-window labels

Research-grade extension:
- Ego4D Episodic Memory for wearable-style memory retrieval
- StreamingBench / SVBench / RTV-Bench for streaming QA behavior
- LongVideoBench / MLVU for long-context video reasoning
- VidHalluc-style negative tests for hallucination
```

Ego4D is the best conceptual match for long-term wearable memory. StreamingBench/SVBench/RTV-Bench are better for online/streaming behavior. LongVideoBench/MLVU are useful for general long-video reasoning but less aligned with “my personal visual memory.”

## Final target dashboard

Eventually your eval report should look like this:

```text
Run: 2026-06-07_23-xx
Model: Qwen2.5-VL / GPT-4o-mini / etc.
Embedding: nomic-embed-text
Frame sampling: 1 fps
Passive interval: 30 sec
Top-K: 8

Live QA
- accuracy: 74%
- hallucination rate: 12%
- p95 latency: 4.8s
- mean frame age: 0.6s

Memory retrieval
- Recall@1: 42%
- Recall@5: 78%
- MRR: 0.55
- mean temporal distance to gold: 18.2s

LTM final answer
- fully correct: 61%
- partially correct: 18%
- wrong: 21%
- unsupported claim rate: 15%
- abstention accuracy: 70%

Failure breakdown
- missed visual event: 8
- memory not written: 5
- retrieval missed: 12
- answer ignored evidence: 4
- hallucinated object/location/time: 7
```

That failure breakdown is where the real engineering value is.

## My honest recommendation

For now, build this in order:

```text
1. Create 10 scripted toy videos.
2. Label event timelines manually.
3. Add scheduled live QA questions.
4. Add scheduled LTM QA questions.
5. Build replay runner with simulated timestamps.
6. Score live QA first.
7. Score retrieval-only LTM second.
8. Score final LTM answers third.
9. Add public datasets only after the local pipeline works.
```

The big trap is trying to make the benchmark “realistic” too early. Start with controlled videos where you know the truth. Once the pipeline catches obvious regressions, scale to Ego4D/StreamingBench-style data.

[1]: https://streamingbench.github.io/ " StreamingBench"
[2]: https://arxiv.org/abs/2502.10810 "[2502.10810] SVBench: A Benchmark with Temporal Multi-Turn Dialogues for Streaming Video Understanding"
[3]: https://arxiv.org/abs/2505.02064 "[2505.02064] RTV-Bench: Benchmarking MLLM Continuous Perception, Understanding and Reasoning through Real-Time Video"
[4]: https://ego4d-data.org/docs/benchmarks/episodic-memory/ "Episodic Memory | Ego4D"
[5]: https://arxiv.org/abs/2407.15754 "[2407.15754] LongVideoBench: A Benchmark for Long-context Interleaved Video-Language Understanding"
[6]: https://github.com/JUNJIE99/MLVU "GitHub - JUNJIE99/MLVU: MLVU: Multi-task Long Video Understanding Benchmark · GitHub"
[7]: https://arxiv.org/abs/2412.03735 "[2412.03735] VidHalluc: Evaluating Temporal Hallucinations in Multimodal Large Language Models for Video Understanding"
