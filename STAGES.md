
## Big picture architecture

Eventually you want this:

```text
Wearable / GoPro / phone camera
        ↓
Streaming ingestion
        ↓
Frame sampler
        ↓
VLM inference
        ↓
Structured scene/memory records
        ↓
Long-term storage + vector DB
        ↓
Text QA over live view + past memory
        ↓
Evaluation + profiling + efficient inference research
```

Your current prototype already covers:

```text
GoPro RTMP → OpenCV → latest frames → VLM answers text question
```

So now we gradually add **memory, retrieval, metadata, evaluation, and efficiency**.

---

# Phase 1 — `vlm_smoke`: stabilize live visual QA

Goal: make the current prototype reliable and reproducible.

You already have the raw version. Now clean it up.

### Features

Add:

```text
- RTMP stream reader
- frame buffer
- text question input
- VLM inference on latest frame / recent frames
- basic logging
- config file
```

### Folder

```text
vlm_smoke/
├── README.md
├── pyproject.toml
├── .env.example
├── src/
│   ├── main.py
│   ├── config.py
│   ├── frame_source.py
│   ├── vlm_client.py
│   └── utils.py
└── outputs/
    └── sampled_frames/
```

### What to implement

Abstract the frame source now:

```python
class FrameSource:
    def read(self):
        ...
```

Then implement:

```text
RTMPFrameSource
WebcamFrameSource
VideoFileFrameSource
```

Even if you only use RTMP now, this abstraction will save you later.

### Success criteria

You should be able to run:

```bash
uv run python -m src.main
```

Then ask:

```text
What do you see?
Is there a person?
What object is closest to the camera?
```

and get reasonable answers.

### Evaluation

Measure:

```text
- stream read success rate
- average VLM latency
- number of frames sent per query
- answer quality by manual inspection
```

At this phase, don’t overdo it. Just make it stable.

---

# Phase 2 — `memory_log`: store visual memory as JSONL

Goal: create long-term memory records, but without vector DB yet.

Do **not** add ChromaDB immediately. First make the memory format good.

### Features

Every N seconds, save:

```text
- timestamp
- image path
- VLM scene summary
- detected objects
- location metadata placeholder
- privacy risk
- whether to store the memory
```

Example record:

```json
{
  "memory_id": "2026-06-04T23-12-30.123",
  "timestamp": "2026-06-04T23:12:30.123+09:00",
  "image_path": "outputs/frames/2026-06-04T23-12-30.jpg",
  "summary": "A desk with a laptop, cable, and GoPro accessories.",
  "objects": ["desk", "laptop", "cable", "camera"],
  "scene_type": "indoor_workspace",
  "location": {
    "lat": null,
    "lon": null,
    "source": "not_available"
  },
  "should_store": true,
  "memory_reason": "Contains meaningful workspace context.",
  "privacy_risk": "low"
}
```

### Folder

```text
memory_log/
├── README.md
├── src/
│   ├── main.py
│   ├── memory_writer.py
│   ├── schema.py
│   ├── vlm_client.py
│   └── frame_source.py
└── outputs/
    ├── frames/
    └── memories.jsonl
```

### Important design decision

Use structured VLM output.

Prompt the VLM to return JSON:

```text
Analyze this frame and return JSON:
{
  "summary": "...",
  "scene_type": "...",
  "objects": ["..."],
  "people_count": 0,
  "text_visible": ["..."],
  "should_store": true,
  "memory_reason": "...",
  "privacy_risk": "low|medium|high"
}
```

### Success criteria

After running for 2 minutes, you should have:

```text
- saved frame images
- memories.jsonl
- valid JSON records
- no crash when stream temporarily fails
```

This phase is boring but extremely important. If your memory records are trash, retrieval will also be trash.

---

# Phase 3 — `memory_search`: keyword/time-based memory QA

Goal: answer past-memory questions **without vector DB** first.

Example questions:

```text
What did I see recently?
What was on my desk 5 minutes ago?
Did I see a laptop today?
Show me memories involving a person.
```

### Features

Implement simple retrieval over `memories.jsonl`:

```text
- filter by time range
- filter by object
- filter by scene type
- search summary text
```

No embeddings yet.

### Why this comes before ChromaDB

Because many memory questions are not semantic search problems. They are often:

```text
time filter + metadata filter + maybe semantic search
```

For example:

```text
Where was I yesterday afternoon?
```

This is mostly timestamp + location, not vector similarity.

### Success criteria

You can ask:

```text
What did I see in the last 10 minutes?
```

and it returns relevant memory records with timestamps and summaries.

---

# Phase 4 — `vector_memory`: add ChromaDB / embedding retrieval

Goal: support semantic memory questions.

Now add vector DB.

### Features

For each memory record, embed:

```text
summary
objects
scene_type
visible text
location description if available
```

Store in ChromaDB with metadata:

```text
memory_id
timestamp
image_path
scene_type
objects
privacy_risk
lat/lon if available
```

### Example questions

```text
When did I see something like a restaurant?
Did I pass by any signs?
Where did I see a red object?
What did I see near the station?
```

### Retrieval pipeline

Use hybrid retrieval:

```text
1. Parse user question
2. Extract filters:
   - time range
   - location
   - objects
   - scene type
3. Retrieve candidate memories
4. Rerank / select top K
5. Send selected memory summaries + maybe images to VLM/LLM
6. Answer with timestamps and evidence
```

### Success criteria

For a memory question, the system returns:

```text
- answer
- relevant memory timestamps
- optionally image paths as evidence
```

Do not just return a vague answer. Always ground it in memory records.

---

# Phase 5 — `location_memory`: add location metadata

Goal: support questions like:

```text
Where was I yesterday afternoon?
What did I see near this location?
What did I see around Shibuya?
```

For now, fake it or manually inject it. Don’t block on smartphone GPS integration.

### Step 5A: manual location

Add CLI flag:

```bash
uv run python -m src.main --location-label "home desk"
```

Store:

```json
{
  "location": {
    "label": "home desk",
    "lat": null,
    "lon": null
  }
}
```

### Step 5B: phone/server location later

Eventually:

```text
phone GPS → backend API → memory record
```

But for the prototype, manual label is enough.

### Success criteria

The system can answer:

```text
What did I see at home desk today?
```

or:

```text
What locations did I visit during this recording?
```

---

# Phase 6 — `backend_api`: make it a real service

Goal: move from script to backend system.

Use FastAPI.

### Endpoints

```text
POST /ask_live
- asks about current/recent frames

POST /ask_memory
- asks about stored memories

POST /ingest_frame
- optional future endpoint for phone/wearable frame upload

GET /memories
- inspect memory records

GET /health
- health check
```

Architecture:

```text
RTMP capture worker
        ↓
shared frame buffer
        ↓
FastAPI server
        ↓
VLM client + memory retriever
```

### Success criteria

You can run:

```bash
uv run uvicorn src.app:app --reload
```

Then call:

```bash
curl -X POST http://localhost:8000/ask_live \
  -H "Content-Type: application/json" \
  -d '{"question": "What do you see right now?"}'
```

This is where the project starts looking like an actual system instead of a script.

---

# Phase 7 — `ui_demo`: simple user-facing prototype

Goal: make a demo people can understand in 30 seconds.

Use Streamlit first. Don’t waste time on a fancy React frontend yet.

### UI

Add:

```text
- live camera preview
- text box for live questions
- text box for memory questions
- recent memory timeline
- selected memory images
- latency display
```

### Success criteria

A teammate can use it without reading your code.

This matters for internship evaluation because demos are judged emotionally too. A working UI makes the project feel real.

---

# Phase 8 — `eval_harness`: evaluate latency, retrieval, and answer quality

Goal: turn your prototype into something measurable.

### Evaluation dimensions

For live QA:

```text
- end-to-end latency
- VLM inference latency
- frame capture latency
- answer correctness
- failure rate
```

For memory QA:

```text
- retrieval recall@k
- answer correctness
- timestamp accuracy
- location accuracy
- hallucination rate
```

For efficiency:

```text
- number of frames sent per query
- image resolution
- number of visual tokens
- cost per query
- latency per query
```

### Dataset

Record short controlled videos:

```text
video_001: desk scene
video_002: walking outside
video_003: convenience store shelf
video_004: train station sign
video_005: object placed then removed
```

Create questions:

```json
{
  "question": "Did I see a red notebook?",
  "answer": "Yes",
  "evidence_time": "00:01:23",
  "type": "object_memory"
}
```

### Success criteria

You can report:

```text
Current baseline:
- live QA latency: 3.2 sec average
- memory retrieval recall@5: 0.78
- answer accuracy: 0.72
- average frames sent per query: 4
```

This becomes presentation material.

---

# Phase 9 — `efficient_vlm`: research/optimization layer

Only after the system works, start research-style experiments.

Possible directions:

## Direction A: adaptive frame selection

Instead of sending every sampled frame to VLM, choose frames based on:

```text
- scene change
- object novelty
- embedding distance
- motion intensity
- user query relevance
```

Baseline:

```text
send latest 4 frames
```

Improved:

```text
send top-k informative frames
```

Evaluate:

```text
accuracy vs latency vs cost
```

## Direction B: visual memory compression

Store different memory levels:

```text
Level 0: raw image
Level 1: thumbnail
Level 2: VLM summary
Level 3: structured metadata
Level 4: embedding only
```

Question: when do you need the raw image again?

This is directly relevant to:

```text
long-term visual memory + efficient inference
```

## Direction C: query-aware memory retrieval

For user question:

```text
Where did I see a red sign?
```

Retrieve by:

```text
semantic text embedding + object metadata + color metadata + location/time filter
```

Compare against naive vector search.

## Direction D: adaptive VLM routing

Use cheaper models for easy frames and stronger models for hard frames:

```text
small VLM → confidence low → large VLM
```

This is practical and research-ish.

---

# Recommended order

I’d do this exact order:

```text
0. Current working RTMP + VLM QA
1. vlm_smoke
2. memory_log
3. memory_search
4. vector_memory
5. backend_api
6. ui_demo
7. eval_harness
8. efficient_vlm experiments
9. smartphone/wearable integration
```

Notice I put **smartphone integration late**. That is intentional. It can eat tons of time and not teach much if the core memory system is weak.

---

# What you should build next

Your next feature should be:

```text
Automatic memory logging every 2–5 seconds
```

Not ChromaDB yet. Not vLLM yet. Not WebRTC yet.

Implement this:

```text
RTMP stream
→ sample frame every 3 sec
→ ask VLM for structured JSON summary
→ save image
→ append memory record to memories.jsonl
```

Once you have 50–100 memory records, then add search.

The first real milestone is:

```text
“I walked around / moved the camera for 5 minutes, and now I can ask what I saw earlier.”
```

That directly matches the project overview and is demo-worthy.
