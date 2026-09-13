---
title: FastAPI
type: entity
domain: mlops
roles: [ml-engineer, ai-engineer, mlops-engineer]
difficulty: core
frequency: high
status: drafted
tags: [tooling]
updated: 2026-09-13
sources: []
---

# FastAPI

## What it is
A modern Python web framework built for building APIs quickly with automatic request validation, async support, and self-generating documentation — it has become the default choice for wrapping a trained model behind an HTTP endpoint because it fits ML serving needs (typed I/O, async I/O for model calls, low boilerplate) almost exactly.

## Core concepts
- **Pydantic models for request/response validation**: define input/output schemas as Pydantic classes with typed fields; FastAPI validates incoming JSON against them automatically and returns a 422 with a precise error before your code ever runs — this is the main reason it's preferred over Flask for model APIs, where malformed input is a real production risk.
- **Async/await support**: endpoints can be defined `async def`, letting the server handle other requests while one is awaiting I/O (a downstream API call, a database query) — for CPU-bound model inference itself, async doesn't parallelize compute, but it matters a lot when the endpoint also calls out to a feature store, vector DB, or another service.
- **Dependency injection (`Depends`)**: shared logic (auth checks, a loaded model instance, a DB session) is declared once and injected into any endpoint that needs it — avoids re-loading a model per request and centralizes cross-cutting concerns.
- **Automatic OpenAPI docs**: `/docs` (Swagger UI) and `/redoc` are generated from your type hints and Pydantic models with zero extra work — useful both for consumers of the API and as living documentation that can't drift from the code.
- **`uvicorn`/`gunicorn` as the ASGI server**: FastAPI is a framework, not a server — it runs under an ASGI server (`uvicorn`, optionally behind `gunicorn` with multiple worker processes) which is what actually binds the port and handles concurrent connections.
- **Startup/shutdown events (lifespan)**: load the model once at process startup (not per-request) via a lifespan context manager — the standard pattern to avoid the latency and memory cost of reloading a model on every call.

## Code
```python
from fastapi import FastAPI
from pydantic import BaseModel, Field
from contextlib import asynccontextmanager
import joblib

ml_models = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    ml_models["classifier"] = joblib.load("model.pkl")   # load once at startup
    yield
    ml_models.clear()

app = FastAPI(lifespan=lifespan)

class PredictRequest(BaseModel):
    features: list[float] = Field(..., min_length=10, max_length=10)

class PredictResponse(BaseModel):
    prediction: int
    probability: float

@app.post("/predict", response_model=PredictResponse)
async def predict(req: PredictRequest):
    model = ml_models["classifier"]
    proba = model.predict_proba([req.features])[0]
    return PredictResponse(prediction=int(proba.argmax()), probability=float(proba.max()))

@app.get("/health")
async def health():
    return {"status": "ok"}
```
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 4
```

## When to use it vs alternatives
- **vs Flask**: Flask is simpler and more established but has no built-in validation or async support — you'd hand-roll request validation that FastAPI gives you for free via type hints, and it's synchronous by default.
- **vs a managed serving platform (Databricks Model Serving, SageMaker endpoints, vLLM/TGI for LLMs)**: those are purpose-built for model serving (batching, autoscaling, GPU scheduling baked in) and usually the better choice for a single model at scale; FastAPI is the right layer when you need custom business logic around the model call (auth, multi-model orchestration, calling other services), or you're building the serving layer yourself.
- **vs gRPC**: gRPC gives lower-latency binary serialization and strict contracts, better for internal service-to-service calls at high throughput; FastAPI/REST is easier to consume from arbitrary clients and to document, the usual choice for external or less latency-critical APIs.

## Interview angle
**Q. Why load the model at startup instead of inside the endpoint function?**
Loading inside the endpoint reloads the model (disk I/O, deserialization, possibly GPU memory allocation) on every single request, adding massive latency and risking resource exhaustion under concurrent load. Loading once via a lifespan/startup hook keeps the model resident in memory, shared across all requests handled by that worker process.

**Q. Does making an endpoint `async def` speed up model inference?**
Not by itself — CPU-bound inference still blocks the event loop unless offloaded (e.g. via `run_in_threadpool` or a separate process/GPU worker). `async` helps when the endpoint does I/O-bound work (calling a feature store, another microservice, a database) concurrently with other requests; for pure synchronous CPU-bound inference, a sync `def` endpoint (which FastAPI runs in a thread pool automatically) or a dedicated worker process is often more appropriate.

**Q. How would you validate that a request's feature vector matches what the model expects, and what should happen when it doesn't?**
Encode the expected shape/types directly in the Pydantic request model (fixed-length list, typed fields, range constraints via `Field`) so FastAPI rejects malformed requests with a 422 before they reach the model — catching schema mismatches at the API boundary is cheaper and clearer than letting the model crash or silently mispredict on bad input.

## Traps
- Loading the model per-request instead of once at startup — the most common performance bug in FastAPI model-serving code.
- Writing CPU-bound inference as `async def` and assuming it's now non-blocking — it still occupies the event loop unless explicitly offloaded.
- Skipping response models (`response_model=...`) — loses the automatic response validation/serialization and the self-documenting OpenAPI schema.
- Running a single `uvicorn` worker in production — one process can't use multiple CPU cores; scale via `--workers` or a process manager, or handle it at the orchestration layer (Kubernetes replica count) instead.

## Related
[[api-design-for-ml]], [[model-serving-patterns]], [[docker]], [[batch-vs-realtime-inference]]
