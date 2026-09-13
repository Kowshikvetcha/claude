---
title: Docker
type: entity
domain: mlops
roles: [ml-engineer, mlops-engineer, ai-engineer]
difficulty: core
frequency: high
status: drafted
tags: [tooling]
updated: 2026-09-13
sources: []
---

# Docker

## What it is
The standard tool for packaging an application (and an ML model's runtime — code, dependencies, weights, system libraries) into a portable, reproducible container image. It solves "works on my machine" by making the machine part of the shipped artifact.

## Core concepts
- **Image vs container**: an image is a read-only, layered filesystem snapshot built from a `Dockerfile`; a container is a running (or stopped) instance of that image — you build an image once and can run many containers from it.
- **Layers & caching**: each `Dockerfile` instruction creates a cached layer; Docker reuses unchanged layers on rebuild, which is why instruction *order* matters — put rarely-changing steps (installing dependencies) before frequently-changing ones (copying application code) to maximize cache hits.
- **`Dockerfile` essentials**: `FROM` (base image), `COPY`/`ADD` (bring files in), `RUN` (execute at build time), `CMD`/`ENTRYPOINT` (what runs when the container starts), `WORKDIR`, `EXPOSE`, `ENV`.
- **Multi-stage builds**: use one stage to compile/install heavy build dependencies, then copy only the needed artifacts into a slim final stage — keeps the shipped image small without giving up a full build toolchain during construction.
- **Volumes vs bind mounts**: volumes are Docker-managed persistent storage outside the container's writable layer; bind mounts map a host path directly in — relevant for mounting model weights or data without baking them into the image itself.
- **Networking**: containers get their own network namespace by default; `-p host:container` publishes a port, and container-to-container communication within a user-defined bridge network resolves by container/service name (this is what Docker Compose and Kubernetes Services build on).
- **Registries**: images are pushed to and pulled from a registry (Docker Hub, ECR, GCR, an internal registry) — the distribution mechanism that makes an image runnable on any machine with Docker installed, including a Kubernetes cluster.

## Code
```dockerfile
# Multi-stage build for a Python model-serving image
FROM python:3.11-slim AS builder
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

FROM python:3.11-slim
WORKDIR /app
COPY --from=builder /root/.local /root/.local
COPY app/ ./app/
COPY model/ ./model/
ENV PATH=/root/.local/bin:$PATH
EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```
```bash
docker build -t model-service:1.0 .
docker run -p 8000:8000 --env MODEL_VERSION=1.0 model-service:1.0
docker push my-registry/model-service:1.0
```

## When to use it vs alternatives
- **vs a plain venv/conda environment**: a venv isolates Python dependencies but not system libraries, OS version, or non-Python binaries (CUDA drivers' userspace libs, `libgomp`, etc.) — Docker isolates the whole runtime, which is what production and cross-team reproducibility actually need.
- **vs Kubernetes**: Docker builds and runs individual containers; Kubernetes orchestrates many containers across many machines (scaling, scheduling, self-healing) — they're complementary, not alternatives; you build with Docker and deploy the image via Kubernetes, see [[kubernetes]].
- **vs Conda environments shipped as-is**: conda-pack/similar can freeze an environment for reproducibility without full OS-level isolation — lighter weight, but doesn't solve system-library or OS-level drift the way a container does.

## Interview angle
**Q. Why does `Dockerfile` instruction order affect build speed, and what's the practical rule?**
Docker caches each layer and invalidates a layer (and everything after it) the moment its inputs change. Put `COPY requirements.txt` + `RUN pip install` before `COPY . .` (application code) — code changes daily and would otherwise bust the cache and force a full dependency reinstall on every build if ordered first.

**Q. How do you keep a model-serving image small when the model weights are several GB?**
Options: mount weights as a volume/bind mount at runtime instead of baking them into the image (keeps the image itself lean and lets you swap model versions without rebuilding); use multi-stage builds to avoid shipping build-time toolchains; choose a slim/distroless base image. Whether to bake weights in vs mount them depends on whether you want the image itself to be the versioned, immutable artifact (bake in) or want to swap models without rebuilding (mount).

**Q. What's the difference between `CMD` and `ENTRYPOINT`, and why does it matter for a serving container?**
`ENTRYPOINT` is the fixed executable the container always runs; `CMD` supplies default arguments that can be overridden at `docker run` time. For a serving container you typically want `ENTRYPOINT` fixed (e.g. `uvicorn`) and `CMD` overridable (e.g. host/port/workers) so operators can tweak runtime parameters without touching the image.

## Traps
- Running the container process as root by default — a real security issue in production; add a non-root `USER` in the Dockerfile.
- Baking secrets (API keys, credentials) into image layers via `ENV` or `COPY` — they persist in the image history even if a later layer "removes" them; use runtime secret injection instead.
- Not pinning base image tags (`python:3.11` instead of a specific digest/patch tag) — silent behaviour drift when the base image updates upstream.
- Assuming a container provides the same isolation guarantees as a VM — containers share the host kernel; this matters for security-sensitive multi-tenant workloads.

## Related
[[model-packaging-and-containers]], [[kubernetes]], [[ci-cd-for-ml]], [[model-serving-patterns]]
