---
title: Kubernetes
type: entity
domain: mlops
roles: [mlops-engineer, ml-engineer]
difficulty: intermediate
frequency: medium
status: drafted
tags: [tooling]
updated: 2026-09-13
sources: []
---

# Kubernetes

## What it is
The standard container-orchestration platform — it schedules containers across a cluster of machines, restarts them on failure, scales them up and down, and routes traffic to them. For ML specifically, it's the substrate under most self-managed model-serving deployments and distributed training jobs that don't live on a fully managed platform.

## Core concepts
- **Pod**: the smallest deployable unit — one or more tightly coupled containers sharing network/storage namespace. Almost always one container per pod in practice, with sidecars as the main exception (e.g. a logging/metrics sidecar next to a model server).
- **Deployment**: manages a set of identical Pod replicas, handling rolling updates and rollback — the standard way to run a stateless service (like a model-serving API) with a desired replica count.
- **Service**: a stable network endpoint (virtual IP + DNS name) in front of a changing set of Pods — Pods come and go (rescheduled, scaled, replaced), the Service address doesn't.
- **ReplicaSet / HPA**: a ReplicaSet keeps N pod replicas alive; a HorizontalPodAutoscaler adjusts N based on metrics (CPU, memory, or custom metrics like request queue depth/GPU utilization) — the mechanism behind autoscaling a model-serving fleet under load.
- **ConfigMap / Secret**: externalize configuration and credentials from the container image so the same image runs across dev/staging/prod with different injected config.
- **Resource requests & limits**: `requests` (what the scheduler guarantees when placing the pod) vs `limits` (the hard ceiling before throttling/OOM-kill) — critical for GPU workloads, where a pod requesting a GPU (`nvidia.com/gpu: 1`) is scheduled only onto a node with a free GPU, and misconfigured limits either waste capacity or get killed under memory pressure.
- **Namespaces & RBAC**: logical cluster partitioning for multi-team isolation, combined with role-based access control — relevant when multiple ML teams share one cluster.

## Code
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: model-service
spec:
  replicas: 3
  selector:
    matchLabels: {app: model-service}
  template:
    metadata:
      labels: {app: model-service}
    spec:
      containers:
        - name: model-service
          image: my-registry/model-service:1.0
          ports: [{containerPort: 8000}]
          resources:
            requests: {cpu: "500m", memory: "1Gi", nvidia.com/gpu: "1"}
            limits: {cpu: "1", memory: "2Gi", nvidia.com/gpu: "1"}
          readinessProbe:
            httpGet: {path: /health, port: 8000}
            initialDelaySeconds: 5
---
apiVersion: v1
kind: Service
metadata:
  name: model-service
spec:
  selector: {app: model-service}
  ports: [{port: 80, targetPort: 8000}]
---
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: model-service-hpa
spec:
  scaleTargetRef: {apiVersion: apps/v1, kind: Deployment, name: model-service}
  minReplicas: 2
  maxReplicas: 10
  metrics:
    - type: Resource
      resource: {name: cpu, target: {type: Utilization, averageUtilization: 70}}
```

## When to use it vs alternatives
- **vs a managed serverless platform (Cloud Run, SageMaker endpoints, Databricks Model Serving)**: managed platforms trade flexibility for far less operational overhead — right choice when the team doesn't need custom scheduling logic (GPU bin-packing across many models, custom autoscaling metrics) or multi-cloud portability. Kubernetes wins when you need that control, run at large scale, or already have platform-team investment in it.
- **vs Ray Serve**: Ray Serve runs *on top of* Kubernetes (or standalone) and is purpose-built for Python model-serving with less YAML boilerplate for common ML patterns (model composition, batching); raw Kubernetes gives more general infra control at the cost of writing more of that logic yourself.
- **vs Docker Compose**: Compose is fine for local development or a single-machine deployment; Kubernetes is for multi-node, self-healing, autoscaled production — Compose has no answer for "what happens when a machine dies."

## Interview angle
**Q. How would you serve a large model that needs a whole GPU per replica, and what does the scheduler need to know?**
Request the GPU explicitly (`nvidia.com/gpu: 1` in both requests and limits, since GPUs aren't fractionally shareable by default in vanilla Kubernetes) so the scheduler only places the pod on a node with an available GPU; set replica count based on expected concurrent load and GPU memory headroom for the model, and use readiness probes so traffic doesn't route to a pod before the model is loaded into GPU memory.

**Q. What's the difference between a liveness probe and a readiness probe, and why does a model-serving pod need both configured differently?**
Liveness determines whether the container should be restarted (it's deadlocked/crashed); readiness determines whether it should receive traffic right now. A model server can be alive (process running) but not ready (still loading a multi-GB model into memory at startup) — without a readiness probe tuned for load time, traffic hits pods before they can serve, causing request failures during every rollout.

**Q. A Deployment rollout is stuck — new pods aren't becoming ready. How do you debug it?**
Check `kubectl describe pod` for scheduling failures (no node with enough resources/GPU), `kubectl logs` on the new pod for application-level startup errors, and readiness probe configuration (too short an `initialDelaySeconds` for model load time is a classic cause) before assuming it's a cluster capacity issue.

## Traps
- Setting no resource `limits` — a runaway process can starve other workloads on the same node; setting `limits` too tight for memory causes OOM-kills under normal load spikes.
- Confusing `requests` (scheduling guarantee) with `limits` (hard ceiling) — a pod can be scheduled fine on `requests` and still get OOM-killed if it exceeds `limits`.
- Not setting a readiness probe on model-serving pods — traffic is routed to a pod the instant it starts, before the model is loaded, causing errors during every scale-up or rollout.
- Treating raw Kubernetes YAML as the natural first step for an ML team — often a higher-level platform (managed serving, or Ray Serve on Kubernetes) removes most of this boilerplate for standard model-serving needs.

## Related
[[kubernetes-for-ml]], [[docker]], [[model-serving-patterns]], [[scalability-patterns]], [[ray]]
