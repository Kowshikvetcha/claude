---
title: Rapid-Fire System-Design Prompts by Role
type: drill
domain: system-design
roles: [data-scientist, ml-engineer, ai-engineer, mlops-engineer, agentic-engineer, fde]
difficulty: advanced
frequency: high
status: drafted
tags: [drill, system-design]
updated: 2026-09-13
---

# Rapid-Fire System-Design Prompts by Role

One-liners only — no scaffolding, no answer key. Pick one, give yourself 45 seconds to structure
an approach out loud (requirements → metrics → data → model → serving → monitoring → failure
modes), then check whether you'd actually cover it that fast in a real round. Use
[[ml-system-design-framework]] or [[llm-system-design-framework]] as the structural checklist when
a prompt has no closer match below.

## Data Scientist

1. Design a system that predicts subscription churn and routes high-risk users to retention offers. — [[case-churn-prediction]]
2. Design an experimentation platform that runs and analyzes A/B tests at scale.
3. How would you design a fraud detection system for online payments? — [[case-fraud-detection]]
4. Design a demand forecasting system for a retail chain's inventory planning. — [[case-demand-forecasting]]
5. Design a system that recommends products to users on an e-commerce site. — [[case-recommendation-system]]
6. How would you design a metric pipeline that catches a regression in a core business KPI before a stakeholder does?
7. Design a lead-scoring system that ranks sales leads by likelihood to convert.
8. How would you design a search ranking system for an internal product catalog? — [[case-search-ranking]]

## ML Engineer

1. Design a system that recommends products to users in real time. — [[case-recommendation-system]]
2. Design a fraud detection system that scores transactions within 100ms. — [[case-fraud-detection]]
3. Design a demand forecasting pipeline that retrains weekly on new sales data. — [[case-demand-forecasting]]
4. How would you design a search ranking system for a marketplace? — [[case-search-ranking]]
5. Design a feature store that serves both training and low-latency online inference. — [[case-realtime-feature-pipeline]]
6. How would you design an ML system to detect and handle training-serving skew?
7. Design a system that predicts equipment failure from streaming sensor data.
8. How would you design an end-to-end pipeline from a Databricks bronze layer to a served model? — [[case-ml-platform-design]]

## AI Engineer

1. Design a RAG-based assistant that answers questions over a company's internal documents. — [[case-rag-assistant]]
2. How would you design a semantic search system over a large product catalog? — [[case-search-ranking]]
3. Design a customer support chatbot that must ground every answer in a knowledge base. — [[case-rag-assistant]]
4. How would you design an LLM-powered "why we recommended this" explainer for an e-commerce app? — [[case-recommendation-system]]
5. Design an evaluation harness for comparing two prompt versions in production.
6. How would you design a router that sends queries to a small or large model based on cost and difficulty? — [[case-llm-cost-reduction]]
7. Design a fraud-alert summarization layer that turns model scores into analyst-readable explanations. — [[case-fraud-detection]]
8. How would you design a pipeline that extracts structured fields from scanned invoices and contracts? — [[case-document-extraction-pipeline]]

## MLOps Engineer

1. Design a model monitoring system that detects data and concept drift across dozens of models.
2. How would you design a CI/CD pipeline that gates a model's promotion to production?
3. Design a retraining pipeline for a fraud detection model that must never silently degrade. — [[case-fraud-detection]]
4. Design a feature store shared across a demand forecasting team and a recommendations team. — [[case-demand-forecasting]]
5. How would you design a shadow/canary rollout for a new search ranking model? — [[case-search-ranking]]
6. Design a cost-aware autoscaling strategy for a real-time recommendation-serving layer. — [[case-recommendation-system]]
7. How would you design an ML platform that supports self-service model deployment for multiple teams? — [[case-ml-platform-design]]
8. Design an incident-response runbook for a demand forecasting model that degraded silently over a quarter. — [[case-demand-forecasting]]

## Agentic Engineer

1. Design a multi-step agent that automates fraud-case investigation for an analyst team. — [[case-fraud-detection]]
2. How would you design a tool-using agent that plans and books a multi-city trip?
3. Design an agent that answers customer questions and escalates to a human when uncertain. — [[case-agentic-support-automation]]
4. How would you design a multi-agent system where one agent retrieves data and another drafts a report?
5. Design a guardrail system that stops an agent from looping or taking an unsafe action.
6. How would you design an agent that recommends and executes inventory reorders from a demand forecast? — [[case-demand-forecasting]]
7. Design an evaluation framework for measuring an agent's task success rate in production.
8. How would you design a search agent that decides when to retrieve versus answer directly? — [[case-search-ranking]]

## FDE

1. How would you design a RAG prototype for a client's internal document set in under a week?
2. Design a fraud-detection proof-of-concept for a client with only 3 months of transaction history. — [[case-fraud-detection]]
3. How would you scope and prototype a demand forecasting tool for a client's supply chain team? — [[case-demand-forecasting]]
4. Design a recommendation feature for a client's e-commerce site given only clickstream logs. — [[case-recommendation-system]]
5. How would you design a search-ranking improvement for a client's catalog under a tight deadline? — [[case-search-ranking]]
6. Design an agentic workflow that automates a client's manual ticket-triage process. — [[case-agentic-support-automation]]
7. How would you propose an architecture for a client with legacy on-prem data and no cloud budget?
8. Design a lightweight monitoring setup for a client prototype with no dedicated MLOps team.

## Related

[[ml-system-design-framework]] · [[llm-system-design-framework]] · [[case-recommendation-system]] ·
[[case-search-ranking]] · [[case-fraud-detection]] · [[case-demand-forecasting]] ·
[[case-churn-prediction]] · [[case-rag-assistant]] · [[case-agentic-support-automation]] ·
[[case-document-extraction-pipeline]] · [[case-llm-cost-reduction]] · [[case-ml-platform-design]] ·
[[case-realtime-feature-pipeline]] · [[vs-vector-db-options]] · [[moc-system-design]]
