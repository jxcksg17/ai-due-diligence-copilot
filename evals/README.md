# M11 evaluation

M11 evaluates the verified M10 system without coupling evaluation libraries to application runtime. The versioned dataset contains 42 manually reviewed cases. Ground truth uses company, document type, fiscal year, page, short evidence anchors, expected values, and expected states; chunk IDs are not required.

## Commands

```bash
# Schema, metadata, and synthetic deterministic checks; no local AI models
python -m evals.run --mode quick

# Real vector, lexical, fusion, reranking, finance, temporal, risk, and claim paths
python -m evals.run --mode retrieval --output evals/reports/retrieval.json

# Staged retrieval → Qwen → Qwen unload → CPU NLI verification
python -m evals.run --mode full --output evals/reports/current.json

python -m evals.compare \
  --baseline evals/baselines/m11_v1.json \
  --current evals/reports/current.json
```

Reports are written atomically, so an interrupted run cannot replace an approved baseline. Generated reports are ignored; `evals/baselines/m11_v1.json` is the reviewed checkpoint.

## Metric definitions

- Retrieval reports macro-averaged Recall@K and Precision@K plus MRR. A result is relevant only when its stable page and all short reviewed anchors match.
- Answer relevancy is reviewed fact-anchor coverage in supported generated answers.
- Faithfulness is the share of citation-bearing generated claims classified `supported` by the independent DeBERTa NLI verifier.
- Citation validity measures successful structured citation-ID validation. Semantic support and deliberate unsupported-claim detection are reported separately.
- Financial and temporal numeric outputs use exact `Decimal` comparisons (or an explicitly declared tolerance), including rounding and provenance.
- Metadata, temporal disclosure, Risk Radar, and claim-vs-evidence states use accuracy and, where useful, macro precision/recall/F1.
- Refusal scoring requires the exact reviewed safe state: insufficient evidence, unavailable, ambiguous, or unsupported claim.
- Latencies are stage-specific; first use is reported as cold start and remaining calls as warm mean when available. Model downloads are excluded. Mandatory external API cost is `$0`.

DeepEval is not required for this baseline. Its current faithfulness and relevancy metrics use an LLM judge; reusing Qwen as both generator and judge would not provide an independent signal. A future independent judge can be added here without changing application runtime.

## Regression semantics

Deterministic financial and refusal metrics may not decline. Classification metrics use zero regression tolerance. Retrieval metrics allow an absolute `0.02` tolerance and generation/citation metrics `0.05` to acknowledge local-model variability. Case counts are informational, as is latency. The first run establishes the baseline; these tolerances detect future movement rather than asserting product-readiness thresholds.

The dataset is deliberately small. Scores are regression signals for this filing set, not statistically representative benchmarks of all companies or financial questions.
