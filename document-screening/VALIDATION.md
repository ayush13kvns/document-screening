# Delivery validation

Verified on Python 3.12.14 with the pinned dependency constraints.

- 20 tests passed, including inference parity, evidence rules, leakage checks, request authentication, payload limits, input errors and demo protection.
- Native training, JSONL batch scoring, folder extraction and ingestion-to-classification completed.
- A real Uvicorn subprocess served HTTP 200 from `/`, `/health/live`, `/health/ready` and authenticated `POST /v1/classify`.
- HTTP output decisions matched the documented five sample decisions.
- `python -m compileall -q app` completed.
- The installed runtime/test packages satisfied requirements and constraints without fetching additional packages.
- Python 3.11/3.13 and Windows are configured/documented but were not executed in this environment. GitHub Actions has not run yet.
- Docker engine was unavailable, so the Docker image was not built or run.
- One test-client deprecation warning originates from the Starlette/AnyIO dependency combination; it did not affect passing tests.

Synthetic held-out metrics are execution checks, not evidence of company-data performance. The 36-row demonstration has only 12 test examples. Calibration and error estimates are unsuitable for production conclusions.

No Git remote was contacted or pushed. No real customer documents or credentials are included.
