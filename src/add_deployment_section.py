with open('README.md', 'r', encoding='utf-8') as f:
    content = f.read()

deployment_section = """

## Deployment

Wrapped the logistic regression model in a Flask REST API (`/predict` endpoint) that reproduces the full preprocessing pipeline (cleaning, feature engineering, encoding, scaling) on a single incoming applicant using the saved training-set fit values, then applies the cost-tuned 0.70 threshold.

Containerized the API with Docker (Python 3.10-slim base image) so it runs identically regardless of host environment.

## Load testing

Load-tested the containerized API with Locust: 50 concurrent simulated users, 5 users/second ramp-up, over a 60-second sustained run.

| Metric | Result |
|---|---|
| Total requests | 2,472 |
| Failures | 0 (0%) |
| Median response time | 340 ms |
| 95th percentile (p95) | 670 ms |
| 99th percentile (p99) | 770 ms |
| Sustained throughput | ~29 requests/second |

Zero failed requests across the full test run. Full results and charts available in `results/`.

Note: this test ran on local Docker Desktop (Windows/WSL2), not production infrastructure - these numbers reflect this specific environment, not a cloud deployment benchmark.
"""

content = content.rstrip() + "\n" + deployment_section

with open('README.md', 'w', encoding='utf-8', newline='\n') as f:
    f.write(content)

print("Added Deployment and Load testing sections to README")