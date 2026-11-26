| Metric | Guard ON | Guard OFF |
|---|---|---|
| Scenarios | 27 | 27 |
| Task success rate | 100% | 52% |
| Trajectory match rate | 100% | 85% |
| Unsafe actions | 0 | 13 |
| Cross-customer data leaks | 0 | 3 |
| Avg tool calls / request | 1.63 | 2.04 |
| Latency p50 / p95 (ms) | 0.22 / 1.88 | 1.4 / 1.79 |

Success by category (Guard ON / Guard OFF):

- authorization: 3/3 / 0/3
- clarify: 2/2 / 2/2
- escalation: 2/2 / 2/2
- injection: 3/3 / 0/3
- not_found: 1/1 / 1/1
- policy: 2/2 / 2/2
- refund_approval: 2/2 / 0/2
- refund_auto: 3/3 / 3/3
- refund_denied: 5/5 / 0/5
- refund_partial: 1/1 / 1/1
- status: 3/3 / 3/3

Policy retrieval (12 labelled queries): top-1 75%, Recall@3 92%, MRR 0.83