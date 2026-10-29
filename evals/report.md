# grounded-rag eval report

backend: **cohere** · n=12 · k=5

| metric | value |
|---|---|
| recall_at_k | 1.0 |
| mrr | 1.0 |
| must_contain_rate | 1.0 |
| cited_rate | 1.0 |
| refusals_correct | True |
| acl_leaks | 0 |
| p50_ms | 18061 |

| id | recall@k | rr | must_contain | cited | refusal_ok | acl_leak | ms | answer |
|---|---|---|---|---|---|---|---|---|
| q1 | 1.0 | 1.0 | True | True | None | False | 12337 | 14 days. |
| q2 | 1.0 | 1.0 | True | True | None | False | 49984 | A Director General can approve business class for flights of 9 hours or longer. |
| q3 | 1.0 | 1.0 | True | True | None | False | 8283 | The threshold above which a contract must be competed on the tendering service is $40,000. |
| q4 | 1.0 | 1.0 | True | True | None | False | 18061 | The Assistant Deputy Minister approves sole-source justifications above 40,000 dollars. |
| q5 | 1.0 | 1.0 | True | True | None | False | 44262 | No. Protected or classified information must never be entered into a tool that is not deployed within the department's own environment. |
| q6 | 1.0 | 1.0 | True | True | None | False | 12794 | The enterprise AI workspace is accredited for Protected B and is deployed on premises on Canadian soil. |
| q7 | 1.0 | 1.0 | True | True | None | False | 15365 | A suspected security incident must be reported to the departmental Security Operations Centre within one hour of discovery. |
| q8 | None | None | None | False | True | False | 1075 | I'm sorry, I could not find any information about reporting a suspected security incident to the SOC. |
| q9 | 1.0 | 1.0 | True | True | None | False | 32477 | Employees accrue 20 days of annual leave per year after 8 years of service. |
| q10 | 1.0 | 1.0 | True | True | None | False | 21122 | A manager must respond to an accommodation request within 10 working days. |
| q11 | 1.0 | 1.0 | True | True | None | False | 11841 | Receipts are not required for meal allowances. |
| q12 | 1.0 | 1.0 | True | True | None | False | 28468 | A supplier must hold a Designated Organization Screening before award of a contract involving Protected B information. |

Gates: acl_leaks must be 0; refusals_correct must be true; recall@k >= 0.9.
