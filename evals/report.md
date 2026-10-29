# grounded-rag eval report

backend: **offline** · n=12 · k=5

| metric | value |
|---|---|
| recall_at_k | 1.0 |
| mrr | 0.788 |
| must_contain_rate | 0.727 |
| cited_rate | 1.0 |
| refusals_correct | False |
| acl_leaks | 0 |
| p50_ms | 2 |

| id | recall@k | rr | must_contain | cited | refusal_ok | acl_leak | ms | answer |
|---|---|---|---|---|---|---|---|---|
| q1 | 1.0 | 1.0 | True | True | None | False | 3 | According to the documents: # Directive on Travel and Hospitality

Employees must book air travel through the departmental travel service at least 14 days befor |
| q2 | 1.0 | 0.5 | True | True | None | False | 2 | According to the documents: Business class may be approved by a Director General for flights of 9 hours or longer. |
| q3 | 1.0 | 0.3333333333333333 | False | True | None | False | 2 | According to the documents: Every output used in a decision affecting a person must be reviewed by a human before use, and the use of the tool must be documente |
| q4 | 1.0 | 1.0 | True | True | None | False | 2 | According to the documents: Sole-source justifications above 40,000 dollars require approval by the Assistant Deputy Minister. |
| q5 | 1.0 | 1.0 | False | True | None | False | 2 | According to the documents: Protected A information requires access control; Protected B information requires encryption at rest and in transit and must stay on |
| q6 | 1.0 | 1.0 | True | True | None | False | 2 | According to the documents: The department's enterprise AI workspace is accredited for Protected B. |
| q7 | 1.0 | 1.0 | True | True | None | False | 3 | According to the documents: # Security Incident Response Standard

A suspected security incident must be reported to the departmental Security Operations Centre |
| q8 | None | None | None | True | False | False | 2 | According to the documents: Every output used in a decision affecting a person must be reviewed by a human before use, and the use of the tool must be documente |
| q9 | 1.0 | 1.0 | True | True | None | False | 2 | According to the documents: # Leave and Accommodation Procedures

Employees accrue 15 days of annual leave per year for the first 8 years of service and 20 days |
| q10 | 1.0 | 0.5 | True | True | None | False | 2 | According to the documents: Requests for workplace accommodation are submitted to the manager, who must respond within 10 working days. |
| q11 | 1.0 | 1.0 | True | True | None | False | 2 | According to the documents: Receipts are not required for meal allowances but are required for every hospitality expense above 50 dollars. |
| q12 | 1.0 | 0.3333333333333333 | False | True | None | False | 2 | According to the documents: Protected A information requires access control; Protected B information requires encryption at rest and in transit and must stay on |

Gates: acl_leaks must be 0; refusals_correct must be true; recall@k >= 0.9 on the offline backend.
