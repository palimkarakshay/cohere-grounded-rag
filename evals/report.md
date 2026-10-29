# grounded-rag eval report

backend: **offline** · n=12 · k=5

| metric | value |
|---|---|
| recall_at_k | 1.0 |
| mrr | 1.0 |
| must_contain_rate | 1.0 |
| cited_rate | 1.0 |
| refusals_correct | True |
| acl_leaks | 0 |
| p50_ms | 3 |

| id | recall@k | rr | must_contain | cited | refusal_ok | acl_leak | ms | answer |
|---|---|---|---|---|---|---|---|---|
| q1 | 1.0 | 1.0 | True | True | None | False | 3 | According to the documents: # Directive on Travel and Hospitality

Employees must book air travel through the departmental travel service at least 14 days befor |
| q2 | 1.0 | 1.0 | True | True | None | False | 2 | According to the documents: Business class may be approved by a Director General for flights of 9 hours or longer. |
| q3 | 1.0 | 1.0 | True | True | None | False | 2 | According to the documents: Contracts above 40,000 dollars must go through a competitive process posted on the government tendering service, unless an exception |
| q4 | 1.0 | 1.0 | True | True | None | False | 2 | According to the documents: Sole-source justifications above 40,000 dollars require approval by the Assistant Deputy Minister. |
| q5 | 1.0 | 1.0 | True | True | None | False | 2 | According to the documents: Protected or classified information must never be entered into a tool that is not deployed within the department's own environment. |
| q6 | 1.0 | 1.0 | True | True | None | False | 2 | According to the documents: The department's enterprise AI workspace is accredited for Protected B. |
| q7 | 1.0 | 1.0 | True | True | None | False | 3 | According to the documents: # Security Incident Response Standard

A suspected security incident must be reported to the departmental Security Operations Centre |
| q8 | None | None | None | False | True | False | 2 | The provided documents do not contain the answer to that question. |
| q9 | 1.0 | 1.0 | True | True | None | False | 3 | According to the documents: # Leave and Accommodation Procedures

Employees accrue 15 days of annual leave per year for the first 8 years of service and 20 days |
| q10 | 1.0 | 1.0 | True | True | None | False | 3 | According to the documents: Requests for workplace accommodation are submitted to the manager, who must respond within 10 working days. |
| q11 | 1.0 | 1.0 | True | True | None | False | 3 | According to the documents: Receipts are not required for meal allowances but are required for every hospitality expense above 50 dollars. |
| q12 | 1.0 | 1.0 | True | True | None | False | 3 | According to the documents: Professional services contracts that involve access to Protected B information require the supplier to hold a Designated Organizatio |

Gates: acl_leaks must be 0; refusals_correct must be true; recall@k >= 0.9 on the offline backend.
