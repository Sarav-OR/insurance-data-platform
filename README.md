# Insurance Data Platform — Azure Databricks

A production-grade cloud data engineering project built on Azure Databricks,
Delta Lake and PySpark. Simulates a real-world insurance group data platform
covering policy, claims, premium and fraud analytics.

---

## Architecture

```
Raw Data (Synthetic)
       │
       ▼
┌─────────────┐
│   BRONZE    │  Raw ingestion, schema enforcement, DQ quarantine
│  Delta Lake │  5 tables — customers, policies, claims, premiums, fraud_signals
└──────┬──────┘
       │
       ▼
┌─────────────┐
│   SILVER    │  Cleansed, typed, enriched, deduplicated
│  Delta Lake │  6 tables including claims_enriched joined view
└──────┬──────┘
       │
       ▼
┌─────────────┐
│    GOLD     │  Business KPIs — loss ratios, CLV, collection rates
│  Delta Lake │  6 tables including monthly executive summary
└──────┬──────┘
       │
       ▼
┌─────────────┐
│    FRAUD    │  Rule-based + statistical + behavioural + network scoring
│  Delta Lake │  5 tables including composite fraud score + investigation queue
└─────────────┘
```

---

## Tech Stack

| Component       | Technology                        |
|-----------------|-----------------------------------|
| Platform        | Azure Databricks (Premium)        |
| Storage         | Delta Lake                        |
| Processing      | PySpark 3.5                       |
| Language        | Python 3.11                       |
| Orchestration   | Databricks Workflows              |
| CI/CD           | GitHub Actions                    |
| Data Generation | Faker + Pandas                    |
| Testing         | pytest + pyspark                  |

---

## Project Structure

```
insurance-data-platform/
├── .github/
│   └── workflows/
│       └── ci.yml              # GitHub Actions CI pipeline
├── notebooks/
│   ├── 01_bronze_layer.py      # Bronze ingestion + DQ
│   ├── 02_silver_layer.py      # Silver transformations
│   ├── 03_gold_layer.py        # Gold KPI aggregations
│   └── 04_fraud_detection.py   # Fraud scoring engine
├── src/
│   ├── config.py               # Centralised configuration
│   ├── utils.py                # Shared utility functions
│   └── dq_rules.py             # Declarative DQ rule definitions
├── tests/
│   └── test_dq_rules.py        # Unit tests for DQ rules
├── workflows/
│   └── insurance_platform_job.json  # Databricks job definition
├── requirements.txt
└── README.md
```

---

## Datasets Generated

| Domain         | Records | Description                              |
|----------------|---------|------------------------------------------|
| Customers      | 10,000  | Demographics, segments, tenure           |
| Policies       | 15,000  | All product types, risk scores           |
| Claims         | 8,000   | Full lifecycle, fraud flags              |
| Premiums       | 50,000  | Payment schedule, arrears tracking       |
| Fraud Signals  | ~400    | Rule-based fraud indicators              |
| **Total**      | **~83,400** |                                      |

---

## Key Business KPIs Delivered

- **Loss Ratio** — claims paid vs premium income per product line
- **Collection Rate** — premium payment health by policy type
- **Fraud Detection Rate** — confirmed fraud as % of total claims
- **Customer Lifetime Value** — premium income minus claims per segment
- **Settlement Rate** — % of claims fully settled
- **Monthly Executive Summary** — time-series KPI table

---

## Setup & Running

### Prerequisites
- Azure Databricks workspace (Standard tier or above)
- Databricks Runtime 13.3 LTS or above
- Python 3.11+

### Local Development
```bash
git clone https://github.com/YOUR_USERNAME/insurance-data-platform
cd insurance-data-platform
pip install -r requirements.txt
pytest tests/
```

### Databricks Execution
1. Import notebooks from `notebooks/` into Databricks Workspace
2. Create cluster using Runtime 13.3 LTS
3. Run notebooks in order: 01 → 02 → 03 → 04
4. Or deploy `workflows/insurance_platform_job.json` as a Databricks Job

---

## Medallion Architecture

### Bronze Layer
- Raw data ingested with explicit schema enforcement
- No schema inference — upstream changes caught immediately
- Every record stamped with `_batch_id`, `_ingestion_timestamp`, `_record_hash`
- Bad records quarantined to `rejected_*` tables — never silently dropped

### Silver Layer
- String columns cast to proper types (Date, Boolean, Double)
- Deduplication via `row_number()` window function
- Derived columns — claim severity, risk bands, tenure years
- Wide enriched join table — claims + policy + customer + fraud context

### Gold Layer
- Business KPI aggregations per product line and customer segment
- Loss ratios, settlement rates, collection rates
- Monthly executive summary table for time-series reporting
- Designed for direct BI tool consumption

### Fraud Layer
- Rule-based scoring — 8 configurable fraud rules
- Statistical anomaly detection — Z-score on claim amounts
- Behavioural analysis — customer claim velocity and history
- Network scoring — duplicate detection across customers
- Composite weighted score — configurable weights per signal type
- Investigation queue — prioritised list for fraud analysts

---

## Author

Saravanakumar Ravichandran
Senior Data Engineer — Zürich, Switzerland
linkedin.com/in/saravanakumarravichandran
