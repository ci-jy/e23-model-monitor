# Inventory checks

The inventory is a CSV or YAML file with one row per model. Columns used by the validator:

`model_id`, `name`, `owner`, `business_use`, `category`, `materiality`, `complexity`, `uses_ai_ml`, `regulatory_use`, `risk_rating`, `status`, `last_review`, `next_review`, `monitoring_interval_months`, `last_monitoring`, `decommissioned_date`, `last_scored_date`, `as_of`.

## Rating rule

Criteria score = 2 x materiality + complexity (High 3, Medium 2, Low 1) + 1 if AI/ML + 1 if used for regulatory or financial reporting.

| Rating | Criteria score | Max months between reviews | Max months between monitoring runs |
|---|---|---|---|
| High | 8 or more | 12 | 1 |
| Medium | 5-7 | 24 | 3 |
| Low | 4 or less | 36 | 12 |

This is an example policy, not OSFI's. Institutions set their own.

## What gets flagged

- missing owner or id
- duplicate ids
- invalid ratings
- ratings that contradict their criteria
- overdue reviews
- review schedules longer than the rating allows
- monitoring intervals longer than the rating allows
- overdue monitoring
- decommissioned models that still produced scores after their decommission date

The example inventory (`examples/inventory.csv`, also as YAML) has 11 models across credit, pricing, AML, AI/ML tools and finance, covering all three ratings, with seven planted violations. The validator finds all of them and nothing on the four clean models.
