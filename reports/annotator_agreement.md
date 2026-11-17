# Inter-Annotator Agreement (Annotator A vs Annotator B)

Documents compared: **5**, clauses compared: **90**. Agreement is measured **before** reconciliation.

| Measure | Percent agreement | Cohen's kappa | Interpretation (Landis & Koch) |
| :--- | :--- | :--- | :--- |
| Clause family | 62.2% | 0.341 | fair |
| Expected flags (clause x rule decisions: 2790) | 92.2% (clause exact match) | 0.199 | slight |

## Attribute exact-match rate

| Attribute | Matches | Compared | Rate |
| :--- | :--- | :--- | :--- |
| adjustment_trigger | 0 | 2 | 0.0% |
| amounts | 0 | 3 | 0.0% |
| anti_dilution_type | 1 | 2 | 50.0% |
| applies_to | 1 | 4 | 25.0% |
| appointment_rights | 0 | 3 | 0.0% |
| automatic_conversion | 0 | 6 | 0.0% |
| broad_based | 1 | 2 | 50.0% |
| cliff_months | 1 | 1 | 100.0% |
| conditions | 0 | 1 | 0.0% |
| consent_threshold | 0 | 1 | 0.0% |
| conversion_ratio | 0 | 1 | 0.0% |
| conversion_triggers | 0 | 1 | 0.0% |
| covered_holders | 0 | 3 | 0.0% |
| eligibility_threshold | 0 | 1 | 0.0% |
| excluded_issuances | 0 | 2 | 0.0% |
| frequency | 0 | 1 | 0.0% |
| inspection_right | 0 | 5 | 0.0% |
| investor_co_sale | 0 | 5 | 0.0% |
| investor_rofr | 1 | 5 | 20.0% |
| liq_multiple | 0 | 1 | 0.0% |
| liq_participation | 0 | 4 | 0.0% |
| liq_seniority | 0 | 2 | 0.0% |
| majority_threshold | 0 | 3 | 0.0% |
| optional_conversion | 0 | 5 | 0.0% |
| protected_actions | 0 | 1 | 0.0% |
| redemption_trigger | 0 | 1 | 0.0% |
| report_types | 0 | 4 | 0.0% |
| repurchase_right | 0 | 2 | 0.0% |
| requires_founder_consent | 0 | 1 | 0.0% |
| requires_preferred_consent | 0 | 2 | 0.0% |
| requires_series_majority | 0 | 1 | 0.0% |
| restricted_activities | 0 | 1 | 0.0% |
| sale_scope | 0 | 1 | 0.0% |
| topic | 0 | 4 | 0.0% |
| transfer_exceptions | 0 | 1 | 0.0% |
| vesting_months | 0 | 1 | 0.0% |

## Unmatched items

- Only annotated by A: ts_010_founder_institute_plain_preferred.doc.gold.json
- Only annotated by A: yc_postmoney_safe_discount_only.docx.gold.json
- Only annotated by A: yc_postmoney_safe_mfn_only.docx.gold.json
- Only annotated by A: yc_postmoney_safe_valuation_cap_only.docx.gold.json
- Only annotated by A: yc_postmoney_safe_valuation_cap_only_canada.docx.gold.json

## Disagreements to reconcile (41)

- `ts_001_nvca_model_term_sheet.docx.gold.json` c_2 / clause_family: A = exclusivity_no_shop, B = other
- `ts_002_wsgr_venture_financing.docx.gold.json` c_4 / clause_family: A = anti_dilution, B = liquidation_preference
- `ts_002_wsgr_venture_financing.docx.gold.json` c_4 / expected_flags: A = ['R-AD-02'], B = []
- `ts_002_wsgr_venture_financing.docx.gold.json` c_5 / clause_family: A = conversion_rights, B = other
- `ts_002_wsgr_venture_financing.docx.gold.json` c_6 / clause_family: A = information_rights, B = rofr_co_sale
- `ts_002_wsgr_venture_financing.docx.gold.json` c_6 / expected_flags: A = [], B = ['R-RO-01']
- `ts_003_series_seed_equity.txt.gold.json` c_2 / expected_flags: A = ['R-LP-02', 'R-LP-03'], B = []
- `ts_004_gunderson_series_seed.pdf.gold.json` c_16 / clause_family: A = other, B = option_pool
- `ts_004_gunderson_series_seed.pdf.gold.json` c_20 / clause_family: A = liquidation_preference, B = other
- `ts_004_gunderson_series_seed.pdf.gold.json` c_20 / expected_flags: A = ['R-LP-02'], B = []
- `ts_004_gunderson_series_seed.pdf.gold.json` c_21 / clause_family: A = other, B = liquidation_preference
- `ts_004_gunderson_series_seed.pdf.gold.json` c_22 / clause_family: A = other, B = liquidation_preference
- `ts_004_gunderson_series_seed.pdf.gold.json` c_26 / clause_family: A = other, B = conversion_rights
- `ts_004_gunderson_series_seed.pdf.gold.json` c_27 / clause_family: A = other, B = conversion_rights
- `ts_004_gunderson_series_seed.pdf.gold.json` c_29 / clause_family: A = other, B = anti_dilution
- `ts_004_gunderson_series_seed.pdf.gold.json` c_30 / clause_family: A = other, B = anti_dilution
- `ts_004_gunderson_series_seed.pdf.gold.json` c_33 / clause_family: A = other, B = anti_dilution
- `ts_004_gunderson_series_seed.pdf.gold.json` c_34 / clause_family: A = redemption_rights, B = protective_provisions
- `ts_004_gunderson_series_seed.pdf.gold.json` c_34 / expected_flags: A = [], B = ['R-PP-02']
- `ts_004_gunderson_series_seed.pdf.gold.json` c_35 / clause_family: A = other, B = protective_provisions
- `ts_004_gunderson_series_seed.pdf.gold.json` c_36 / clause_family: A = other, B = protective_provisions
- `ts_004_gunderson_series_seed.pdf.gold.json` c_38 / clause_family: A = board_composition, B = other
- `ts_004_gunderson_series_seed.pdf.gold.json` c_42 / clause_family: A = rofr_co_sale, B = other
- `ts_004_gunderson_series_seed.pdf.gold.json` c_43 / clause_family: A = conversion_rights, B = other
- `ts_004_gunderson_series_seed.pdf.gold.json` c_44 / clause_family: A = conversion_rights, B = other
- `ts_004_gunderson_series_seed.pdf.gold.json` c_47 / clause_family: A = information_rights, B = other
- `ts_004_gunderson_series_seed.pdf.gold.json` c_49 / clause_family: A = other, B = information_rights
- `ts_004_gunderson_series_seed.pdf.gold.json` c_50 / clause_family: A = other, B = information_rights
- `ts_004_gunderson_series_seed.pdf.gold.json` c_52 / clause_family: A = other, B = information_rights
- `ts_004_gunderson_series_seed.pdf.gold.json` c_54 / clause_family: A = other, B = board_composition
- `ts_004_gunderson_series_seed.pdf.gold.json` c_55 / clause_family: A = founder_vesting, B = other
- `ts_004_gunderson_series_seed.pdf.gold.json` c_56 / expected_flags: A = ['R-VS-01'], B = []
- `ts_004_gunderson_series_seed.pdf.gold.json` c_57 / clause_family: A = founder_vesting, B = other
- `ts_004_gunderson_series_seed.pdf.gold.json` c_59 / clause_family: A = board_composition, B = other
- `ts_004_gunderson_series_seed.pdf.gold.json` c_62 / expected_flags: A = [], B = ['R-RO-01']
- `ts_004_gunderson_series_seed.pdf.gold.json` c_64 / clause_family: A = other, B = rofr_co_sale
- `ts_004_gunderson_series_seed.pdf.gold.json` c_65 / clause_family: A = drag_along, B = rofr_co_sale
- `ts_004_gunderson_series_seed.pdf.gold.json` c_66 / clause_family: A = other, B = drag_along
- `ts_004_gunderson_series_seed.pdf.gold.json` c_67 / clause_family: A = other, B = drag_along
- `ts_004_gunderson_series_seed.pdf.gold.json` c_68 / clause_family: A = other, B = drag_along
- `ts_004_gunderson_series_seed.pdf.gold.json` c_75 / clause_family: A = conversion_rights, B = other
