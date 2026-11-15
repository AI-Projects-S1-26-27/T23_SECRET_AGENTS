# Member 6 — S4 Risk-Flag Error Analysis

Documents analyzed: 30
True positives: 63
False positives: 74
False negatives: 133
Precision: 45.99%
Recall: 32.14%
F1: 37.84%
Documents with execution errors: 0

## Most common false-positive rules

- R-LP-05: 26
- R-LP-02: 12
- R-VS-03: 12
- R-VS-04: 9
- R-LP-04: 7
- R-CT-04: 5
- R-AD-03: 2
- R-LP-01: 1

## Most common missed rules

- R-DV-01: 17
- R-EX-01: 17
- R-OP-01: 15
- R-RD-01: 15
- R-CT-02: 12
- R-LP-03: 12
- R-DA-01: 9
- R-LP-01: 9
- R-AD-01: 8
- R-CT-03: 8
- R-AD-02: 4
- R-VS-02: 4
- R-VS-01: 2
- R-LP-02: 1

## Interpretation

These results identify risk-rule ID mismatches against the gold annotations. They do not independently identify whether each mismatch originated in extraction, grounding, rule coverage, or inference.

## Document-level errors

### ts_synth_adv_001.txt

- False positives: R-LP-04, R-LP-05, R-VS-03
- Missed flags: R-DA-01, R-DV-01, R-EX-01, R-LP-03, R-OP-01, R-RD-01
- Grounding rate: 1.0
- Execution errors: []

### ts_synth_adv_002.txt

- False positives: R-LP-02, R-LP-05, R-VS-04
- Missed flags: R-AD-01, R-CT-02, R-CT-03, R-DA-01, R-DV-01, R-EX-01, R-OP-01, R-RD-01
- Grounding rate: 1.0
- Execution errors: []

### ts_synth_adv_003.txt

- False positives: R-LP-04, R-LP-05, R-VS-03
- Missed flags: R-DA-01, R-DV-01, R-EX-01, R-OP-01, R-RD-01
- Grounding rate: 1.0
- Execution errors: []

### ts_synth_adv_004.txt

- False positives: R-CT-04, R-LP-02, R-LP-04, R-LP-05, R-VS-03
- Missed flags: R-AD-01, R-DV-01, R-EX-01, R-LP-03, R-RD-01
- Grounding rate: 1.0
- Execution errors: []

### ts_synth_adv_005.txt

- False positives: R-CT-04, R-LP-02, R-LP-04, R-LP-05, R-VS-04
- Missed flags: R-CT-02, R-DA-01, R-DV-01, R-EX-01, R-LP-03, R-OP-01, R-RD-01
- Grounding rate: 1.0
- Execution errors: []

### ts_synth_adv_006.txt

- False positives: R-LP-02, R-LP-04, R-LP-05
- Missed flags: R-AD-01, R-CT-02, R-DA-01, R-DV-01, R-EX-01, R-OP-01, R-RD-01
- Grounding rate: 1.0
- Execution errors: []

### ts_synth_adv_007.txt

- False positives: R-LP-02, R-LP-05, R-VS-03
- Missed flags: R-AD-01, R-CT-02, R-EX-01, R-LP-01, R-RD-01
- Grounding rate: 0.6666666666666666
- Execution errors: []

### ts_synth_adv_008.txt

- False positives: R-LP-05
- Missed flags: R-CT-02, R-CT-03, R-DV-01, R-EX-01, R-LP-01, R-LP-03, R-OP-01, R-RD-01
- Grounding rate: 1.0
- Execution errors: []

### ts_synth_adv_009.txt

- False positives: R-LP-04, R-LP-05
- Missed flags: R-CT-02, R-EX-01, R-LP-03, R-OP-01, R-RD-01
- Grounding rate: 1.0
- Execution errors: []

### ts_synth_adv_010.txt

- False positives: R-LP-05, R-VS-03
- Missed flags: R-AD-01, R-CT-02, R-CT-03, R-DA-01, R-DV-01, R-EX-01, R-LP-01, R-LP-03, R-OP-01, R-RD-01
- Grounding rate: 1.0
- Execution errors: []

### ts_synth_adv_011.txt

- False positives: R-LP-04, R-LP-05, R-VS-03
- Missed flags: R-CT-02, R-CT-03, R-DA-01, R-DV-01, R-EX-01, R-LP-03, R-OP-01, R-RD-01, R-VS-02
- Grounding rate: 1.0
- Execution errors: []

### ts_synth_adv_012.txt

- False positives: R-LP-05, R-VS-03
- Missed flags: R-CT-02, R-CT-03, R-DV-01, R-EX-01, R-OP-01, R-RD-01, R-VS-02
- Grounding rate: 1.0
- Execution errors: []

### ts_synth_adv_013.txt

- False positives: R-VS-04
- Missed flags: R-AD-01, R-CT-02, R-CT-03, R-DA-01, R-DV-01, R-EX-01, R-LP-01, R-LP-02, R-LP-03, R-OP-01, R-RD-01
- Grounding rate: 1.0
- Execution errors: []

### ts_synth_adv_014.txt

- False positives: R-LP-02, R-LP-05
- Missed flags: R-CT-02, R-CT-03, R-EX-01, R-LP-01, R-LP-03, R-OP-01, R-RD-01, R-VS-02
- Grounding rate: 0.5
- Execution errors: []

### ts_synth_adv_015.txt

- False positives: R-LP-05
- Missed flags: R-AD-01, R-CT-02, R-CT-03, R-DA-01, R-DV-01, R-EX-01, R-OP-01, R-RD-01, R-VS-02
- Grounding rate: 1.0
- Execution errors: []

### ts_synth_std_001.txt

- False positives: R-VS-04
- Missed flags: R-AD-01, R-LP-01
- Grounding rate: 0.5
- Execution errors: []

### ts_synth_std_002.txt

- False positives: R-LP-02, R-LP-05
- Missed flags: R-EX-01
- Grounding rate: 1.0
- Execution errors: []

### ts_synth_std_003.txt

- False positives: R-LP-05, R-VS-03
- Missed flags: R-OP-01
- Grounding rate: 1.0
- Execution errors: []

### ts_synth_std_004.txt

- False positives: R-AD-03, R-LP-02, R-LP-05, R-VS-03
- Missed flags: R-LP-01
- Grounding rate: 1.0
- Execution errors: []

### ts_synth_std_005.txt

- False positives: R-LP-02, R-LP-05, R-VS-03
- Missed flags: R-DV-01
- Grounding rate: 1.0
- Execution errors: []

### ts_synth_std_006.txt

- False positives: R-CT-04, R-LP-01, R-LP-05
- Missed flags: R-DV-01
- Grounding rate: 1.0
- Execution errors: []

### ts_synth_std_007.txt

- False positives: R-LP-02, R-LP-05
- Missed flags: R-LP-03, R-VS-01
- Grounding rate: 1.0
- Execution errors: []

### ts_synth_std_008.txt

- False positives: R-VS-04
- Missed flags: R-AD-02
- Grounding rate: 1.0
- Execution errors: []

### ts_synth_std_009.txt

- False positives: R-VS-04
- Missed flags: R-AD-02, R-LP-03
- Grounding rate: 1.0
- Execution errors: []

### ts_synth_std_010.txt

- False positives: R-CT-04, R-LP-02, R-LP-05, R-VS-03
- Missed flags: R-DV-01
- Grounding rate: 1.0
- Execution errors: []

### ts_synth_std_011.txt

- False positives: R-AD-03, R-LP-05, R-VS-03
- Missed flags: R-LP-01
- Grounding rate: 0.75
- Execution errors: []

### ts_synth_std_012.txt

- False positives: R-LP-05, R-VS-04
- Missed flags: R-AD-02, R-DV-01, R-LP-03
- Grounding rate: 1.0
- Execution errors: []

### ts_synth_std_013.txt

- False positives: R-LP-05, R-VS-04
- Missed flags: R-DV-01
- Grounding rate: 1.0
- Execution errors: []

### ts_synth_std_014.txt

- False positives: R-CT-04, R-LP-05
- Missed flags: R-EX-01, R-OP-01, R-VS-01
- Grounding rate: 1.0
- Execution errors: []

### ts_synth_std_015.txt

- False positives: R-LP-02, R-LP-05, R-VS-04
- Missed flags: R-AD-02, R-LP-01
- Grounding rate: 1.0
- Execution errors: []

