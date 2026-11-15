# Demo video script (target 5 to 6 minutes)

Record the deployed version if M1 has one, otherwise `streamlit run app.py` locally.
Before recording: close other tabs, set the browser zoom so text is readable, and run the
real pipeline on one document first so you know it works.

| Time | Screen | Say |
| :--- | :--- | :--- |
| 0:00 | Title slide or the app home | Who you are, the problem: first-time founders get dense term sheets with days to respond and no lawyer. |
| 0:30 | Architecture diagram (from the report) | Five stages: ingest, classify (SBERT + k-NN), extract (LLM with source spans), reason (forward-chaining rules), explain (plain English with a number checker). The neural part reads; the symbolic part decides. |
| 1:15 | Sidebar: demo result | State clearly that this is bundled mock data. Show the risk score and severity counts. |
| 1:45 | Clauses tab | Original clause beside plain-English summary. Point at a flagged clause, a low-confidence clause marked "needs manual review", and a summary with the unverified-numbers warning. |
| 2:45 | Flags tab: open "Founders removable with loss of equity" | The proof trace: the rule R-CT-05 only fires because R-CT-02 first derived InvestorControl. This is the multi-step reasoning a plain LLM cannot show. |
| 3:30 | Facts tab | Every extracted number has a quoted source span and a grounded check. |
| 4:00 | Upload mode: real document, press Analyse | Show the live pipeline result and any pipeline errors in the Debug tab. |
| 4:45 | Investor tab | Look up a fund. Mention found / not found / ambiguous, and that "not found" is not a risk flag. |
| 5:15 | reports/figures: confusion matrix and flag precision/recall | Report only what the numbers show, including the weak spots and that synthetic gold labels come from the generator. |
| 5:45 | Closing slide | Limitations, and that it is an educational tool, not legal advice. |

## Questions you should be ready for in the viva (your module)

- Why is the UI separated from the pipeline? (`data_source.py` is the only place that chooses mock or real data.)
- How is the proof tree built? (`trace.py` links a rule's premise to another rule's conclusion by name; cycles are guarded.)
- What does Cohen's kappa measure and what did you find? (Agreement beyond chance, before reconciliation.)
- Why are synthetic-gold results optimistic? (The same generator made the text and the labels.)
- What does the investor panel not tell you? (Absence of a Form D filing is not evidence of risk.)
