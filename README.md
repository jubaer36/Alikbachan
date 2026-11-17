# Alikbachan: A Stratified Benchmark for Bengali LLM Hallucination

A binary hallucination-detection benchmark for Bengali. Each item is a
(context, prompt, response) triple, and the task is to label the response
**faithful (1)** or **hallucinated (0)**. Phase 2 items are also tagged by cultural band,
task type and a 5-family / 32-mode failure taxonomy, so results can be read per stratum.

Paper: *One Framework to Diagnose Them All: A Stratified Benchmark for Bengali LLM
Hallucination* (EACL 2027 Findings). <!-- TODO: add paper / arXiv link -->

## Origin

The data comes from the **অলীকবচন: Bengali LLM Hallucination Detection Challenge**
([Kaggle](https://www.kaggle.com/competitions/bengali-hallucination/overview), Datathon 2.0). It was organised by IUTCS at IUT, powered by the Institute of Policy
Dynamics (IPD) and sponsored by the Enhancement of Bengali Language in ICT through Research
and Development (EBLICT) project.

- **Phase 1** (Kaggle leaderboard): `dataset/phase1/test` was the scored test set.
  `dataset/phase1/sample` was the small labelled set for local validation.
- **Phase 2** (final round): a held-out fold, disjoint from Phase 1, used to run the top
  teams' offline notebooks before the in-person final.

## Release

| Split | File | Items | % faithful | Labels |
|---|---|---:|---:|---|
| Phase 1 test | `dataset/phase1/test.jsonl` | 2,516 | 53.1 | public |
| Phase 1 sample | `dataset/phase1/sample.jsonl` | 299 | 54.5 | public |
| Phase 2 public | `dataset/phase2/public.jsonl` | 1,500 | 46.5 | public |
| Phase 2 hidden | not distributed | 3,500 | 45.3 | private |

Phase 2 (5,000 items) is split 30 / 70 into public and hidden. The hidden split stays
private as a contamination-resistant test set. Every file is also available as CSV.

```
dataset/
├── phase1/             test, sample
├── phase2/             public
├── metadata/           taxonomy.json, phase2_split_statistics.csv, release_summary.json
└── submission_format/  sample submission files
scripts/                evaluate.py, LLM-judge prediction scripts
```

## Task

Input: `context` (a passage, or `[NULL]` for none), `prompt_bn`, `response_bn`.
Output: `label` ∈ {0, 1}.

- `1` faithful: correct and, if a context is given, supported by it.
- `0` hallucinated: fabricated, unsupported or factually wrong.

With `[NULL]` context the response is judged against world knowledge. Otherwise it is judged
against the context, even where the context deliberately disagrees with reality.

## Data fields

### Phase 1 (`test`, `sample`)

| Field | Type | Description |
|---|---|---|
| `id` | str | Item id (`P1-T-…` test, `P1-S-…` sample) |
| `competition_id` | int | Original Kaggle row id (test only) |
| `task_type` | str | `closed_book_qa` or `context_grounded_qa` |
| `context` | str | Source passage, or `[NULL]` |
| `prompt_bn` | str | Bengali prompt |
| `response_bn` | str | Candidate response |
| `label` | int | 1 faithful, 0 hallucinated |
| `overlaps_test` | bool | Sample only: item also appears in test (5 items) |

### Phase 2 (`public`)

| Field | Type | Description |
|---|---|---|
| `id` | str | Item id (`P2-…`) |
| `source` | str | Source corpus |
| `task_type` | str | `short_ob_qa`, `long_ob_qa`, `short_cb_qa`, `summarization`, `translation` or `adversarial` |
| `family` | str | Hallucination family, `A` to `E` |
| `failure_mode` | str | One of 32 failure modes |
| `cultural_band` | str | `C0`, `C1` or `C2` |
| `provenance` | str | `theory_authored` or `theory_confirmed` |
| `context` | str | Source passage, or `[NULL]` |
| `prompt_bn` | str | Bengali prompt |
| `response_bn` | str | Candidate response |
| `label` | int | 1 faithful, 0 hallucinated |
| `rationale` | str | English explanation of the label |

> Detectors should see only `context`, `prompt_bn` and `response_bn`. The other Phase 2
> fields are annotations for slicing results, and they leak the label (`rationale` states it
> outright).

## Loading

```python
import pandas as pd
p1_test = pd.read_json("dataset/phase1/test.jsonl", lines=True)
p2_public = pd.read_json("dataset/phase2/public.jsonl", lines=True)
```

## Submission format

A CSV with exactly two columns, `id` and `label`, one row per item. `label` is `0`
(hallucinated) or `1` (faithful). Examples are in `dataset/submission_format/`:

```
id,label
P2-00002,1
P2-00012,0
```

An optional `score` column (P(faithful), 0 to 1) adds ROC-AUC and ECE to the report.

## Evaluation

```bash
python3 scripts/evaluate.py --gold dataset/phase2/public.jsonl --pred preds.csv \
    --by family failure_mode cultural_band task_type
```

The main metric is macro-F1. The scorer also reports accuracy, false-flag rate and miss
rate, overall and per stratum. To run an LLM as a zero-shot judge, use `scripts/run_local.py`
(local) or `scripts/run_openrouter.py` (OpenRouter).

**Hidden-test evaluation:** we will launch an evaluation system for the hidden Phase 2
split soon.

## License

Released under [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/).

## Citation

```bibtex
@inproceedings{alikbachan,
  title  = {One Framework to Diagnose Them All: A Stratified Benchmark for Bengali LLM Hallucination},
  author = {Asif Shahriyar Sushmit and Shefayat E Shams Adib and Abdullah Al Jubaer Gem and
            Ahmed Shafin Ruhan and MD. Nazmuddoha Ansary and Biplab Kumar Sarkar and
            Syed Rifat Raiyan and Ajwad Abrar and Abdullah Bin Hasan and Adib Sakhawat and
            M. Sakeef Hossain and Ahmed Alfey Sani and Arefin Mizan and Sourab Chakma and
            Quazi Adibur Rahman Adib and Wassie Haque and Sanjay Kumar Sen and Saad Tahmid and
            Mohammad Mamun Or Rashid},
  booktitle = {Findings of the Association for Computational Linguistics: EACL 2027},
  year   = {2027}
}
```
