"""Alikbachan scorer. Labels: 1 = faithful, 0 = hallucinated.

Usage:
    python3 scripts/evaluate.py --gold dataset/phase2/public.jsonl --pred preds.csv --by family failure_mode

preds.csv columns:
    id,label         required
    score            optional: the model's probability that the item is faithful (0..1).
                     If present, ROC-AUC and ECE are also reported.

How to get `score` from an LLM judge that answers "1" or "0":
    take the log-probabilities of the "1" and "0" tokens at the answer position, then
        score = exp(lp_1) / (exp(lp_1) + exp(lp_0))
    Without token probabilities, leave the column out; ROC-AUC and ECE need real probabilities.
"""
import argparse
import csv
import json


def metrics(gold, pred, score=None):
    tp = sum(g == 1 and p == 1 for g, p in zip(gold, pred))  # faithful, judged faithful
    tn = sum(g == 0 and p == 0 for g, p in zip(gold, pred))  # hallucinated, caught
    fn = sum(g == 1 and p == 0 for g, p in zip(gold, pred))  # faithful, wrongly flagged
    fp = sum(g == 0 and p == 1 for g, p in zip(gold, pred))  # hallucinated, missed

    f1_faithful = 2 * tp / (2 * tp + fp + fn) if tp else 0.0
    f1_halluc = 2 * tn / (2 * tn + fp + fn) if tn else 0.0
    classes = set(gold) | set(pred)  # macro-F1 over classes present (sklearn convention)

    result = {
        "n": len(gold),
        "accuracy": (tp + tn) / len(gold),
        "macro_f1": (f1_faithful * (1 in classes) + f1_halluc * (0 in classes)) / len(classes),
        "false_flag_rate": fn / (tp + fn) if tp + fn else None,
        "miss_rate": fp / (tn + fp) if tn + fp else None,
    }
    if score:
        result["roc_auc"] = roc_auc(gold, score)
        result["ece"] = ece(gold, pred, score)
    return result


def roc_auc(gold, score):
    # Chance that a random faithful item gets a higher score than a random hallucinated one.
    pos = [s for s, g in zip(score, gold) if g == 1]
    neg = [s for s, g in zip(score, gold) if g == 0]
    if not pos or not neg:
        return None
    wins = sum((p > n) + 0.5 * (p == n) for p in pos for n in neg)
    return wins / (len(pos) * len(neg))


def ece(gold, pred, score, n_bins=15):
    # Group items by confidence (max of score, 1 - score), then average |accuracy - confidence|
    # over the bins, weighted by bin size. 0 = perfectly calibrated.
    bins = [[] for _ in range(n_bins)]
    for g, p, s in zip(gold, pred, score):
        conf = max(s, 1 - s)
        bins[min(int(conf * n_bins), n_bins - 1)].append((conf, g == p))
    total = 0.0
    for b in bins:
        if b:
            avg_conf = sum(c for c, _ in b) / len(b)
            accuracy = sum(ok for _, ok in b) / len(b)
            total += len(b) / len(gold) * abs(accuracy - avg_conf)
    return total


def print_row(name, values):
    cells = [f"{v:.3f}" if isinstance(v, float) else str(v if v is not None else "-") for v in values]
    print(f"{str(name):40s}" + "".join(f"{c:>17s}" for c in cells))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gold", required=True)
    parser.add_argument("--pred", required=True)
    parser.add_argument("--by", nargs="*", default=[])
    args = parser.parse_args()

    items = [json.loads(line) for line in open(args.gold, encoding="utf-8")]
    rows = list(csv.DictReader(open(args.pred, encoding="utf-8")))
    labels = {r["id"]: int(r["label"]) for r in rows}
    scores = {r["id"]: float(r["score"]) for r in rows if r.get("score")}

    def score(group):
        return metrics([x["label"] for x in group],
                       [labels[x["id"]] for x in group],
                       [scores[x["id"]] for x in group] if scores else None)

    overall = score(items)
    print_row("", overall.keys())
    print_row("overall", overall.values())

    for field in args.by:
        print(f"\n-- by {field}")
        for value in sorted({x[field] for x in items}):
            print_row(value, score([x for x in items if x[field] == value]).values())


if __name__ == "__main__":
    main()
