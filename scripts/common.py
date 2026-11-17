import json
import math

SYSTEM_PROMPT = (
    "You are a strict binary classifier for Bengali hallucination detection. "
    "Given a Bengali (context, prompt, response) triple, you must output "
    "EXACTLY ONE character: the digit 1 or the digit 0. No words, no JSON, "
    "no explanation, no whitespace, no quotes — just the digit.\n\n"
    "OUTPUT 1  if the response is FAITHFUL: it is supported by the context, "
    "or (when the context is [NULL]) is factually correct.\n"
    "OUTPUT 0  if the response is HALLUCINATED: it contains factual errors, "
    "contradicts the context, or invents information not supported by it.\n"
)

USER_TEMPLATE = (
    "Context: {context}\n"
    "Prompt: {prompt_bn}\n"
    "Response: {response_bn}\n\n"
    "Label (1 or 0 only):"
)

HEADER = ["id", "label", "score", "raw", "logprobs_json"]


def load_items(path):
    return [json.loads(line) for line in open(path, encoding="utf-8")]


def build_messages(item):
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": USER_TEMPLATE.format(**item)},
    ]


def parse_label(text):
    for ch in text.strip().strip("\"'"):
        if ch in "01":
            return int(ch)
    return None


def score_from_logprobs(top_logprobs):
    lp = {}
    for t in top_logprobs:
        lp.setdefault(t["token"], t["logprob"])
    if "1" in lp and "0" in lp:
        m = max(lp["1"], lp["0"])
        p1, p0 = math.exp(lp["1"] - m), math.exp(lp["0"] - m)
        return p1 / (p1 + p0)
    if "1" in lp:
        return min(math.exp(lp["1"]), 1.0)
    if "0" in lp:
        return max(1.0 - math.exp(lp["0"]), 0.0)
    return 0.5


def make_row(item_id, raw, top_logprobs):
    score = score_from_logprobs(top_logprobs)
    label = parse_label(raw)
    if label is None:
        label = int(score >= 0.5)
    return {
        "id": item_id,
        "label": label,
        "score": round(score, 6),
        "raw": raw[:60],
        "logprobs_json": json.dumps(top_logprobs, ensure_ascii=False),
    }
