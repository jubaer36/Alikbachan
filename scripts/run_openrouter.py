import argparse
import csv
import os
from concurrent.futures import ThreadPoolExecutor

from openai import OpenAI

from common import HEADER, build_messages, load_items, make_row

LOGPROB_PROVIDERS = {
    "meta-llama/llama-3.1-8b-instruct": ["Novita", "CoreWeave"],
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=os.environ["OPENROUTER_API_KEY"])

    extra_body = {}
    if args.model in LOGPROB_PROVIDERS:
        extra_body["provider"] = {"order": LOGPROB_PROVIDERS[args.model], "allow_fallbacks": False}

    def judge(item):
        resp = client.chat.completions.create(
            model=args.model,
            messages=build_messages(item),
            max_tokens=8,
            temperature=0,
            logprobs=True,
            top_logprobs=20,
            extra_body=extra_body,
        )
        choice = resp.choices[0]
        top_logprobs = [
            {"token": t.token, "logprob": t.logprob}
            for t in choice.logprobs.content[0].top_logprobs
        ]
        return make_row(item["id"], choice.message.content or "", top_logprobs)

    items = load_items(args.data)
    with ThreadPoolExecutor(args.workers) as pool:
        rows = list(pool.map(judge, items))

    with open(args.out, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=HEADER)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
