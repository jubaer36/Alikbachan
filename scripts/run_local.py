import argparse
import csv

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from common import HEADER, SYSTEM_PROMPT, build_messages, load_items, make_row


def render_prompt(tokenizer, item):
    messages = build_messages(item)
    try:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    except Exception:
        merged = [{"role": "user", "content": SYSTEM_PROMPT + "\n\n" + messages[1]["content"]}]
        return tokenizer.apply_chat_template(merged, tokenize=False, add_generation_prompt=True)


def judge(model, tokenizer, item, top_k=20):
    inputs = tokenizer(render_prompt(tokenizer, item), return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=6,
            do_sample=False,
            output_scores=True,
            return_dict_in_generate=True,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )
    raw = tokenizer.decode(out.sequences[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    log_probs = torch.log_softmax(out.scores[0][0].float(), dim=-1)
    top = torch.topk(log_probs, top_k)
    top_logprobs = [
        {"token": tokenizer.decode([tid]), "logprob": lp}
        for tid, lp in zip(top.indices.tolist(), top.values.tolist())
    ]
    return make_row(item["id"], raw, top_logprobs)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16, device_map="auto")
    model.eval()

    items = load_items(args.data)
    with open(args.out, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=HEADER)
        writer.writeheader()
        for i, item in enumerate(items, 1):
            writer.writerow(judge(model, tokenizer, item))
            if i % 100 == 0:
                print(f"{i}/{len(items)}")


if __name__ == "__main__":
    main()
