#!/usr/bin/env python
# coding: utf-8

# # CSE 151B Competition — Starter Notebook
# 
# Welcome to the **CSE 151B Spring 2026 Math Reasoning Competition**!  
# This notebook walks you through the full pipeline end-to-end:
# 
# 1. Setting up the Python environment with `uv`
# 2. Loading the competition dataset
# 3. Running inference with **Qwen3-4B-Thinking** via vLLM (INT8 quantized)
# 4. Scoring responses against ground-truth answers
# 5. Saving results to JSONL for submission
# 
# The public dataset (`public.jsonl`) contains questions **with** answers so you can measure accuracy locally.  
# The private test set used for the leaderboard does **not** include answers — for that, skip evaluation and submit the raw responses.

# ## 1. Environment Setup
# 
# We use [`uv`](https://github.com/astral-sh/uv) for fast, reproducible package management.
# 
# The steps below:
# 1. Install `uv` into `~/.local/bin`
# 2. Create a virtual environment at `.venv/`
# 3. Install all required packages (This might take a while)
# 
# > **After running this cell, restart the kernel** so that the newly installed packages (especially `vllm` and `transformers`) are picked up by the current Python session.

# ### Comment Out the cell below after first installation.

# In[ ]:


"""# Install uv
!wget -qO- https://astral.sh/uv/install.sh | sh

# Create a virtual environment
!uv venv .venv --seed --clear

# Install dependencies — this is fast thanks to uv's parallel resolver
!.venv/bin/python -m pip install sympy numpy transformers vllm tqdm bitsandbytes antlr4-python3-runtime==4.11.1 ipykernel jupyter

# Install Jupyter Kernel
!.venv/bin/python -m ipykernel install --user --name cse151b --display-name "Python (cse151b)"

print("Done. Restart the kernel before proceeding.")
print("Selection process: on top right, click on current kernel '(ususally named python)' -> 'select another kernel' -> 'Jupyter Kernel' -> 'Python (cse151b)'.")
"""


# ### Run the cell below every time to activate the installed environment. 

# In[ ]:


# activate venv after installation. This needs to be run everytime.
#get_ipython().system('source ./.venv/bin/activate')


# ## 2. Imports & Configuration
# 
# All key settings are collected in one place.  
# - `DATA_PATH` — public dataset with ground-truth answers (use this to measure accuracy)
# - `OUTPUT_PATH` — where per-question results will be written
# - `GPU_ID` — which GPU to use (update if your machine has a different device index)
# - `MAX_TOKENS` — maximum tokens the model may generate per response

# In[ ]:


import json
import os

# ── Configuration ─────────────────────────────────────────────────────────────
MODEL_ID    = "Qwen/Qwen3-4B-Thinking-2507"
GPU_ID      = "0"                    # CUDA_VISIBLE_DEVICES
DATA_PATH   = "data/private.jsonl"
OUTPUT_PATH = "results/private_results.jsonl"
MAX_TOKENS  = 32768

os.environ["CUDA_VISIBLE_DEVICES"] = GPU_ID

import re
import sys
from pathlib import Path
from typing import Optional

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from tqdm import tqdm


# ## 3. Load the Dataset
# 
# The dataset is stored as newline-delimited JSON (`.jsonl`). Each line is one question with the following fields:
# 
# | Field | Description |
# |---|---|
# | `id` | Unique question identifier |
# | `question` | Problem statement |
# | `options` | List of answer choices — present for **MCQ**, absent for **free-form** |
# | `answer` | Ground-truth answer (letter for MCQ, value/list for free-form) |

# In[ ]:

def run_inference():
    data = [json.loads(line) for line in open(DATA_PATH)]

    n_mcq  = sum(bool(d.get("options")) for d in data)
    n_free = sum(not d.get("options")   for d in data)
    print(f"Loaded {len(data)} questions  ({n_mcq} MCQ, {n_free} free-form)")

    # Preview one MCQ and one free-form item
    mcq_sample  = next(d for d in data if d.get("options"))
    free_sample = next(d for d in data if not d.get("options"))

    print("\n── MCQ sample ──")
    print(json.dumps(mcq_sample, indent=2))
    print("\n── Free-form sample ──")
    print(json.dumps(free_sample, indent=2))


    # ## 4. Prompt Construction
    # 
    # We use two system prompts depending on the question type:
    # 
    # - **MCQ** — the model must select the best answer letter and wrap it in `\boxed{}`
    # - **Free-form** — the model solves step-by-step and puts the final answer in `\boxed{}`
    # 
    # `build_prompt()` returns the appropriate `(system, user)` pair for each item.

    # In[ ]:


    SYSTEM_PROMPT_MATH = (
        "You are an expert mathematician. Solve the problem step-by-step, show all work clearly."
        "Rules:"
        "- Keep full symbolic/fractional precision throughout; only evaluate numerically at the very last step"
        "- Do NOT round intermediate values; carry at least 8 significant figures in any decimal intermediate"
        "- Do NOT round the final answer unless the problem explicitly asks for rounding"
        "- For multi-part answers, put ALL parts in a single \\boxed{}, separated by commas"
        "- Output exactly ONE \\boxed{} total. Never box intermediate results or repeat the box."
        "- Double-check exponential and logarithm calculations by substituting back"
    )

    SYSTEM_PROMPT_MCQ = (
        "You are an expert problem solver. Work through the problem step by step, analyzing each option carefully."
        "Eliminate wrong answers with brief reasoning."
        "After your analysis, state your final answer as a single letter inside \\boxed{}, eg. \\boxed{C}. Do not include any text after the boxed answer."
    )

    def build_prompt(question: str, options: Optional[list]) -> tuple[str, str]:
        """Return (system_prompt, user_prompt) for a question."""
        if options:
            labels    = [chr(65 + i) for i in range(len(options))]
            opts_text = "\n".join(f"{lbl}. {opt.strip()}" for lbl, opt in zip(labels, options))
            return SYSTEM_PROMPT_MCQ, f"{question}\n\nOptions:\n{opts_text}"
        return SYSTEM_PROMPT_MATH, question


    # Verify with samples
    for label, item in [("MCQ", mcq_sample), ("Free-form", free_sample)]:
        sys_p, usr_p = build_prompt(item["question"], item.get("options"))
        print(f"── {label} user prompt (first 200 chars) ──")
        print(usr_p[:200], "...\n")


    # ## 5. Load Model with vLLM (for general case, vLLM is faster)
    # 
    # We load **Qwen3-4B-Thinking-2507** with **INT8 quantization** via BitsAndBytes.  
    # Setting `load_format="bitsandbytes"` tells vLLM to apply on-the-fly INT8 weight quantization, roughly halving GPU memory usage compared to BF16.
    # 
    # Key parameters:
    # - `gpu_memory_utilization` — fraction of GPU VRAM reserved for the model and KV cache
    # - `max_model_len` — maximum sequence length (prompt + generation)
    # - `max_num_seqs` — maximum number of sequences processed in parallel

    # In[ ]:


    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    tokenizer.pad_token = tokenizer.eos_token

    llm = LLM(
        model=MODEL_ID,
        quantization="bitsandbytes",
        load_format="bitsandbytes",
        enable_prefix_caching=False,
        gpu_memory_utilization=0.50,
        max_model_len=16384,
        trust_remote_code=True,
        max_num_seqs=256,
        max_num_batched_tokens=32768,
    )

    sampling_params = SamplingParams(
        max_tokens=MAX_TOKENS,
        temperature=0.6,
        top_p=0.95,
        top_k=20,
        min_p=0.0,
        presence_penalty=0.0,
        repetition_penalty=1.05,
    )

    print("Model loaded.")


    # ## 5. Load Model with Transformers (alternative to vLLM for DataHub)
    # 
    # We load **Qwen3-4B-Thinking-2507** with **INT4 quantization** via BitsAndBytes.  
    # 
    # Key parameters:
    # - `load_in_4bit` — quantization strategy of INT4

    # In[7]:


    """import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

    MODEL_ID = "Qwen/Qwen3-4B-Thinking-2507"

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    llm = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        trust_remote_code=True,
        quantization_config=bnb_config,
        device_map="auto",
    )
    """


    # ## 6. Generate Responses
    # 
    # We format every question into a chat-template prompt, then call `llm.generate()` in one batched pass.  
    # vLLM handles batching and scheduling internally — no manual batching needed.

    # ### Generate with vLLM

    # In[8]:

    # Open output file in append mode
    out_path = Path(OUTPUT_PATH)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    f_out = open(out_path, "w")
    for i, item in enumerate(data):
        item_id = item.get("id")
        print(f"\n── Question {i} (id={item_id}) ──")

        system, user = build_prompt(item["question"], item.get("options"))
        prompt_text = tokenizer.apply_chat_template(
            [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            tokenize=False,
            add_generation_prompt=True,
        )

        output = llm.generate([prompt_text], sampling_params)
        response = output[0].outputs[0].text.strip()

        record = {
            "id": item_id,
            "is_mcq": bool(item.get("options")),
            "response": response,
        }

        f_out.write(json.dumps(record) + "\n")
        f_out.flush()

        print(f"── Finished {i} ──")

    f_out.close()
    print("Done.")


    # In[ ]:

    
    responses = []
    id_to_response = {}

    # Load generated results
    if os.path.exists(OUTPUT_PATH):
        with open(OUTPUT_PATH, "r") as f:
            for line in f:
                r = json.loads(line)
                id_to_response[r["id"]] = r["response"]

    # Align with data order (IMPORTANT for zip(data, responses))
    for item in data:
        item_id = item.get("id")
        responses.append(id_to_response.get(item_id, ""))  # fallback if missing

    print(f"Loaded {len(responses)} responses for Judger")


    # ### Generate with Transformers (for Datahub)

    # In[ ]:


    """# Build prompts for first 5 entries
    prompts = []
    for item in data[:5]:
        system, user = build_prompt(item["question"], item.get("options"))
        prompt_text = tokenizer.apply_chat_template(
            [{"role": "system", "content": system},
            {"role": "user",   "content": user}],
            tokenize=False,
            add_generation_prompt=True,
        )
        prompts.append(prompt_text)

    # Tokenize (padded batch)
    print(f"Generating responses for {len(prompts)} questions...")
    inputs = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=16384,
    ).to(llm.device)

    # Generate
    with torch.no_grad():
        output_ids = llm.generate(
            **inputs,
            max_new_tokens=MAX_TOKENS,
            temperature=0.6,
            top_p=0.95,
            top_k=20,
            repetition_penalty=1.0,
            do_sample=True,
        )

    # Decode only the new tokens (strip the prompt)
    responses = []
    for i, out in enumerate(output_ids):
        new_tokens = out[inputs["input_ids"].shape[1]:]
        responses.append(tokenizer.decode(new_tokens, skip_special_tokens=True).strip())

    # Preview first 3
    for i in range(min(3, len(responses))):
        print(f"\n── Response {i} (id={data[i].get('id')}) ──")
        print(responses[i][:400], "..." if len(responses[i]) > 400 else "")"""


    # ## 7. Score Responses
    # 
    # Scoring differs by question type:
    # 
    # - **MCQ**: extract the predicted letter from `\boxed{}` and compare to the gold letter (exact match).
    # - **Free-form**: use `Judger.auto_judge()` which handles symbolic and numeric equivalence.
    # 
    # Each result record contains `{id, is_mcq, gold, response, correct}`.

    # In[ ]:

    """
    def extract_letter(text: str) -> str:
        m = re.search(r"\\boxed\{([A-Za-z])\}", text)
        if m:
            return m.group(1).upper()

        matches = re.findall(r"\b([A-Z])\b", text.upper())
        return matches[-1] if matches else ""


    def score_mcq(response: str, gold_letter: str) -> bool:
        return extract_letter(response) == gold_letter.strip().upper()


    # Load Judger for free-form scoring
    sys.path.insert(0, ".")
    from judger import Judger
    judger = Judger(strict_extract=False)

    results = []
    for item, response in tqdm(zip(data, responses), total=len(data), desc="Scoring"):
        is_mcq = bool(item.get("options"))
        gold   = item["answer"]

        if is_mcq:
            correct = score_mcq(response, str(gold))
        else:
            gold_list = gold if isinstance(gold, list) else [gold]
            try:
                correct = judger.auto_judge(
                    pred=response,
                    gold=gold_list,
                    options=[[]] * len(gold_list),
                )
            except Exception:
                correct = False

        results.append({
            "id":       item.get("id"),
            "is_mcq":   is_mcq,
            "gold":     gold,
            "response": response,
            "correct":  correct,
        })

    print(f"Scoring complete. {len(results)} results.")


    # ## 8. Summary
    # 
    # Print accuracy broken down by question type.

    # In[ ]:


    mcq_res  = [r for r in results if r["is_mcq"]]
    free_res = [r for r in results if not r["is_mcq"]]

    def acc(subset):
        return sum(r["correct"] for r in subset) / len(subset) * 100 if subset else 0.0

    print("=" * 50)
    print("EVALUATION RESULTS")
    print("=" * 50)
    print(f"  MCQ        : {sum(r['correct'] for r in mcq_res):4d} / {len(mcq_res):4d}  ({acc(mcq_res):.2f}%)")
    print(f"  Free-form  : {sum(r['correct'] for r in free_res):4d} / {len(free_res):4d}  ({acc(free_res):.2f}%)")
    print(f"  Overall    : {sum(r['correct'] for r in results):4d} / {len(results):4d}  ({acc(results):.2f}%)")
    print("=" * 50)
    """

    # In[ ]:


    """with open("incorrect_traces.txt", "w") as f:
        for r in results:
            if not r["correct"]:
                f.write("=" * 60 + "\n")
                f.write(f"ID: {r['id']}\n")
                f.write(f"Type: {'MCQ' if r['is_mcq'] else 'Free-form'}\n")
                f.write(f"Gold: {r['gold']}\n")
                import re
                boxed = re.findall(r'\\boxed\{([^}]+)\}', r['response'])
                f.write(f"Model boxed answers: {boxed}\n")
                f.write(f"Full response:\n{r['response']}\n\n")

    print("Saved incorrect_traces.txt")"""


    # ## 9. Save Results
    # 
    # Results are written as newline-delimited JSON.
    # 
    # **With evaluation** (public set — you have ground-truth):  
    # Each line: `{id, is_mcq, gold, response, correct}`
    # 
    # **Without evaluation** (private test set — no ground-truth available):  
    # Each line: `{id, is_mcq, response}` — omit `gold` and `correct`.
    # 
    # Toggle `SAVE_EVAL` below accordingly.

    # In[ ]:


    """SAVE_EVAL = True   # Set to False when running on the private test set

    out_path = Path(OUTPUT_PATH)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w") as f:
        for r in results:
            if SAVE_EVAL:
                record = {"id": r["id"], "is_mcq": r["is_mcq"], "gold": r["gold"],
                        "response": r["response"], "correct": r["correct"]}
            else:
                record = {"id": r["id"], "is_mcq": r["is_mcq"], "response": r["response"]}
            f.write(json.dumps(record) + "\n")

    print(f"Saved {len(results)} records to {out_path}")"""


    # ## Next Steps
    # 
    # This notebook gives you a working baseline. Here are directions to improve your score:
    # 
    # - **Prompt engineering** — try different system prompts or few-shot examples inside the user turn
    # - **Sampling parameters** — adjust `temperature`, `top_p`, or use majority voting across multiple samples
    # - **Fine-tuning** — the competition allows model fine-tuning; see the course resources for guidance
    # 
    # Good luck!

    # In[ ]:


    import json
    import csv

    # Load all private IDs
    all_data = [json.loads(line) for line in open("data/private.jsonl")]

    # Load completed results
    completed = {}
    with open("results/private_results.jsonl") as f:
        for line in f:
            r = json.loads(line)
            completed[r["id"]] = r["response"]

    print(f"Completed: {len(completed)} / {len(all_data)}")

    # Write CSV with newlines replaced
    with open("results/submission.csv", "w", newline="") as f:
        writer = csv.writer(f, quoting=csv.QUOTE_ALL)
        writer.writerow(["id", "response"])
        for d in all_data:
            response = completed.get(d["id"], "\\boxed{}")
            # Replace newlines so each response stays on one row
            response = response.replace("\n", " ").replace("\r", " ")
            writer.writerow([d["id"], response])

    print(f"Saved submission.csv with {len(all_data)} rows")

run_inference()