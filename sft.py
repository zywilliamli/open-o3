from unsloth import FastLanguageModel
from huggingface_hub import create_repo, upload_folder
import os
import torch

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Qwen2.5-14B-Instruct",
    load_in_4bit=False,  # 4bit uses much less memory
    load_in_8bit=True,  # A bit more accurate, uses 2x memory
    full_finetuning=False,  # We have full finetuning now!
    # token = "hf_...",      # use one if using gated models
)

model = FastLanguageModel.get_peft_model(
    model,
    r=16,  # Choose any number > 0! Suggested 8, 16, 32, 64, 128
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],
    lora_alpha=32,  # Best to choose alpha = rank or rank*2
    lora_dropout=0,  # Supports any, but = 0 is optimized
    bias="none",  # Supports any, but = "none" is optimized
    # [NEW] "unsloth" uses 30% less VRAM, fits 2x larger batch sizes!
    use_gradient_checkpointing="unsloth",  # True or "unsloth" for very long context
    random_state=3407,
    use_rslora=False,  # We support rank stabilized LoRA
    loftq_config=None,  # And LoftQ
)

import json
from datasets import Dataset as HFDataset
import copy


def clean_messages(messages):
    msgs = copy.deepcopy(messages)
    for m in msgs:
        # drop empty tool_calls arrays
        if m.get("tool_calls") == []:
            m.pop("tool_calls", None)
        # ensure content exists and is a string
        if "content" not in m or m.get("content") is None:
            m["content"] = ""
    return msgs


all_conversations = []
with open("training-trajectories-new.jsonl", "r") as f:
    for line in f:
        all_conversations.append(json.loads(line))

dataset_list = []
for conversation in all_conversations:
    messages = clean_messages(conversation["messages"])  # <-- use the cleaner
    tools = conversation.get("tools", None)
    dataset_list.append(
        tokenizer.apply_chat_template(
            messages,
            tools=tools,
            add_generation_prompt=False,
            tokenize=False,
            enable_thinking=False,
        )
    )

full_dataset = HFDataset.from_dict({"text": dataset_list})
split_dataset = full_dataset.train_test_split(test_size=0.05, seed=42)
train_dataset = split_dataset["train"]
val_dataset = split_dataset["test"]

print(train_dataset[0])
from trl import SFTTrainer, SFTConfig

# Keep most recent context if truncation happens
tokenizer.truncation_side = "right"
tokenizer.padding_side = "right"

MAX_LEN = 16384

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    args=SFTConfig(
        group_by_length=True,
        dataset_text_field="text",
        max_seq_length=MAX_LEN,
        packing=False,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=16,
        warmup_steps=10,
        num_train_epochs=8,
        learning_rate=5e-5,
        logging_steps=5,
        eval_steps=10,
        eval_strategy="steps",
        save_strategy="steps",
        save_steps=10,
        load_best_model_at_end=True,
        optim="adamw_torch",
        weight_decay=0.01,
        lr_scheduler_type="cosine",
        seed=3407,
        report_to="none",
        max_grad_norm=0.5,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        save_total_limit=2,
        dataset_num_proc=4,
        dataloader_num_workers=2
    ),
)

QWEN_INSTRUCTION_PART = "<|im_start|>user\n"
QWEN_RESPONSE_PART = "<|im_start|>assistant\n"

# Always include ALL assistant spans (including tool invocations)
# in the supervised labels by widening the mask to every
# <|im_start|>assistant ... <|im_end|> block.
base_collator = trainer.data_collator

# Literal token sequences for assistant header and end marker
resp_tokens = tokenizer.encode(QWEN_RESPONSE_PART, add_special_tokens=False)
end_tokens = tokenizer.encode("<|im_end|>", add_special_tokens=False)


def _find_all(subseq, seq):
    starts = []
    n, m = len(seq), len(subseq)
    if m == 0:
        return starts
    i = 0
    while i <= n - m:
        if seq[i:i + m] == subseq:
            starts.append(i)
            i += m
        else:
            i += 1
    return starts


class CollatorAllAssistant:
    def __init__(self, base):
        self.base = base

    def __call__(self, features):
        batch = self.base(features)
        input_ids = batch["input_ids"]  # [B, T]
        # Rebuild labels from scratch: only assistant spans supervised
        for b in range(input_ids.size(0)):
            ids_t = input_ids[b]
            labs_t = torch.full_like(ids_t, -100)
            ids = ids_t.tolist()
            # Find every assistant block and label its content
            starts = _find_all(resp_tokens, ids)
            for s in starts:
                tail = ids[s + len(resp_tokens):]
                end_rel_starts = _find_all(end_tokens, tail)
                if not end_rel_starts:
                    continue
                e = s + len(resp_tokens) + end_rel_starts[0]
                start_idx = s + len(resp_tokens)
                end_idx = e
                labs_t[start_idx:end_idx] = ids_t[start_idx:end_idx]
            batch["labels"][b] = labs_t
        return batch


trainer.data_collator = CollatorAllAssistant(base_collator)

trainer_stats = trainer.train()

model.save_pretrained("model")
tokenizer.save_pretrained("model")

try:
    import os
    import subprocess

    aws_cmd = "/usr/local/bin/aws"

    print("Starting S3 upload...")
    result = subprocess.run([
        aws_cmd, "s3", "sync",
        "./model",
        f"s3://{os.environ.get('BACKUP_BUCKET')}/models/open-o3-sft-3",
        "--storage-class", "STANDARD_IA"
    ], capture_output=True, text=True, timeout=600)

    if result.returncode == 0:
        print("✅ S3 upload completed successfully!")
    else:
        print(f"❌ S3 upload failed: {result.stderr}")
        raise Exception(f"S3 sync failed: {result.stderr}")
except Exception as e:
    print(f"S3 upload failed with exception: {e}")

repo_id = "twelvehertz/open-o3-sft-3"

create_repo(repo_id, token=os.environ["HF_TOKEN"], private=False, exist_ok=True)

upload_folder(
    folder_path="model",
    repo_id=repo_id,
    token=os.environ["HF_TOKEN"],
    commit_message="qwen 2.5 14b instruct lora module for web search",
)
print(f"Pushed to https://huggingface.co/{repo_id}")
