import copy
import json
import os

import torch
from unsloth import FastLanguageModel
from unsloth_zoo.dataset_utils import train_on_responses_only
from datasets import Dataset as HFDataset
from huggingface_hub import create_repo, upload_folder
from peft import PeftModel
from transformers import AutoTokenizer, AutoModelForCausalLM, DataCollatorForSeq2Seq
from trl import SFTTrainer, SFTConfig


def add_special_toks(model, tokenizer):
    SPECIALS = ["</tool_response>", "<tool_response>", "</tool_call>", "<tool_call>"]
    to_add = [t for t in SPECIALS if t not in tokenizer.get_vocab()]
    if to_add:
        print(f"Adding special tokens to tokenizer: {to_add}")
        tokenizer.add_special_tokens({"additional_special_tokens": to_add})
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model.resize_token_embeddings(len(tokenizer))


def clean_messages(messages):
    msgs = copy.deepcopy(messages)
    for m in msgs:
        if m.get("tool_calls") == []:
            m.pop("tool_calls", None)
        if "content" not in m or m.get("content") is None:
            m["content"] = ""
    return msgs


def keep_last_assistant_only(example):
    labs = example["labels"]
    if labs is None:
        return example
    n = len(labs)
    i = n - 1
    while i >= 0 and labs[i] == -100:
        i -= 1
    if i < 0:
        return example
    end = i
    start = end
    while start > 0 and labs[start - 1] != -100:
        start -= 1
    new = [-100] * n
    new[start:end + 1] = labs[start:end + 1]
    example["labels"] = new
    return example


BASE_MODEL_ID = "unsloth/Qwen2.5-14B-Instruct"

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=BASE_MODEL_ID,
    load_in_4bit=False,
    load_in_8bit=True,
    full_finetuning=False,
    # token=os.getenv("HF_TOKEN") # for gated models
)

model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",  # attn
                    "gate_proj", "up_proj", "down_proj"],  # mlp
    lora_alpha=32,
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=3407,
    use_rslora=False,
    loftq_config=None,
)

if os.getenv("ADD_SPECIAL_TOKENS"):
    add_special_toks(model, tokenizer)

all_conversations = []
with open("training-trajectories-dummy.jsonl", "r") as f:
    for line in f:
        all_conversations.append(json.loads(line))

dataset_list = []
for conversation in all_conversations:
    messages = clean_messages(conversation["messages"])
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

tokenizer.truncation_side = "right"
tokenizer.padding_side = "right"

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer),
    args=SFTConfig(
        group_by_length=True,
        dataset_text_field="text",
        max_seq_length=16384,
        packing=False,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=16,
        warmup_steps=10,
        num_train_epochs=5,
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
        save_total_limit=10,
        dataset_num_proc=4,
        dataloader_num_workers=2
    ),
)

trainer = train_on_responses_only(
    trainer,
    instruction_part="<|im_start|>user\n",
    response_part="<|im_start|>assistant\n",
)

if os.getenv("TRAIN_ON_LAST_MESSAGE_ONLY"):
    trainer.train_dataset = trainer.train_dataset.map(keep_last_assistant_only)
    if getattr(trainer, "eval_dataset", None) is not None:
        trainer.eval_dataset = trainer.eval_dataset.map(keep_last_assistant_only)

if os.getenv("DEBUG"):
    print("Sample data row w/ chat template: ")
    print(train_dataset[0])
    print("=" * 100)

    ids = trainer.train_dataset[0]["input_ids"]
    labels = trainer.train_dataset[0]["labels"]
    space = tokenizer("*", add_special_tokens=False).input_ids[0]
    unmasked_sample = tokenizer.decode([space if l == -100 else t for t, l in zip(ids, labels)])
    print("Sample data row after masking: ")
    print(unmasked_sample)
    print("=" * 100)

trainer_stats = trainer.train()

model.save_pretrained("model")
tokenizer.save_pretrained("model")

tok = AutoTokenizer.from_pretrained(BASE_MODEL_ID, use_fast=True)
base = AutoModelForCausalLM.from_pretrained(BASE_MODEL_ID, torch_dtype="auto")

if os.getenv("ADD_SPECIAL_TOKENS"):
    add_special_toks(base, tok)

merged = PeftModel.from_pretrained(base, 'model')
merged = merged.merge_and_unload()

merged.save_pretrained("model_full")
tok.save_pretrained("model_full")

repo_id = "twelvehertz/open-o3-sft"

create_repo(repo_id, token=os.environ["HF_TOKEN"], private=False, exist_ok=True)

upload_folder(
    folder_path="model_full",
    repo_id=repo_id,
    token=os.environ["HF_TOKEN"],
    commit_message="qwen 2.5 14b instruct lora module for web search",
)
print(f"Pushed to https://huggingface.co/{repo_id}")
