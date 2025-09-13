import asyncio
import base64
import csv
import hashlib
import json
import os
import random
from tqdm import tqdm
import pandas as pd

from datasets import load_dataset, Dataset, concatenate_datasets

from search_agent import SearchAgent
from art import Trajectory
from art.langgraph import wrap_rollout
from art.trajectories import get_messages
import art

RANDOM_SEED = 0
random.seed(RANDOM_SEED)


def load_browse_comp(train_test_split: float = 0.8):
    def derive_key(password: str, length: int) -> bytes:
        """Derive a fixed-length key from the password using SHA256."""
        hasher = hashlib.sha256()
        hasher.update(password.encode())
        key = hasher.digest()
        return key * (length // len(key)) + key[: length % len(key)]

    def decrypt(ciphertext_b64: str, password: str) -> str:
        """Decrypt base64-encoded ciphertext with XOR."""
        encrypted = base64.b64decode(ciphertext_b64)
        key = derive_key(password, len(encrypted))
        decrypted = bytes(a ^ b for a, b in zip(encrypted, key))
        return decrypted.decode()

    df = pd.read_csv(
        "https://openaipublic.blob.core.windows.net/simple-evals/browse_comp_test_set.csv"
    )
    examples = [row.to_dict() for _, row in df.iterrows()]
    qa_pairs = [{"query": decrypt(example.get("problem", ""), example.get("canary", "")),
                 "answer": decrypt(example.get("answer", ""), example.get("canary", ""))} for example in examples]
    random.shuffle(qa_pairs)

    return {"train": qa_pairs[:int(len(qa_pairs) * train_test_split)],
            "test": qa_pairs[int(len(qa_pairs) * train_test_split):]}


def load_simple_qa(train_test_split: float = 0.8):
    df = pd.read_csv(
        "https://openaipublic.blob.core.windows.net/simple-evals/simple_qa_test_set.csv"
    )
    examples = [row.to_dict() for _, row in df.iterrows()]
    qa_pairs = [{"query": example.get("problem", ""),
                 "answer": example.get("answer", "")} for example in examples]
    random.shuffle(qa_pairs)

    return {"train": qa_pairs[:int(len(qa_pairs) * train_test_split)],
            "test": qa_pairs[int(len(qa_pairs) * train_test_split):]}


browse_comp = load_browse_comp(0.8)
simple_qa = load_simple_qa(0.8)
hotpot_qa = load_dataset("hotpotqa/hotpot_qa", "fullwiki").rename_column("question", "query")
ms_marco = load_dataset("microsoft/ms_marco", 'v2.1')

sft_sample_dataset = concatenate_datasets([
    Dataset.from_list(browse_comp['train'][:500]).add_column("source", ["browse_comp"] * 500),
    Dataset.from_list(simple_qa['train'][:500]).add_column("source", ["simple_qa"] * 500),
    hotpot_qa['train'].select(range(500)).add_column("source", ["hotpot_qa"] * 500),
    ms_marco['train'].select(range(500)).add_column("source", ["ms_marco"] * 500)])

sft_sample_dataset = sft_sample_dataset.shuffle(RANDOM_SEED).select(range(2000))


async def rollout(user_input):
    agent = SearchAgent()
    graph = await agent.build_trainable_graph()
    await graph.ainvoke({"messages": [{"role": "user", "content": user_input}]}, agent.config)
    return Trajectory(messages_and_choices=[], reward=0)


model = art.Model(
    name="o4-mini", project="o4-mini-web-search",
    inference_model_name="o4-mini",
    inference_api_key=os.getenv("OPENAI_API_KEY"),
    inference_base_url="https://api.openai.com/v1/",
)

write_header = not os.path.exists("sft-training-data.csv") or os.path.getsize("sft-training-data.csv") == 0
csv_header = ['dataset', 'query', 'reference_answer', 'agent_answer', 'trajectory']

if write_header:
    with open("sft-training-data.csv", 'a') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=csv_header)
        writer.writeheader()

processed_queries = pd.read_csv("sft-training-data.csv")['query'].values

for row in tqdm(sft_sample_dataset):
    query = row["query"]
    reference_answer = row["answer"] if row["answer"] else row["answers"]  # ms marco answer is a list
    dataset = row["source"]

    if query in processed_queries:
        print("skipping query: '{}'".format(query[:100]))
        continue
    try:
        print("processing query: '{}'".format(query[:100]))
        traj = asyncio.run(wrap_rollout(model, rollout)(query))
        training_data = {"messages": get_messages(traj.messages_and_choices), "tools": traj.tools}

        with open("training-trajectories.jsonl", 'a', newline='') as f:
            f.write(json.dumps(training_data) + '\n')

        with open("sft-training-data.csv", 'a') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=csv_header)
            writer.writerow({"dataset": dataset, "query": query, "reference_answer": reference_answer,
                             "agent_answer": traj.messages_and_choices[-1].message.content,
                             "trajectory": training_data})
    except Exception as error:
        with open("sft-training-data.csv", 'a') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=csv_header)
            writer.writerow({"dataset": dataset, "query": query, "reference_answer": reference_answer,
                             "agent_answer": "failed", "trajectory": {}})
        print(error)
