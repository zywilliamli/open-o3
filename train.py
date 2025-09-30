import random

import pandas as pd
from dotenv import load_dotenv

from eval.o_chat_completion_sampler import OChatCompletionSampler
from eval.simpleqa_eval import SimpleQAEval

load_dotenv()

from langchain_core.messages import AIMessage

_original_init = AIMessage.__init__


def _patched_init(self, **kwargs):
    tool_calls = kwargs.get("tool_calls", [])
    for call in tool_calls:
        if isinstance(call.get("args"), str):
            try:
                print(call['args'])
                call["args"] = json.loads(call["args"])
            except json.JSONDecodeError:
                call["args"] = {}
    _original_init(self, **kwargs)


AIMessage.__init__ = _patched_init

import art
from art.utils.output_dirs import get_output_dir_from_model_properties
from art.local import LocalBackend
import asyncio
from art.utils import iterate_dataset
from art.langgraph import wrap_rollout
from tqdm.asyncio import tqdm
import json
import os, glob, re
import polars as pl
from search_agent import SearchAgent
from art import Trajectory
from huggingface_hub import create_repo, upload_folder, snapshot_download


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


async def train():
    HF_MODEL_ID = "twelvehertz/open-o3-sft-13-merged-full"

    train_set = load_simple_qa(0.8)['train'][500:700]
    val_set = load_simple_qa(0.8)['train'][700:720]

    model = art.TrainableModel(
        name="rl-open-o3",
        project="open-o3",
        base_model=HF_MODEL_ID
    )
    backend = LocalBackend()
    await model.register(backend)

    train_iterator = iterate_dataset(
        train_set,
        groups_per_step=1,
        num_epochs=1,
        initial_step=await model.get_step(),
    )

    harness = SimpleQAEval(OChatCompletionSampler(model="o4-mini"))

    async def rollout(row):
        try:
            agent = SearchAgent()
            graph = await agent.build_trainable_graph()
            result = await graph.ainvoke({"messages": [{"role": "user", "content": row['query']}]}, agent.config)
            response = result["messages"][-1].content
            grade_result = harness.grade_sample(row['query'], row['answer'], response)
            return Trajectory(messages_and_choices=[], reward=1 if grade_result == "A" else 0)
        except Exception as e:
            print(e)
            return Trajectory(messages_and_choices=[], reward=0)

    async def benchmark_model(model, data):
        val_trajectories = await tqdm.gather(
            *(wrap_rollout(model, rollout)(row) for row in data),
            desc=f"validation {model.name}",
        )
        for t in val_trajectories:
            try:
                print(len(t.messages_and_choices))
            except:
                print('error')

        valid_trajectories = [t for t in val_trajectories if isinstance(t, art.Trajectory)]

        if model._backend is not None:
            await model.log(valid_trajectories)

        metrics = pl.DataFrame(
            [{**t.metrics, "reward": t.reward} for t in valid_trajectories]
        )

        avg_metrics = metrics.select(
            [pl.mean(c).alias(c) for c in metrics.columns]
        ).with_columns(pl.lit(len(valid_trajectories)).alias("n_trajectories"))
        print(avg_metrics)

        return avg_metrics

    for batch in train_iterator:
        if (batch.step) % 20 == 0:
            print(f"\n--- Evaluating at Iteration {batch.step} ---")
            await benchmark_model(model, val_set)

            repo_id = "twelvehertz/open-o3-simpleqa-rl"

            art_path = get_output_dir_from_model_properties(name="rl-open-o3", project="open-o3")
            ckpt_root = os.path.join(art_path, "checkpoints")
            numeric_dirs = [
                d for d in os.listdir(ckpt_root)
                if d.isdigit() and os.path.isdir(os.path.join(ckpt_root, d))
            ]

            if not numeric_dirs:
                print("⚠️ No numeric checkpoint folders found; skipping Hub upload this iteration.")
            else:
                latest_ckpt = max(numeric_dirs, key=lambda d: int(d))
                latest_path = os.path.join(ckpt_root, latest_ckpt)
                print(f"Uploading checkpoint folder: {latest_path}")

                create_repo(repo_id, token=os.environ["HF_TOKEN"], private=False, exist_ok=True)

                upload_folder(
                    folder_path=latest_path,
                    token=os.environ["HF_TOKEN"],
                    repo_id=repo_id,
                    commit_message=f"Upload RL checkpoint {latest_ckpt} (LoRA adapter)"
                )

        groups = await art.gather_trajectory_groups(
            (
                art.TrajectoryGroup(
                    (
                        wrap_rollout(model, rollout)(row)
                        for _ in range(16)
                    )
                )
                for row in batch.items
            ),
        )

        await model.train(
            groups,
            _config=art.dev.TrainConfig(
                precalculate_logprobs=False
            ),
        )
    print("Training finished.")


if __name__ == "__main__":
    asyncio.run(train())
