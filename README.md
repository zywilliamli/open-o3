Repo for training GPT O3 style multi step web search capability into open source models

Any code involving GPU (inferencing or training model) is run via SkyPilot and RunPod
 - `collect_sft.py` can be run locally to collect search tool use trajectories from gpt models, available query sets are BrowseComp, SimpleQA, HotpotQA and MS Marco
 - `run_sft.py` is the skypilot code that runs `sft.py`, the sft training script that uses the trajectories collected earlier
 - `run_train.py` is the skypilot code that runs `train.py`, the rl training script that uses grpo on either a base model or an sft checkpoint
 - `run_eval.py` is the skypilot code that runs `eval.py`, which loads a training checkpoint from hf hub and evaluates on either BrowseComp or SimpleQA

Copy `.env.example` into `.env` and add your API keys 

This repo uses `uv`