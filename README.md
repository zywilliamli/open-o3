Repo for training GPT O3 style multi step web search capability into open source models

Any code involving GPU (inferencing or training model) is run using [SkyPilot](https://docs.skypilot.co/en/latest/overview.html#bringing-your-infra) and [RunPod](https://docs.runpod.io/integrations/skypilot)
 - `collect_sft.py` can be run locally to collect trajectories from gpt models, using web search tools to answer queries from a selection of Q&A evals (BrowseComp, SimpleQA, HotpotQA and MS Marco)
 - `run_sft.py` is the skypilot code that runs `sft.py`, the SFT training script that uses the trajectories collected earlier to clone the behaviour of the GPT model
   - in practice this step improves tool call accuracy, and increases the number of times the model searches for information before answering
 - `run_train.py` is the skypilot code that runs `train.py`, the RL training script that uses GRPO on either a base model or an SFT checkpoint, reward is binary for outcome correctness
 - `run_eval.py` is the skypilot code that runs `eval.py`, which loads a training checkpoint from hf hub and evaluates on either BrowseComp or SimpleQA

Copy `.env.example` into `.env` and add your API keys 

This repo uses `uv`, use this [link](https://docs.astral.sh/uv/getting-started/installation/) to set up uv, then run `uv sync` to install the project deps