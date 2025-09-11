import argparse

from eval.browsecomp_eval import BrowseCompEval
from eval.hf_search_agent_sampler import HFSearchAgentSampler
from eval.simpleqa_eval import SimpleQAEval
from eval.oai_search_agent_sampler import OAISearchAgentSampler
from eval.o_chat_completion_sampler import OChatCompletionSampler
from dotenv import load_dotenv

SUPPORTED_EVAL_SETS = ['browsecomp', 'simpleqa']


def run_eval(args):
    load_dotenv()
    grader = OChatCompletionSampler(model=args.grader_model)

    if args.eval_set == "browsecomp":
        harness = BrowseCompEval(grader_model=grader, num_examples=args.num_samples)
    elif args.eval_set == "simpleqa":
        harness = SimpleQAEval(grader_model=grader, num_examples=args.num_samples)
    else:
        raise ValueError(f"Invalid eval set name, currently only supporting {SUPPORTED_EVAL_SETS}")

    if args.sampler == "oai":
        sampler = OAISearchAgentSampler(args.sampler_model)
    elif args.sampler == "hf":
        sampler = HFSearchAgentSampler(args.sampler_model)
    else:
        raise ValueError("Invalid sampler, must be 'oai' or 'hf'")

    print(harness(sampler))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--grader-model", default="o4-mini", help="openai model for grading response")
    parser.add_argument("--sampler", default="oai",
                        help="'oai' for OpenAI models via API or 'hf' for HuggingFace models run locally")
    parser.add_argument("--sampler-model", default="o4-mini",
                        help="openai model name if sampler set to 'oai', or huggingface model name if sampler set to 'hf'")
    parser.add_argument("--num-samples", default=1, help="number of samples to run")
    parser.add_argument("--eval-set", default="simpleqa",
                        help=f"eval set name, currently supporting {SUPPORTED_EVAL_SETS}")
    args = parser.parse_args()

    run_eval(args)


if __name__ == "__main__":
    main()
