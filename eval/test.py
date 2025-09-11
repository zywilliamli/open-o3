from eval.browsecomp_eval import BrowseCompEval
from eval.hf_search_agent_sampler import HFSearchAgentSampler
from eval.simpleqa_eval import SimpleQAEval
from eval.oai_search_agent_sampler import OAISearchAgentSampler
from eval.o_chat_completion_sampler import OChatCompletionSampler
from dotenv import load_dotenv

load_dotenv()
oai_sampler = OChatCompletionSampler(model="o4-mini")
search_sampler = HFSearchAgentSampler()
# harness = BrowseCompEval(grader_model=oai_sampler, num_examples=1)
harness = SimpleQAEval(grader_model=oai_sampler, num_examples=1)
print(harness(search_sampler))
