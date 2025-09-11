import asyncio
import json
import os
from typing import Any


from eval.interfaces import MessageList, SamplerBase, SamplerResponse
from search_agent import SearchAgent

from art import Trajectory
from art.langgraph import wrap_rollout
from art.trajectories import get_messages
import art


class OAISearchAgentSampler(SamplerBase):

    def __init__(self, model_name: str):
        self.model_name = model_name

    def _handle_image(
            self,
            image: str,
            encoding: str = "base64",
            format: str = "png",
            fovea: int = 768,
    ):
        new_image = {
            "type": "image_url",
            "image_url": {
                "url": f"data:image/{format};{encoding},{image}",
            },
        }
        return new_image

    def _handle_text(self, text: str):
        return {"type": "text", "text": text}

    def _pack_message(self, role: str, content: Any):
        return {"role": str(role), "content": content}

    async def rollout(self, message_list):
        agent = SearchAgent()
        graph = await agent.build_trainable_graph()
        await graph.ainvoke({"messages": message_list}, agent.config)
        return Trajectory(messages_and_choices=[], reward=0)

    async def _build_and_invoke(self, message_list: MessageList):
        model = art.Model(
            name=self.model_name, project=f"{self.model_name}-eval",
            inference_model_name=self.model_name,
            inference_api_key=os.getenv("OPENAI_API_KEY"),
            inference_base_url="https://api.openai.com/v1/",
        )
        traj = await wrap_rollout(model, self.rollout)(message_list)
        training_data = []

        training_data.append({"messages": get_messages(traj.messages_and_choices), "tools": traj.tools})
        for histroy in traj.additional_histories:
            training_data.append({"messages": get_messages(histroy.messages_and_choices), "tools": histroy.tools})

        with open("training-data.jsonl", 'a') as f:
            for data in training_data:
                f.write(json.dumps(data) + '\n')

        return traj

    def __call__(self, message_list: MessageList) -> SamplerResponse:
        try:
            response = asyncio.run(self._build_and_invoke(message_list))
            content = response.messages_and_choices[-1].message.content
            return SamplerResponse(
                response_text=content,
                response_metadata={"usage": None},
                actual_queried_message_list=message_list,
            )
        except Exception as e:
            print(f"Error executing search agent: {e}")
            return SamplerResponse(
                response_text="",
                response_metadata={"usage": None},
                actual_queried_message_list=message_list,
            )
