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


class HFSearchAgentSampler(SamplerBase):

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
        return await graph.ainvoke({"messages": message_list}, agent.config)

    def __call__(self, message_list: MessageList) -> SamplerResponse:
        try:
            response = asyncio.run(self.rollout(message_list))
            content = response["messages"][-1].content
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
