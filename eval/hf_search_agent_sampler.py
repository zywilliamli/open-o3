import asyncio
import json
from typing import Any

from eval.interfaces import MessageList, SamplerBase, SamplerResponse
from search_agent import SearchAgent
import art
from art import Trajectory
from art.langgraph import wrap_rollout
from art.trajectories import get_messages
from art.local import LocalBackend
from art.utils.output_dirs import get_output_dir_from_model_properties
import os


class HFSearchAgentSampler(SamplerBase):
    def __init__(self, use_trained_peft: bool = False):
        self.model = art.TrainableModel(
            name="sft-open-o3",
            project="open-o3",
            base_model="unsloth/Qwen2.5-14B-Instruct"
        )
        if use_trained_peft:
            art_path = get_output_dir_from_model_properties(name="sft-open-o3", project="open-o3")
            os.makedirs(art_path, exist_ok=True)

            try:
                import subprocess

                aws_cmd = "/usr/local/bin/aws"

                print("Starting S3 download...")
                result = subprocess.run([
                    aws_cmd, "s3", "sync",
                    art_path,
                    f"s3://{os.environ.get('BACKUP_BUCKET')}/models/open-o3-sft-3",
                    "--storage-class", "STANDARD_IA"
                ], capture_output=True, text=True, timeout=600)

                if result.returncode == 0:
                    print("✅ S3 download completed successfully!")
                else:
                    print(f"❌ S3 download failed: {result.stderr}")
                    raise Exception(f"S3 sync failed: {result.stderr}")
            except Exception as e:
                print(f"S3 download failed with exception: {e}")
        backend = LocalBackend()
        asyncio.run(self.model.register(backend))
        print('finished registering model')

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
        traj = await wrap_rollout(self.model, self.rollout)(message_list)
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
