import asyncio
import os
from enum import Enum
from typing import Annotated

from typing_extensions import TypedDict

import dotenv

dotenv.load_dotenv()

from langchain_google_community import GoogleSearchAPIWrapper
from langchain_exa import ExaSearchResults
from langchain_tavily import TavilySearch
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langchain.chat_models import init_chat_model
from langchain_openai import ChatOpenAI
from art.langgraph import init_chat_model as train_model
from langchain_core.messages import SystemMessage
from langchain_core.tools import tool

SYSTEM_PROMPT = """"You are an AI assistant, your task is to answer the user's query.
You have access to a web search tool to help you gather information when needed. """


@tool
def google_web_search(query: str) -> list:
    """search the web using Google Search"""
    return GoogleSearchAPIWrapper().results(query, num_results=5)


@tool
def exa_web_search(query: str) -> str:
    """search the web using Exa Search"""
    return ExaSearchResults()._run(query=query, num_results=5, highlights=True)


@tool
def tavily_web_search(query: str) -> str:
    """search the web using Tavily Search"""
    res = TavilySearch(max_results=5).invoke({"query": query})
    clean_res = ''
    for result in res['results']:
        clean_res += f"url: {result['url']}\n"
        clean_res += f"title: {result['title']}\n"
        clean_res += f"content: {result['content']}\n\n"
    return clean_res


class SearchProvider(Enum):
    GOOGLE = google_web_search
    EXA = exa_web_search
    TAVILY = tavily_web_search


class State(TypedDict):
    messages: Annotated[list, add_messages]


class SearchAgent:
    def __init__(self, search_limit: int = 5, search_provider: SearchProvider = SearchProvider.TAVILY) -> None:
        self.config = {"configurable": {"thread_id": "1"}, "recursion_limit": 50}
        self.search_limit = search_limit
        self.search_provider = search_provider
        self.graph = None

    async def build_trainable_graph(self):
        return await self._build_graph(train_model())

    async def build_graph(self, model_name: str = "openai:o3"):
        if model_name.startswith("openai"):
            llm = init_chat_model(model_name)
        else:
            llm = ChatOpenAI(
                model=model_name,
                base_url="https://openrouter.ai/api/v1",
                api_key=os.environ["OPENROUTER_API_KEY"],
            )

        return await self._build_graph(llm)

    async def _build_graph(self, llm):
        graph_builder = StateGraph(State)
        llm_with_tools = llm.bind_tools([self.search_provider.value])

        async def chatbot(state: State):
            messages = state["messages"]
            # Prepend the system message if it's not already present
            if not any(
                    (getattr(m, "type", None) == "system") or (isinstance(m, dict) and m.get("role") == "system")
                    for m in messages
            ):
                messages = [SystemMessage(content=SYSTEM_PROMPT), *messages]
            msg = await llm_with_tools.ainvoke(messages)
            return {"messages": [msg]}

        graph_builder.add_node("chatbot", chatbot)
        graph_builder.add_node("tools", ToolNode(tools=[self.search_provider.value]))
        graph_builder.add_conditional_edges("chatbot", tools_condition)
        graph_builder.add_edge("tools", "chatbot")
        graph_builder.set_entry_point("chatbot")
        return graph_builder.compile(checkpointer=InMemorySaver())


async def main():
    agent = SearchAgent()
    graph = await agent.build_graph()
    print("Type 'quit' to exit.")

    while True:
        user_input = await asyncio.to_thread(input, "User: ")
        if user_input.lower() in {"quit", "exit", "q"}:
            print("Goodbye!")
            break

        try:
            stream = graph.astream(
                {"messages": [{"role": "user", "content": user_input}]},
                agent.config,
                stream_mode="values",
            )
            async for event in stream:
                try:
                    event["messages"][-1].pretty_print()
                except Exception:
                    content = event["messages"][-1].get("content")
                    if content:
                        print(content)
        except Exception as e:
            print(f"Error: {e}")
            break


if __name__ == "__main__":
    asyncio.run(main())
