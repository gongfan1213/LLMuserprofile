import os

from langchain_openai import ChatOpenAI


def create_llm():
    return ChatOpenAI(
        api_key=os.getenv("DASHSCOPE_APIKEY"),
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        model="qwen-max",
        verbose=True,
    )
