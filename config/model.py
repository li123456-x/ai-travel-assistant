import os

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model


load_dotenv()

if not os.getenv("DEEPSEEK_API_KEY"):
    raise RuntimeError("缺少必要环境变量：DEEPSEEK_API_KEY")

model = init_chat_model(
    model="deepseek-v4-flash",
    model_provider="deepseek",
    extra_body={
        "thinking": {
            "type": "disabled",
        }
    },
)
