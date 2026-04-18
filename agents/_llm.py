from langchain_openai import ChatOpenAI

from config import Settings

_settings = Settings()


def chat_model() -> ChatOpenAI:
    return ChatOpenAI(
        model=_settings.model_name,
        temperature=0.7,
        api_key=_settings.api_key.get_secret_value(),
    )
