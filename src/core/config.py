from pydantic_settings import BaseSettings, SettingsConfigDict
from src.services.chat.llm import GenAIAgentClient
import os


class Settings(BaseSettings):
    BASE_ULR: str
    API_KEY: str
    LOG_LEVEL: str = "INFO"

    ORIGINS: str
    GPT_MODEL: str
    TEMPERATURE: float = 0.25

    @property
    def client(self):
        return GenAIAgentClient(base_url=self.BASE_ULR, api_key=self.API_KEY)

    @property
    def tree(self):
        if os.path.exists("data/info"):
            tree_path = "data/info"
        else:
            tree_path = None
        return tree_path

    model_config = SettingsConfigDict(env_file=".env")


settings = Settings()
