import os

class Settings:
    BASE_ULR: str = os.environ.get("BASE_ULR")
    API_KEY: str = os.environ.get("API_KEY")

settings = Settings()