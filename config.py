from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    openai_api_key: str = ""
    hf_auth_token: str = ""
    whisperx_model_size: str = "large-v2"
    whisperx_device: str = "cpu"
    whisperx_compute_type: str = "int8"
    whisperx_batch_size: int = 4
    reports_dir: str = "reports"
    uploads_dir: str = "uploads"
    openai_model: str = "gpt-4o"

    class Config:
        env_file = ".env"


settings = Settings()
