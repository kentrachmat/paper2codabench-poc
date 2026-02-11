"""
Configuration module for Azure OpenAI and other settings.
Loads credentials from environment variables.
"""
import os
from dotenv import load_dotenv
from pathlib import Path

# Load .env file from project root
project_root = Path(__file__).parent.parent
env_path = project_root / ".env"
load_dotenv(env_path)


class Config:
    """Configuration settings"""

    # Azure OpenAI Settings
    AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
    AZURE_OPENAI_KEY = os.getenv("AZURE_OPENAI_KEY")
    AZURE_OPENAI_DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4")
    AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION")  # Optional

    # Project Paths
    PROJECT_ROOT = project_root
    PAPERS_DIR = project_root / "papers"
    TASKSPEC_DIR = project_root / "taskspec"
    BUNDLES_DIR = project_root / "bundles"
    EXAMPLES_DIR = project_root / "examples"
    TEMPLATES_DIR = project_root / "src" / "templates"

    @classmethod
    def validate(cls):
        """Validate required configuration"""
        missing = []

        if not cls.AZURE_OPENAI_ENDPOINT:
            missing.append("AZURE_OPENAI_ENDPOINT")
        if not cls.AZURE_OPENAI_KEY:
            missing.append("AZURE_OPENAI_KEY")
        if not cls.AZURE_OPENAI_DEPLOYMENT:
            missing.append("AZURE_OPENAI_DEPLOYMENT")

        if missing:
            raise ValueError(
                f"Missing required environment variables: {', '.join(missing)}\n"
                f"Please copy .env.example to .env and fill in your credentials."
            )

        return True

    @classmethod
    def ensure_directories(cls):
        """Ensure all required directories exist"""
        for dir_path in [
            cls.TASKSPEC_DIR,
            cls.BUNDLES_DIR,
            cls.EXAMPLES_DIR,
            cls.TEMPLATES_DIR
        ]:
            dir_path.mkdir(parents=True, exist_ok=True)


# Validate configuration on import (will raise error if credentials missing)
# Comment out during development if you don't have credentials yet
# Config.validate()
Config.ensure_directories()
