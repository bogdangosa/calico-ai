import os

from dotenv import load_dotenv
from dynaconf import Dynaconf
from dynaconf.utils.boxing import DynaBox

from loguru import logger

from src.services.exceptions import ConfigurationError

load_dotenv()

profile = os.getenv("DYNACONF_APP_PROFILE")

if profile is None:
    raise ConfigurationError("Project env not set")

settings = Dynaconf(
    envvar_prefix="DYNACONF",
    settings_files=[
        "config/config.yaml",
        "config/input_case_schemas.yaml",
        f"config/config.{profile}.yaml",
        f"config/secrets.{profile}.yaml",
        f"config/.secrets.{profile}.yaml",
    ],
    load_dotenv=True,
)

def fetch_from_env(user: str) -> str:
    # Try the requested name first, then fall back to POSTGRES_PASSWORD if appropriate
    value = os.getenv(user.upper())
    if value is None and user.lower() == "database_password":
        value = os.getenv("POSTGRES_PASSWORD")
        
    if value is None:
        logger.warning(f"Environment variable '{user}' not set")
        raise ConfigurationError(f"Missing env variable: {user}")
    return value

if __name__ == "__main__":
    print(f"config/config.{profile}.yaml")
    print(DynaBox(settings.as_dict()).to_yaml())
    print("Loaded files:", settings._loaded_files)
