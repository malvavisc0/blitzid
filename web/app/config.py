"""BlitzID public website and hosted dashboard configuration."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent
STATIC_DIR = APP_DIR / "static"
TEMPLATES_DIR = APP_DIR / "templates"


@dataclass(frozen=True)
class Settings:
    """Resolved web configuration (deployment environment variables)."""

    openapi_path: Path = STATIC_DIR / "openapi.json"
    environment: str = "development"
    version: str = "0.0.0"
    database_url: str = "sqlite+pysqlite:///./web.db"
    session_secret: str = ""
    app_base_url: str = "http://localhost:18081"
    redis_url: str = "redis://localhost:6379/0"
    google_client_id: str = ""
    google_client_secret: str = ""
    github_client_id: str = ""
    github_client_secret: str = ""
    stripe_secret_key: str = ""
    stripe_webhook_secret: str = ""
    stripe_price_starter: str = ""
    stripe_price_pro: str = ""
    local_billing: bool = False
    secure_cookies: bool = False

    @classmethod
    def from_env(cls) -> Settings:
        """Read the deployment environment with development defaults."""
        app_base_url = os.environ.get("APP_BASE_URL", "http://localhost:18081")
        environment = os.environ.get(
            "ENVIRONMENT", os.environ.get("BLITZID_WEB_ENVIRONMENT", "development")
        )
        return cls(
            openapi_path=Path(
                os.environ.get(
                    "BLITZID_WEB_OPENAPI_PATH", str(STATIC_DIR / "openapi.json")
                )
            ),
            environment=environment,
            version=os.environ.get(
                "BLITZID_WEB_VERSION", os.environ.get("ENVIRONMENT_VERSION", "0.0.0")
            ),
            database_url=os.environ.get("DATABASE_URL", "sqlite+pysqlite:///./web.db"),
            session_secret=os.environ.get("SESSION_SECRET", ""),
            app_base_url=app_base_url.rstrip("/"),
            redis_url=os.environ.get("REDIS_URL", "redis://localhost:6379/0"),
            google_client_id=os.environ.get("GOOGLE_CLIENT_ID", ""),
            google_client_secret=os.environ.get("GOOGLE_CLIENT_SECRET", ""),
            github_client_id=os.environ.get("GITHUB_CLIENT_ID", ""),
            github_client_secret=os.environ.get("GITHUB_CLIENT_SECRET", ""),
            stripe_secret_key=os.environ.get("STRIPE_SECRET_KEY", ""),
            stripe_webhook_secret=os.environ.get("STRIPE_WEBHOOK_SECRET", ""),
            stripe_price_starter=os.environ.get("STRIPE_PRICE_STARTER_MONTHLY", ""),
            stripe_price_pro=os.environ.get("STRIPE_PRICE_PRO_MONTHLY", ""),
            local_billing=os.environ.get("BLITZID_WEB_LOCAL_BILLING", "")
            .strip()
            .lower()
            in {"1", "true", "yes", "on"},
            secure_cookies=app_base_url.startswith("https://")
            or environment == "production",
        )
