"""Public page tests: templates render, routes resolve, schema is served."""

from __future__ import annotations

import json
from pathlib import Path
from urllib.robotparser import RobotFileParser

import pytest
from fastapi.testclient import TestClient

from app.config import Settings
from app.main import create_app


@pytest.fixture
def schema_file(tmp_path: Path) -> Path:
    path = tmp_path / "openapi.json"
    path.write_text(
        json.dumps(
            {
                "openapi": "3.1.0",
                "info": {"title": "BlitzID API", "version": "9.9.9"},
                "paths": {},
            }
        ),
        encoding="utf-8",
    )
    return path


@pytest.fixture
def client(schema_file: Path) -> TestClient:
    return TestClient(
        create_app(
            Settings(
                openapi_path=schema_file,
                environment="test",
                version="9.9.9",
                database_url="sqlite+pysqlite:///:memory:",
                session_secret="test-session-secret",
                app_base_url="http://testserver",
            )
        )
    )


def test_landing_renders(client: TestClient) -> None:
    response = client.get("/")
    assert response.status_code == 200
    assert "You keep the rules" in response.text
    assert "/static/passport-specimen.webp" in response.text
    assert 'href="/docs">Read the docs</a>' in response.text
    assert 'href="#pricing">Hosted plans from $5/month</a>' in response.text
    assert '<section class="retention-panel" aria-labelledby="retention-title">' in (
        response.text
    )
    assert "Self-hosted job lifecycle" in response.text
    assert 'id="faq"' in response.text
    assert '<ol class="retention">' in response.text
    assert (
        'href="mailto:sales@blitzid.io?subject=BlitzID%20Enterprise"' in response.text
    )
    footer = response.text.split('<footer class="site-footer">', 1)[1]
    assert "mailto:" not in footer
    assert 'href="/docs">Docs</a>' in footer
    assert 'href="/#pricing">Pricing</a>' in footer
    assert 'href="https://github.com/malvavisc0/blitzid#http-api"' not in response.text
    assert "{{" not in response.text


def test_docs_render_the_interactive_reference(client: TestClient) -> None:
    response = client.get("/docs")
    assert response.status_code == 200
    assert "/static/scalar.js" in response.text
    assert 'url: "/openapi.json"' in response.text
    assert 'showDeveloperTools: "never"' in response.text
    assert '<header class="site-header">' not in response.text
    assert 'class="skip-link"' not in response.text
    assert "{{" not in response.text


@pytest.mark.parametrize(
    ("path", "title"),
    [("/", "ID Document Extraction API | BlitzID"), ("/docs", "BlitzID API docs")],
)
def test_public_share_metadata(client: TestClient, path: str, title: str) -> None:
    head = " ".join(client.get(path).text.split("</head>", 1)[0].split())
    assert f"<title>{title}</title>" in head
    assert f'<meta property="og:title" content="{title}"' in head
    assert f'<link rel="canonical" href="https://blitzid.io{path}"' in head
    assert f'<meta property="og:url" content="https://blitzid.io{path}"' in head
    image = "https://blitzid.io/static/og.png"
    assert f'<meta property="og:image" content="{image}"' in head
    assert head.count('property="og:image:alt"') == 1
    assert "a specimen passport's document number is shown as a JSON field" in head
    if path == "/":
        description = (
            "Extract names, document numbers, and expiry dates from passports, "
            "ID cards, and driver's licenses as JSON. Use the hosted API or self-host."
        )
        assert f'name="description" content="{description}"' in head
        assert f'property="og:description" content="{description}"' in head


def test_share_image_is_a_large_png(client: TestClient) -> None:
    response = client.get("/static/og.png")
    assert response.status_code == 200
    assert response.headers["content-type"] == "image/png"
    assert response.content.startswith(b"\x89PNG\r\n\x1a\n")
    assert int.from_bytes(response.content[16:20], "big") == 1200
    assert int.from_bytes(response.content[20:24], "big") == 630


def test_reference_endpoint_is_removed(client: TestClient) -> None:
    assert client.get("/reference").status_code == 404


def test_login_preview(client: TestClient) -> None:
    response = client.get("/login")
    assert response.status_code == 200
    header = response.text.split("</header>", 1)[0]
    assert '<header class="site-header">' in header
    assert 'href="/" aria-label="BlitzID home"' in header
    assert '<nav class="nav-menu"' not in header
    assert 'href="/login"' not in header
    assert 'id="login-title"' in response.text
    assert 'href="https://blitzid.io/login"' in response.text
    assert "Sign-in isn't available on this deployment" in response.text
    assert "/auth/login/" not in response.text
    assert '<meta name="robots" content="noindex" />' in response.text
    assert "{{" not in response.text


def test_crawlers_can_read_the_login_noindex(client: TestClient) -> None:
    response = client.get("/robots.txt")
    assert response.status_code == 200
    robots = RobotFileParser()
    robots.parse(response.text.splitlines())
    assert robots.can_fetch("Googlebot", "https://blitzid.io/login")
    assert not robots.can_fetch("Googlebot", "https://blitzid.io/dashboard")
    assert not robots.can_fetch("Googlebot", "https://blitzid.io/billing")


def test_pricing_redirects_to_the_landing_section(client: TestClient) -> None:
    response = client.get("/pricing", follow_redirects=False)
    assert response.status_code == 307
    assert response.headers["location"] == "/#pricing"


def test_legal_pages_render(client: TestClient) -> None:
    for path, title in (("/privacy", "Privacy"), ("/terms", "Terms")):
        response = client.get(path)
        assert response.status_code == 200
        assert f"<h1>{title}</h1>" in response.text


def test_openapi_schema_is_served(client: TestClient) -> None:
    response = client.get("/openapi.json")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("application/json")
    assert response.json()["info"]["title"] == "BlitzID API"


def test_openapi_missing_is_a_clear_503(tmp_path: Path) -> None:
    client = TestClient(create_app(Settings(openapi_path=tmp_path / "missing.json")))
    response = client.get("/openapi.json")
    assert response.status_code == 503
    assert "export_openapi.py" in response.json()["detail"]


def test_healthz_reports_build_info(client: TestClient) -> None:
    response = client.get("/healthz")
    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "service": "web",
        "version": "9.9.9",
        "environment": "test",
    }


def test_unknown_path_gets_the_branded_404(client: TestClient) -> None:
    response = client.get("/missing")
    assert response.status_code == 404
    assert "Page not found" in response.text
    assert 'href="/docs"' in response.text


def test_security_headers(client: TestClient) -> None:
    response = client.get("/")
    assert response.headers["x-content-type-options"] == "nosniff"
    assert response.headers["x-frame-options"] == "DENY"
    assert response.headers["referrer-policy"] == "strict-origin-when-cross-origin"


@pytest.mark.parametrize("path", ["/", "/privacy", "/missing"])
def test_public_pages_share_the_signed_out_header(
    client: TestClient, path: str
) -> None:
    response = client.get(path)
    assert response.text.count('<header class="site-header">') == 1
    header = response.text.split("</header>", 1)[0]
    assert 'href="/docs"' not in header
    assert 'href="/#pricing">Pricing</a>' in header
    assert 'href="/#faq">FAQs</a>' in header
    assert 'href="https://github.com/malvavisc0/blitzid"' in header
    assert "<span>GitHub</span>" in header
    assert 'href="/login">Sign in</a>' in header
    assert 'href="/dashboard"' not in header
    assert "csrf_token" not in header
    assert "set-cookie" not in response.headers
    assert response.headers["cache-control"] == "no-store"


def test_favicon_redirects_to_svg(client: TestClient) -> None:
    response = client.get("/favicon.ico", follow_redirects=False)
    assert response.status_code == 308
    assert response.headers["location"] == "/static/favicon.svg"
