# BlitzID website and hosted dashboard

The public site and the paid hosted product:

- Landing, login, legal pages, and one `/docs` page with the quickstart and the
  interactive Scalar reference.
- Google/GitHub OAuth, personal organizations, and signed sessions.
- API keys (create once, rotate, revoke) with plan limits and audit rows.
- Stripe Checkout, plan changes, Customer Portal, webhooks, and entitlements.
- Usage for the current month, read from `usage_daily`.

Self-hosting the API does not require any of this: leave
`BLITZID_API_AUTH_REQUIRED=0` and run `docker compose up -d --no-deps api redis`.

```
web/
  alembic.ini
  migrations/          # schema owned by the web app
  app/
    main.py            # app, sessions, middleware, routes, health
    config.py          # environment settings
    db.py models.py    # SQLAlchemy/Postgres
    auth.py            # OAuth providers, sessions, CSRF, provisioning
    keys.py plans.py   # key format/hash/status, plan limits
    billing.py         # Stripe adapter, entitlements, event handling
    cache.py           # Redis key-cache invalidation shared with the API
    audit.py
    routes/            # pages, docs, auth, dashboard, billing, webhooks
    templates/         # public + dashboard/key templates
    static/            # app.css, app.js, fonts, images, baked docs assets
  tests/
  Dockerfile
```

`app/static/passport-specimen.webp` is a web-sized copy of
`images/td3_passport_specimen.jpg` — an ICAO specimen data page with
synthetic content (no real person), attributed in `images/README.md`.

## Local development

The docs pages need two generated files in `app/static/`: the OpenAPI schema
(from the API source) and the Scalar bundle. Neither is committed.

```bash
# from the repo root
uv run python scripts/export_openapi.py web/app/static/openapi.json
curl -L -o web/app/static/scalar.js \
  https://cdn.jsdelivr.net/npm/@scalar/api-reference@1.83.0/dist/browser/standalone.js

cd web
uv sync
uv run alembic upgrade head          # SQLite by default: ./web.db
uv run uvicorn app.main:app --reload --port 18081
```

Open <http://localhost:18081/>, `/docs`, and `/login`.

Without OAuth credentials the login page stays in preview mode. To sign in
locally, create Google/GitHub OAuth apps with the redirect URI
`http://localhost:18081/auth/callback/{provider}` and set the four
`GOOGLE_*`/`GITHUB_*` variables. For real billing, set the Stripe variables and
forward webhooks with
`stripe listen --forward-to localhost:18081/webhooks/stripe`.

With the local compose overlay (`docker-compose.local.yml`, which sets
`ENVIRONMENT=development` and `BLITZID_WEB_LOCAL_BILLING=1`), accounts without a Stripe customer can choose
and switch between Starter and Pro locally without a charge. They can also
cancel the preview plan immediately without deleting their account or keys.
These shortcuts are disabled in production and whenever
`BLITZID_WEB_LOCAL_BILLING` is unset.

In Stripe's default Customer Portal configuration, enable **Switch plan** and
add both configured monthly prices. Configure proration and effective timing
there. **Change plan** opens Stripe to confirm the update to the existing
subscription. Downgrades keep existing keys and current-month usage; the lower
limits apply once the plan change takes effect. **Manage billing** opens the
portal for payment methods and invoices.

Enable **Cancel subscriptions** in the default Stripe Customer Portal and
select **At the end of the billing period**. **Cancel plan** opens Stripe's
confirmation for the existing subscription. The `customer.subscription.updated`
and `customer.subscription.deleted` webhooks sync the scheduled end date and
final status. API access continues until the scheduled end date, then stops;
login and invoice history remain available. Run `alembic upgrade head` against
the shared database before updating the hosted API so it can enforce scheduled
cancellation dates.

Stripe Checkout collects the billing address, optional company name, and
optional VAT/tax ID where supported. Stripe generates subscription invoices.
Keep **Invoice history** visible in the portal and let customers update their
name, billing address, and tax ID for future invoices. Before taking payments,
configure your seller details, invoice tax IDs, and applicable Stripe Tax
registrations, and verify a business invoice in a Stripe sandbox. Keep purchase
orders and payment terms sales-assisted through Stripe Invoicing; paying a
standalone invoice does not automatically activate a BlitzID subscription.

Both docs assets are baked into the Docker image at build time, so the deployed
schema always matches the API release in the same build context.

## Settings

| Variable | Default | Purpose |
| --- | --- | --- |
| `DATABASE_URL` | `sqlite+pysqlite:///./web.db` | Postgres in production (`postgresql+psycopg://…`) |
| `SESSION_SECRET` | random per process | Signs the session cookie; required in production |
| `APP_BASE_URL` | `http://localhost:18081` | OAuth redirect base and Stripe return URLs |
| `REDIS_URL` | `redis://localhost:6379/0` | API-key cache invalidation |
| `GOOGLE_CLIENT_ID` / `GOOGLE_CLIENT_SECRET` | empty | Enables Google OIDC |
| `GITHUB_CLIENT_ID` / `GITHUB_CLIENT_SECRET` | empty | Enables GitHub OAuth |
| `STRIPE_SECRET_KEY` | empty | Enables checkout/portal |
| `STRIPE_WEBHOOK_SECRET` | empty | Verifies `/webhooks/stripe` |
| `STRIPE_PRICE_STARTER_MONTHLY` | empty | $5/month price ID |
| `STRIPE_PRICE_PRO_MONTHLY` | empty | $19/month price ID |
| `ENVIRONMENT` | `production` in compose | `production` forces secure cookies; `docker-compose.local.yml` sets `development` |
| `BLITZID_WEB_LOCAL_BILLING` | off | Enables no-charge plan activation in development |
| `BLITZID_WEB_OPENAPI_PATH` | `app/static/openapi.json` | Bundled schema path |
| `BLITZID_WEB_VERSION` | `0.0.0` | Reported by `/healthz` |

Hosted API enforcement (API service):

```dotenv
BLITZID_SAAS_DATABASE_URL=postgresql://...
BLITZID_API_AUTH_REQUIRED=1
```

## Docker

```bash
# from the repo root — local development overlay
echo "POSTGRES_PASSWORD=change-me" >> .env    # untracked; never commit
docker compose -f docker-compose.yml -f docker-compose.local.yml \
  up -d postgres web
# http://localhost:8001 — the migrate service applies migrations first

# production shape (Coolify uses this file alone)
echo "SESSION_SECRET=$(openssl rand -hex 32)" >> .env
echo "APP_BASE_URL=https://blitzid.io" >> .env
docker compose -f docker-compose.yml up -d postgres web
```

Set `SESSION_SECRET` for anything you sign into. Production refuses to
start without it; development falls back to a per-process random key.
The local overlay enables no-charge preview billing; the base file is
production-only.

The Dockerfile has two stages: a schema stage that installs the API project
and runs `scripts/export_openapi.py` (no model weights or Redis needed), and a
runtime stage that installs only the web dependencies, downloads the pinned
Scalar bundle, and runs `alembic upgrade head` before Uvicorn.

## Deploy (Coolify)

On a `v*` tag, `.github/workflows/docker.yml` builds and pushes
`ghcr.io/malvavisc0/blitzid` and `ghcr.io/malvavisc0/blitzid-web`. To trigger
a Coolify deployment after the push, set repository variable
`COOLIFY_DEPLOY_ENABLED=true`, secrets `COOLIFY_WEBHOOK` (the full Coolify
deploy hook URL, including its UUID) and `COOLIFY_TOKEN`, and optionally
`PUBLIC_WEB_URL` if the site is not `https://blitzid.io`.

Use `docker-compose.coolify.yml` for the Coolify resource — a pull-only
file that joins the external `coolify` network, publishes no host ports, and
routes `BLITZID_WEB_DOMAIN` to `web:8080` and `BLITZID_API_DOMAIN` to
`api:8000` through Coolify's Caddy labels. Enable **Raw Compose deployment**
so Coolify runs the file verbatim. Do not use `docker-compose.yml` (loopback
host ports), `docker-compose.local.yml` (development mode plus no-charge
preview billing), or `docker-compose.dev.yml`.

Set `POSTGRES_PASSWORD`, `SESSION_SECRET`, `APP_BASE_URL`,
`BLITZID_WEB_DOMAIN`, `BLITZID_API_DOMAIN`, and `BLITZID_IMAGE_TAG` to the
release tag (the web image tag defaults to it). Keep
`BLITZID_API_AUTH_REQUIRED=1` for the hosted API and add the OAuth, Stripe,
and optional LLM/Langfuse variables. Make both GHCR packages public or run
`docker login ghcr.io` on the server.

The file is capped for an 8 vCPU / 16 GB server: API 8g, Redis 2.5g,
Postgres 2g, web 768m; add 4–8 GB swap because model loading peaks before
steady state. API concurrency defaults to 4; lower
`BLITZID_API_MAX_CONCURRENT_JOBS` to 2 if jobs are latency-sensitive.

## Checks

```bash
cd web
uv run ruff check app tests
uv run ruff format --check app tests
uv run mypy app
DATABASE_URL=sqlite+pysqlite:///./ci.db uv run alembic upgrade head
DATABASE_URL=sqlite+pysqlite:///./ci.db uv run alembic check
uv run pytest
```

## Routing

Route `blitzid.io` to the `web` service (`web:8080`, or `127.0.0.1:8001` for a
proxy on the Docker host). Keep `api.blitzid.io` routed to the separate `api`
service. Set `ENVIRONMENT=production` and `APP_BASE_URL=https://blitzid.io` on
the web service. Register `https://blitzid.io/auth/callback/google` and
`https://blitzid.io/auth/callback/github` in the respective OAuth apps, and use
`https://blitzid.io/webhooks/stripe` for the web's Stripe endpoint. Set Stripe's
default Customer Portal return link to `https://blitzid.io/dashboard` as well.

The session cookie is host-only. HTML responses are `no-store` because the
shared header shows the session's navigation and logout form; the one-time key
page is never cached. Keep `/static/` and `/openapi.json` cacheable as usual.
The web container serves the schema locally, so docs don't need to reach the
API at runtime.
