# VPS recommendations for the MVP deployment

Source plans: [ApexNodes Virtual Servers (promo)](https://apexnodes.xyz/store/vds-promo),
[Virtual Servers (HI-CPU)](https://apexnodes.xyz/store/vds-hi-cpu),
and [High-Bandwidth](https://apexnodes.xyz/store/high-bandwidth).
Prices are the listed USD monthly rates as of 2026-10.

## Client profile (first client: Uber-like rideshare)

Two onboarding personas with very different weights:

- **Drivers (heavy):** license/ID front (crop + face + OCR/MRZ,
  ~2.3 s) + document back (crop + MRZ, ~1.2 s) + portrait-vs-selfie
  verify (~0.25 s) + liveness (~0.1 s) ≈ **~4 s CPU per driver**.
  Real-world resubmissions (blur, glare, cut-off cards) run 1.5–2x
  attempts: **~6–8 s effective per approved driver**.
- **Passengers (light):** selfie + liveness ≈ **~0.4 s CPU** — even
  ~10k passengers/hour barely registers against document flows.
- **Traffic shape:** bursty, not smooth — city launches and fleet
  batches dump hundreds of drivers at once, then long quiet periods.

Client-facing consequences:

1. **Fraud is the threat model.** Document forgery and face-swap
   attacks concentrate on driver onboarding (that is where the payout
   account is created). Run the full suite for drivers — `mrz`,
   `consistency`, `antispoof`, and the `/liveness` challenge — and
   keep passenger checks light. Skip `types=ocr` when `mrz` suffices:
   MRZ runs its own OCR pass internally, so requesting both doubles
   the cost.
2. **DDoS protection is required, not optional** — a KYC outage
   blocks driver earnings for the whole client. Either take
   VDS-HI-CPU-DE-3 (includes L3-4 Anti-DDoS) or put Cloudflare in
   front of the reverse proxy on the cheaper promo plan.
3. **Queue tuned for batch traffic:** raise
   `BLITZID_API_MAX_QUEUED_JOBS` to ~100 and cut
   `BLITZID_API_MAX_UPLOAD_MB` to 10 (phone photos are 2–8 MB; 20 MB
   is wasted headroom). Worst-case Redis memory stays ~1 GB,
   typical ~300 MB.

## Requirements derived from the codebase

- **RAM:** all engines (SCRFD, ArcFace, FairFace, MiniFASNet, RapidOCR)
  load at service startup and stay resident — roughly 1.5–2 GB for the
  API process alone, plus Redis, the OS, and Docker overhead. 4 GB is
  the floor; 8 GB is comfortable.
- **CPU:** inference is CPU-bound on onnxruntime and mostly
  single-image, so per-core speed matters more than core count. 2 vCPU
  is the minimum; 4 covers `BLITZID_API_MAX_CONCURRENT_JOBS=2` plus
  queue/sweeper threads. An OCR pass takes ~1 s per image on a laptop
  i7; older/slower cores stretch that.
- **Disk:** the Docker image bakes every engine's weights (~1 GB with
  layers); any listed plan's NVMe is sufficient.
- **External LLM:** structured extraction and consistency checks call
  an OpenAI-compatible endpoint via `BLITZID_LLM_*` — host that
  elsewhere (e.g. a separate vLLM box or a managed API). Do not plan to
  co-locate it with the API on these plans.

## Top 5

| # | Plan | Specs | Price/mo | Why |
|----|------|-------|----------|-----|
| 1 | VDS-PROMO-DE-3 | 4 vCPU i9-9900K, 8 GB, 128 GB | $4.05 | Best value: 8 GB gives headroom over the ~2 GB engine footprint; 4 cores cover 2 concurrent jobs plus OCR's ~1 s passes. FastAPI, Redis, and baked weights all fit without swapping. |
| 2 | VDS-PROMO-DE-2 | 2 vCPU i9-9900K, 4 GB, 64 GB | $2.37 | Minimum viable: runs the full stack (engines ~2 GB + Redis + OS) with little slack. Fine for pilot traffic; expect to upgrade when concurrency matters. |
| 3 | VDS-HI-CPU-DE-3 | 4 vCPU i9-12900K, 8 GB, 100 GB | $8.99 | Same shape as #1 but ~25% faster per core (OCR/MRZ latency drops accordingly) and includes L3-4 DDoS protection — relevant for a public KYC endpoint. Costs ~2x. |
| 4 | VDS-PROMO-DE-4 | 8 vCPU i9-9900K, 16 GB, 256 GB | $7.39 | Only if you raise `BLITZID_API_MAX_CONCURRENT_JOBS` and expect real parallel load. Cheaper than HI-CPU-DE-3 with more RAM/disk, but the 9900K is the older, slower core. |
| 5 | VDS-HI-CPU-DE-2 | 2 vCPU i9-12900K, 4 GB, 60 GB | $4.99 | The "snappy small box": fastest single core, good latency per job, but 4 GB leaves no room to grow and it costs more than the 4-core promo. |

## Not recommended for this stack

- **1 vCPU / 2 GB tiers** (DE-1 both lines, PL-1): the resident engines
  alone would swap; the OCR pass would block the sync endpoints.
- **PL (Ryzen 7700, DDR5) tiers**: solid hardware, but priced above
  the Frankfurt equivalents with no Anti-DDoS line listed.
- **High-bandwidth line** ([`/store/high-bandwidth`](https://apexnodes.xyz/store/high-bandwidth),
  EPYC 7763, Amsterdam): 10 Gbit/s is a non-asset for this stack — a
  full onboarding flow uploads at most 20 MB and returns a few
  hundred KB of JSON, so even 50 flows/minute is ~1–2 Mbit/s
  sustained and the 1 Gbit on the other lines is already overkill.
  Meanwhile the EPYC 7763 is a 64-core server part at 3.5 GHz with
  single-core performance roughly 30–40% below the i9-12900K — and
  our inference latency is per-core-bound. The closest match,
  VDS-EPYC-NL-2 (4 vCPU, 8 GB, $9.00), costs the same as
  VDS-HI-CPU-DE-3 while being slower at it. Only consider this line
  if we later serve heavy media (video liveness streams) from the
  same box.

## Capacity: how many users can the recommended box serve

Numbers assume VDS-PROMO-DE-3 (4 vCPU i9-9900K), benchmark timings
from `scripts/benchmark.py` (the 9900K performs like the benchmark's
i7-11850H), and default queue settings.

### Hard caps from the code

- Async `/analyze`: `BLITZID_API_MAX_CONCURRENT_JOBS=2` processing,
  `BLITZID_API_MAX_QUEUED_JOBS=50` waiting — 52 jobs in the system;
  the 53rd submission gets `503` + `Retry-After`.
- Sync `/crop`, `/verify`, `/liveness`: no explicit cap, but face
  detections serialize on the shared detector lock, so they queue
  rather than parallelize.

### Per-operation cost and sustained throughput

| Operation | Server time | Sustained rate (2 workers) |
|-----------|-------------|----------------------------|
| `/crop` | ~50 ms | sync, ~4–10/s |
| `/verify` | ~200–250 ms | sync, ~4–6/s |
| `/liveness` (3 frames) | ~100 ms | sync, ~8–15/s |
| `/analyze face` only | ~40 ms | ~25–40 jobs/s |
| `/analyze face+ocr+mrz` | ~2.2 s (two OCR passes) | ~1–2 jobs/s |

### Full onboarding flow

A complete KYC flow (crop → analyze face+ocr+mrz → verify →
liveness) costs **~2.5–3 s of CPU** per user. On 4 cores that is:

- **Steady state: ~1 flow/second ≈ 60–90 flows/minute
  (~4,000–5,000/hour)** — the theoretical ceiling is ~1.3/s, ~1/s
  realistic since one OCR job only saturates 1–2 cores.
- **Live users mid-flow**: a user takes 30–60 s of *human* time for
  ~3 s of *server* time, so several hundred users can be
  mid-onboarding simultaneously without degradation.
- **Bursts**: the 50-slot queue absorbs ~25–50 simultaneous
  submissions; the 503 + `Retry-After` backpressure protects the rest.

### Sizing rule of thumb

- 4 vCPU ≈ **one completed flow per second, sustained**.
- Comfortable up to ~50k users/month with peak-hour demand under
  ~2,000 flows/hour. Most MVPs run at hundreds of flows/day — the
  CPU sits ~99% idle at that level, and **8 vCPU is not needed**:
  extra cores do not speed up a single flow (OCR uses 1–2 cores);
  they only help once sustained traffic exceeds ~1 flow/s, and the
  queue already converts spikes into graceful waiting.
- Upgrade signal: sustained 503s or minutes-long queue growth —
  then move to the 8 vCPU plan and raise
  `BLITZID_API_MAX_CONCURRENT_JOBS`, or add a second identical box
  behind a load balancer (the API is stateless).

## Concurrency: MAX_CONCURRENT_JOBS vs replicas

We deploy via Docker through Coolify. On a single host, use
`BLITZID_API_MAX_CONCURRENT_JOBS`, not compose/Coolify replicas:

- **Threads are cheap.** Workers are threads inside one process
  sharing one resident engine set (~2 GB). On 4 vCPU, set
  `BLITZID_API_MAX_CONCURRENT_JOBS=3` — each OCR job only uses 1–2
  cores, so the default of 2 leaves cores idle. One env var, no
  extra RAM, no redeploy.
- **Replicas are expensive on one box.** Each replica is a separate
  process that loads its own ~2 GB of weights: 2 replicas ≈ 4–5 GB
  of engines alone, which an 8 GB box can barely take.
- **If you do scale out with replicas** (multi-node, or
  zero-downtime deploys): the Redis queue is claim-once with
  lease/heartbeat, so N replicas safely pull from one shared queue —
  but all replicas must point at the **same Redis**
  (`BLITZID_API_REDIS_URL`); do not let each replica spin its own.

**Decision for the MVP: single replica on VDS-HI-CPU-DE-4 with
`BLITZID_API_MAX_CONCURRENT_JOBS=5`.** Revisit replicas when we add
a second node.

## Final sizing decision

**VDS-HI-CPU-DE-4 ($13.49/mo, 6 vCPU i9-12900K, 12 GB RAM,
L3-4 Anti-DDoS, Frankfurt).**

Why one tier above the 4 vCPU options is the safe call:

- With a paying rideshare client from day one, launch-week bursts
  (fleet batches, city launches) are the known traffic shape, and
  doubling headroom costs $4.50/mo — cheap insurance against the
  sizing being wrong.
- 12 GB RAM removes memory as a constraint: engines (~2 GB) +
  worst-case Redis (~1 GB at the tuned queue settings) + Docker/OS
  leave ~8 GB of slack.
- Capacity: ~1.5–2 full flows/second ≈ **~750–1,200 approved
  drivers/hour** including resubmissions; a 500-driver city launch
  drains in ~15–25 minutes of queue time. Passengers remain
  negligible load.
- Caveat: "6 vCPU" on a 12900K host mixes P-core and E-core threads,
  so per-worker latencies are uneven — normal, not a defect. Watch
  queue depth in `/health`, not individual job times.

Settings for this box:

| Knob | Value | Why |
|------|-------|-----|
| `BLITZID_API_MAX_CONCURRENT_JOBS` | 5 | 6 cores minus headroom for sync endpoints and the sweeper |
| `BLITZID_API_MAX_QUEUED_JOBS` | 100 | absorb fleet batches without 503s |
| `BLITZID_API_MAX_UPLOAD_MB` | 10 | phone photos are 2–8 MB; halves worst-case Redis memory |

Downgrade path if the box sits idle for a quarter: VDS-HI-CPU-DE-3
($8.99) with `MAX_CONCURRENT_JOBS=3`. Upgrade path if `/health`
shows sustained queue depth >20 or 503s during batches: a second
DE-4 behind a load balancer (the API is stateless; replicas must
share one Redis).

## Deployment checklist

- Provisioned box: **VDS-HI-CPU-DE-4 ($13.49/mo)** — see "Final
  sizing decision" above for the settings table.
- Coolify: point the service at `docker-compose.yml` and set
  `BLITZID_LLM_BASE_URL` / `BLITZID_LLM_API_KEY` / `BLITZID_LLM_MODEL`
  in the environment tab (required by the `structured` and
  `consistency` engines; `/health` reports them).
- Put a reverse proxy (Caddy or nginx) in front for TLS and basic
  rate limiting — the API ships without authentication or request
  throttling (`docs/integration.md` assumes a gateway in front).
- Use the compose healthcheck and `/health` for orchestration, and
  keep `BLITZID_API_MAX_UPLOAD_MB` at its 20 MB default unless there
  is a reason — it bounds worst-case Redis memory.
- Do not request `types=ocr` when `mrz` suffices: MRZ runs its own
  OCR pass internally, so asking for both doubles the cost.
