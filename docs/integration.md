# Composing verification flows with the BlitzID API

Blocks, not a pipeline: every endpoint stands alone, and each product
picks only what its signup needs. Two kinds of input go in and facts
come out: photos of a person's face and photos of their documents.
This guide lists the blocks, then recipes of increasing size, and it
is for the two people who need it: developers wiring a flow into a
product, and the admin running the service. Every command below runs
against a local stack.

## What the service does, and what stays yours

The service quality-checks document photos, reads the printed text and
the machine-readable code strip, validates the strip's check digits,
compares a selfie to a document portrait, estimates age group, gender,
and race per face, and cross-checks the copies inside one document.

It does not check liveness (the selfie may be a photo of a screen), it
does not do tamper forensics on card artwork, it does not join several
documents into one verdict, and it does not decide who is allowed in.
Those stay yours. The data for them comes back; the policy does not.

## For self-hosted admins

### Run the stack

```bash
# production shape (Coolify points here too): baked weights, no mounts
docker compose up

# local iteration: hot code, --reload on edits under ./src
docker compose -f docker-compose.dev.yml up

curl http://localhost:8000/health
```

The image carries every engine's weights, so cold start needs no
downloads and the container runs as a non-root user against a RAM-only
Redis. In Coolify, point the service at `docker-compose.yml` and set
the variables below in its environment tab.

### Configure it

| Variable | Required | Purpose |
|---|---|---|
| `BLITZID_LLM_BASE_URL` | for `structured` and `consistency` | your OpenAI-compatible chat endpoint |
| `BLITZID_LLM_API_KEY` | for `structured` and `consistency` | its key |
| `BLITZID_LLM_MODEL` | for `structured` and `consistency` | the model name it serves |
| `LANGFUSE_PUBLIC_KEY` / `LANGFUSE_SECRET_KEY` / `LANGFUSE_BASE_URL` | no | trace the LLM calls (spans carry OCR text, so point this somewhere you trust) |
| `BLITZID_API_MAX_UPLOAD_MB` | no (20) | upload cap per file |
| `BLITZID_API_JOB_TTL_SECONDS` | no (900) | how long results live |
| `BLITZID_API_MAX_QUEUED_JOBS` | no (50) | queue depth before `503` |
| `BLITZID_API_MAX_CONCURRENT_JOBS` | no (2) | worker threads |
| `BLITZID_API_JOB_LEASE_SECONDS` | no (120) | crash-recovery lease |
| `BLITZID_MODELS_DIR` | no (`/models`) | where the baked weights live |

The LLM endpoint sees the OCR text of every structured call. That is
names, birth dates, and document numbers, so choose an endpoint you
would trust with that.

### Check it is healthy

`GET /health` returns `200` with seven model flags: `face`, `verify`,
`attributes`, `antispoof`, `ocr`, `mrz`, `structured`. A `false` flag
means that analysis returns `400` with the missing piece named. `503`
means the job store is down and the container cannot accept work.

## The blocks

| Block | Call | Gives you |
|---|---|---|
| Photo quality gate | `POST /crop` | `pass`/`warn`/`reject` verdict, per-check detail, and a corrected crop. Works on documents and selfies alike |
| Data extraction | `POST /analyze -F types=ocr,mrz,structured` | printed fields, code-strip fields (check digits validated), typed records |
| Age signal | `POST /analyze -F types=attributes` | age band, gender, and race per face |
| Spoof signal | `POST /analyze -F types=antispoof` | per-face scores: live, printed photo, screen photo |
| Document self-check | `POST /analyze -F types=consistency` | printed text vs code strip vs portrait, per-field verdicts |
| Live selfie | `POST /liveness/challenge` + `POST /liveness/session` | one random action (`turn_head`, `smile`, `move_closer`) verified across three frames |
| Face match | `POST /verify` | similarity, verdict, and both compared crops as evidence |
| Health | `GET /health` | which engines are loaded |

## Recipes

**+18 age gate (one call).** A social site or content platform that
only needs to keep minors out:

```bash
curl -F image=@id.jpg -F types=structured localhost:8000/analyze
```

Read `date_of_birth` from the `structured` record (or `age` from
`types=attributes` when only a face is available) and apply your
policy, e.g. today minus birth date at least 18 years. No selfie, no
liveness, no MRZ needed. If the age comes from a face rather than a
document, run the liveness block first.

**Live selfie signup.** A product that accepts people but must see
them live: run the liveness block as the gate, then `types=antispoof`
as a second opinion (a `screen_score` or `paper_score` near 1.0 means
someone pointed a camera at a photo or display). Add `types=attributes`
for an age band and `POST /verify` when a previous photo exists.

**Full document onboarding.** Marketplace, mobility, finance: the
document-by-document path in the next section.

## Recipe: a full document onboarding

### 1. Capture the selfie first

It anchors the person, and the final match needs it. When the product
must see a live person, run the liveness block here: issue a
challenge, show its action in the live camera preview, and record
three frames while the user performs it. Liveness beyond that is your
app's job: refuse gallery uploads and capture from a live preview.

### 2. Capture the identity document

One document, both sides where the card has a back:

- **Passport**: one photo, the data page. The code strip is at the
  bottom of the same page.
- **ID card**: front and back. The back carries the code strip.
- **Driver license**: front and back. The back carries the barcode
  (see the gap list at the end).

Capture paths depend on what the user must prove. "Selfie, then ID
front and back" is the base. "Selfie, then passport" is the same flow
with one capture. If the license is a separate card from the ID, or
your use case involves a vehicle (car sharing, delivery, ride-hailing),
keep going document by document the same way: registration, insurance
certificate, a photo of the plate on the vehicle.

Capture rules to put in your camera UI: the whole card in frame, glare
off the holograms, a flat dark surface, no fingers over the edges.

### 3. Check each photo before you trust it

```bash
curl -F image=@id-front.jpg -F side=front localhost:8000/crop
```

The response carries `verdict` (`pass`, `warn`, `reject`) plus the
individual `checks`, and `crop_base64`: the perspective-corrected crop.
Analyze the crop rather than the raw photo. A `reject` means the card
was not found or is too small in frame.

A user whose photo just failed is mid-signup and ready to blame their
phone. Lead with the fix: "Retake the photo: the whole card must be in
frame and the glare off the window."

### 4. Read everything at once

```bash
curl -F image=@id-front-crop.jpg \
    -F types=face,ocr,mrz,structured,consistency \
    localhost:8000/analyze
# {"job_id": "<uuid>", "status": "queued", "status_url": "/jobs/<uuid>"}
```

Pick types per document:

| Document | Types |
|---|---|
| Passport data page | `face,ocr,mrz,structured,consistency` |
| ID card (back photographed) | `face,ocr,mrz,structured,consistency` |
| ID card (front only) | `face,ocr,structured` (no code strip to read) |
| Driver license | `face,ocr,structured,attributes` |
| Vehicle registration / insurance | `ocr,structured` |
| Plate photo | `structured` |

Analyses run as jobs so a slow LLM call never blocks your request
thread. Poll `GET /jobs/{job_id}` until `status` is `done`, then store
the body immediately: reading a result claims it, the next read gets
`410 Gone`, and results expire after `BLITZID_API_JOB_TTL_SECONDS`.
One job's sections share detection, OCR, and the LLM, so asking for
all six types costs about one of each.

The `structured` section returns typed fields. Its prompts cover
identity documents, code strips, and plates; other layouts (vehicle
papers, insurance certificates) come out as generic field lists, so
expect to map those names once.

### 5. Cross-check within each document

The `consistency` section returns two blocks. `comparisons` holds one
row per field shared by the printed text and the code strip:
`document_number`, `date_of_birth`, `date_of_expiry`, `surname`,
`given_names`, `sex`. `photo_comparisons` holds two rows: `age` (the
birth date's exact age against the portrait's age band) and `sex` (the
M/F/X marker against the predicted gender). Every row keeps both
values and the model's confidence.

`verdict` is `match`, `partial`, `mismatch`, or `unavailable`. Names
compare word by word and ignore order, so an extra surname word comes
back `partial`. The age row allows one band of slack because the model
is coarse and portraits age at issue time. An `X` sex marker comes
back `partial`: a binary model cannot confirm it. Race is never
cross-checked; documents do not carry it.

Treat the code strip as the trustworthy copy. It is machine-written
and check-digit validated, while everything else has been through the
camera and a model.

### 6. Match the selfie to the document portrait

```bash
curl -F image1=@id-front-crop.jpg -F image2=@selfie.jpg \
    -F face1_bbox=49,91,84,116 localhost:8000/verify
```

Pin the portrait with `face1_bbox` from the `face` section. ID cards
and some licenses carry a faint second portrait (the ghost image), and
an unpinned comparison may pick it. Match against the identity
document's portrait first; a second document's portrait is a useful
second opinion.

The response is the verdict plus the evidence: `face1` and `face2`
with their `bbox`, `confidence`, a `crop_base64` thumbnail, and
`alternatives` listing the candidates not picked. Keep both crops with
the application. When a user disputes a decision, those two thumbnails
are what actually happened.

### 7. Join multiple documents yourself

The service reads one document per job. When the flow collects a set
(ID, license, vehicle papers, plate), the joins live in your backend
today, and they are where multi-document fraud actually hides: the
name must be the same person across documents, the plate photo must
match the plate on the registration, dates must agree. All those
values come back normalized enough to compare directly.

## When things work

| Status | What happened | What to do next |
|---|---|---|
| `202` from `/analyze` | The job is queued | Poll `GET /jobs/{id}`; keep the id |
| `200` from `/crop`, `verdict: pass` | Card found, sharp, bright | Keep the crop; analyze the crop |
| `200` from `/crop`, `verdict: warn` | Usable, one weak check | Continue and lower your confidence in the data |
| Job `status: done` | One section per requested type | Store the body now; the next read is `410 Gone` |
| `consistent: true` | Printed text, code strip, and portrait agree | Continue to the selfie match |
| `200` from `/verify`, `verified: true` | Same person at your threshold | Accept; keep both crops as evidence |
| `similarity` near your accept/reject edges | Borderline pair | Human review with both crops on screen |

## When things fail

| Status or signal | What happened | What to do | What to show the user |
|---|---|---|---|
| `422` | Bad request: unknown type, malformed `face*_bbox`, threshold out of range, or no face where one is needed | Fix the request, or ask for a retake | The retake rule in the first words |
| `400` | Undecodable bytes, or that engine is missing | Check the file; check `/health` | "We could not read that photo, please try again" |
| `413` | Over `BLITZID_API_MAX_UPLOAD_MB` | Downscale before upload | "That photo is too large" |
| `415` | PDF upload | Ask for a photo | "Take a photo instead of a file" |
| `410` / `404` | Result claimed, expired, or unknown id | Store results at first read | (internal) |
| `503` + `Retry-After` | Queue full or Redis down | Retry with backoff | "One moment, please try again" |
| Document row `mismatch` | A misread character or an altered card | Retake, or human review with both values | "Let's double-check this document together" |
| Photo row `mismatch` | The photo does not look like the document's person, or the model is wrong about a tricky portrait (artwork, heavy filters) | Human review. Never auto-reject on this alone | "Please take a fresh selfie in good light" |
| Any row `partial` | Near-match: multi-part names, age inside the slack, an `X` sex marker | Your call; usually continue | (nothing) |

## Deciding with thresholds

`verified` is `similarity >= threshold`. The service default is `0.4`;
treat it as a starting point. Set three bands from your own traffic:
accept above some score, reject below some score, and send the middle
to a human. Two data points to calibrate against: the same photo twice
scores close to `1.0`, two different people scored `0.05`. Where real
selfie-versus-document pairs land depends on your users' phones.

You can pass `threshold` per request to trial a policy without
redeploying. Return `similarity`, not just `verified`, to your support
team: a user arguing "that is me in both photos" needs the score and
the two crops, not a boolean.

## What is not covered yet

Read this before you promise a flow:

- **Barcode (PDF417/AAMVA) on US and Canadian licenses.** The back of
  those cards carries a machine copy with its own checksums, better
  than any OCR. Today the `ocr` and `structured` passes read the
  printed front only.
- **Video replay and generated faces.** The liveness block stops
  static spoofs and the `antispoof` block flags cameras pointed at
  photos and displays. Filming a screen that plays a live video, or a
  real-time deepfake, is beyond what these catch: that last mile is
  certified presentation-attack detection (ISO 30107-3 SDKs such as
  iProov or FaceTec).
- **Cross-document matching.** Step 7 is yours today.
- **Eligibility policy.** Age minimums, validity windows, expiry
  cutoffs: compute them from the dates returned. The service reports;
  it does not judge.
- **Document kinds beyond IDs, code strips, and plates.** The
  structured reader extracts fields from any document text, but its
  prompts are tuned for those three. Expect generic field names
  elsewhere.
- **NFC chip (ePassport).** Needs native mobile code. The photo of the
  data page is what you get.

## Data and privacy

Uploads and results live in RAM end to end: jobs are raw image bytes in
a Redis with persistence disabled, results are claim-once and
TTL-bounded, and the service writes nothing to disk. Two boundaries to
choose deliberately: the `structured` and `consistency` analyses send
OCR text to the endpoint named in `BLITZID_LLM_BASE_URL`, and tracing
exports spans (including that text) to Langfuse when the `LANGFUSE_*`
keys are set. Point both at infrastructure you trust with document
data. The face attributes include race: keep them out of eligibility
and risk decisions (they are a protected characteristic in most
jurisdictions), and store them as display metadata at most.
