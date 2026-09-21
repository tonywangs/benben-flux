# Benben’s World 🐾

Create new scenes and artwork from real photos of **Benben the Maltese**, using FLUX.1-Kontext-dev on Modal. The private website now uses reference-image editing. The earlier FLUX.1-dev LoRA experiments remain available for research, but did not pass the likeness tests and are not the website's default.

## Use the website

Open [Benben’s World](https://jordan-tony-collab--benben-flux-web.modal.run), sign in with `benben-web-auth`, choose Portrait, Standing, or Smiling, and describe a change. The selected reference is displayed next to the result. The application includes instructions to preserve Benben's facial features and ponytail.

Garden and watercolor edits showed a clear visual improvement over the LoRA samples. Exact likeness is not guaranteed; large changes to pose, clothing, or headwear can still alter his appearance. Start with a reference close to the pose you want. The first request after inactivity takes longer while the model loads.

## Workspace and credentials

All project compute uses **jordan-tony-collab**, environment **main**. Explicit profile flags keep other projects' defaults unchanged.

```sh
uv sync --locked --dev
uv run modal token new --profile jordan-tony-collab --no-activate
```

In [Hugging Face](https://huggingface.co/black-forest-labs/FLUX.1-Kontext-dev), obtain access to FLUX.1-Kontext-dev and FLUX.1-dev. In [Modal Secrets](https://modal.com/secrets), under this workspace and environment, configure:

- `huggingface-secret`: `HF_TOKEN` with read access to the models.
- `benben-web-auth`: `USERNAME` and `PASSWORD` for the website.

Both are already configured in the current workspace. Tokens and photos do not belong in Git or chat.

## Deploy the reference editor

The current `benben-flux` Volume contains the shared base model, Kontext transformer, training artifacts, and reference photos. Kontext reuses the base text encoders and VAE, and does not load either experimental LoRA.

```sh
uv run modal run --profile jordan-tony-collab --env main modal_app.py::download_model
uv run modal run --profile jordan-tony-collab --env main kontext_app.py::download
uv run modal deploy --profile jordan-tony-collab --env main kontext_app.py
uv run modal deploy --profile jordan-tony-collab --env main modal_app.py
```

The `benben-kontext` deployment provides the GPU editor and private reference retrieval. The `benben-flux` deployment provides the authenticated Gradio website. Both must be deployed for the site to work. Reference mappings live in `benben.py`; they currently point to the curated photos stored under `/runs/benben-v2/photos/`. Retain that dataset even though its LoRA was rejected. Moving to a fresh workspace also requires transferring these photos or updating the mappings to an uploaded dataset.

To reproduce the three initial reference edits (this uses GPU inference):

```sh
uv run modal run --profile jordan-tony-collab --env main kontext_app.py::test
```

Results and generation settings are written to `outputs/kontext/`, which is excluded from Git. The original astronaut instruction added a helmet despite asking for a visible ponytail; a more explicit live test kept the ponytail visible but still added unwanted headwear. Astronaut clothing is therefore not a recommended example.

## Experimental LoRA training

Read the [quality plan and observed results](docs/quality-plan.md) before spending credits on another run. A completed training job is not evidence of subject likeness.

The training implementation supports local JPEG, PNG, WebP and HEIC/HEIF files, normalization and metadata removal, unique run IDs, checkpoints, and resumption. Put 3–40 distinct images in an ignored `photos/` folder. Each dimension must be at least 512px.

For individual captions, create `metadata.jsonl` next to pre-cropped square images, with one record per image:

```json
{"file_name": "01.png", "text": "a photo of benbenmaltese the Maltese dog, sitting on carpet with his ponytail visible"}
```

Captioned mode validates file alignment locally and checks the uploaded dataset on CPU before allocating the training GPU. For example:

```sh
uv run modal run --profile jordan-tony-collab --env main --detach modal_app.py::train \
  --photos photos/benben-v2 --run-id YOUR-NEW-RUN-ID --steps 400 \
  --resolution 768 --captioned --center-crop --lora-alpha 16 \
  --learning-rate 0.00008 --checkpointing-steps 200 --max-sequence-length 256
```

Resume an interrupted run with `--run-id YOUR-NEW-RUN-ID --resume`. Resumption uses the saved configuration, including the total step target. A usable checkpoint must exist; a failure before the first checkpoint requires a fresh run ID. Completed runs cannot be overwritten.

For controlled comparisons of stored adapters:

```sh
uv run modal run --profile jordan-tony-collab --env main experiments.py::evaluate \
  --spec eval_specs/v2-checkpoints.json --name v2-checkpoints
```

The checked-in specs refer to the stored v1/v2 runs. Edit paths for a different run. All images and prompt/seed/settings are saved in ignored `outputs/` and `/evaluations/` on the Volume. Comparisons are limited to 12 images per invocation and never change the live default automatically. `reference_experiment.py` preserves the ordinary img2img comparison that failed to change the background adequately.

## Cost controls and verification

GPU services use A100 80GB, no minimum warm containers, one container per service (or per parameter pool for the legacy LoRA class), and a 60-second idle scale-down window. Training has a 20-minute timeout; comparisons have 15 minutes; individual editing calls have 10 minutes. These are execution limits, not a dollar budget. Warm idle time, CPU and storage can still incur charges.

```sh
uv run ruff check .
uv run pytest
# CPU dependency/import and Gradio construction checks; no secrets required:
uv run modal run --profile jordan-tony-collab --env main check_environment.py
```

Local tests cover validation, caption alignment, trainer arguments and evaluation bounds. Actual identity and prompt-following quality require visual review. The pinned Diffusers trainer is [v0.35.1](https://github.com/huggingface/diffusers/tree/0f252be0ed42006c125ef4429156cb13ae6c1d60). Runtime packages are pinned; base-model downloads use the model repository's current revision on first download and are then cached.

References: [Modal pet LoRA example](https://modal.com/docs/examples/diffusers_lora_finetune), [Diffusers FLUX training](https://github.com/huggingface/diffusers/blob/v0.35.1/examples/dreambooth/README_flux.md), [FLUX Kontext model](https://huggingface.co/black-forest-labs/FLUX.1-Kontext-dev).
