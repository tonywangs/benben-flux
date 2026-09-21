# Benben’s World 🐾

Generate photos and artwork of **Benben the Maltese** with a FLUX.1-dev DreamBooth LoRA, trained and served on Modal. Photos stay out of Git; training runs, checkpoints, and weights live in a Modal Volume. The Gradio app requires a password and GPU containers scale to zero.

## Setup

Install [uv](https://docs.astral.sh/uv/getting-started/installation/), then:

```sh
uv sync --locked
uv run modal setup
```

1. Sign into Hugging Face and accept the [FLUX.1-dev access agreement](https://huggingface.co/black-forest-labs/FLUX.1-dev). Review its model license for your intended use.
2. Create a read token with access to this gated model.
3. In [Modal Secrets](https://modal.com/secrets), create `huggingface-secret` containing `HF_TOKEN`. Use the same Modal workspace as your CLI. Keep tokens out of this repository and chat.
4. For the web app, create `benben-web-auth` with `USERNAME` and a strong `PASSWORD`.

Create both secrets before running `modal_app.py`: Modal resolves all app resources at startup. Only the download function receives the Hugging Face secret; only the web function receives the web credentials. The independent `check_environment.py` check needs neither secret.

## Add Benben’s photos

Put 10–20 different photos in `photos/` (3–40 supported). JPEG, PNG, WebP and iPhone HEIC/HEIF work. Each dimension must be at least 512 pixels. Include close-ups and full-body shots, different angles and backgrounds, and keep Benben clearly visible. Avoid near duplicates, other dogs, heavy filters, and outfits in every photo. Keep uncropped originals elsewhere: the trainer uses square crops, so leave room around Benben.

The local loader fixes orientation, strips metadata, reduces large images, and removes exact normalized duplicates before uploading. It does not inspect whether a photo actually depicts Benben. Images and generated outputs are gitignored.

## Check the environment

```sh
uv run pytest
uv run ruff check .
# Builds the remote image and checks the actual trainer imports; CPU only:
uv run modal run check_environment.py
```

## Train

```sh
uv run modal run --detach modal_app.py::train --photos ./photos --run-id benben-v1
```

Starts with 500 steps, rank 16, 512px, batch size 1, gradient checkpointing, and a learning rate of 0.0001 on an A100 80GB. These are starting settings, not a guarantee of likeness. The first run downloads the base model; later runs reuse it. Watch logs and costs in [Modal](https://modal.com/apps). Do not run multiple training jobs against the same run ID.

Each new run needs a unique ID. Artifacts are stored in the `benben-flux` Volume:

```text
base/                   shared pretrained model
runs/benben-v1/
  photos/               normalized private training photos
  config.json           reproducible training settings
  command.json          exact trainer command
  lora/                 adapter, last 3 checkpoints and TensorBoard logs
  COMPLETE              written only after successful training
```

Checkpoints are saved every 100 steps. If a job fails after saving one:

```sh
uv run modal run --detach modal_app.py::train --run-id benben-v1 --resume
```

Resume uses the run’s original config, including its total step target; CLI hyperparameter overrides are ignored on resume. Modal periodically persists the Volume and the wrapper commits on normal completion or a caught trainer error. A hard timeout/crash can lose the most recent writes. If failure occurs before a usable checkpoint, start a fresh run ID.

Compare a few fixed-seed prompts after training. If likeness is weak, try a separate run with 800 steps; if Benben repeats poses/backgrounds or ignores prompts, try 250–350. Better photos usually help more than spending more GPU credits. This uses DreamBooth LoRA on transformer weights; it does **not** train new textual-inversion embeddings or text encoders. The learned trigger phrase is `benbenmaltese the Maltese dog`.

```sh
uv run modal run --detach modal_app.py::train --run-id benben-v2 --steps 800 --rank 16
```

## Generate and share

```sh
uv run modal run modal_app.py::generate --run-id benben-v1 \
  --prompt 'a watercolor portrait in a sunny garden' --seed 117
# Saves outputs/benben.png and generation settings in outputs/benben.json.

uv run modal serve modal_app.py
# Or create a persistent URL:
uv run modal deploy modal_app.py
```

Open the Modal-provided URL, sign in with `benben-web-auth`, and enter the completed training run ID. Share the URL and credentials only with people you want to generate images. Cold starts take longer than subsequent requests. Each inference container handles one generation at a time; the UI queues up to 10 requests.

For an adapter download:

```sh
uv run modal volume get benben-flux /runs/benben-v1/lora/pytorch_lora_weights.safetensors ./outputs/
```

The model uses an A100 80GB for inference as well as training. There are no minimum warm containers; idle inference containers shut down after approximately 60 seconds. Warm idle time and Volume storage can still cost money. Each distinct run ID has its own inference parameter pool, so avoid opening many runs simultaneously. GPU functions have a 2-hour training timeout and a 10-minute generation timeout. These limits are **not** a dollar budget; use Modal billing controls and usage monitoring. There is no reason to use all $5k for one pet LoRA.

## Development

```sh
uv sync --locked --dev
uv run pytest
uv run ruff check .
```

Local tests check dataset preprocessing, run path validation, and training arguments. The remote environment check verifies dependency imports and the upstream trainer CLI; it does not prove CUDA execution, gated model access, or image quality. Those require a real training and inference run with Benben’s photos.

The trainer is fetched from a pinned [Diffusers v0.35.1 commit](https://github.com/huggingface/diffusers/tree/0f252be0ed42006c125ef4429156cb13ae6c1d60); runtime packages and local dependencies are pinned. The initial base-model download follows the model repository’s default revision and reuses its local cache. Remote image build logs show the complete installed dependency resolution.

Inspired by [Modal’s pet LoRA example](https://modal.com/docs/examples/diffusers_lora_finetune), with [Diffusers FLUX training](https://github.com/huggingface/diffusers/blob/v0.35.1/examples/dreambooth/README_flux.md), [Modal Volumes](https://modal.com/docs/guide/volumes), and [scale-to-zero controls](https://modal.com/docs/guide/scale).
