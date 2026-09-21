"""Independent CPU checks that can run before configuring account secrets."""

import modal

from modal_app import image, web_image

app = modal.App("benben-environment-check")


@app.function(image=image, timeout=600)
def training():
    import subprocess
    import diffusers
    import torch

    subprocess.run(
        ["python", "/opt/diffusers/examples/dreambooth/train_dreambooth_lora_flux.py", "--help"],
        check=True,
        stdout=subprocess.DEVNULL,
    )
    return {"torch": torch.__version__, "diffusers": diffusers.__version__, "trainer": "OK"}


@app.function(image=web_image.add_local_python_source("modal_app"), timeout=300)
def web():
    import os
    from modal_app import web as web_function

    os.environ["USERNAME"] = "environment-check"
    os.environ["PASSWORD"] = "environment-check"
    asgi = web_function.local()
    return {"web": type(asgi).__name__, "routes": len(asgi.routes)}


@app.local_entrypoint()
def main():
    print(training.remote())
    print(web.remote())
