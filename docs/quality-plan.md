# Benben identity experiment

The first deployment passed engineering tests but failed its actual purpose: generating Benben. A successful GPU job is not an identity evaluation.

## Questions and evidence

1. Is the v1 adapter active? A fixed-seed base/v1 comparison shows materially different results, and the saved adapter contains 684 tensors with nonzero LoRA B matrices. This rules out a completely ignored adapter.
2. Does the inference description matter? On the diagnostic seed, explicitly describing the tied-up ponytail restores it. The base model can also generate a ponytail, so this alone is not evidence of Benben's identity.
3. Is increasing adapter strength sufficient? A 2x weight on that same diagnostic prompt produces severe degradation. Do not ship a blanket strength increase.
4. Is the training data preserving the important features? V1 used a mixture of grooming styles, some small or obscured faces, and random square crops. V2 uses 12 reviewed square crops with head, ponytail and facial proportions preserved, and pose/background captions matched by filename.

These observations do not identify a single causal explanation. Several interventions change together in the targeted retrain; this is a practical repair experiment, not an isolated hyperparameter study.

## Bounded next experiment

- Base model and trigger remain FLUX.1-dev and `benbenmaltese`.
- 12 curated 768px square images; center crop leaves the reviewed square intact.
- Individual captions identify the same dog and describe variable pose/background and visible topknot.
- Rank 16, explicit alpha 16 (v1 inherited alpha 4), AdamW learning rate 0.00008.
- 400 steps, snapshots at 200 and 400; training timeout 20 minutes.
- Frozen text encoders; no prior-preservation images or synthetic training photos.
- Original photos remain unchanged and excluded from Git. Crops/captions stay in ignored `photos/`; evaluation images/specifications stay in ignored `outputs/` and the private Modal Volume.

## Evaluation before promotion

Use fixed prompts, seeds, resolution, steps and guidance across checkpoints. Inspect the earlier and later checkpoint on a neutral garden photo, a neutral indoor portrait, and an astronaut scene. The neutral prompts must not mention a ponytail. Inspect those alongside reference photos from other sessions (e.g. IMG_7495.HEIC and IMG_7648.HEIC), which were not included in v2 training.

Check the visible high tied-up topknot, rounded dark eyes and compact muzzle, long hanging ear hair, white coat and small body proportions, natural anatomy, and the requested setting. Compare these features against the references rather than assigning a misleading automated identity percentage.

Do not count a generic Maltese with a ponytail as success. Do not select only one flattering sample while ignoring failures. Prefer the earlier checkpoint if later training loses prompt following or degrades anatomy. Do not replace the live default until the small comparison materially improves identity; owner feedback is still necessary to judge exact likeness.

This experiment does not run an automatic training sweep or repeated retraining loop. One bounded retrain is followed by inspection and a written result.


## Recorded result

The 400-step run completed in approximately 8 minutes. Neither the 200-step nor 400-step checkpoint passed the neutral-prompt identity tests. The later checkpoint did not consistently improve the face and produced poor fur/ear shapes in some samples. Three additional conditioning checks restored a ponytail but still did not establish Benben's likeness. Neither checkpoint was promoted.

A three-image ordinary FLUX img2img comparison using the actual portrait preserved face and topknot better, but failed the garden/background instruction. It was not selected as a finished solution either.

The next targeted test uses FLUX.1-Kontext-dev, which is specifically trained to edit reference images. Access was verified using the existing Hugging Face token before requesting model weights. The download fetches only the Kontext transformer and configuration, reusing the already cached FLUX text encoders and VAE. This test performs no training. Three edits cover a garden, astronaut clothing, and watercolor style. Its result must be evaluated before changing the live site.


## Reference-editor result and deployment

The Kontext garden image retained the visible ponytail, dark eyes, compact muzzle and hanging ears while changing the background to a garden. The watercolor image retained the smiling reference's expression and tied topknot. Both were visibly closer than the LoRA samples. The astronaut image changed the scene and clothing but added a helmet, obscuring the ponytail; this is a recorded failure, not a successful identity sample.

The website now uses Kontext with an actual visible reference photo. It does not use the rejected v2 LoRA. A rewritten astronaut instruction was checked through the authenticated live site. It kept a ponytail visible but still added unwanted headwear, failing that instruction. The astronaut suggestion was removed from the UI; garden and watercolor are the tested starting examples. This is a change in method, not a claim that the LoRA was fixed. Owner assessment of exact likeness remains necessary.
