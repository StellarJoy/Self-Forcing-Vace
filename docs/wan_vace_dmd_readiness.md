# GT-fed VACE teacher DMD experiment notes

This repository is already wired to run the intended experiment: use Self-Forcing as the student/generator, switch the DMD teacher (`real_score`) to Wan VACE, and feed GT video tensors as VACE reference conditioning.

## Minimal switches for the experiment

In `configs/self_forcing_dmd.yaml`:

```yaml
use_vace_teacher: true
gt_video_root: /path/to/gt_video_tensors
gt_video_ext: .pt
vace_use_mask: true
# optional: only needed if you want a spatial/temporal edit mask
gt_mask_root: null
```

The GT tensor can be either `[C,T,H,W]` or `[T,C,H,W]`; the current loader converts `[T,C,H,W]` to `[C,T,H,W]` when it sees the expected channel count.

## What the mask means here

Wan VACE uses the mask to split the reference video into two latent branches:

- `inactive = video * (1 - mask)`
- `reactive = video * mask`

Then it concatenates the inactive latent, reactive latent, and a downsampled mask latent-like tensor into `vace_context`.

For your current goal — **the GT video is only used as the reference video for teacher guidance** — you do **not** need to provide a mask file. If `gt_mask_root` is `null`, the training code now follows native `WanVace.build_vace_context`: it creates an all-ones mask, meaning the whole GT video is treated as the reference/conditioned region.

Use an explicit mask only if you want VACE to distinguish edited/conditioned regions. With the current convention:

- mask value `1`: this pixel/frame region goes to the reactive/reference branch;
- mask value `0`: this pixel/frame region goes to the inactive branch.

## Current path in code

1. `use_vace_teacher: true` makes `real_score` load `VaceWanModel`.
2. `TextDataset` emits index-aligned `gt_video_path` and optional `gt_mask_path`.
3. `Trainer._build_vace_context` loads the GT video, creates/loads the mask, encodes both branches with the Wan VAE, packs the downsampled mask, and writes `conditional_dict["vace_context"]`.
4. `WanDiffusionWrapper.forward` passes `vace_context` and `vace_context_scale` into the VACE model only when those fields exist.

So for a first runnable experiment, keep `vace_use_mask: true` and leave `gt_mask_root: null`; that gives you a full-video reference mask without extra dataset files.
