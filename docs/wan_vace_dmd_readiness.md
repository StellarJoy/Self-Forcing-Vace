# GT-fed VACE teacher DMD experiment notes

This repository supports two lightweight ways to feed Wan VACE teacher reference information during Self-Forcing DMD training:

1. **Pixel-video references**: `gt_video_root/{idx}.pt`, optionally with masks.
2. **Precomputed latent references**: `shard.*/sample_id/{prompt.txt,embed.pt,latent.pt}`.

## Sharded latent/embed data layout

For precomputed data, set:

```yaml
data_format: sharded_latent_embed
data_path: /path/to/reference_root
use_vace_teacher: true
vace_use_mask: true
latent_filename: latent.pt
embed_filename: embed.pt
prompt_filename: prompt.txt
```

Expected directory structure:

```text
reference_root/
  shard.01/
    001/
      prompt.txt
      embed.pt
      latent.pt
    002/
      prompt.txt
      embed.pt
      latent.pt
  shard.02/
    ...
```

`embed.pt` should be the Wan text embedding for the prompt. Training uses it directly as `prompt_embeds`, so the T5 text encoder does not need to recompute the conditional prompt embedding for these samples. `prompt.txt` is still useful metadata for debugging, logging, and evaluation.

`latent.pt` should be a Wan VAE latent for the reference video, either `[16,T,H,W]` or `[T,16,H,W]`. The trainer treats this as a full-video VACE reference by building:

```text
inactive_latent = zeros_like(latent)
reactive_latent = latent
mask_latent = ones([64,T,H,W])
vace_context = cat([inactive_latent, reactive_latent, mask_latent], dim=0)
```

So the final VACE context has shape `[96,T,H,W]`.

## Pixel video reference path

For raw/video-tensor references, keep:

```yaml
use_vace_teacher: true
gt_video_root: /path/to/gt_video_tensors
gt_video_ext: .pt
vace_use_mask: true
gt_mask_root: null
```

If no `gt_mask_root` is provided, the trainer creates an all-ones mask, meaning the whole video is treated as the reference/conditioned region. Explicit masks are only needed for spatial or temporal editing experiments.

## Current training semantics

- `real_score` becomes Wan VACE when `use_vace_teacher: true`.
- The causal Self-Forcing generator and plain fake-score model ignore VACE kwargs.
- VACE reference context is teacher-side guidance for DMD; it is not an extra runtime input to the student generator.
