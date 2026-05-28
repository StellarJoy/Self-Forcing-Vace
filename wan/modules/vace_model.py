from .model import WanModel


class VaceWanModel(WanModel):
    """
    Minimal VACE-compatible model shim.

    This keeps compatibility with checkpoints/API that expect a VACE model
    forward signature, while reusing WanModel internals in this repository.
    """

    def _forward(
        self,
        x,
        t,
        context,
        seq_len,
        vace_context=None,
        vace_context_scale=1.0,
        classify_mode=False,
        concat_time_embeddings=False,
        register_tokens=None,
        cls_pred_branch=None,
        gan_ca_blocks=None,
        clip_fea=None,
        y=None,
    ):
        # NOTE: vace_context arguments are currently accepted for compatibility.
        # Actual VACE fusion will be implemented in a follow-up change.
        return super()._forward(
            x=x,
            t=t,
            context=context,
            seq_len=seq_len,
            classify_mode=classify_mode,
            concat_time_embeddings=concat_time_embeddings,
            register_tokens=register_tokens,
            cls_pred_branch=cls_pred_branch,
            gan_ca_blocks=gan_ca_blocks,
            clip_fea=clip_fea,
            y=y,
        )
