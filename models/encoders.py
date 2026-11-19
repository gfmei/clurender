"""Point cloud encoders shared by pretraining and the downstream tasks.

``build_encoder`` makes a DGCNN or a batched OctFormer from argument-style
settings, ``add_octformer_arguments`` defines the OctFormer settings for every
script, and ``downstream_encoder`` rebuilds the architecture a checkpoint was
pretrained with.
"""

import torch

from models.dgcnn import DGCNN
from models.octformer import OctFormer
from models.serialization import CURVES

ENCODERS = ("dgcnn", "octformer")
OCTFORMER_SETTINGS = ("octformer_channels", "octformer_blocks", "octformer_heads", "octformer_patch_size",
                      "octformer_dilation", "octformer_stride", "octformer_k", "octformer_voxel",
                      "octformer_drop_path", "octformer_fpn_channels", "octformer_full_attention_from",
                      "serialization")


def add_octformer_arguments(parser):
    group = parser.add_argument_group("OctFormer encoder (--model octformer)")
    group.add_argument("--octformer-channels", type=int, nargs="+", default=[64, 128, 256, 256])
    group.add_argument("--octformer-blocks", type=int, nargs="+", default=[2, 2, 6, 2])
    group.add_argument("--octformer-heads", type=int, nargs="+", default=[4, 8, 16, 16])
    group.add_argument("--octformer-patch-size", type=int, default=32, help="Points per attention window")
    group.add_argument("--octformer-dilation", type=int, default=4, help="Dilation of every second block")
    group.add_argument("--octformer-stride", type=int, default=4, help="Points pooled per downsampling step")
    group.add_argument("--octformer-k", type=int, default=16, help="Neighbors of the 3 x 3 x 3 convolutions")
    group.add_argument("--octformer-voxel", type=float, default=0.0625,
                       help="Level-0 voxel size for clouds normalized to the unit sphere")
    group.add_argument("--octformer-drop-path", type=float, default=0.1)
    group.add_argument("--octformer-fpn-channels", type=int, default=128)
    group.add_argument("--octformer-full-attention-from", type=int, default=1,
                       help="First stage with full attention; earlier stages use (dilated) windows")
    group.add_argument("--serialization", choices=CURVES, default="z",
                       help="Curve that orders points: z (octree order, as in OctFormer) or hilbert")


def encoder_name(encoder):
    if isinstance(encoder, OctFormer):
        return "octformer"
    if isinstance(encoder, DGCNN):
        return "dgcnn"
    raise ValueError(f"Unsupported encoder {type(encoder).__name__}")


def build_encoder(settings):
    """``settings``: a mapping with model, emb_dims, k and, for OctFormer, the octformer_* entries."""
    if settings["model"] == "dgcnn":
        return DGCNN(settings["emb_dims"], settings["k"], num_cls=-1)
    if settings["model"] == "octformer":
        return OctFormer(
            emb_dims=settings["emb_dims"], channels=tuple(settings["octformer_channels"]),
            blocks=tuple(settings["octformer_blocks"]), heads=tuple(settings["octformer_heads"]),
            patch_size=settings["octformer_patch_size"], dilation=settings["octformer_dilation"],
            stride=settings["octformer_stride"], k=settings["octformer_k"], voxel=settings["octformer_voxel"],
            curve=settings["serialization"], drop_path=settings["octformer_drop_path"],
            fpn_channels=settings["octformer_fpn_channels"],
            full_attention_from=settings["octformer_full_attention_from"])
    raise ValueError(f"Unsupported encoder {settings['model']}; choose from {ENCODERS}")


def checkpoint_settings(path):
    """Arguments saved with a main_pretrain.py checkpoint, or None for a plain state dict."""
    state = torch.load(path, map_location="cpu", weights_only=True)
    return dict(state["args"]) if "args" in state and "model" in state else None


def downstream_encoder(args):
    """The encoder a downstream task trains or probes.

    An OctFormer is rebuilt exactly as it was pretrained. DGCNN keeps the
    task's ``--k``, since its neighbor count is a per-task choice that does
    not change its weights.
    """
    settings = vars(args).copy()
    saved = checkpoint_settings(args.pretrained) if getattr(args, "pretrained", None) else None
    if saved is not None:
        settings["model"] = saved.get("model", "dgcnn")
        if settings["model"] == "octformer":
            settings.update({key: saved[key] for key in ("emb_dims", *OCTFORMER_SETTINGS)})
    return build_encoder(settings)


def encoder_checkpoint(model, args):
    """What main_pretrain.py saves as backbone.pth.

    DGCNN keeps its plain state dict; other encoders also carry their
    settings, in the full-checkpoint layout that load_encoder reads.
    """
    if args.model == "dgcnn":
        return model.backbone.state_dict()
    settings = {key: getattr(args, key) for key in ("model", "emb_dims", "k", *OCTFORMER_SETTINGS)}
    return {"args": settings,
            "model": {"backbone." + name: value for name, value in model.backbone.state_dict().items()}}
