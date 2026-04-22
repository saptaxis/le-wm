import os
from functools import partial
from pathlib import Path

import hydra
import lightning as pl
import numpy as np
import stable_pretraining as spt
import stable_worldmodel as swm
import torch
from lightning.pytorch.loggers import TensorBoardLogger, WandbLogger
from omegaconf import OmegaConf, open_dict

from jepa import JEPA
from module import ARPredictor, Embedder, LinearStateHead, MLP, SIGReg
from utils import get_column_normalizer, get_img_preprocessor, ModelObjectCallBack


def lejepa_forward(self, batch, stage, cfg):
    """encode observations, predict next states, compute losses."""

    ctx_len = cfg.wm.history_size
    n_preds = cfg.wm.num_preds
    lambd = cfg.loss.sigreg.weight

    # Replace NaN values with 0 (occurs at sequence boundaries)
    batch["action"] = torch.nan_to_num(batch["action"], 0.0)

    output = self.model.encode(batch)

    emb = output["emb"]  # (B, T, D)
    act_emb = output["act_emb"]

    ctx_emb = emb[:, :ctx_len]
    ctx_act = act_emb[:, : ctx_len]

    tgt_emb = emb[:, n_preds:] # label
    pred_emb = self.model.predict(ctx_emb, ctx_act) # pred

    # Dedicated kinematic subspace (v2): reserve the first `kin_block` dims of
    # z for the aux-decoded kinematic state. SIGReg then applies only to the
    # remaining z_rest dims so it doesn't fight the aux loss's pull toward
    # real state statistics. If kin_block is unset or equals embed_dim, this
    # falls back to v1 behavior (sigreg on full z, aux reads full z).
    aux_cfg = cfg.wm.get("aux_loss", None)
    kin_block = int(aux_cfg.get("kin_block", emb.size(-1))) if aux_cfg is not None else emb.size(-1)
    kin_block = min(kin_block, emb.size(-1))

    # LeWM loss
    output["pred_loss"] = (pred_emb - tgt_emb).pow(2).mean()
    if kin_block < emb.size(-1):
        output["sigreg_loss"] = self.sigreg(emb[..., kin_block:].transpose(0, 1))
    else:
        output["sigreg_loss"] = self.sigreg(emb.transpose(0, 1))
    output["loss"] = output["pred_loss"] + lambd * output["sigreg_loss"]

    # Optional auxiliary kinematic loss: decode predicted z's to GT state and
    # penalize mismatch. Forces action effects into the kinematic z-subspace.
    if (
        aux_cfg is not None
        and bool(aux_cfg.get("enabled", False))
        and getattr(self.model, "state_head", None) is not None
        and "state" in batch
    ):
        kin_dim = int(aux_cfg.get("state_dim", 6))
        lam = float(aux_cfg["lambda"])
        # pred_emb predicts positions [ctx_len - (ctx_len - n_preds) ... T-1].
        # With ctx_len=3, n_preds=1 this gives the last 3 positions of the T=4
        # window, same as tgt_emb. GT state targets slice the same positions.
        gt_state = batch["state"][:, n_preds:, :kin_dim].float()
        target = self.model.state_head.normalize_target(gt_state)
        # State head reads from the reserved kinematic subspace (first kin_block
        # dims). If kin_block == D, this is just the full z (v1 behavior).
        z_kin = pred_emb[..., :kin_block]
        decoded = self.model.state_head(z_kin)
        output["aux_kin_loss"] = (decoded - target).pow(2).mean()
        output["loss"] = output["loss"] + lam * output["aux_kin_loss"]

    losses_dict = {f"{stage}/{k}": v.detach() for k, v in output.items() if "loss" in k}
    is_train = (stage == "fit")
    self.log_dict(losses_dict, on_step=is_train, on_epoch=True, sync_dist=True)
    return output

@hydra.main(version_base=None, config_path="./config/train", config_name="lewm")
def run(cfg):
    #########################
    ##       dataset       ##
    #########################

    # Support single dataset name or list of names for ConcatDataset
    ds_cfg = dict(cfg.data.dataset)
    ds_name = ds_cfg.pop("name")
    if isinstance(ds_name, str):
        ds_names = [ds_name]
    else:
        ds_names = list(ds_name)

    datasets = [swm.data.HDF5Dataset(name=n, **ds_cfg, transform=None) for n in ds_names]
    if len(datasets) == 1:
        dataset = datasets[0]
    else:
        dataset = swm.data.ConcatDataset(datasets)
        print(f"ConcatDataset: {len(ds_names)} datasets, {len(dataset)} total clips")

    transforms = [get_img_preprocessor(source='pixels', target='pixels', img_size=cfg.img_size)]

    # Use first dataset for normalizer stats
    ref_dataset = datasets[0]
    with open_dict(cfg):
        for col in cfg.data.dataset.keys_to_load:
            if col.startswith("pixels"):
                continue
            # `state` is consumed by the auxiliary kinematic loss, which computes
            # its own normalization over just the first `state_dim` kinematic
            # dims (the per-column normalizer would otherwise try to normalize
            # all 15 dims, including near-constant physics params with std~=0).
            if col == "state":
                continue

            normalizer = get_column_normalizer(ref_dataset, col, col)
            transforms.append(normalizer)

            setattr(cfg.wm, f"{col}_dim", ref_dataset.get_dim(col))

    transform = spt.data.transforms.Compose(*transforms)
    # Propagate transform to each sub-dataset explicitly. swm.data.ConcatDataset
    # does NOT invoke a transform stored on itself; its __getitem__ is pure
    # delegation to self.datasets[i], which DO honor their own .transform.
    # Without this loop, every multi-dataset config silently skips the pixel
    # preprocessor AND the per-column normalizers: ViT sees raw uint8 pixels,
    # action_encoder sees raw [-1, 1] actions. Single-dataset configs are
    # unaffected because `dataset is datasets[0]` and the line below sets the
    # transform on the HDF5Dataset directly. See e5-05 notebook.
    for ds in datasets:
        ds.transform = transform
    dataset.transform = transform  # harmless: preserved for introspection

    rnd_gen = torch.Generator().manual_seed(cfg.seed)
    train_set, val_set = spt.data.random_split(
        dataset, lengths=[cfg.train_split, 1 - cfg.train_split], generator=rnd_gen
    )

    train = torch.utils.data.DataLoader(train_set, **cfg.loader,shuffle=True, drop_last=True, generator=rnd_gen)
    val = torch.utils.data.DataLoader(val_set, **cfg.loader, shuffle=False, drop_last=False)
    
    ##############################
    ##       model / optim      ##
    ##############################

    encoder = spt.backbone.utils.vit_hf(
        cfg.encoder_scale,
        patch_size=cfg.patch_size,
        image_size=cfg.img_size,
        pretrained=False,
        use_mask_token=False,
    )

    hidden_dim = encoder.config.hidden_size
    embed_dim = cfg.wm.get("embed_dim", hidden_dim)
    effective_act_dim = cfg.data.dataset.frameskip * cfg.wm.action_dim

    predictor = ARPredictor(
        num_frames=cfg.wm.history_size,
        input_dim=embed_dim,
        hidden_dim=hidden_dim,
        output_dim=hidden_dim,
        **cfg.predictor,
    )

    action_encoder = Embedder(input_dim=effective_act_dim, emb_dim=embed_dim)
    
    projector = MLP(
        input_dim=hidden_dim,
        output_dim=embed_dim,
        hidden_dim=2048,
        norm_fn=torch.nn.BatchNorm1d,
    )

    predictor_proj = MLP(
        input_dim=hidden_dim,
        output_dim=embed_dim,
        hidden_dim=2048,
        norm_fn=torch.nn.BatchNorm1d,
    )

    # Auxiliary kinematic state head (optional).
    state_head = None
    aux_cfg = cfg.wm.get("aux_loss", None)
    if aux_cfg is not None and bool(aux_cfg.get("enabled", False)):
        kin_dim = int(aux_cfg.get("state_dim", 6))
        # Compute per-dim kinematic mean/std POOLED across ALL datasets in the
        # training mix, not just datasets[0]. Prior bug (fixed 2026-04-19):
        # used ref_dataset only, which undersized angle/ang_vel std by ~2×
        # relative to pooled (heuristic has much narrower rotation than
        # impulse-side, etc.). The undersized std made non-heuristic samples
        # contribute disproportionately large normalized targets to the aux
        # loss, biasing the optimizer. Pooling equalizes per-sample weight.
        # HDF5 state has shape (total_frames, 15); we only care about the
        # first `kin_dim` (kinematic) dimensions.
        state_parts = [
            np.asarray(d.get_col_data("state"))[:, :kin_dim] for d in datasets
        ]
        state_data = np.concatenate(state_parts, axis=0)
        # Drop any rows with NaN to match the action normalizer's behavior.
        mask = ~np.isnan(state_data).any(axis=1)
        state_data = state_data[mask]
        kin_mean = torch.from_numpy(state_data.mean(axis=0)).float()
        kin_std = torch.from_numpy(state_data.std(axis=0)).float()
        # Safety clamp against degenerate dims. With normal kinematic data from
        # lunar-lander trajectories, no clamp should actually fire.
        kin_std = torch.clamp(kin_std, min=1e-4)
        # Dedicated kinematic subspace: state head reads from a reserved block
        # of z (z_kin). Default kin_block = embed_dim preserves v1 behavior
        # (head reads full z). v2 sets kin_block=16 to reserve the first 16
        # dims of z for kinematic decoding and excludes those dims from SIGReg.
        kin_block = int(aux_cfg.get("kin_block", embed_dim))
        assert 0 < kin_block <= embed_dim, (
            f"aux_loss.kin_block={kin_block} must be in (0, embed_dim={embed_dim}]"
        )
        assert kin_block >= kin_dim, (
            f"aux_loss.kin_block={kin_block} must be >= aux_loss.state_dim={kin_dim}"
            f" (head maps kin_block→state_dim dims)"
        )
        print(
            f"Aux kinematic head: pooled across {len(datasets)} datasets, "
            f"kin_block={kin_block}/{embed_dim}, "
            f"mean={kin_mean.tolist()}, std={kin_std.tolist()}, "
            f"lambda={float(aux_cfg['lambda'])}"
        )
        state_head = LinearStateHead(
            in_dim=kin_block,
            out_dim=kin_dim,
            target_mean=kin_mean,
            target_std=kin_std,
        )

    world_model = JEPA(
        encoder=encoder,
        predictor=predictor,
        action_encoder=action_encoder,
        projector=projector,
        pred_proj=predictor_proj,
        state_head=state_head,
    )

    optimizers = {
        'model_opt': {
            "modules": 'model',
            "optimizer": dict(cfg.optimizer),
            "scheduler": {"type": "LinearWarmupCosineAnnealingLR"},
            "interval": "epoch",
        },
    }

    data_module = spt.data.DataModule(train=train, val=val)
    world_model = spt.Module(
        model = world_model,
        sigreg = SIGReg(**cfg.loss.sigreg.kwargs),
        forward=partial(lejepa_forward, cfg=cfg),
        optim=optimizers,
    )

    ##########################
    ##       training       ##
    ##########################

    run_id = cfg.get("subdir") or ""
    if cfg.get("run_dir"):
        run_dir = Path(cfg.run_dir) / run_id
    else:
        run_dir = Path(swm.data.utils.get_cache_dir(), run_id)

    logger = None
    if cfg.wandb.enabled:
        logger = WandbLogger(**cfg.wandb.config)
        logger.log_hyperparams(OmegaConf.to_container(cfg))
    else:
        logger = TensorBoardLogger(save_dir=str(run_dir), name="tb_logs", version="")

    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / "config.yaml", "w") as f:
        OmegaConf.save(cfg, f)

    object_dump_callback = ModelObjectCallBack(
        dirpath=run_dir, filename=cfg.output_model_name, epoch_interval=1,
    )

    trainer = pl.Trainer(
        **cfg.trainer,
        callbacks=[object_dump_callback],
        num_sanity_val_steps=1,
        logger=logger,
        enable_checkpointing=True,
    )

    manager = spt.Manager(
        trainer=trainer,
        module=world_model,
        data=data_module,
        ckpt_path=run_dir / f"{cfg.output_model_name}_weights.ckpt",
    )

    manager()
    return


if __name__ == "__main__":
    run()
