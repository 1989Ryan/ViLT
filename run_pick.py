import os
import copy
import pytorch_lightning as pl
import torch
from pytorch_lightning.loggers import WandbLogger

from vilt.config import ex
from vilt.modules import ViLTransformerSS
from vilt.datamodules.multitask_datamodule import MTDataModule
from tabletop_dataset import tabletop_gym_objpick_dataset


@ex.automain
def main(_config):
    _config = copy.deepcopy(_config)
    pl.seed_everything(_config["seed"])
    wandb_logger = WandbLogger(name="VILT_Picking")
    # dm = MTDataModule(_config, dist=True)
    dataset = tabletop_gym_objpick_dataset(
        _config =_config,
        device ="cuda:0",
        root= "/home/zirui/tabletop_gym/dataset", 
        num=8000)
    train_dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=_config["batch_size"], shuffle=True, num_workers=0)
    valdataset = tabletop_gym_objpick_dataset(
        _config =_config,
        device ="cuda:0",
        root= "/home/zirui/tabletop_gym/dataset", 
        test = True,
        num=None)
    val_dataloader = torch.utils.data.DataLoader(
        valdataset, batch_size=_config["batch_size"], shuffle=True, num_workers=0)
    model = ViLTransformerSS(_config)
    exp_name = f'{_config["exp_name"]}'

    os.makedirs(_config["log_dir"], exist_ok=True)
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        # save_top_k=1,
        dirpath='baseline/vilt/pick_ckpt',
        # filename='placing_finetune',
        verbose=True,
        save_last=True,
        # save_on_train_epoch_end=True
    )
    # logger = pl.loggers.TensorBoardLogger(
    #     _config["log_dir"],
    #     name=f'{exp_name}_seed{_config["seed"]}_from_{_config["load_path"].split("/")[-1][:-5]}',
    # )
    # wandb.init(project="ParaGon_train_n{}".format(cfg['dataset']['data_num']))
    lr_callback = pl.callbacks.LearningRateMonitor(logging_interval="step")
    callbacks = [checkpoint_callback, lr_callback]

    num_gpus = (
        _config["num_gpus"]
        if isinstance(_config["num_gpus"], int)
        else len(_config["num_gpus"])
    )

    grad_steps = _config["batch_size"] // (
        _config["per_gpu_batchsize"] * num_gpus * _config["num_nodes"]
    )

    max_steps = _config["max_steps"] if _config["max_steps"] is not None else None

    trainer = pl.Trainer(
        gpus=_config["num_gpus"],
        num_nodes=_config["num_nodes"],
        precision=_config["precision"],
        accelerator="cuda",
        benchmark=True,
        deterministic=True,
        max_epochs=_config["max_epoch"] if max_steps is None else 1000,
        max_steps=max_steps,
        callbacks=callbacks,
        logger=wandb_logger,
        # prepare_data_per_node=False,
        # replace_sampler_ddp=False,
        # accumulate_grad_batches=grad_steps,
        log_every_n_steps=10,
        # flush_logs_every_n_steps=10,
        # resume_from_checkpoint=_config["resume_from"],
        # weights_summary="top",
        fast_dev_run=_config["fast_dev_run"],
        val_check_interval=_config["val_check_interval"],
    )

    if not _config["test_only"]:
        trainer.fit(model, train_dataloader, val_dataloader)
