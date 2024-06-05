# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
# Training the model
```sh
python ms_speech_to_text_aed.py \
    # (Optional: --config-path=<path to dir of configs> --config-name=<name of config without .yaml>) \
    model.train_ds.tarred_audio_filepaths=<path to tar files with audio> \
    model.train_ds.manifest_filepath=<path to audio data manifest> \
    model.train_ds.batch_duration=360 \
    model.train_ds.num_buckets=30 \
    model.train_ds.bucket_duration_bins=<optional list of precomputed float bins for bucket durations, speeds up init> \
    model.validation_ds.manifest_filepath=<path to validation manifest> \
    model.test_ds.manifest_filepath=<path to test manifest> \
    model.model_defaults.asr_enc_hidden=1024 \
    model.model_defaults.lm_enc_hidden=512 \
    model.model_defaults.lm_dec_hidden=1024 \
    model.tokenizer.langs.spl_tokens.dir=<path to the directory of prompt special tokens tokenizer> \
    model.tokenizer.langs.spl_tokens.type=bpe \
    model.tokenizer.langs.en.dir=<path to the directory of en language tokenizer (add new langs the same way)> \
    model.tokenizer.langs.en.type=bpe \
    model.prompt_format="canary" \
    trainer.devices=-1 \
    trainer.accelerator="ddp" \
    trainer.max_steps=100000 \
    +trainer.limit_train_batches=20000 \
    trainer.val_check_interval=5000 \
    +trainer.use_distributed_sampler=false \
    model.optim.name="adamw" \
    model.optim.lr=0.001 \
    model.optim.betas=[0.9,0.999] \
    model.optim.weight_decay=0.0001 \
    model.optim.sched.warmup_steps=2000 \
    exp_manager.create_wandb_logger=True \
    exp_manager.wandb_logger_kwargs.name="<Name of experiment>" \
    exp_manager.wandb_logger_kwargs.project="<Name of project>"
```


"""
import os
from dataclasses import is_dataclass

import pytorch_lightning as pl
from omegaconf import DictConfig, OmegaConf, open_dict

from nemo.collections.asr.models import MSEncDecMultiTaskModelAdapter as EncDecMultiTaskModel
from nemo.core import adapter_mixins
from nemo.core.config import hydra_runner
from nemo.utils import logging, model_utils
from nemo.utils.exp_manager import clean_exp_ckpt, exp_manager

def update_model_config_to_support_adapter(model_cfg, current_cfg):
    with open_dict(model_cfg):
        # Override prediction logging in config
        model_cfg.log_prediction = current_cfg.model.get('log_prediction', False)

        # Update encoder adapter compatible config
        adapter_metadata = adapter_mixins.get_registered_adapter(model_cfg.encoder._target_)
        if adapter_metadata is not None:
            model_cfg.encoder._target_ = adapter_metadata.adapter_class_path


def update_model_cfg(original_cfg, new_cfg):
    with open_dict(original_cfg), open_dict(new_cfg):
        # force inject some keys into the config
        whitelist_keys = ['num_workers', 'pin_memory']
        for wkey in whitelist_keys:
            if wkey in new_cfg:
                original_cfg[wkey] = new_cfg[wkey]
                print(f"Injecting white listed key `{wkey}` into config")

        # drop keys which don't exist in old config and are not whitelisted
        new_keys = list(new_cfg.keys())
        for key in new_keys:
            if key not in original_cfg:
                new_cfg.pop(key)
                print("Removing unavailable key from config :", key)

        new_cfg = OmegaConf.merge(original_cfg, new_cfg)
    return new_cfg


def add_global_adapter_cfg(model, global_adapter_cfg):
    # Convert to DictConfig from dict or Dataclass
    if is_dataclass(global_adapter_cfg):
        global_adapter_cfg = OmegaConf.structured(global_adapter_cfg)

    if not isinstance(global_adapter_cfg, DictConfig):
        global_adapter_cfg = DictConfig(global_adapter_cfg)

    # Update the model.cfg with information about the new adapter global cfg
    with open_dict(global_adapter_cfg), open_dict(model.cfg):
        if 'adapters' not in model.cfg:
            model.cfg.adapters = OmegaConf.create({})

        # Add the global config for adapters to the model's internal config
        model.cfg.adapters[model.adapter_global_cfg_key] = global_adapter_cfg

        # Update all adapter modules (that already exist) with this global adapter config
        model.update_adapter_cfg(model.cfg.adapters)


@hydra_runner(config_path="../conf/speech_multitask/", config_name="fast-conformer_aed")
def main(cfg):
    logging.info(f'Hydra config: {OmegaConf.to_yaml(cfg)}')

    trainer = pl.Trainer(**cfg.trainer)
    exp_manager(trainer, cfg.get("exp_manager", None))

    # Check for spl tokens to create spl_tokenizer.
    if cfg.get("spl_tokens"):
        logging.info("Detected spl_tokens config. Building tokenizer.")
        spl_cfg = cfg["spl_tokens"]
        spl_tokenizer_cls = model_utils.import_class_by_path(cfg.model.tokenizer.custom_tokenizer["_target_"])
        spl_tokenizer_cls.build_special_tokenizer(
            spl_cfg["tokens"], spl_cfg["model_dir"], force_rebuild=spl_cfg["force_rebuild"]
        )
        cfg.model.tokenizer.langs.spl_tokens.dir = spl_cfg["model_dir"]

    aed_model = EncDecMultiTaskModel(cfg=cfg.model, trainer=trainer)

    # Initialize the weights of the model from another model, if provided via config
    aed_model.maybe_init_from_pretrained_checkpoint(cfg)
    aed_model.encoder.freeze()
    aed_model = aed_model.train()
    aed_model.unfreeze_enabled_adapters()
    trainer.fit(aed_model)

    if hasattr(cfg.model, 'test_ds') and cfg.model.test_ds.manifest_filepath is not None:
        if aed_model.prepare_test(trainer):
            trainer.test(aed_model)


if __name__ == '__main__':
    main()
