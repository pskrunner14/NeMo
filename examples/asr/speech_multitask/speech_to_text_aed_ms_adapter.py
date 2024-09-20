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
python speech_to_text_aed.py \
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
import pytorch_lightning as pl
from omegaconf import OmegaConf

from nemo.collections.asr.models import EncDecMultiTaskModel, MSEncDecMultiTaskModel
from nemo.core.config import hydra_runner
from nemo.utils import logging, model_utils
from nemo.utils.exp_manager import exp_manager

from nemo.collections.asr.parts.submodules.adapters import TransformerMultiHeadAttentionAdapterConfig, MultiHeadAttentionAdapterConfig
from nemo.collections.common.parts.adapter_modules import LinearAdapterConfig


@hydra_runner(config_path="../conf/speech_multitask/", config_name="fast-conformer_aed_ms")
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


    # aed_model = EncDecMultiTaskModel(cfg=cfg.model, trainer=trainer)
    aed_model = MSEncDecMultiTaskModel(cfg=cfg.model, trainer=trainer)
    aed_model.replace_adapter_compatible_modules()
    # Initialize the weights of the model from another model, if provided via config
    aed_model.maybe_init_from_pretrained_checkpoint(cfg)

    # Linear Adapters
    encoder_adapter_cfg = LinearAdapterConfig(
        in_features=cfg.model.model_defaults.asr_enc_hidden,
        dim=cfg.model.adapter.dim,
    )
    decoder_adapter_cfg = LinearAdapterConfig(
        in_features=cfg.model.model_defaults.asr_enc_hidden,
        dim=cfg.model.adapter.dim,
    )
    # encoder_adapter_cfg = MultiHeadAttentionAdapterConfig(
    #     n_feat=cfg.model.model_defaults.asr_enc_hidden,
    #     n_head=1,
    #     proj_dim=cfg.model.adapter.dim,
    # )
    # decoder_adapter_cfg = TransformerMultiHeadAttentionAdapterConfig(
    #     hidden_size=cfg.model.model_defaults.asr_enc_hidden,
    #     num_attention_heads=1,
    #     proj_dim=cfg.model.adapter.dim,
    # )
    print(aed_model)
    aed_model.add_adapter("encoder:ms_encoder_adapter", encoder_adapter_cfg)
    aed_model.add_adapter("transf_decoder:ms_decoder_adapter", decoder_adapter_cfg)
    aed_model.set_enabled_adapters(enabled=False)  # disable all adapters prior to training
    aed_model.set_enabled_adapters("encoder:ms_encoder_adapter", enabled=True)  # enable just one adapter by name
    aed_model.set_enabled_adapters("transf_decoder:ms_decoder_adapter", enabled=True)  # enable just one adapter by name

    aed_model.freeze()
    aed_model = aed_model.train()
    aed_model.unfreeze_enabled_adapters()
    # aed_model.cfg = aed_model.cfg

    #added by Kunal for metacat params:
    for name, param in aed_model.named_parameters():
        if 'metacat' in name:
            param.requires_grad = True
    # import ipdb; ipdb.set_trace()

    trainer.fit(aed_model)

    if hasattr(cfg.model, 'test_ds') and cfg.model.test_ds.manifest_filepath is not None:
        if aed_model.prepare_test(trainer):
            trainer.test(aed_model)


if __name__ == '__main__':
    main()