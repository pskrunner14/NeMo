# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
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
import os
from collections import OrderedDict
from dataclasses import is_dataclass

import pytorch_lightning as pl
import torch
from omegaconf import DictConfig, OmegaConf, open_dict
from pytorch_lightning import seed_everything

from nemo.collections.asr.models.eesd_models_heh import EncDecEESDModel
from nemo.collections.asr.modules.multi_layer_feat import ConformerMultiLayerFeaturePreprocessor
from nemo.core import adapter_mixins
from nemo.core.classes.common import typecheck
from nemo.core.config import hydra_runner
from nemo.utils import logging
from nemo.utils.exp_manager import exp_manager

typecheck.set_typecheck_enabled(enabled=False)

"""
CUDA_VISIBLE_DEVICES="0" python encdec_eesd_train.py \
    --config-path="../conf/neural_diarizer" \
    --config-name="fastconformer_large" \
    model.train_ds.manifest_filepath=$train_manifests \
    model.validation_ds.manifest_filepath=$dev_manifests \
    model.train_ds.batch_size=$batch_size \
    model.validation_ds.batch_size=$batch_size \
    ++model.validation_ds.use_der_metric=true \
    model.train_ds.num_workers=$num_workers \
    model.validation_ds.num_workers=$num_workers \
    model.optim.lr=$lr \
    model.optim.sched.warmup_steps=$warmup_steps \
    model.optim.sched.min_lr=$min_lr \
    ++exp_manager.checkpoint_callback_params.monitor="val_der" \
    ++exp_manager.checkpoint_callback_params.mode="min" \
    exp_manager.name=$exp_name
"""


def load_ssl_encoder(model: EncDecEESDModel, cfg: DictConfig):
    if cfg.get("init_from_ptl_ckpt", None) is not None and isinstance(cfg.init_from_ptl_ckpt, str):
        state_dict = torch.load(cfg.init_from_ptl_ckpt, map_location='cpu')['state_dict']
        logging.info(f"Loading encoder from PyTorch Lightning checkpoint: {cfg.init_from_ptl_ckpt}")
    elif cfg.get("init_from_ptl_ckpt", None) is not None and isinstance(cfg.init_from_ptl_ckpt, DictConfig):
        model.maybe_init_from_pretrained_checkpoint(cfg)
        return model
    else:
        logging.info("No model checkpoint or pretrained model specified for encoder initialization.")
        return model

    if isinstance(model.preprocessor, ConformerMultiLayerFeaturePreprocessor):
        logging.info("Loading encoder for ConformerMultiLayerFeaturePreprocessor.")
        encoder_state_dict = OrderedDict()
        for key, value in state_dict.items():
            if key.startswith('encoder.'):
                encoder_state_dict[f'preprocessor.feature_extractor.{key}'] = value
    else:
        encoder_state_dict = state_dict

    model.load_state_dict(encoder_state_dict, strict=False)
    logging.info("Loaded ssl encoder state dict.")

    return model


@hydra_runner(config_path="../conf/neural_diarizer", config_name="fastconformer_large.yaml")
def main(cfg):
    logging.info(f'Hydra config: {OmegaConf.to_yaml(cfg)}')
    trainer = pl.Trainer(**cfg.trainer)
    exp_manager(trainer, cfg.get("exp_manager", None))

    model = EncDecEESDModel(cfg=cfg.model, trainer=trainer)

    model = load_ssl_encoder(model, cfg)

    if cfg.model.get("freeze_encoder", False):
        logging.info("Freezing encoder weights.")
        model.encoder.freeze()

    trainer.fit(model)


if __name__ == '__main__':

    main()
