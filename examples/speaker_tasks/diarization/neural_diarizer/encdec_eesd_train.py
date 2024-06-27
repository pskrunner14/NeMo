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
import torch
import pytorch_lightning as pl
from omegaconf import OmegaConf, DictConfig, open_dict
from pytorch_lightning import seed_everything

from nemo.collections.asr.modules.multi_layer_feat import ConformerMultiLayerFeaturePreprocessor
from nemo.collections.asr.models.eesd_models_heh import EncDecEESDModel
from nemo.core.config import hydra_runner
from nemo.utils import logging
from nemo.utils.exp_manager import exp_manager
from nemo.core import adapter_mixins

from nemo.core.classes.common import typecheck
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

seed_everything(42)

def update_model_config_to_support_adapter(model_cfg):
    with open_dict(model_cfg):
        # Update encoder adapter compatible config
        adapter_metadata = adapter_mixins.get_registered_adapter(model_cfg.encoder._target_)
        if adapter_metadata is not None:
            model_cfg.encoder._target_ = adapter_metadata.adapter_class_path


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


def setup_adapters(model: EncDecEESDModel, cfg: DictConfig):
    # Setup adapters
    with open_dict(cfg.model.adapter):
        # Extract the name of the adapter (must be give for training)
        adapter_name = cfg.model.adapter.pop("adapter_name")
        adapter_type = cfg.model.adapter.pop("adapter_type")
        adapter_module_name = cfg.model.adapter.pop("adapter_module_name", None)
        adapter_state_dict_name = cfg.model.adapter.pop("adapter_state_dict_name", None)

        # Resolve the config of the specified `adapter_type`
        if adapter_type not in cfg.model.adapter.keys():
            raise ValueError(
                f"Adapter type ({adapter_type}) config could not be found. Adapter setup config - \n"
                f"{OmegaConf.to_yaml(cfg.model.adapter)}"
            )

        adapter_type_cfg = cfg.model.adapter[adapter_type]
        print(f"Found `{adapter_type}` config :\n" f"{OmegaConf.to_yaml(adapter_type_cfg)}")

        # Augment adapter name with module name, if not provided by user
        if adapter_module_name is not None and ':' not in adapter_name:
            adapter_name = f'{adapter_module_name}:{adapter_name}'

        # Extract the global adapter config, if provided
        adapter_global_cfg = cfg.model.adapter.pop(model.adapter_global_cfg_key, None)
        if adapter_global_cfg is not None:
            add_global_adapter_cfg(model, adapter_global_cfg)

    model.add_adapter(adapter_name, cfg=adapter_type_cfg)
    assert model.is_adapter_available()

    # Disable all other adapters, enable just the current adapter.
    model.set_enabled_adapters(enabled=False)  # disable all adapters prior to training
    model.set_enabled_adapters(adapter_name, enabled=True)  # enable just one adapter by name
    return adapter_state_dict_name


@hydra_runner(config_path="../conf/neural_diarizer", config_name="fastconformer_large.yaml")
def main(cfg):
    logging.info(f'Hydra config: {OmegaConf.to_yaml(cfg)}')
    trainer = pl.Trainer(**cfg.trainer)
    exp_log_dir = exp_manager(trainer, cfg.get("exp_manager", None))

    if "adapter" in cfg.model:
        update_model_config_to_support_adapter(cfg.model)

    model = EncDecEESDModel(cfg=cfg.model, trainer=trainer)

    model = load_ssl_encoder(model, cfg)

    if cfg.model.get("freeze_encoder", False):
        logging.info("Freezing encoder weights.")
        model.encoder.freeze()

    adapter_state_dict_name = None
    if "adapter" in cfg.model:
        adapter_state_dict_name = setup_adapters(model, cfg)
    
    trainer.fit(model)

    # Save the adapter state dict
    if adapter_state_dict_name is not None:
        state_path = exp_log_dir if exp_log_dir is not None else os.getcwd()
        ckpt_path = os.path.join(state_path, "checkpoints")
        if os.path.exists(ckpt_path):
            state_path = ckpt_path
        state_path = os.path.join(state_path, adapter_state_dict_name)

        # Save the adapter modules in a seperate file
        model.save_adapters(str(state_path))



if __name__ == '__main__':
    
    main()
