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

from dataclasses import dataclass
from omegaconf import OmegaConf
import pytorch_lightning as pl

from typing import Optional
import torch
from nemo.collections.asr.models.eesd_models_heh import EncDecEESDModel
from nemo.core.config import hydra_runner
from nemo.utils import logging
from nemo.utils.exp_manager import exp_manager
from nemo.core.classes.common import typecheck
typecheck.set_typecheck_enabled(enabled=False)

"""
CUDA_VISIBLE_DEVICES="1" python eval_eesd.py \
    manifest_filepath=$dev_manifests \
    model_filepath=$model_filepath \
    batch_size=$batch_size \
    num_workers=$num_workers \
    output_dir="./outputs" \
    threshold=0.5
"""

@dataclass
class EESDEvalConfig:
    model_filepath: str = ""
    manifest_filepath: str = ""
    uem_filepath: str = ""
    collar: float = 0.25
    threshold: float = 0.5
    skip_overlap: bool = False
    batch_size: int = 1
    num_workers: int = 4
    random_seed: Optional[int] = None
    cuda: Optional[int] = None
    matmul_precision: str = "highest"  # Literal["highest", "high", "medium"]
    use_der_metric: bool = True
    output_dir: Optional[str] = "outputs/"



def load_eesd_model(model_filepath):
    model_filepath = str(model_filepath).replace('\\', '')
    if model_filepath.endswith('.nemo'):
        model = EncDecEESDModel.restore_from(model_filepath, map_location='cpu')
    elif model_filepath.endswith('.ckpt'):
        model = EncDecEESDModel.load_from_checkpoint(model_filepath, map_location='cpu')
    else:
        raise ValueError(f"Unsupported model file format: {model_filepath}")
    return model



@hydra_runner(config_name="EESDEvalConfig", schema=EESDEvalConfig)
def main(cfg):
    logging.info(f'Hydra config: {OmegaConf.to_yaml(cfg)}')
    if cfg.random_seed:
        pl.seed_everything(cfg.random_seed)

    torch.set_float32_matmul_precision(cfg.matmul_precision)
    if cfg.cuda is None:
        if torch.cuda.is_available():
            device = [0]  # use 0th CUDA device
            accelerator = 'gpu'
            map_location = torch.device('cuda:0')
        elif cfg.allow_mps and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            logging.warning(
                "MPS device (Apple Silicon M-series GPU) support is experimental."
                " Env variable `PYTORCH_ENABLE_MPS_FALLBACK=1` should be set in most cases to avoid failures."
            )
            device = [0]
            accelerator = 'mps'
            map_location = torch.device('mps')
        else:
            device = 1
            accelerator = 'cpu'
            map_location = torch.device('cpu')
    else:
        device = [cfg.cuda]
        accelerator = 'gpu'
        map_location = torch.device(f'cuda:{cfg.cuda}')

    logging.info(f"Inference will be done on device: {map_location}")

    model = load_eesd_model(cfg.model_filepath)

    trainer = pl.Trainer(devices=device, accelerator=accelerator)
    model.set_trainer(trainer)
    model = model.eval()

    dataloader = model.get_inference_dataloader(cfg)

    trainer.test(model, dataloaders=dataloader)



if __name__ == '__main__':
    
    main()
