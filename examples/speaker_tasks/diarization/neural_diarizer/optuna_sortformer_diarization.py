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
"""

python $BASEPATH/neural_diarizer/sortformer_diarization.py \
    model_path=/path/to/sortformer_model.nemo \
    batch_size=4 \
    interpolated_scale=0.16 \
    save_tensor_images=True \
    tensor_image_dir=/path/to/tensor_image_dir \
    dataset_manifest=/path/to/diarization_path_to_manifest.json

"""
import pytorch_lightning as pl
from omegaconf import OmegaConf
from pytorch_lightning import seed_everything

from nemo.collections.asr.models import SortformerEncLabelModel
from nemo.core.config import hydra_runner
from nemo.collections.asr.metrics.der import score_labels

import os

from dataclasses import dataclass, is_dataclass
from tqdm import tqdm
from typing import Optional, Union, List, Tuple, Dict

from pyannote.core import Segment, Timeline
from nemo.collections.asr.parts.utils.vad_utils import binarization, filtering
from nemo.collections.asr.parts.utils.speaker_utils import audio_rttm_map as get_audio_rttm_map
from nemo.collections.asr.parts.utils.speaker_utils import (
labels_to_pyannote_object,
generate_diarization_output_lines,
rttm_to_labels,
)



import pytorch_lightning as pl
import torch
import logging
from omegaconf import OmegaConf
from nemo.core.config import hydra_runner

import optuna
import os
import tempfile
import time
import json
import subprocess
import logging


from sortformer_diarization import (
DiarizationConfig,
VadParams,
convert_pred_mat_to_segments,
get_uem_object,
)

seed_everything(42)
torch.backends.cudnn.deterministic = True


def optuna_suggest_params(vad_cfg, trial):
    vad_cfg.onset=trial.suggest_float("onset", 0.1, 0.9, step=0.01)
    vad_cfg.offset=trial.suggest_float("offset", 0.1, 0.9, step=0.01)
    vad_cfg.pad_onset=trial.suggest_float("pad_onset", 0.0, 0.25, step=0.01)
    vad_cfg.pad_offset=trial.suggest_float("pad_offset", 0.0, 0.25, step=0.01)
    vad_cfg.min_duration_on=trial.suggest_float("min_duration_on", 0.0, 0.25, step=0.01)
    vad_cfg.min_duration_off=trial.suggest_float("min_duration_off", 0.0, 0.25, step=0.01)
    return vad_cfg


def diarization_objective(
    trial, 
    vad_cfg, 
    temp_out_dir, 
    infer_audio_rttm_dict, 
    diar_model_preds_total_list, 
    collar: float=0.25, 
    ignore_overlap: bool=False
    ):
    with tempfile.TemporaryDirectory(dir=temp_out_dir, prefix="Diar_PostProcessing_") as local_temp_out_dir:
        if trial is not None:
            vad_cfg = optuna_suggest_params(vad_cfg, trial) 
        all_hyps, all_refs, all_uems = convert_pred_mat_to_segments(infer_audio_rttm_dict, 
                                                                    vad_cfg=vad_cfg, 
                                                                    batch_preds_list=diar_model_preds_total_list, 
                                                                    unit_10ms_frame_count=8,
                                                                    bypass_postprocessing=False)
        metric, mapping_dict, itemized_errors = score_labels(AUDIO_RTTM_MAP=infer_audio_rttm_dict, 
                                                            all_reference=all_refs, 
                                                            all_hypothesis=all_hyps, 
                                                            all_uem=all_uems, 
                                                            collar=collar, 
                                                            ignore_overlap=ignore_overlap
                                                            )
        der = abs(metric)
    return der
    

@hydra_runner(config_name="DiarizationConfig", schema=DiarizationConfig)
def main(cfg: DiarizationConfig) -> Union[DiarizationConfig]:

    for key in cfg:
        cfg[key] = None if cfg[key] == 'None' else cfg[key]

    if is_dataclass(cfg):
        cfg = OmegaConf.structured(cfg)

    if cfg.random_seed:
        pl.seed_everything(cfg.random_seed)

    if cfg.model_path is None and cfg.pretrained_name is None:
        raise ValueError("Both cfg.model_path and cfg.pretrained_name cannot be None!")
    if cfg.audio_dir is None and cfg.dataset_manifest is None:
        raise ValueError("Both cfg.audio_dir and cfg.dataset_manifest cannot be None!")

    # setup GPU
    torch.set_float32_matmul_precision(cfg.matmul_precision)
    if cfg.cuda is None:
        if torch.cuda.is_available():
            device = [0]  # use 0th CUDA device
            accelerator = 'gpu'
            map_location = torch.device('cuda:0')
        elif cfg.allow_mps and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
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

    if cfg.model_path.endswith(".ckpt"):
        diar_model = SortformerEncLabelModel.load_from_checkpoint(checkpoint_path=cfg.model_path, map_location=map_location, strict=False)
    elif cfg.model_path.endswith(".nemo"):
        diar_model = SortformerEncLabelModel.restore_from(restore_path=cfg.model_path, map_location=map_location)
    else:
        raise ValueError("cfg.model_path must end with.ckpt or.nemo!")
    diar_model._cfg.diarizer.out_dir = cfg.tensor_image_dir
    diar_model._cfg.test_ds.session_len_sec = cfg.session_len_sec
    trainer = pl.Trainer(devices=device, accelerator=accelerator)
    diar_model.set_trainer(trainer)
    if cfg.eval_mode:
        diar_model = diar_model.eval()
    diar_model._cfg.test_ds.manifest_filepath = cfg.dataset_manifest
    infer_audio_rttm_dict = get_audio_rttm_map(cfg.dataset_manifest)
    diar_model._cfg.test_ds.batch_size = cfg.batch_size
    diar_model.use_new_pil = cfg.use_new_pil
    
    # Force the model to use the designated hop length
    scale_n = len(diar_model.msdd_multiscale_args_dict['scale_dict'])
    diar_model.msdd_multiscale_args_dict['scale_dict'][scale_n-1] = (float(cfg.interpolated_scale), float(cfg.interpolated_scale/2))
    
    # Model setup for inference 
    diar_model._cfg.test_ds.num_workers = cfg.num_workers
    diar_model.setup_test_data(test_data_config=diar_model._cfg.test_ds)    
    diar_model.streaming_mode = cfg.streaming_mode
    diar_model.sortformer_diarizer.step_len = cfg.step_len
    diar_model.sortformer_diarizer.mem_len = cfg.mem_len
    diar_model.save_tensor_images = cfg.save_tensor_images
   
    # Check if the saved tensor exists:
    tensor_filename = os.path.basename(cfg.dataset_manifest).replace("manifest.", "").replace(".json", "")
    model_base_path = os.path.dirname(cfg.model_path)
    tensor_path = f"{model_base_path}/pred_tensors/{tensor_filename}.pt"
    if os.path.exists(tensor_path):
        logging.info(f"Loading the saved tensors from {tensor_path}...")
        diar_model_preds_total_list = torch.load(tensor_path)
    else:
        diar_model.test_batch()
        torch.save(diar_model.preds_total_list, tensor_path)
    # if temp_out_dir does not exist, create it:
    temp_out_dir = os.path.join(model_base_path, "temp_out_dir")
    if not os.path.exists(temp_out_dir):
        os.makedirs(temp_out_dir)
    
    vad_cfg = VadParams() 
    worker_function = lambda trial: diarization_objective(
        trial=trial,
        vad_cfg=vad_cfg,
        temp_out_dir=temp_out_dir,
        infer_audio_rttm_dict=infer_audio_rttm_dict, 
        diar_model_preds_total_list=diar_model_preds_total_list,
    )
    study = optuna.create_study(
        direction="minimize", 
        study_name=cfg.optuna_study_name, 
        storage=cfg.storage, 
        load_if_exists=True
    )
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)  # Setup the root logger.
    if cfg.output_log_file is not None:
        logger.addHandler(logging.FileHandler(cfg.output_log_file, mode="a"))
    logger.addHandler(logging.StreamHandler())
    optuna.logging.enable_propagation()  # Propagate logs to the root logger.
    study.optimize(worker_function, n_trials=cfg.optuna_n_trials)

if __name__ == '__main__':
    main()
