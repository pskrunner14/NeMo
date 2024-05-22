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
    session_len_sec=600 \
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
from typing import Optional, Union

from pyannote.core import Segment, Timeline
from nemo.collections.asr.parts.utils.vad_utils import binarization, filtering
from nemo.collections.asr.parts.utils.speaker_utils import audio_rttm_map as get_audio_rttm_map
from nemo.collections.asr.parts.utils.speaker_utils import (
labels_to_pyannote_object,
generate_cluster_labels,
rttm_to_labels,
get_overlap_range,
is_overlap,
)



import pytorch_lightning as pl
import torch
from omegaconf import OmegaConf
from nemo.core.config import hydra_runner
"""
Example of end-to-end diarization inference 
"""

seed_everything(42)

@dataclass
class DiarizationConfig:
    # Required configs
    model_path: Optional[str] = None  # Path to a .nemo file
    pretrained_name: Optional[str] = None  # Name of a pretrained model
    audio_dir: Optional[str] = None  # Path to a directory which contains audio files
    tensor_image_dir: Optional[str] = None  # Path to a directory which contains tensor images
    save_tensor_images: bool = False  # If True, saves tensor images to disk for debugging purposes
    dataset_manifest: Optional[str] = None  # Path to dataset's JSON manifest
    channel_selector: Optional[
        Union[int, str]
    ] = None  # Used to select a single channel from multichannel audio, or use average across channels
    audio_key: str = 'audio_filepath'  # Used to override the default audio key in dataset_manifest
    eval_config_yaml: Optional[str] = None  # Path to a yaml file of config of evaluation
    presort_manifest: bool = True  # Significant inference speedup on short-form data due to padding reduction
    interpolated_scale:float=0.16
    
    # General configs
    output_filename: Optional[str] = None
    session_len_sec: float = 60 # End-to-end diarization session length in seconds
    batch_size: int = 4
    num_workers: int = 0
    random_seed: Optional[int] = None  # seed number going to be used in seed_everything()
    
    # Streaming diarization configs
    streaming_mode: bool = True # If True, streaming diarization will be used. For long-form audio, set mem_len=step_len
    mem_len: int = 2000
    step_len: int = 2000

    # If `cuda` is a negative number, inference will be on CPU only.
    cuda: Optional[int] = None
    allow_mps: bool = False  # allow to select MPS device (Apple Silicon M-series GPU)
    amp: bool = False
    amp_dtype: str = "float16"  # can be set to "float16" or "bfloat16" when using amp
    matmul_precision: str = "highest"  # Literal["highest", "high", "medium"]
    audio_type: str = "wav"



@dataclass
class VadParams:
    window_length_in_sec: float = 0.15
    shift_length_in_sec: float = 0.01
    smoothing: str = False
    overlap: float = 0.5
    # onset: float = 0.725
    # offset: float = 0.55
    onset: float = 0.65
    offset: float = 0.55
    pad_onset: float = 0.05
    pad_offset: float = 0.05
    min_duration_on: float = 0.2
    min_duration_off: float = 0.2
    filter_speech_first: bool = True

def get_overlapping_list(source_range_list, target_range):
    out_range = []
    for line_str in source_range_list:
        _stt, _end, spk = line_str.split()
        s_range = (float(_stt), float(_end))
        if is_overlap(s_range, target_range):
            ovl_range = get_overlap_range(s_range, target_range)
            out_range.append(f"{ovl_range[0]} {ovl_range[1]} {spk}")
    return out_range

def timestamps_to_pyannote_object(timestamps, cluster_labels, uniq_id, audio_rttm_values, all_hypothesis, all_reference, all_uems):
    offset, dur = float(audio_rttm_values.get('offset', None)), float(audio_rttm_values.get('duration', None))
    if offset is not None:
        labels, lines = generate_cluster_labels(timestamps, cluster_labels, offset=offset)
    else:
        labels, lines = generate_cluster_labels(timestamps, cluster_labels)
    hypothesis = labels_to_pyannote_object(labels, uniq_name=uniq_id)
    all_hypothesis.append([uniq_id, hypothesis])
    rttm_file = audio_rttm_values.get('rttm_filepath', None)
    if rttm_file is not None and os.path.exists(rttm_file):
        uem_lines = [[offset, dur+offset]] 
        org_ref_labels = rttm_to_labels(rttm_file)
        ref_labels = org_ref_labels
        
        reference = labels_to_pyannote_object(ref_labels, uniq_name=uniq_id)
        uem_obj = get_uem_object(uem_lines, uniq_id=uniq_id)
        all_uems.append(uem_obj)
        all_reference.append([uniq_id, reference])
    return all_hypothesis, all_reference, all_uems

def ts_vad_post_processing(ts_vad_binary_vec, cfg_vad_params, hop_length: int=8):
    ts_vad_binary_frames = torch.repeat_interleave(ts_vad_binary_vec, hop_length)
    speech_segments = binarization(ts_vad_binary_frames, cfg_vad_params)
    speech_segments = filtering(speech_segments, cfg_vad_params)
    return speech_segments

def get_uem_object(uem_lines, uniq_id):
    """
    Generate pyannote timeline segments for uem file
    
     <UEM> file format
     UNIQ_SPEAKER_ID CHANNEL START_TIME END_TIME
     
    Args:
        uem_lines (list): list of session ID and start, end times.
    """
    timeline = Timeline(uri=uniq_id)
    for uem_stt_end in uem_lines:
        start_time, end_time = uem_stt_end 
        timeline.add(Segment(float(start_time), float(end_time)))
    return timeline


def convert_pred_mat_to_segments(
    audio_rttm_map_dict, 
    batch_preds: torch.Tensor, 
    offset: float, 
    hop_length:int = 5
    ):
    batch_pred_ts_segs, all_hypothesis, all_reference, all_uems = [], [], [], []
    thres_offset = {0: 0, 1: 0, 2: 0, 3: 0}
    for sample_idx, (uniq_id, audio_rttm_values) in enumerate(audio_rttm_map_dict.items()):
        spk_ts, timestamps, cluster_labels = [], [], []
        speaker_assign_mat = batch_preds[sample_idx]
        for spk_id in range(speaker_assign_mat.shape[-1]):
            cfg_vad_params = OmegaConf.structured(VadParams())
            cfg_vad_params.onset = cfg_vad_params.onset + thres_offset[spk_id]
            cfg_vad_params.offset = cfg_vad_params.offset + thres_offset[spk_id]
            ts_mat = ts_vad_post_processing(speaker_assign_mat[:, spk_id], cfg_vad_params, hop_length=8)
            ts_mat = ts_mat + offset
            ts_seg_list = ts_mat.tolist()
            spk_ts.append(ts_seg_list)
            cluster_labels.extend([ spk_id for _ in range(len(ts_seg_list))])
            timestamps.extend(ts_seg_list)
        all_hypothesis, all_reference, all_uems = timestamps_to_pyannote_object(timestamps, cluster_labels, uniq_id, audio_rttm_values, all_hypothesis, all_reference, all_uems)
        batch_pred_ts_segs.append(spk_ts) 
    return batch_pred_ts_segs, all_hypothesis, all_reference, all_uems



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

    diar_model = SortformerEncLabelModel.load_from_checkpoint(checkpoint_path=cfg.model_path, map_location=map_location)
    diar_model._cfg.diarizer.out_dir = cfg.tensor_image_dir
    diar_model._cfg.test_ds.session_len_sec = cfg.session_len_sec
    trainer = pl.Trainer(devices=device, accelerator=accelerator)
    diar_model.set_trainer(trainer)
    diar_model._cfg.test_ds.manifest_filepath = cfg.dataset_manifest
    infer_audio_rttm_dict = get_audio_rttm_map(cfg.dataset_manifest)
    diar_model._cfg.test_ds.batch_size = cfg.batch_size
    
    # Force the model to use the designated hop length
    scale_n = len(diar_model.msdd_multiscale_args_dict['scale_dict'])
    diar_model.msdd_multiscale_args_dict['scale_dict'][scale_n-1] = (float(cfg.interpolated_scale), float(cfg.interpolated_scale/2))
    
    # Model setup for inference 
    diar_model.setup_test_data(test_data_config=diar_model._cfg.test_ds)    
    diar_model.streaming_mode = cfg.streaming_mode
    diar_model.sortformer_diarizer.step_len = cfg.step_len
    diar_model.sortformer_diarizer.mem_len = cfg.mem_len
    diar_model.save_tensor_images = cfg.save_tensor_images
    diar_model.test_batch()
    
    # Evaluation
    output_list, all_hyps, all_refs, all_uems = convert_pred_mat_to_segments(infer_audio_rttm_dict, 
                                                                             batch_preds=diar_model.preds_total, 
                                                                             offset=0, 
                                                                             hop_length=5)
    metric, mapping_dict, itemized_errors = score_labels(AUDIO_RTTM_MAP=infer_audio_rttm_dict, 
                                                         all_reference=all_refs, 
                                                         all_hypothesis=all_hyps, 
                                                         all_uem=all_uems, 
                                                         collar=0.25, 
                                                         ignore_overlap=False)
    print("VadParams:", VadParams())

if __name__ == '__main__':
    main()
