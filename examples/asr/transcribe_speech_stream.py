# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
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

import contextlib
import glob
import json
import os
import re

import matplotlib.pyplot as plt
import numpy as np
from nemo.collections.asr.parts.utils.manifest_utils import get_ctm_line

from dataclasses import dataclass, is_dataclass
from tempfile import NamedTemporaryFile
from typing import List, Optional, Union

import pytorch_lightning as pl
import torch
from omegaconf import OmegaConf, open_dict

from nemo.collections.asr.models import EncDecCTCModel, EncDecHybridRNNTCTCModel, EncDecMultiTaskModel, MSEncDecMultiTaskModel
from nemo.collections.asr.modules.conformer_encoder import ConformerChangeConfig
from nemo.collections.asr.parts.submodules.ctc_decoding import CTCDecodingConfig
from nemo.collections.asr.parts.submodules.multitask_decoding import MultiTaskDecoding, MultiTaskDecodingConfig
from nemo.collections.asr.parts.submodules.rnnt_decoding import RNNTDecodingConfig
from nemo.collections.asr.parts.utils.eval_utils import cal_write_wer
from nemo.collections.asr.parts.utils.rnnt_utils import Hypothesis
from nemo.collections.asr.parts.utils.transcribe_utils import (
    compute_output_filename,
    prepare_audio_data,
    read_and_maybe_sort_manifest,
    restore_transcription_order,
    setup_model,
    transcribe_partial_audio,
    write_transcription,
)
from nemo.collections.common.parts.preprocessing.manifest import get_full_path
from nemo.core.config import hydra_runner
from nemo.utils import logging

"""
Transcribe audio file on a single CPU/GPU. Useful for transcription of moderate amounts of audio data.

# Arguments
  model_path: path to .nemo ASR checkpoint
  pretrained_name: name of pretrained ASR model (from NGC registry)
  audio_dir: path to directory with audio files
  dataset_manifest: path to dataset JSON manifest file (in NeMo format)

  compute_timestamps: Bool to request greedy time stamp information (if the model supports it)
  compute_langs: Bool to request language ID information (if the model supports it)

  (Optionally: You can limit the type of timestamp computations using below overrides)
  ctc_decoding.ctc_timestamp_type="all"  # (default all, can be [all, char, word])
  rnnt_decoding.rnnt_timestamp_type="all"  # (default all, can be [all, char, word])

  (Optionally: You can limit the type of timestamp computations using below overrides)
  ctc_decoding.ctc_timestamp_type="all"  # (default all, can be [all, char, word])
  rnnt_decoding.rnnt_timestamp_type="all"  # (default all, can be [all, char, word])

  output_filename: Output filename where the transcriptions will be written
  batch_size: batch size during inference

  cuda: Optional int to enable or disable execution of model on certain CUDA device.
  allow_mps: Bool to allow using MPS (Apple Silicon M-series GPU) device if available
  amp: Bool to decide if Automatic Mixed Precision should be used during inference
  audio_type: Str filetype of the audio. Supported = wav, flac, mp3

  overwrite_transcripts: Bool which when set allows repeated transcriptions to overwrite previous results.

  ctc_decoding: Decoding sub-config for CTC. Refer to documentation for specific values.
  rnnt_decoding: Decoding sub-config for RNNT. Refer to documentation for specific values.

  calculate_wer: Bool to decide whether to calculate wer/cer at end of this script
  clean_groundtruth_text: Bool to clean groundtruth text
  langid: Str used for convert_num_to_words during groundtruth cleaning
  use_cer: Bool to use Character Error Rate (CER)  or Word Error Rate (WER)

# Usage
ASR model can be specified by either "model_path" or "pretrained_name".
Data for transcription can be defined with either "audio_dir" or "dataset_manifest".
append_pred - optional. Allows you to add more than one prediction to an existing .json
pred_name_postfix - optional. The name you want to be written for the current model
Results are returned in a JSON manifest file.

python transcribe_speech.py \
    model_path=null \
    pretrained_name=null \
    audio_dir="<remove or path to folder of audio files>" \
    dataset_manifest="<remove or path to manifest>" \
    output_filename="<remove or specify output filename>" \
    clean_groundtruth_text=True \
    langid='en' \
    batch_size=32 \
    compute_timestamps=False \
    compute_langs=False \
    cuda=0 \
    amp=True \
    append_pred=False \
    pred_name_postfix="<remove or use another model name for output filename>"
"""


@dataclass
class ModelChangeConfig:

    # Sub-config for changes specific to the Conformer Encoder
    conformer: ConformerChangeConfig = ConformerChangeConfig()


@dataclass
class TranscriptionConfig:
    # Required configs
    model_path: Optional[str] = None  # Path to a .nemo file
    asr_model_path: Optional[str] = model_path  # Path to a .nemo file
    diar_pred_model_path: Optional[str] = None  # Path to a diarization model
    pretrained_name: Optional[str] = None  # Name of a pretrained model
    audio_dir: Optional[str] = None  # Path to a directory which contains audio files
    dataset_manifest: Optional[str] = None  # Path to dataset's JSON manifest
    channel_selector: Optional[
        Union[int, str]
    ] = None  # Used to select a single channel from multichannel audio, or use average across channels
    audio_key: str = 'audio_filepath'  # Used to override the default audio key in dataset_manifest
    eval_config_yaml: Optional[str] = None  # Path to a yaml file of config of evaluation
    presort_manifest: bool = True  # Significant inference speedup on short-form data due to padding reduction

    # General configs
    output_filename: Optional[str] = None
    batch_size: int = 32
    num_workers: int = 0
    append_pred: bool = False  # Sets mode of work, if True it will add new field transcriptions.
    pred_name_postfix: Optional[str] = None  # If you need to use another model name, rather than standard one.
    random_seed: Optional[int] = None  # seed number going to be used in seed_everything()

    # Set to True to output greedy timestamp information (only supported models)
    compute_timestamps: bool = False
    # set to True if need to return full alignment information
    preserve_alignment: bool = False

    # Set to True to output language ID information
    compute_langs: bool = False

    # Set `cuda` to int to define CUDA device. If 'None', will look for CUDA
    # device anyway, and do inference on CPU only if CUDA device is not found.
    # If `cuda` is a negative number, inference will be on CPU only.
    cuda: Optional[int] = None
    allow_mps: bool = False  # allow to select MPS device (Apple Silicon M-series GPU)
    amp: bool = False
    amp_dtype: str = "float16"  # can be set to "float16" or "bfloat16" when using amp
    matmul_precision: str = "highest"  # Literal["highest", "high", "medium"]
    audio_type: str = "wav"

    # Recompute model transcription, even if the output folder exists with scores.
    overwrite_transcripts: bool = True

    # Decoding strategy for CTC models
    ctc_decoding: CTCDecodingConfig = CTCDecodingConfig()

    # Decoding strategy for RNNT models
    rnnt_decoding: RNNTDecodingConfig = RNNTDecodingConfig(fused_batch_size=-1)

    # Decoding strategy for AED models
    multitask_decoding: MultiTaskDecodingConfig = MultiTaskDecodingConfig()

    # decoder type: ctc or rnnt, can be used to switch between CTC and RNNT decoder for Hybrid RNNT/CTC models
    decoder_type: Optional[str] = None
    # att_context_size can be set for cache-aware streaming models with multiple look-aheads
    att_context_size: Optional[list] = None

    # Use this for model-specific changes before transcription
    model_change: ModelChangeConfig = ModelChangeConfig()

    # Config for word / character error rate calculation
    calculate_wer: bool = True
    clean_groundtruth_text: bool = False
    langid: str = "en"  # specify this for convert_num_to_words step in groundtruth cleaning
    use_cer: bool = False

    # can be set to True to return list of transcriptions instead of the config
    # if True, will also skip writing anything to the output file
    return_transcriptions: bool = False

    # Set to False to return text instead of hypotheses from the transcribe function, so as to save memory
    return_hypotheses: bool = True

    # key for groundtruth text in manifest
    gt_text_attr_name: str = "text"
    gt_lang_attr_name: str = "lang"

    # Use model's transcribe() function instead of transcribe_partial_audio() by default
    # Only use transcribe_partial_audio() when the audio is too long to fit in memory
    # Your manifest input should have `offset` field to use transcribe_partial_audio()
    allow_partial_transcribe: bool = False
    extract_nbest: bool = False  # Extract n-best hypotheses from the model

def min_max_token_wise_normalization(matrix):
    row_min = matrix.min(dim=1, keepdim=True).values
    row_max = matrix.max(dim=1, keepdim=True).values

    # Apply Min-Max normalization to each row
    normalized_matrix = (matrix - row_min) / (row_max - row_min)
    return normalized_matrix

def get_attn_frames(idx, dp_frame_inds, matrix, l_margin, r_margin, last_spk_frame_idx, thres=0.5):
    left = max(0, dp_frame_inds[0] - l_margin, last_spk_frame_idx+1)
    right = min(matrix.size(1) - 1, dp_frame_inds[-1] + r_margin)
    target_inds = torch.arange(left, right + 1)
    valid_mask = matrix[idx, target_inds] > thres
    if len(valid_mask) == 0 or torch.all(valid_mask).item() == False:
        frame_inds = dp_frame_inds
    else:
        frame_inds = target_inds[valid_mask]
    attn_vals = matrix[idx, frame_inds]
    return frame_inds, attn_vals

def get_word_alignment(
    matrix, 
    path_2d, 
    token_seq, 
    feat_frame_len_sec=0.08, 
    sil_token=None, 
    spl_token_pattern=r'<\|spltoken\d+\|>',
    decimal=2,
    l_margin: int = 8,
    r_margin: int = 0,
    max_spks: int = 10,
    is_multispeaker: bool = False,
    ):
    matrix = torch.tensor(matrix)
    attn_map = torch.zeros_like(matrix)
    word_seq_dict = {}
    word_count = -1
    spk = 'unknown'
    word_open = False
    rn_matrix = min_max_token_wise_normalization(matrix)
    spk_wise_last_frame_inds = { spl_token_pattern.replace('\\', '').replace('d+', f"{idx}"): -1 for idx in range(max_spks)}
    spk_wise_last_frame_inds[spk] = -1
    for idx, tok in enumerate(token_seq):
        dp_frame_inds = np.sort(path_2d[path_2d[:, 0] == idx, 1])
        is_spl_token = contains_pattern = bool(re.search(spl_token_pattern, tok))
        if is_spl_token:
            spk = str(tok)
        frame_inds, attn_vals = get_attn_frames(idx, dp_frame_inds, rn_matrix, l_margin, r_margin, last_spk_frame_idx=spk_wise_last_frame_inds[spk])
        # print(f"idx:{idx}, dp_frame_inds: {dp_frame_inds}, frame_inds: {frame_inds}, attn_vals: {attn_vals} spk: {spk}, spk_wise_last_frame_inds[spk] {spk_wise_last_frame_inds[spk]}, {frame_inds[-1]}, token:{tok}")
        attn_map[idx, frame_inds] = attn_vals

        if not (sil_token in tok and len(tok) == 1) and sil_token in tok:
            word_count += 1
            word_seq_dict[word_count] = {'word':[tok], 
                                         'start': round(feat_frame_len_sec * float(frame_inds[0]), decimal), 
                                         'end': round(feat_frame_len_sec * float(frame_inds[-1] + 1), decimal),
                                         'spk': spk}
            word_open = True
            spk_wise_last_frame_inds[spk] = frame_inds[-1].item()
        elif not is_spl_token and sil_token not in tok and word_open:
            word_seq_dict[word_count]['word'].append(tok)
            word_seq_dict[word_count]['end'] = round(feat_frame_len_sec * float(frame_inds[-1] + 1), decimal)
            spk_wise_last_frame_inds[spk] = frame_inds[-1].item()
        else:
            word_open = False
    # Last item handler
    if word_open and word_seq_dict[word_count]['end'] is None:
        word_seq_dict[word_count]['end'] = round(feat_frame_len_sec * float(frame_inds[-1] + 1), decimal)
        word_open = False
        
    word_alignment = []
    for word_count, word_info in word_seq_dict.items():
        word =''.join(word_info['word']).replace(sil_token, '')
        word_alignment.append([word,
                               word_info['start'], 
                               word_info['end'], 
                               word_info['spk']]
                              )
    return word_alignment, attn_map
        

def backtrack(direction, rows, cols):
    # Backtrack to find the path
    path = []
    i, j = rows - 1, cols - 1
    while i > 0 or j > 0:
        path.append((i, j))
        if direction[i, j] == -1: # 'left':
            j -= 1
        else:
            i -= 1
    path.append((0, 0))  # add the starting point
    path.reverse()
    return path

def run_dp_for_alignment(matrix):
    # matrix = torch.tensor(matrix)
    rows, cols = matrix.shape
    # Initialize DP table and direction table
    dp = torch.zeros_like(matrix, dtype=torch.float32)
    direction = torch.zeros((rows, cols), dtype=torch.int8)

    # Fill the DP table
    for i in range(rows):
        for j in range(cols):
            if i == 0 and j == 0:
                dp[i, j] = matrix[i, j]
            elif i == 0:
                dp[i, j] = dp[i, j-1] + matrix[i, j]
                direction[i, j] = -1 # 'left'
            elif j == 0:
                dp[i, j] = dp[i-1, j] + matrix[i, j]
                direction[i, j] = 1 # 'above'
            else:
                if dp[i-1, j] > dp[i, j-1]:
                    dp[i, j] = dp[i-1, j] + matrix[i, j]
                    direction[i, j] =  1 # 'above'
                else:
                    dp[i, j] = dp[i, j-1] + matrix[i, j]
                    direction[i, j] = -1 # 'left'

    path = backtrack(direction, rows, cols)
    path_2d = np.array(path)
    return path_2d

def save_plot_imgs(path_2d, layer_info, matrix, filename, tokens, FS=1.5):
    # Set the y-axis ticks and labels based on the tokens
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    for axis in ['top','bottom','left','right']:
        ax1.spines[axis].set_linewidth(0.5)
        
    plt.plot(path_2d[:, 1], path_2d[:, 0], marker='o', color='black', linewidth=1.0, alpha=0.25, markersize=1)
    plt.imshow(matrix, cmap='jet', interpolation='none')
    plt.text(x=100, y=50, s=f'Layer {layer_info}', color='white', fontsize=12, 
         bbox=dict(facecolor='red', alpha=0.5))  # Adjust x, y, and text properties
    y_ticks = np.arange(len(tokens))  # Assuming `tokens` is the list of strings for each y-tick
    plt.yticks(y_ticks, tokens, fontsize=FS)
    plt.savefig(filename, bbox_inches='tight', pad_inches=0.1, dpi=600)
    print(f"Saved the image to | {filename}")
    plt.clf() 
    plt.close()


def save_attn_imgs(
    cfg,
    asr_model, 
    transcriptions, 
    num_layers: int, 
    tensor_attn_dict, 
    tokens, 
    idx: int = 0,
    pre_tokens_len=5,
    end_cut = 2,
    layer_stt = 13,
    layer_end= 16,
    is_multispeaker=True,
    ): 
    """
    https://github.com/openai/whisper/blob/main/notebooks/Multilingual_ASR.ipynb
    """
    if cfg.get("diar_pred_model_path", None) is None:
        is_multispeaker = False
    # Skip the end token because end token does not generate an attention step.
    token_seq = asr_model.tokenizer.ids_to_tokens(transcriptions[0].y_sequence.cpu().numpy())[pre_tokens_len:-1]
    token_seq = np.array(token_seq)
    
    token_ids = transcriptions[0].y_sequence[pre_tokens_len:]
    (spl_stt, spl_end) = (13, 30)
    spk_token_inds = torch.nonzero((token_ids >= spl_stt) & (token_ids <= spl_end)).squeeze()
    
    # if attn_block is not None:
    avg_list = []
    for k in range(layer_stt, layer_end+1):
        attn_block = tensor_attn_dict[k]
        if layer_stt <= k <= layer_end:
            avg_list.append( torch.mean( attn_block[idx, :, 0, :, :], dim=0).unsqueeze(0) )
        
        output_dir = "/home/taejinp/Downloads/multispeaker_canary_imshow/"
        _img = attn_block[idx, :, 0, :, :]
        img = torch.mean(_img, dim=0).cpu()
        non_spk_mask = torch.ones(img.size(0), dtype=bool)
        all_mask = torch.ones(img.size(0), dtype=bool)
        non_spk_mask[spk_token_inds] = False
        
        # Convert tensor to numpy array
        img = img.numpy()
        # if False:
        if True:
            # for (token_type, mask) in [('Words', non_spk_mask), ('Spks', spk_token_inds), ('All', all_mask)]:
            for (token_type, mask) in [('All', all_mask)]:
                if not is_multispeaker and token_type == 'Spks':
                    continue
                filename = os.path.join(output_dir, f'image_layer-{k+1}th-{token_type}_batch-{idx}.png')
                matrix = torch.tensor(img[mask, :-end_cut])
                path_2d = run_dp_for_alignment(matrix)
                save_plot_imgs(path_2d=path_2d, layer_info=f"layer-{k+1}th", matrix=matrix, filename=filename, tokens=token_seq[mask])

    # for (token_type, mask) in [('Words', non_spk_mask), ('Spks', spk_token_inds), ('All', all_mask)]:
    for (token_type, mask) in [('All', all_mask)]:
        if not is_multispeaker and token_type == 'Spks':
            continue
        layer_avg_attn_block = torch.mean(torch.vstack(avg_list), dim=0).cpu().numpy()
        matrix = torch.tensor(layer_avg_attn_block[mask, :-end_cut])
        filename = os.path.join(output_dir, f'image_layerAVG-{token_type}-stt{layer_stt}_end{layer_end}.png')
        path_2d = run_dp_for_alignment(matrix)
        save_plot_imgs(path_2d=path_2d, layer_info=f"layer-avg-stt{layer_stt}_end{layer_end}", matrix=matrix, filename=filename, tokens=token_seq[mask])
    matrix_all = layer_avg_attn_block[:, :-end_cut]
    path_2d_all = run_dp_for_alignment(matrix)
    sil_token = asr_model.tokenizer.ids_to_tokens([31])[0]
    word_alignment, attn_map = get_word_alignment(matrix=matrix_all, path_2d=path_2d_all, token_seq=token_seq, is_multispeaker=is_multispeaker, sil_token=sil_token)
    thr_filename = os.path.join(output_dir, f'image_THRattn-{token_type}-stt{layer_stt}_end{layer_end}.png')
    save_plot_imgs(path_2d=path_2d, layer_info=f"layer-avg-stt{layer_stt}_end{layer_end}", matrix=attn_map, filename=thr_filename, tokens=token_seq[mask])
    return word_alignment

def write_ctm(filepaths, output_filename, word_alignments, decimal=2, str_pattern=r'<\|spltoken\d+\|>'):
    str_pattern = str_pattern.replace("\\", '')
    left_str, right_str = str_pattern.split('d+')[0], str_pattern.split('d+')[1]
    
    filepath_list = open(filepaths).readlines()
    ctm_output_list, rttm_output_list = [], []
    for idx, json_string in enumerate(filepath_list):
        meta_dict = json.loads(json_string)
        uniq_id = os.path.basename(meta_dict['audio_filepath']).split('.')[0]
        output_folder = os.path.dirname(output_filename)
        ctm_filename = os.path.join(output_folder, f"{uniq_id}.ctm")
        rttm_filename = os.path.join(output_folder, f"{uniq_id}.rttm")
        for word_line in word_alignments:
            word = word_line[0]
            dur = round(word_line[2] - word_line[1], decimal) - 0.01
            start = round(word_line[1], decimal)
            spk_str = word_line[3]
            if 'unknown' in spk_str or spk_str is None:
                spk_token_int = 0
            else: 
                spk_token_int = int(spk_str.replace(left_str,'').replace(right_str, ''))
            rttm_line = f"SPEAKER {uniq_id} {spk_token_int} {start:.3f} {dur:.3f} <NA> <NA> {spk_str} <NA> <NA> {word}\n"
            ctm_line = get_ctm_line(
                    source=uniq_id + f"_spk{spk_token_int}",
                    channel=f"{spk_token_int}",
                    start_time=start,
                    duration=dur,
                    token=word,
                    conf=1.0,
                    type_of_token="lex",
                    speaker=f"{spk_str}",
            )
            ctm_output_list.append(ctm_line)
            rttm_output_list.append(rttm_line)
    with open(ctm_filename, 'w') as f:
        f.write(''.join(ctm_output_list))
    with open(rttm_filename, 'w') as f:
        f.write(''.join(rttm_output_list))
    
@hydra_runner(config_name="TranscriptionConfig", schema=TranscriptionConfig)
def main(cfg: TranscriptionConfig) -> Union[TranscriptionConfig, List[Hypothesis]]:
    logging.info(f'Hydra config: {OmegaConf.to_yaml(cfg)}')
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

    # Load augmentor from exteranl yaml file which contains eval info, could be extend to other feature such VAD, P&C
    augmentor = None
    if cfg.eval_config_yaml:
        eval_config = OmegaConf.load(cfg.eval_config_yaml)
        augmentor = eval_config.test_ds.get("augmentor")
        logging.info(f"Will apply on-the-fly augmentation on samples during transcription: {augmentor} ")

    # setup GPU
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

    # cfg.asr_model_path = cfg.model_path
    asr_model, model_name = setup_model(cfg, map_location)

    trainer = pl.Trainer(devices=device, accelerator=accelerator)
    asr_model.set_trainer(trainer)
    asr_model = asr_model.eval()

    
    if cfg.get('diar_pred_model_path', None) is not None:
        asr_model.setup_diar_pred_model(cfg.diar_pred_model_path, map_location)

    # we will adjust this flag if the model does not support it
    compute_timestamps = cfg.compute_timestamps
    compute_langs = cfg.compute_langs
    # has to be True if timestamps are required
    preserve_alignment = True if cfg.compute_timestamps else cfg.preserve_alignment

    # Check whether model and decoder type match
    if isinstance(asr_model, EncDecCTCModel):
        if cfg.decoder_type and cfg.decoder_type != 'ctc':
            raise ValueError('CTC model only support ctc decoding!')
    elif isinstance(asr_model, EncDecHybridRNNTCTCModel):
        if cfg.decoder_type and cfg.decoder_type not in ['ctc', 'rnnt']:
            raise ValueError('Hybrid model only support ctc or rnnt decoding!')
    else:  # rnnt model, there could be other models needs to be addressed.
        if cfg.decoder_type and cfg.decoder_type != 'rnnt':
            raise ValueError('RNNT model only support rnnt decoding!')

    if cfg.decoder_type and hasattr(asr_model.encoder, 'set_default_att_context_size'):
        asr_model.encoder.set_default_att_context_size(cfg.att_context_size)

    # Setup decoding strategy
    if hasattr(asr_model, 'change_decoding_strategy') and hasattr(asr_model, 'decoding'):
        if isinstance(asr_model.decoding, MultiTaskDecoding):
            cfg.multitask_decoding.compute_langs = cfg.compute_langs
            cfg.multitask_decoding.preserve_alignments = cfg.preserve_alignment
            if cfg.extract_nbest:
                cfg.multitask_decoding.beam.return_best_hypothesis = False
                cfg.return_hypotheses = True
            asr_model.change_decoding_strategy(cfg.multitask_decoding)
        elif cfg.decoder_type is not None:
            # TODO: Support compute_langs in CTC eventually
            if cfg.compute_langs and cfg.decoder_type == 'ctc':
                raise ValueError("CTC models do not support `compute_langs` at the moment")

            decoding_cfg = cfg.rnnt_decoding if cfg.decoder_type == 'rnnt' else cfg.ctc_decoding
            if cfg.extract_nbest:
                decoding_cfg.beam.return_best_hypothesis = False
                cfg.return_hypotheses = True
            decoding_cfg.compute_timestamps = cfg.compute_timestamps  # both ctc and rnnt support it
            if 'preserve_alignments' in decoding_cfg:
                decoding_cfg.preserve_alignments = preserve_alignment
            if 'compute_langs' in decoding_cfg:
                decoding_cfg.compute_langs = cfg.compute_langs
            if hasattr(asr_model, 'cur_decoder'):
                asr_model.change_decoding_strategy(decoding_cfg, decoder_type=cfg.decoder_type)
            else:
                asr_model.change_decoding_strategy(decoding_cfg)

        # Check if ctc or rnnt model
        elif hasattr(asr_model, 'joint'):  # RNNT model
            if cfg.extract_nbest:
                cfg.rnnt_decoding.beam.return_best_hypothesis = False
                cfg.return_hypotheses = True
            cfg.rnnt_decoding.fused_batch_size = -1
            cfg.rnnt_decoding.compute_timestamps = cfg.compute_timestamps
            cfg.rnnt_decoding.compute_langs = cfg.compute_langs
            if 'preserve_alignments' in cfg.rnnt_decoding:
                cfg.rnnt_decoding.preserve_alignments = preserve_alignment

            asr_model.change_decoding_strategy(cfg.rnnt_decoding)
        else:
            if cfg.compute_langs:
                raise ValueError("CTC models do not support `compute_langs` at the moment.")
            cfg.ctc_decoding.compute_timestamps = cfg.compute_timestamps
            if cfg.extract_nbest:
                cfg.ctc_decoding.beam.return_best_hypothesis = False
                cfg.return_hypotheses = True

            asr_model.change_decoding_strategy(cfg.ctc_decoding)

    # Setup decoding config based on model type and decoder_type
    with open_dict(cfg):
        if isinstance(asr_model, EncDecCTCModel) or (
            isinstance(asr_model, EncDecHybridRNNTCTCModel) and cfg.decoder_type == "ctc"
        ):
            cfg.decoding = cfg.ctc_decoding
        elif isinstance(asr_model.decoding, MultiTaskDecoding):
            cfg.decoding = cfg.multitask_decoding
        else:
            cfg.decoding = cfg.rnnt_decoding

    remove_path_after_done = None
    if isinstance(asr_model, EncDecMultiTaskModel) or isinstance(asr_model, MSEncDecMultiTaskModel):
        # Special case for EncDecMultiTaskModel, where the input manifest is directly passed into the model's transcribe() function
        partial_audio = False
        if cfg.audio_dir is not None and not cfg.append_pred:
            filepaths = list(glob.glob(os.path.join(cfg.audio_dir, f"**/*.{cfg.audio_type}"), recursive=True))
        else:
            assert cfg.dataset_manifest is not None
            if cfg.presort_manifest:
                with NamedTemporaryFile("w", suffix=".json", delete=False) as f:
                    for item in read_and_maybe_sort_manifest(cfg.dataset_manifest, try_sort=True):
                        item["audio_filepath"] = get_full_path(item["audio_filepath"], cfg.dataset_manifest)
                        print(json.dumps(item), file=f)
                    cfg.dataset_manifest = f.name
                    remove_path_after_done = f.name
            filepaths = cfg.dataset_manifest
    else:
        # prepare audio filepaths and decide wether it's partial audio
        filepaths, partial_audio = prepare_audio_data(cfg)

    if not cfg.allow_partial_transcribe:
        # by defatul, use model's transcribe() function, unless partial audio is required
        partial_audio = False

    # setup AMP (optional)
    if cfg.amp and torch.cuda.is_available() and hasattr(torch.cuda, 'amp') and hasattr(torch.cuda.amp, 'autocast'):
        logging.info("AMP enabled!\n")
        autocast = torch.cuda.amp.autocast
    else:

        @contextlib.contextmanager
        def autocast(dtype=None):
            yield

    # Compute output filename
    cfg = compute_output_filename(cfg, model_name)

    # if transcripts should not be overwritten, and already exists, skip re-transcription step and return
    if not cfg.return_transcriptions and not cfg.overwrite_transcripts and os.path.exists(cfg.output_filename):
        logging.info(
            f"Previous transcripts found at {cfg.output_filename}, and flag `overwrite_transcripts`"
            f"is {cfg.overwrite_transcripts}. Returning without re-transcribing text."
        )
        return cfg

    # transcribe audio

    amp_dtype = torch.float16 if cfg.amp_dtype == "float16" else torch.bfloat16
    attn_block = None
    with autocast(dtype=amp_dtype):
        with torch.no_grad():
            if partial_audio:
                transcriptions = transcribe_partial_audio(
                    asr_model=asr_model,
                    path2manifest=cfg.dataset_manifest,
                    batch_size=cfg.batch_size,
                    num_workers=cfg.num_workers,
                    return_hypotheses=cfg.return_hypotheses,
                    channel_selector=cfg.channel_selector,
                    augmentor=augmentor,
                    decoder_type=cfg.decoder_type,
                )
            else:
                override_cfg = asr_model.get_transcribe_config()
                override_cfg.batch_size = cfg.batch_size
                override_cfg.num_workers = cfg.num_workers
                override_cfg.return_hypotheses = cfg.return_hypotheses
                override_cfg.channel_selector = cfg.channel_selector
                override_cfg.augmentor = augmentor
                override_cfg.text_field = cfg.gt_text_attr_name
                override_cfg.lang_field = cfg.gt_lang_attr_name
                transcriptions = asr_model.transcribe(audio=filepaths, override_config=override_cfg,)
                attn_layer_dict = asr_model.transf_decoder._decoder.last_layer_attention_saved_list
                num_layers = len(attn_layer_dict)
                # attn_block = torch.vstack(attn_list[1:])
                tensor_attn_dict = {}
                for i, attn_list in attn_layer_dict.items():
                    if i == 0:
                        pre_tokens_len = attn_list[0].shape[2]
                    attn_block = torch.stack(attn_list[1:], 3)
                    tensor_attn_dict[i] = attn_block
    
    if cfg.dataset_manifest is not None:
        logging.info(f"Finished transcribing from manifest file: {cfg.dataset_manifest}")
        if cfg.presort_manifest:
            transcriptions = restore_transcription_order(cfg.dataset_manifest, transcriptions)
    else:
        logging.info(f"Finished transcribing {len(filepaths)} files !")
    logging.info(f"Writing transcriptions into file: {cfg.output_filename}")

    # if transcriptions form a tuple of (best_hypotheses, all_hypotheses)
    if type(transcriptions) == tuple and len(transcriptions) == 2:
        if cfg.extract_nbest:
            # extract all hypotheses if exists
            transcriptions = transcriptions[1]
        else:
            # extract just best hypothesis
            transcriptions = transcriptions[0]
    if cfg.return_transcriptions:
        return transcriptions
    
    word_alignments = save_attn_imgs(cfg, asr_model, transcriptions, num_layers, tensor_attn_dict, pre_tokens_len)
    write_ctm(filepaths, cfg.output_filename, word_alignments)

    # write audio transcriptions
    output_filename, pred_text_attr_name = write_transcription(
        transcriptions,
        cfg,
        model_name,
        filepaths=filepaths,
        compute_langs=compute_langs,
        compute_timestamps=compute_timestamps,
    )
    logging.info(f"Finished writing predictions to {output_filename}!")

if __name__ == '__main__':
    main()  # noqa pylint: disable=no-value-for-parameter
