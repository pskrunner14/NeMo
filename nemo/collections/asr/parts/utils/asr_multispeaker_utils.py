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

import omegaconf
import os
import torch.utils.data
from pathlib import Path
from lhotse import CutSet
from lhotse.cut import MixedCut, MonoCut
from lhotse.dataset import AudioSamples
from lhotse.dataset.collation import collate_vectors, collate_matrices
from lhotse.utils import compute_num_samples
from lhotse import SupervisionSet
import numpy as np

from typing import Dict, Optional, Tuple, List

import torch.utils.data
import re
from lhotse.dataset import AudioSamples
from lhotse.dataset.collation import collate_vectors, collate_matrices
from lhotse.utils import compute_num_samples
from lhotse import SupervisionSet

from pathlib import Path
import numpy as np
from copy import deepcopy
from nemo.collections.asr.data.audio_to_text_lhotse import TokenizerWrapper
from nemo.collections.common.tokenizers.aggregate_tokenizer import AggregateTokenizer
from nemo.collections.common.tokenizers.tokenizer_spec import TokenizerSpec
from nemo.core.neural_types import AudioSignal, LabelsType, LengthsType, NeuralType


def apply_spk_mapping(diar_preds: torch.Tensor, spk_mappings: torch.Tensor) -> torch.Tensor:
    """ 
    Applies a speaker mapping to diar predictions.

    Args:
        diar_preds (Tensor): The diar predictions tensor.   
            Dimension: (batch_size, num_frames, num_speakers)
        spk_mappings (Tensor): The speaker mappings tensor.
            Dimension: (batch_size, num_speakers)
    
    Returns:
        permuted_diar_preds (Tensor): The permuted diar predictions tensor with the given speaker mappings.
    """
    expanded_mappings = spk_mappings.unsqueeze(1).expand(-1, diar_preds.size(1), -1)
    permuted_diar_preds = torch.gather(diar_preds, 2, expanded_mappings)
    return permuted_diar_preds

def shuffle_spk_mapping(cuts: list, num_speakers: int, shuffle_spk_mapping: bool = False, pattern= r'<\|spltoken\d+\|>') -> Tuple[CutSet, torch.Tensor]:
    """ 
    Applies a shuffle mapping to speaker labels in the cuts.
    Debug with:
    # print(f"idx: {idx} : spk mapping {spk_mappings[idx]} | word: {word} -> new_word: {new_word}")

    Args:
        cuts (List[MonoCut, MixedCut]): A list of Cut instances.
        num_speakers (int): The total number of speakers.
        shuffle_spk_mapping (bool): Whether to shuffle the speaker mappings.
        pattern (str): A regular expression pattern for speaker tokens.

    Returns:
        cuts (list): The updated CutSet with shuffled speaker mappings.
        spk_mappings (Tensor): The shuffled speaker mappings.
    """ 
    if shuffle_spk_mapping:
        batch_size = len(cuts) 
        numbers = torch.tensor([x for x in range(num_speakers)]).repeat(batch_size, 1)
        permuted_indices = torch.rand(batch_size, 4).argsort(dim=1)
        spk_mappings = torch.gather(numbers, 1, permuted_indices)
        str_pattern = pattern.replace("\\", '')
        left_str, right_str = str_pattern.split('d+')[0], str_pattern.split('d+')[1]
        for idx, cut in enumerate(cuts):
            new_cut_text = deepcopy(cut.text)
            word_list = []
            for word in new_cut_text.split(): 
                if len(re.findall(pattern, word)) > 0:
                    spk_token_int = int(word.replace(left_str,'').replace(right_str, ''))
                    new_spk = spk_mappings[idx][spk_token_int]
                    new_word = f'{left_str}{new_spk}{right_str}'
                    word_list.append(new_word)
                else:
                    word_list.append(word)
            cuts[idx].text = ' '.join(word_list)
    else:
        spk_mappings = torch.arange(num_speakers).unsqueeze(0).repeat(len(cuts), 1)    
    return cuts, spk_mappings 

def speaker_to_target(
    a_cut,
    num_speakers: int = 4, 
    num_sample_per_mel_frame: int = 160, 
    num_mel_frame_per_asr_frame: int = 8, 
    spk_tar_all_zero: bool = False
    ):
    '''
    Get rttm samples corresponding to one cut, generate speaker mask numpy.ndarray with shape (num_speaker, hidden_length)
    This function is needed for speaker diarization with ASR model trainings.

    Args:
        a_cut (MonoCut, MixedCut): Lhotse Cut instance which is MonoCut or MixedCut instance.
        num_speakers (int): max number of speakers for all cuts ("mask" dim0), 4 by default
        num_sample_per_mel_frame (int): number of sample per mel frame, sample_rate / 1000 * window_stride, 160 by default (10ms window stride)
        num_mel_frame_per_asr_frame (int): encoder subsampling_factor, 8 by default
        spk_tar_all_zero (Tensor): set to True gives all zero "mask"
    
    Returns:
        mask (Tensor): speaker mask with shape (num_speaker, hidden_lenght)
    '''
    # get cut-related segments from rttms
    rttms = SupervisionSet.from_rttm(a_cut.rttm_filepath)
    basename = os.path.basename(a_cut.rttm_filepath).replace('.rttm', '')
    if isinstance(a_cut, MixedCut):
        cut_list = [track.cut for track in cut.tracks] 
    elif isinstance(a_cut, MonoCut):
        cut_list = [a_cut]
    else:
        raise ValueError(f"Unsupported cut type type{cut}: only MixedCut and MonoCut are supported")
    
    segments_total = []
    for cut in cut_list:
        segments_iterator = rttms.find(recording_id=cut.recording_id, start_after=cut.start, end_before=cut.end, adjust_offset=True)
        segments = [s for s in segments_iterator]
        segments_total.extend(segments)
    # apply arrival time sorting to the existing segments
    segments_total.sort(key = lambda rttm_sup: rttm_sup.start)

    seen = set()
    seen_add = seen.add
    speaker_ats = [s.speaker for s in segments if not (s.speaker in seen or seen_add(s.speaker))]
    
    speaker_to_idx_map = {
            spk: idx
            for idx, spk in enumerate(speaker_ats)
    }
    if len(speaker_to_idx_map) > num_speakers:
        raise ValueError(f"Number of speakers {len(speaker_to_idx_map)} is larger than the maximum number of speakers {num_speakers}")
    # initialize mask matrices (num_speaker, encoder_hidden_len)
    if spk_tar_all_zero: 
        mask = np.zeros((num_speakers, get_hidden_length_from_sample_length(cut.num_samples, num_sample_per_mel_frame, num_mel_frame_per_asr_frame)))
    else:
        mask = get_mask_from_segments(segments, cut, speaker_to_idx_map, num_speakers, num_sample_per_mel_frame, num_mel_frame_per_asr_frame)
    return mask

def get_mask_from_segments(segments, cut, speaker_to_idx_map, num_speakers=4, num_sample_per_mel_frame=160, num_mel_frame_per_asr_frame=8):
    """ 
    Generate mask matrix from segments list.
    This function is needed for speaker diarization with ASR model trainings.
    Debug with:
    # print(f"id: {cut.recording_id} spk_idx: {speaker_idx} st: {stt}, et: {ent}, st_encoder_loc: {st_encoder_loc}, et_encoder_loc: {et_encoder_loc}")
    
    Args:
        segments: A list of Lhotse Supervision segments iterator.
        cut (MonoCut, MixedCut): Lhotse MonoCut or MixedCut instance.
        speaker_to_idx_map (dict): A dictionary mapping speaker names to indices.
        num_speakers (int): max number of speakers for all cuts ("mask" dim0), 4 by default
        num_sample_per_mel_frame (int): number of sample per mel frame, sample_rate / 1000 * window_stride, 160 by default (10ms window stride)
        num_mel_frame_per_asr_frame (int): encoder subsampling_factor, 8 by default
    
    Returns:
        mask (Tensor): A numpy array of shape (num_speakers, encoder_hidden_len).
            Dimension: (num_speakers, num_frames)
    """
    encoder_hidden_len = get_hidden_length_from_sample_length(cut.num_samples, num_sample_per_mel_frame, num_mel_frame_per_asr_frame)
    mask = np.zeros((num_speakers, encoder_hidden_len))
    for rttm_sup in segments:
        speaker_idx = speaker_to_idx_map[rttm_sup.speaker]
        # only consider the first <num_speakers> speakers
        stt = (
                    compute_num_samples(rttm_sup.start, cut.sampling_rate)
                    if rttm_sup.start > 0
                    else 0
                )
        ent = (
                    compute_num_samples(rttm_sup.end, cut.sampling_rate)
                    if rttm_sup.end < cut.duration
                    else compute_num_samples(rttm_sup.duration, cut.sampling_rate)
                )                   

        # map start time (st) and end time (et) to encoded hidden location
        st_encoder_loc = get_hidden_length_from_sample_length(stt, num_sample_per_mel_frame, num_mel_frame_per_asr_frame)
        et_encoder_loc = get_hidden_length_from_sample_length(ent, num_sample_per_mel_frame, num_mel_frame_per_asr_frame)
        mask[speaker_idx, st_encoder_loc:et_encoder_loc] = 1
    return mask 

def get_hidden_length_from_sample_length(
    num_samples: int, 
    num_sample_per_mel_frame: int = 160, 
    num_mel_frame_per_asr_frame: int = 8
) -> int:
    """ 
    Calculate the hidden length from the given number of samples.
    This function is needed for speaker diarization with ASR model trainings.

    This function computes the number of frames required for a given number of audio samples,
    considering the number of samples per mel frame and the number of mel frames per ASR frame.

    Parameters:
        num_samples (int): The total number of audio samples.
        num_sample_per_mel_frame (int, optional): The number of samples per mel frame. Default is 160.
        num_mel_frame_per_asr_frame (int, optional): The number of mel frames per ASR frame. Default is 8.

    Returns:
        hidden_length (int): The calculated hidden length in terms of the number of frames.
    """
    mel_frame_count = np.ceil((num_samples + 1) / num_sample_per_mel_frame)
    hidden_length = np.ceil(mel_frame_count / num_mel_frame_per_asr_frame)
    return int(hidden_length)