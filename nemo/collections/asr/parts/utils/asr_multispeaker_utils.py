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

import os
import re
import math
from copy import deepcopy
from typing import Dict, Optional, Tuple, List

import torch.utils.data
from lhotse import CutSet
from lhotse.cut import MixedCut, MonoCut
from lhotse.utils import compute_num_samples
from lhotse import SupervisionSet

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
    Applies a shuffle mapping to speaker text labels in the cuts.
    Example:
        Original cut.text:
            "<|spltoken0|> we do shuffle <|spltoken1|> and map speakers <|spltoken0|> yes <|spltoken2|> we keep dimensions" 
        Speaker Mapping: [3, 0, 1, 2]
        Shuffled cut.text:
            "<|spltoken3|> we do shuffle <|spltoken0|> and map speakers <|spltoken3|> yes <|spltoken1|> we keep dimensions" 

    Args:
        cuts (List[MonoCut, MixedCut]): A list of Cut instances.
        num_speakers (int): The total number of speakers.
        shuffle_spk_mapping (bool): Whether to shuffle the speaker mappings.
        pattern (str): A regular expression pattern for speaker tokens.

    Returns:
        cuts (list): The updated CutSet with shuffled speaker mappings.
        spk_mappings (Tensor): The shuffled speaker mappings in batch.
    """ 
    if shuffle_spk_mapping:
        batch_size = len(cuts) 
        permuted_indices = torch.rand(batch_size, 4).argsort(dim=1)
        spk_mappings = torch.gather(torch.arange(num_speakers).repeat(batch_size, 1), 1, permuted_indices)
        str_pattern = pattern.replace("\\", '')
        left_str, right_str = str_pattern.split('d+')[0], str_pattern.split('d+')[1]
        for idx, cut in enumerate(cuts):
            word_list = []
            for word in deepcopy(cut.text).split(): 
                if len(re.findall(pattern, word)) > 0:
                    spk_token_int = int(word.replace(left_str,'').replace(right_str, ''))
                    new_spk = spk_mappings[idx][spk_token_int]
                    word_list.append(f'{left_str}{new_spk}{right_str}')
                else:
                    word_list.append(word)
            cuts[idx].text = ' '.join(word_list)
    else:
        spk_mappings = torch.arange(num_speakers).unsqueeze(0).repeat(batch_size, 1)    
    return cuts, spk_mappings 

def find_segments_from_rttm(
        recording_id: str, 
        rttms, 
        start_after: float, 
        end_before: float, 
        adjust_offset=True, 
        tolerance=0.001):
    """ 
    Finds segments from the given rttm file.
    This function is designed to replace rttm

    Args:
        recording_id (str): The recording ID in string format.
        rttms (SupervisionSet): The SupervisionSet instance.
        start_after (float): The start time after which segments are selected.
        end_before (float): The end time before which segments are selected.
        adjust_offset (bool): Whether to adjust the offset of the segments.
        tolerance (float): The tolerance for time matching. 0.001 by default.
    
    Returns:
        segments (List[SupervisionSegment]): A list of SupervisionSegment instances.
    """
    segment_by_recording_id = rttms._segments_by_recording_id
    if segment_by_recording_id is None:
        from cytoolz import groupby
        segment_by_recording_id = groupby(lambda seg: seg.recording_id, rttms)

    return [
            # We only modify the offset - the duration remains the same, as we're only shifting the segment
            # relative to the Cut's start, and not truncating anything.
            segment.with_offset(-start_after) if adjust_offset else segment
            for segment in segment_by_recording_id.get(recording_id, [])
            if segment.start < end_before + tolerance
            and segment.end > start_after + tolerance
        ]

def speaker_to_target(
    a_cut,
    num_speakers: int = 4, 
    num_sample_per_mel_frame: int = 160, 
    num_mel_frame_per_asr_frame: int = 8, 
    spk_tar_all_zero: bool = False,
    boundary_segments: bool = False
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
        boundary_segments (bool): set to True to include segments containing the boundary of the cut, False by default for multi-speaker ASR training
    
    Returns:
        mask (Tensor): speaker mask with shape (num_speaker, hidden_lenght)
    '''
    # get cut-related segments from rttms
    rttms = SupervisionSet.from_rttm(a_cut.rttm_filepath)
    basename = os.path.basename(a_cut.rttm_filepath).replace('.rttm', '')
    if isinstance(a_cut, MixedCut):
        cut_list = [track.cut for track in a_cut.tracks] 
    elif isinstance(a_cut, MonoCut):
        cut_list = [a_cut]
    else:
        raise ValueError(f"Unsupported cut type type{cut}: only MixedCut and MonoCut are supported")
    
    segments_total = []
    for cut in cut_list:
        if boundary_segments: # segments with seg_start < total_end and seg_end > total_start are included
            segments_iterator = find_segments_from_rttm(recording_id=cut.recording_id, rttms=rttms, start_after=cut.start, end_before=cut.end)
        else: # segments with seg_start > total_start and seg_end < total_end are included
            segments_iterator = rttms.find(recording_id=cut.recording_id, start_after=cut.start, end_before=cut.end, adjust_offset=True)
        segments = [s for s in segments_iterator]
        segments_total.extend(segments)
    
    # apply arrival time sorting to the existing segments
    segments_total.sort(key = lambda rttm_sup: rttm_sup.start)

    seen = set()
    seen_add = seen.add
    speaker_ats = [s.speaker for s in segments_total if not (s.speaker in seen or seen_add(s.speaker))]
     
    speaker_to_idx_map = {
            spk: idx
            for idx, spk in enumerate(speaker_ats)
    }
    if len(speaker_to_idx_map) > num_speakers:
        raise ValueError(f"Number of speakers {len(speaker_to_idx_map)} is larger than the maximum number of speakers {num_speakers}")
    # initialize mask matrices (num_speaker, encoder_hidden_len)
    if spk_tar_all_zero: 
        mask = torch.zeros((num_speakers, get_hidden_length_from_sample_length(a_cut.num_samples, num_sample_per_mel_frame, num_mel_frame_per_asr_frame)))
    else:
        mask = get_mask_from_segments(segments_total, cut_list[0], speaker_to_idx_map, num_speakers, num_sample_per_mel_frame, num_mel_frame_per_asr_frame)
    return mask

def get_mask_from_segments(segments: list, cut, speaker_to_idx_map: torch.Tensor, num_speakers: int =4, num_sample_per_mel_frame: int=160, num_mel_frame_per_asr_frame:int=8):
    """ 
    Generate mask matrix from segments list.
    This function is needed for speaker diarization with ASR model trainings.
    
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
    mask = torch.zeros((num_speakers, encoder_hidden_len))
    for rttm_sup in segments:
        speaker_idx = speaker_to_idx_map[rttm_sup.speaker]
        # only consider the first <num_speakers> speakers
        stt = compute_num_samples(max(rttm_sup.start, 0), cut.sampling_rate)
        ent = compute_num_samples(min(rttm_sup.end, cut.duration), cut.sampling_rate)
        # map start time (st) and end time (et) to encoded hidden location
        st_encoder_loc = 0 if stt == 0 else get_hidden_length_from_sample_length(stt, num_sample_per_mel_frame, num_mel_frame_per_asr_frame)
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
    mel_frame_count = math.ceil((num_samples + 1) / num_sample_per_mel_frame)
    hidden_length = math.ceil(mel_frame_count / num_mel_frame_per_asr_frame)
    return int(hidden_length)