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
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import torch

from nemo.collections.asr.data.audio_to_eesd_label import extract_seg_info_from_rttm, get_global_seg_spk_labels, get_speaker_labels_from_diar_rttms, get_frame_targets_from_rttm
from nemo.collections.asr.parts.preprocessing.features import WaveformFeaturizer
from nemo.collections.asr.parts.utils.offline_clustering import get_argmin_mat
from nemo.collections.asr.parts.utils.speaker_utils import convert_rttm_line, get_subsegments
from nemo.collections.common.parts.preprocessing.collections import DiarizationSpeechLabel
from nemo.core.classes import Dataset
from nemo.core.neural_types import AudioSignal, LengthsType, NeuralType, ProbsType
from nemo.collections.asr.parts.utils.audio_utils import ChannelSelectorType
from nemo.collections.asr.parts.preprocessing.perturb import process_augmentations

import numpy as np

@dataclass
class EESDAudioRTTMItem:
    sample_id: str
    audio_signal: torch.Tensor
    audio_signal_length: torch.Tensor
    labels: torch.Tensor
    num_spkeaers: torch.Tensor
    speaker_id_map: Dict[int, str]
    uem: Optional[Tuple[float, float]] = None
    processed_signal: torch.Tensor = None
    processed_signal_length: torch.Tensor = None
    rttm_file: Optional[str] = None


@dataclass
class EESDAudioRTTMBatch:
    sample_id: torch.Tensor
    audio_signal: torch.Tensor
    audio_signal_length: torch.Tensor
    labels: torch.Tensor
    num_spkeaers: torch.Tensor
    processed_signal: torch.Tensor = None
    processed_signal_length: torch.Tensor = None
    uem: Optional[Tuple[float, float]] = None
    rttm_file: Optional[List[str]] = None


def pad_audio_to_length(audio, length, value = 0):
    """
    Pad audio signal to given length.
    """
    curr_length = audio.size(0)
    if curr_length >= length:
        return audio
    if len(audio.shape) > 1:
        # multi-channel audio, shape: [time, channels]
        audio = torch.nn.functional.pad(audio, (0, 0, 0, length - curr_length), mode='constant', value=value)
    else:
        # single-channel audio
        audio = torch.nn.functional.pad(audio, (0, length - curr_length), 'constant', value=value)
    return audio


def collate_fn_eesd(batch: List[EESDAudioRTTMItem])->EESDAudioRTTMBatch:
    sample_id = [x.sample_id for x in batch]
    max_audio_len = max([x.audio_signal_length for x in batch])
    max_label_len = max([x.labels.size(0) for x in batch])
    audio_signal = torch.stack([pad_audio_to_length(x.audio_signal, max_audio_len) for x in batch])
    audio_signal_length = torch.stack([x.audio_signal_length for x in batch])
    labels = torch.stack([pad_audio_to_length(x.labels, max_label_len, value=0) for x in batch])
    num_spkeaers = torch.stack([x.num_spkeaers for x in batch])
    uem = [x.uem for x in batch]
    rttm_file = [x.rttm_file for x in batch]
    return EESDAudioRTTMBatch(
        sample_id=sample_id, 
        audio_signal=audio_signal, 
        audio_signal_length=audio_signal_length, 
        labels=labels, 
        num_spkeaers=num_spkeaers,
        uem=uem,
        rttm_file=rttm_file
    )


class AudioToEESDLabelDataset(Dataset):
    """
    Dataset that takes audio signal and RTTM file and returns audio signal and EESD labels.
    """
    def __init__(
        self,
        manifest_filepath: str,
        sample_rate: int,
        window_stride: float = 0.01,
        int_values: bool = False,
        trim: bool = False,
        max_speakers: int = 4,
        session_len_sec: Optional[float] = None,
        random_offset: bool = False,
        round_digits: int = 2,
        augmentor: 'nemo.collections.asr.parts.perturb.AudioAugmentor' = None,
        return_sample_id: bool = False,
        channel_selector: Optional[ChannelSelectorType] = None,
        normalize_db: Optional[float] = None,
        uem_filepath: Optional[str] = None,
    ):
        super().__init__()
        if isinstance(manifest_filepath, str):
            manifest_filepath = manifest_filepath.split(',')
        self.collection = DiarizationSpeechLabel(
            manifests_files=manifest_filepath,
            emb_dict=None,
            clus_label_dict=None,
            pairwise_infer=False,
        )

        self.featurizer = WaveformFeaturizer(sample_rate=sample_rate, int_values=int_values, augmentor=augmentor)
        self.trim = trim
        self.return_sample_id = return_sample_id
        self.channel_selector = channel_selector
        self.normalize_db = normalize_db
        self.session_len_sec = session_len_sec
        self.random_offset = random_offset
        self.round_digits = round_digits
        self.max_speakers = max_speakers
        self.feat_per_sec = 1 / window_stride
        self.global_speaker_label_table = get_speaker_labels_from_diar_rttms(self.collection)
        self.uem_map = self.get_uem_map(uem_filepath) if uem_filepath else {}

    def get_uem_map(self, uem_filepath):
        """
        Get UEM map from UEM file.
        """
        uem_map = {}
        with open(uem_filepath, 'r') as f:
            for line in f:
                line = line.strip().split()
                if len(line) != 3:
                    raise ValueError(f"UEM file should have 3 columns, found {len(line)} columns")
                audio_id = line[0]
                start_time = float(line[1])
                end_time = float(line[2])
                uem_map[audio_id] = (start_time, end_time)
        return uem_map

    def get_uniq_id_with_range(self, sample, deci=3):
        """
        Generate unique training sample ID from unique file ID, offset and duration. The start-end time added
        unique ID is required for identifying the sample since multiple short audio samples are generated from a single
        audio file. The start time and end time of the audio stream uses millisecond units if `deci=3`.

        Args:
            sample:
                `DiarizationSpeechLabel` instance from collections.

        Returns:
            uniq_id (str):
                Unique sample ID which includes start and end time of the audio stream.
                Example: abc1001_3122_6458
        """
        bare_uniq_id = os.path.splitext(os.path.basename(sample.rttm_file))[0]
        sample_offset = sample.offset if sample.offset is not None else 0.0
        offset = str(int(round(sample_offset, deci) * pow(10, deci)))
        if not sample.duration:
            endtime = 'NA'
        else:
            endtime = str(int(round(float(sample_offset) + float(sample.duration), deci) * pow(10, deci)))
        uniq_id = f"{bare_uniq_id}_off{offset}_dur{endtime}"
        return uniq_id, bare_uniq_id

    def parse_rttm_for_frame_labels(self, uniq_id, rttm_file, offset, duration):
        """
        Parse RTTM file to get speaker labels and speaker segments.
        """
        rttm_lines = open(rttm_file).readlines()
        rttm_timestamps, sess_to_global_spkids = extract_seg_info_from_rttm(uniq_id, offset, duration, rttm_lines)

        # shape = [num_frames, num_speakers]
        frame_targets = get_frame_targets_from_rttm(rttm_timestamps=rttm_timestamps, 
                                                      offset=offset,
                                                      duration=duration,
                                                      round_digits=self.round_digits, 
                                                      feat_per_sec=self.feat_per_sec, 
                                                      max_spks=self.max_speakers) 
        return frame_targets, sess_to_global_spkids


    def __len__(self):
        return len(self.collection)
    
    def __getitem__(self, index: int) -> EESDAudioRTTMItem:
        sample = self.collection[index]
        offset = sample.offset if sample.offset is not None else 0
        duration = sample.duration if sample.duration is not None else None

        uniq_id, base_id = self.get_uniq_id_with_range(sample, self.round_digits)
        audio_signal = self.featurizer.process(sample.audio_file, offset=offset, duration=duration, trim=self.trim, channel_selector=self.channel_selector, normalize_db=self.normalize_db)
        curr_duration = audio_signal.size(0) / self.featurizer.sample_rate

        session_len_samples = int(self.session_len_sec * self.featurizer.sample_rate) if self.session_len_sec else audio_signal.size(0)
        random_offset_samples = 0
        if self.session_len_sec and self.random_offset and curr_duration > self.session_len_sec:
            random_offset = np.random.uniform(0, curr_duration - self.session_len_sec)
            random_offset_samples = int(random_offset * self.featurizer.sample_rate)
            audio_signal = audio_signal[random_offset_samples:random_offset_samples + session_len_samples]
            uniq_id = f"{uniq_id}_RandOffset{str(int(round(random_offset, self.round_digits) * pow(10, self.round_digits)))}"
        if self.session_len_sec:
                audio_signal = audio_signal[:session_len_samples] 

        audio_signal_length = torch.tensor(audio_signal.size(0)).long()
        total_offset = offset + random_offset_samples / self.featurizer.sample_rate
        total_duration = audio_signal.size(0) / self.featurizer.sample_rate
        labels, speaker_id_map = self.parse_rttm_for_frame_labels(uniq_id, sample.rttm_file, total_offset, total_duration)
        num_speakers = torch.tensor(len(speaker_id_map)).long()
        uem = self.uem_map.get(base_id, None)
        return EESDAudioRTTMItem(
            sample_id=uniq_id,
            audio_signal=audio_signal,
            audio_signal_length=audio_signal_length,
            labels=labels,
            num_spkeaers=num_speakers,
            speaker_id_map=speaker_id_map,
            uem=uem,
            rttm_file=sample.rttm_file
        )

    def collate_fn(self, batch):
        return collate_fn_eesd(batch)


def get_audio_to_eesd_label_dataset_from_config(
    config: Dict, 
    local_rank: int, 
    global_rank: int, 
    world_size: int,
) -> AudioToEESDLabelDataset:
    if config.get('is_tarred', False):
        raise ValueError("Tarred dataset not supported for EESD dataset yet")

    augmentor = process_augmentations(config.get('augmentation', None), global_rank, world_size)
    
    return AudioToEESDLabelDataset(
        manifest_filepath=config['manifest_filepath'],
        sample_rate=config['sample_rate'],
        window_stride=config['window_stride'],
        int_values=config.get('int_values', False),
        trim=config.get('trim_silence', False),
        max_speakers=config.get('max_speakers', 4),
        session_len_sec=config.get('session_len_sec', None),
        random_offset=config.get('random_offset', False),
        round_digits=config.get('round_digits', 2),
        augmentor=augmentor,
        return_sample_id=config.get('return_sample_id', False),
        channel_selector=config.get('channel_selector', None),
        normalize_db=config.get('normalize_db', None),
        uem_filepath=config.get('uem_filepath', None),
    )
