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

from typing import Dict, Optional, Tuple

import torch.utils.data
from lhotse import SupervisionSet
from lhotse.dataset import AudioSamples
from lhotse.dataset.collation import collate_matrices
from lhotse.utils import compute_num_samples

# from nemo.collections.common.tokenizers.aggregate_tokenizer import AggregateTokenizer
# from nemo.collections.common.tokenizers.tokenizer_spec import TokenizerSpec
from nemo.core.neural_types import AudioSignal, LabelsType, LengthsType, NeuralType

import numpy as np
class LhotseSpeechToDiarizationLabelDataset(torch.utils.data.Dataset):
    """
    This dataset is based on diarization datasets from audio_to_eesd_label.py.
    Unlike native NeMo datasets, Lhotse dataset defines only the mapping from
    a CutSet (meta-data) to a mini-batch with PyTorch tensors.
    Specifically, it performs tokenization, I/O, augmentation, and feature extraction (if any).
    Managing data, sampling, de-duplication across workers/nodes etc. is all handled
    by Lhotse samplers instead.
    """

    @property
    def output_types(self) -> Optional[Dict[str, NeuralType]]:
        return {
            'audio_signal': NeuralType(('B', 'T'), AudioSignal()),
            'a_sig_length': NeuralType(tuple('B'), LengthsType()),
            'targets': NeuralType(('B', 'T', 'N'), LabelsType()),
            'target_length': NeuralType(tuple('B'), LengthsType()),
            'sample_id': NeuralType(tuple('B'), LengthsType(), optional=True),
        }
    
    def __init__(self, cfg):
        super().__init__()
        self.load_audio = AudioSamples(fault_tolerant=True)
        self.cfg = cfg
        self.num_speakers = self.cfg.get('num_speakers', 4)

        self.num_sample_per_mel_frame = int(self.cfg.get('window_stride', 0.01) * self.cfg.get('sample_rate', 16000)) # 160
        self.num_mel_frame_per_target_frame = int(self.cfg.get('subsampling_factor', 8))

    def __getitem__(self, cuts) -> Tuple[torch.Tensor, ...]:
        audio, audio_lens, cuts = self.load_audio(cuts)
        
        speaker_activities = torch.stack([self.get_speaker_activity(cut) for cut in cuts])
        targets = collate_matrices(speaker_activities).transpose(1, 2)
        target_lens = torch.tensor([[self.sample_length_to_target_length(l, self.num_sample_per_mel_frame, self.num_mel_frame_per_target_frame)] for l in audio_lens])

        return audio, audio_lens, target_lens, targets
    
    def get_speaker_activity(self, cut):
        # source from: https://github.com/lhotse-speech/lhotse/blob/0a4aed49754d61b781c14de85f7772dda71c6226/lhotse/cut/base.py#L882
        
        n_frames = self.sample_length_to_target_length(cut.num_samples, self.num_sample_per_mel_frame, self.num_mel_frame_per_target_frame)
        speaker_activity = torch.zeros((self.num_speakers, n_frames))

        supervisions = SupervisionSet.from_rttm(cut.rttm_filepath)
        speaker_to_idx_map = {
            spk: idx
            for idx, spk in enumerate(
                sorted(set(s.speaker for s in supervisions))
            )
        }

        for supervision in supervisions:
            speaker_idx = speaker_to_idx_map[supervision.speaker]
            if speaker_idx >= self.num_speakers:
                raise ValueError(
                    f"The number of speakers is larger than the number of speakers allowed by the model: {speaker_idx} >= {self.num_speakers}"
                )
            sample_start = (
                compute_num_samples(supervision.start, cut.sampling_rate)
                if supervision.start > 0
                else 0
            )
            sample_end = (
                compute_num_samples(supervision.end, cut.sampling_rate)
                if supervision.end < cut.duration
                else compute_num_samples(supervision.duration, cut.sampling_rate)
            )   
            target_start = self.sample_length_to_target_length(sample_start, self.num_sample_per_mel_frame, self.num_mel_frame_per_target_frame)
            target_end = self.sample_length_to_target_length(sample_end, self.num_sample_per_mel_frame, self.num_mel_frame_per_target_frame)
            speaker_activity[speaker_idx, target_start:target_end] = 1

        return speaker_activity

    @staticmethod
    def sample_length_to_target_length(num_samples: int, num_sample_per_mel_frame: int = 160, num_mel_frame_per_target_frame: int = 8):
        '''
        This function solves the mismatch between the number of feature length and output length.
        input: the number of samples in the audio signal
        output: the number of frames in the output
        '''
        return int(np.ceil(np.ceil((num_samples + 1) / num_sample_per_mel_frame) / num_mel_frame_per_target_frame))