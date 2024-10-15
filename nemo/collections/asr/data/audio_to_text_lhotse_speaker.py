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
import soundfile

import torch.utils.data
from lhotse.cut import MixedCut, MonoCut, MixTrack
from lhotse.dataset import AudioSamples
from lhotse.dataset.collation import collate_vectors, collate_matrices
from lhotse.utils import compute_num_samples
from lhotse import SupervisionSet, SupervisionSegment, MonoCut, Recording, CutSet, AudioSource

import numpy as np

from nemo.collections.asr.data.audio_to_text_lhotse import TokenizerWrapper
from nemo.collections.common.tokenizers.aggregate_tokenizer import AggregateTokenizer
from nemo.collections.common.tokenizers.tokenizer_spec import TokenizerSpec
from nemo.core.neural_types import AudioSignal, LabelsType, LengthsType, NeuralType

from nemo.collections.asr.parts.utils.asr_multispeaker_utils import (
    speaker_to_target, 
    get_hidden_length_from_sample_length, 
    get_mask_from_segments,
    shuffle_spk_mapping,
)

class LhotseSpeechToTextSpkBpeDataset(torch.utils.data.Dataset):
    """
    This dataset is based on BPE datasets from audio_to_text.py. It has the same functionality of LhotseSpeechToTextBpeDataset but also yield speaker target tensor.
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
            'transcripts': NeuralType(('B', 'T'), LabelsType()),
            'transcript_length': NeuralType(tuple('B'), LengthsType()),
            'spk_tar_id': NeuralType(('B','T'), LabelsType()),
            'sample_id': NeuralType(tuple('B'), LengthsType(), optional=True),
        }

    def __init__(self, cfg, tokenizer):
        super().__init__()
        self.tokenizer = TokenizerWrapper(tokenizer)
        self.load_audio = AudioSamples(fault_tolerant=True)
        self.cfg = cfg
        self.spk_tar_all_zero = self.cfg.get('spk_tar_all_zero',False)
        self.num_speakers = self.cfg.get('num_speakers', 4)
        self.num_sample_per_mel_frame = self.cfg.get('num_sample_per_mel_frame', 160)
        self.num_mel_frame_per_asr_frame = self.cfg.get('num_mel_frame_per_asr_frame', 8)
        self.shuffle_spk_mapping = self.cfg.get('shuffle_spk_mapping', True)
        self.spk_token_pattern= r'<\|spltoken\d+\|>' 
        self.inference_mode = self.cfg.get('inference_mode', False)

    def __getitem__(self, cuts) -> Tuple[torch.Tensor, ...]:
        # cuts, spk_mappings = shuffle_spk_mapping(cuts=cuts, num_speakers=self.num_speakers, shuffle_spk_mapping=self.
        # shuffle_spk_mapping, pattern=self.spk_token_pattern)
        
        spk_mappings = torch.arange(self.num_speakers).unsqueeze(0).repeat(len(cuts),1)
        # if cuts[0].dataset_id == "training_list":
            # cuts = self.cuts2mixcuts(cuts)
        if self.inference_mode:
            spk_targets = [torch.transpose(torch.zeros(self.num_speakers, get_hidden_length_from_sample_length(cut.num_samples, self.num_sample_per_mel_frame, self.num_mel_frame_per_asr_frame)), 0, 1) for cut in cuts]
        else:                    
            spk_targets = [torch.as_tensor(speaker_to_target(cut, self.num_speakers, self.num_sample_per_mel_frame, self.num_mel_frame_per_asr_frame, self.spk_tar_all_zero), dtype=torch.float32) for cut in cuts]
        spk_targets = collate_matrices(spk_targets)
        audio, audio_lens, cuts = self.load_audio(cuts)
        tokens = [torch.as_tensor(self.tokenizer(c.supervisions[0].text, c.supervisions[0].language)) for c in cuts]
        token_lens = torch.tensor([t.size(0) for t in tokens], dtype=torch.long)
        tokens = collate_vectors(tokens, padding_value=0)

        return audio, audio_lens, tokens, token_lens, spk_targets, spk_mappings

    def cuts2mixcuts(self, cuts):
        cut_set = []
        for cut in cuts:
            offsets = cut.delays
            durations = cut.durations
            wavs = cut.wavs
            text = cut.text
            rttm_filepath = cut.rttm_filepath
            # speakers = cut.speakers

            tracks = []
            for i, (offset, duration, wav) in enumerate(zip(offsets, durations, wavs)):
                wav_dur = soundfile.info(wav).duration
                wav_samples = soundfile.info(wav).frames
                cut_1spk = MonoCut(
                    id=wav.split('/')[-1].replace('.wav', ''),
                    start=0,
                    duration=duration,
                    channel=0,
                    supervisions=[],
                    recording=Recording(
                        id=wav.split('/')[-1].replace('.wav', ''),
                        sources=[
                            AudioSource(
                                type='file',
                                channels=[0],
                                source=wav
                            )
                        ],
                        sampling_rate=16000, 
                        num_samples=wav_samples,
                        duration=wav_dur
                    ),
                )

                tracks.append(MixTrack(cut=cut_1spk, type=type(cut_1spk), offset=offset))
            custom = {
                        'text': text,
                        'rttm_filepath': rttm_filepath,
                    }
            sup = SupervisionSegment(
                id=cut.id,
                recording_id=cut.recording_id,
                start=0,
                duration=offset+wav_dur,
                text=cut.text,
                custom = custom
            )

            tracks[0].cut.supervisions.append(sup)
            cut_multi_spk = MixedCut(id=cut.rttm_filepath.split('/')[-1][:-5], tracks=tracks)

            cut_set.append(cut_multi_spk)
        
        return CutSet.from_cuts(cut_set)
