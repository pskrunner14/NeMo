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
import copy
import filecmp
import json
import os
import shutil
import tempfile
from unittest import mock

import numpy as np
import pytest
import soundfile as sf
import torch.cuda
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader


from nemo.collections.asr.data.audio_to_diar_label import AudioToSpeechE2ESpkDiarDataset
from nemo.collections.asr.parts.preprocessing.features import WaveformFeaturizer
from nemo.collections.asr.modules import AudioToMelSpectrogramPreprocessor
from nemo.collections.asr.parts.utils.speaker_utils import read_rttm_lines, get_offset_and_duration, get_vad_out_from_rttm_line
from nemo.tests.collections.asr.test_diar_datasets import is_rttm_length_too_long

class TestLhotseAudioToSpeechE2ESpkDiarDataset:

    @pytest.mark.unit
    def test_e2e_speaker_diar_dataset(self, test_data_dir):
        manifest_path = os.path.abspath(os.path.join(test_data_dir, 'asr/diarizer/lsm_val.json'))

        batch_size = 4
        num_samples = 8
    
        device = 'gpu' if torch.cuda.is_available() else 'cpu'
        data_dict_list = []
        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8') as f:
            with open(manifest_path, 'r', encoding='utf-8') as mfile:
                for ix, line in enumerate(mfile):
                    if ix >= num_samples:
                        break

                    line = line.replace("tests/data/", test_data_dir + "/").replace("\n", "")
                    f.write(f"{line}\n")
                    data_dict = json.loads(line)
                    data_dict_list.append(data_dict)

            f.seek(0)
            featurizer = WaveformFeaturizer(sample_rate=16000, int_values=False, augmentor=None)
            preprocessor = AudioToMelSpectrogramPreprocessor(normalize="per_feature",
                                                            window_size=0.025,
                                                            sample_rate=16000,
                                                            window_stride=0.01,
                                                            window="hann",
                                                            features=128,
                                                            n_fft=512,
                                                            frame_splicing=1,
                                                            dither=0.00001
                                                        )

            
            dataloader_instance = get_lhotse_dataloader_from_config(
                config,
                global_rank=self.global_rank,
                world_size=self.world_size,
                dataset=LhotseAudioToSpeechE2ESpkDiarDataset(cfg=config),
            )

            assert len(dataloader_instance) == (num_samples / batch_size)  # Check if the number of batches is correct
            batch_counts = len(dataloader_instance)
            
            deviation_thres_rate = 0.01  # 1% deviation allowed
            for batch_index, batch in enumerate(dataloader_instance):
                if batch_index != batch_counts - 1:
                    assert len(batch) == batch_size, "Batch size does not match the expected value"
                audio_signals, audio_signal_len, targets, target_lens = batch
                for sample_index in range(audio_signals.shape[0]):
                    dataloader_audio_in_sec = audio_signal_len[sample_index].item()
                    data_dur_in_sec = abs(data_dict_list[batch_size*batch_index + sample_index]['duration'] * featurizer.sample_rate - dataloader_audio_in_sec) 
                    assert data_dur_in_sec <= deviation_thres_rate * dataloader_audio_in_sec, "Duration deviation exceeds 1%"
                assert not torch.isnan(audio_signals).any(), "audio_signals tensor contains NaN values"
                assert not torch.isnan(audio_signal_len).any(), "audio_signal_len tensor contains NaN values"
                assert not torch.isnan(targets).any(), "targets tensor contains NaN values"
                assert not torch.isnan(target_lens).any(), "target_lens tensor contains NaN values"
            