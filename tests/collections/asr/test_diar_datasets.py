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

class TestUtilityFunctions:
    pass

class TestAudioToSpeechE2ESpkDiarDataset:

    # @pytest.mark.unit
    # def test_mismatch_in_model_dataloader_config(self, caplog):
    #     logging._logger.propagate = True
    #     caplog.set_level(logging.WARNING)

    #     model_cfg = OmegaConf.create(dict(labels=OmegaConf.create(["a", "b", "c"])))
    #     dataloader_cfg = OmegaConf.create(dict(labels=copy.deepcopy(self.labels)))

    #     inject_dataloader_value_from_model_config(model_cfg, dataloader_cfg, key='labels')

    #     assert (
    #         """`labels` is explicitly provided to the data loader, and is different from the `labels` provided at the model level config."""
    #         in caplog.text
    #     )

    #     logging._logger.propagate = False


    @pytest.mark.unit
    def test_e2e_speaker_diar_dataset(self, test_data_dir):
        manifest_path = os.path.abspath(os.path.join(test_data_dir, 'asr/diarizer/an4_manifest.json'))

        num_samples = 1
        batch_size = 1
        device = 'gpu' if torch.cuda.is_available() else 'cpu'
        texts, rttms = [], []
        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8') as f:
            with open(manifest_path, 'r', encoding='utf-8') as mfile:
                for ix, line in enumerate(mfile):
                    if ix >= num_samples:
                        break

                    # line = line.replace("tests/data/", "tests/.data/").replace("\n", "")
                    line = line.replace("tests/data/", f"{test_data_dir}/tests/.data/").replace("\n", "")
                    f.write(f"{line}\n")

                    data = json.loads(line)
                    texts.append(data['text'])
                    rttms.append(data['rttm_filepath'])

            f.seek(0)
            featurizer = WaveformFeaturizer(sample_rate=16000, int_values=False, augmentor=None)
            preprocessor = AudioToMelSpectrogramPreprocessor(normalize="per_feature",
                                                            window_size=0.025,
                                                            sample_rate=16000,
                                                            window_stride=0.01,
                                                            window="hann",
                                                            features=80,
                                                            n_fft=512,
                                                            frame_splicing=1,
                                                            dither=0.00001
                                                        )

            dataset = AudioToSpeechE2ESpkDiarDataset(
                manifest_filepath=f.name,
                preprocessor=preprocessor,
                soft_label_thres=0.5,
                session_len_sec=90,
                num_spks=4,
                featurizer=featurizer,
                window_stride=0.01,
                global_rank=0,
                soft_targets=False,
                )
            import ipdb; ipdb.set_trace()

          