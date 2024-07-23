# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
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

import numpy as np
import pytest
import torch
from omegaconf import DictConfig

from nemo.collections.asr.parts.utils.asr_multispeaker_utils import (
    apply_spk_mapping,
    shuffle_spk_mapping,
    speaker_to_target,
    get_mask_from_segments,
    get_hidden_length_from_sample_length,
)

# class TestShuffleSpeakerMapping:
    
# class TestSpeakerToTarget:
    
# class TestGetMaskFromSegments:
    
class TestGetHiddenLengthFromSampleLength:
    @pytest.mark.parametrize(
        "num_samples, num_sample_per_mel_frame, num_mel_frame_per_asr_frame, expected_hidden_length",
        [
            (160, 160, 8, 1),
            (1280, 160, 8, 2),
            (0, 160, 8, 1),
            (159, 160, 8, 1),
            (129, 100, 5, 1),
            (300, 150, 3, 1),
        ]
    )
    def test_various_cases(self, num_samples, num_sample_per_mel_frame, num_mel_frame_per_asr_frame, expected_hidden_length):
        result = get_hidden_length_from_sample_length(num_samples, num_sample_per_mel_frame, num_mel_frame_per_asr_frame)
        assert result == expected_hidden_length

    def test_default_parameters(self):
        assert get_hidden_length_from_sample_length(160) == 1
        assert get_hidden_length_from_sample_length(1280) == 2
        assert get_hidden_length_from_sample_length(0) == 1
        assert get_hidden_length_from_sample_length(159) == 1

    def test_edge_cases(self):
        assert get_hidden_length_from_sample_length(159, 160, 8) == 1
        assert get_hidden_length_from_sample_length(160, 160, 8) == 1
        assert get_hidden_length_from_sample_length(161, 160, 8) == 1
        assert get_hidden_length_from_sample_length(1279, 160, 8) == 1

    def test_real_life_examples(self):
        # The samples tried when this function was designed.
        assert get_hidden_length_from_sample_length(160000) == 126
        assert get_hidden_length_from_sample_length(159999) == 125
        assert get_hidden_length_from_sample_length(158720) == 125
        assert get_hidden_length_from_sample_length(158719) == 124
        
        assert get_hidden_length_from_sample_length(158880) == 125
        assert get_hidden_length_from_sample_length(158879) == 125
        assert get_hidden_length_from_sample_length(1600) == 2
        assert get_hidden_length_from_sample_length(1599) == 2