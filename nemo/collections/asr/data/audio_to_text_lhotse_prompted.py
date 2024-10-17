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
from dataclasses import dataclass
from typing import Callable, Optional, Union

import torch.utils.data
from lhotse import CutSet
from lhotse.dataset import AudioSamples
from lhotse.dataset.collation import collate_vectors, collate_matrices
from lhotse.utils import compute_num_samples
from lhotse import SupervisionSet
from nemo.collections.asr.parts.utils.asr_multispeaker_utils import (
    speaker_to_target, 
    get_hidden_length_from_sample_length, 
    get_mask_from_segments,
    shuffle_spk_mapping
)
from nemo.collections.common.tokenizers import TokenizerSpec
from nemo.collections.asr.data.audio_to_text_lhotse import TokenizerWrapper

@dataclass
class PromptedAudioToTextMiniBatch:
    audio: torch.Tensor
    audio_lens: torch.Tensor
    transcript: torch.Tensor
    transcript_lens: torch.Tensor
    prompt: torch.Tensor
    prompt_lens: torch.Tensor
    prompted_transcript: torch.Tensor
    prompted_transcript_lens: torch.Tensor
    cuts: Optional[CutSet] = None

    def get_decoder_inputs_outputs(self) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns the inputs and outputs of transformer decoder for training.
        The input is ``prompted_transcript`` (minus last token),
        and the output is ``prompted_transcript`` (minus first token).
        """
        return self.prompted_transcript[:, :-1], self.prompted_transcript[:, 1:]


@dataclass
class PromptedAudioToTextSpkMiniBatch:
    audio: torch.Tensor
    audio_lens: torch.Tensor
    transcript: torch.Tensor
    transcript_lens: torch.Tensor
    prompt: torch.Tensor
    prompt_lens: torch.Tensor
    prompted_transcript: torch.Tensor
    prompted_transcript_lens: torch.Tensor
    spk_targets: torch.Tensor
    spk_mappings: torch.Tensor
    
    def get_decoder_inputs_outputs(self) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns the inputs and outputs of transformer decoder for training.
        The input is ``prompted_transcript`` (minus last token),
        and the output is ``prompted_transcript`` (minus first token).
        """
        return self.prompted_transcript[:, :-1], self.prompted_transcript[:, 1:]

class PromptedAudioToTextLhotseDataset(torch.utils.data.Dataset):
    """
    This dataset is based on :class:`~nemo.collections.asr.data.audio_to_text_lhotse.LhotseSpeechToTextBpeDataset`.
    It is a Lhotse-style dataset that converts a mini-batch of Cuts into tensors.
    The main difference from ``LhotseSpeechToTextBpeDataset`` is that we introduce
    a special prompt format for multitask encoder-decoder models.

    To perform the prompt formatting, we accept a ``prompt_format_fn``.
    It's expected to accept:
    * a ``CutSet`` which it will internally iterate over for utterances, and
    * a ``TokenizerWrapper`` object that will be internally used to tokenize the utterances

    Tokenized utterances will be extended with special prompt tokens according to ``prompt_format_fn`` logic.
    We support cuts with multiple supervision segments -- their tokenized texts will be concatenated before we add the prompt tokens.
    This is useful, for example, in code-switched scenarios where each segment is spoken in a different language.
    """

    def __init__(
        self,
        tokenizer: TokenizerSpec,
        prompt_format_fn: Callable[
            [CutSet, TokenizerSpec], tuple[list[torch.Tensor], list[torch.Tensor], list[torch.Tensor]]
        ],
    ):
        super().__init__()
        self.tokenizer = tokenizer
        self.load_audio = AudioSamples(fault_tolerant=True)
        self.padding_value = self.tokenizer.pad
        self.prompt_format_fn = prompt_format_fn

    def __getitem__(self, cuts: CutSet) -> PromptedAudioToTextMiniBatch:
        audio, audio_lens, cuts = self.load_audio(cuts)

        # Fast-path: the tokenization and prompt formatting was already done before sampling.
        attrs = ("tokenized_prompt", "tokenized_transcript", "tokenized_prompted_transcript")
        pre_formatted = all(hasattr(c, a) for c in cuts for a in attrs)
        if pre_formatted:
            prompts_with_answers, prompts, answers = zip(
                *((c.tokenized_prompted_transcript, c.tokenized_prompt, c.tokenized_transcript) for c in cuts)
            )
        else:
            prompts_with_answers, prompts, answers = self.prompt_format_fn(cuts, self.tokenizer)

        transcript, transcript_lens = self._collate_tokens(answers)
        prompts_with_answers, prompts_with_answers_lens = self._collate_tokens(prompts_with_answers)
        prompts, prompt_lens = self._collate_tokens(prompts)

        return PromptedAudioToTextMiniBatch(
            audio=audio,
            audio_lens=audio_lens,
            transcript=transcript,
            transcript_lens=transcript_lens,
            prompt=prompts,
            prompt_lens=prompt_lens,
            prompted_transcript=prompts_with_answers,
            prompted_transcript_lens=prompts_with_answers_lens,
            cuts=cuts.drop_in_memory_data(),
        )

    def _collate_tokens(self, tokens: list[Union[list[int], torch.Tensor]]) -> tuple[torch.Tensor, torch.Tensor]:
        tokens = [torch.as_tensor(t) for t in tokens]
        token_lens = torch.tensor([t.size(0) for t in tokens], dtype=torch.long)
        tokens = collate_vectors(tokens, padding_value=self.padding_value)
        return tokens, token_lens


class ProbablyIncorrectLanguageKeyError(RuntimeError):
    pass


class PromptedAudioToTextSpkLhotseDataset(torch.utils.data.Dataset):
    """
    This dataset is based on :class:`~nemo.collections.asr.data.audio_to_text_lhotse.LhotseSpeechToTextBpeDataset`.
    It is a Lhotse-style dataset that converts a mini-batch of Cuts into tensors.
    The main difference from ``LhotseSpeechToTextBpeDataset`` is that we introduce
    a special prompt format for multitask encoder-decoder models.

    To perform the prompt formatting, we accept a ``prompt_format_fn``.
    It's expected to accept:
    * a ``CutSet`` which it will internally iterate over for utterances, and
    * a ``TokenizerWrapper`` object that will be internally used to tokenize the utterances

    Tokenized utterances will be extended with special prompt tokens according to ``prompt_format_fn`` logic.
    We support cuts with multiple supervision segments -- their tokenized texts will be concatenated before we add the prompt tokens.
    This is useful, for example, in code-switched scenarios where each segment is spoken in a different language.
    """
    def __init__(
        self,
        cfg,
        tokenizer: TokenizerSpec,
        prompt_format_fn: Callable[
            [CutSet, TokenizerSpec], tuple[list[torch.Tensor], list[torch.Tensor], list[torch.Tensor]]
        ],
        inference: bool = False,
    ):
        super().__init__()
        self.cfg = cfg
        self.tokenizer = tokenizer
        # self.tokenizer = TokenizerWrapper(tokenizer)
        self.load_audio = AudioSamples(fault_tolerant=True)
        self.padding_value = self.tokenizer.pad
        self.prompt_format_fn = prompt_format_fn
        self.inference = inference
        
        self.spk_tar_all_zero = self.cfg.get('spk_tar_all_zero',False)
        self.shuffle_spk_mapping = self.cfg.get('shuffle_spk_mapping', False)
        self.num_speakers = self.cfg.get('num_speakers', 4)
        self.num_sample_per_mel_frame = self.cfg.get('num_sample_per_mel_frame', 160)
        self.num_mel_frame_per_asr_frame = self.cfg.get('num_mel_frame_per_asr_frame', 8)
        self.spk_token_pattern= r'<\|spltoken\d+\|>'
    
    def _collate_tokens(self, tokens: list[Union[list[int], torch.Tensor]]) -> tuple[torch.Tensor, torch.Tensor]:
        tokens = [torch.as_tensor(t) for t in tokens]
        token_lens = torch.tensor([t.size(0) for t in tokens], dtype=torch.long)
        tokens = collate_vectors(tokens, padding_value=self.padding_value)
        return tokens, token_lens
 
    def __getitem__(self, cuts: CutSet) -> PromptedAudioToTextSpkMiniBatch:
        cuts, spk_mappings = shuffle_spk_mapping(cuts=cuts, 
                                                 num_speakers=self.num_speakers, 
                                                 shuffle_spk_mapping=self.shuffle_spk_mapping, 
                                                 pattern=self.spk_token_pattern
                                                )
        audio, audio_lens, cuts = self.load_audio(cuts)
        spk_targets = [torch.as_tensor(speaker_to_target(_cut, self.num_speakers, self.num_sample_per_mel_frame, self.num_mel_frame_per_asr_frame, self.spk_tar_all_zero), dtype=torch.float32) for _cut in cuts]
        spk_targets = collate_matrices(spk_targets)
        
        # Fast-path: the tokenization and prompt formatting was already done before sampling.
        attrs = ("tokenized_prompt", "tokenized_transcript", "tokenized_prompted_transcript")
        pre_formatted = all(hasattr(c, a) for c in cuts for a in attrs)
        if pre_formatted:
            prompts_with_answers, prompts, answers = zip(
                *((c.tokenized_prompted_transcript, c.tokenized_prompt, c.tokenized_transcript) for c in cuts)
            )
        else:
            prompts_with_answers, prompts, answers = self.prompt_format_fn(cuts, self.tokenizer)

        transcript, transcript_lens = self._collate_tokens(answers)
        prompts_with_answers, prompts_with_answers_lens = self._collate_tokens(prompts_with_answers)
        prompts, prompt_lens = self._collate_tokens(prompts)

        return PromptedAudioToTextSpkMiniBatch(
            audio=audio,
            audio_lens=audio_lens,
            transcript=transcript,
            transcript_lens=transcript_lens,
            prompt=prompts,
            prompt_lens=prompt_lens,
            prompted_transcript=prompts_with_answers,
            prompted_transcript_lens=prompts_with_answers_lens,
            spk_targets=spk_targets,
            spk_mappings=spk_mappings,
        )