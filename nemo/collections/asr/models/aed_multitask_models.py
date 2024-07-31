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
from dataclasses import dataclass, field
from math import ceil
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf, open_dict
from pytorch_lightning import Trainer

from nemo.collections.asr.data.audio_to_text_lhotse_prompted import (
    PromptedAudioToTextLhotseDataset,
    PromptedAudioToTextSpkLhotseDataset,
    get_prompt_format_fn,
)
from nemo.collections.asr.metrics import BLEU, WER
from nemo.collections.asr.metrics.der import concat_perm_word_error_rate as cpWER
from nemo.collections.asr.models.asr_model import ASRModel, ExportableEncDecModel
from nemo.collections.asr.models.label_models import EncDecSpeakerLabelModel
# from nemo.collections.asr.models import SortformerEncLabelModel
from nemo.collections.asr.models.eesd_models import SortformerEncLabelModel
from nemo.collections.asr.parts.mixins import ASRBPEMixin, ASRTranscriptionMixin
from nemo.collections.asr.parts.mixins.transcription import (
    GenericTranscriptionType,
    InternalTranscribeConfig,
    TranscribeConfig,
)
from nemo.collections.asr.parts.submodules.multitask_decoding import MultiTaskDecoding, MultiTaskDecodingConfig
from nemo.collections.asr.parts.submodules.token_classifier import TokenClassifier
from nemo.collections.asr.parts.utils import manifest_utils
from nemo.collections.asr.parts.utils.audio_utils import ChannelSelectorType
from nemo.collections.asr.parts.utils.rnnt_utils import Hypothesis
from nemo.collections.common import tokenizers
from nemo.collections.asr.parts.utils.asr_multispeaker_utils import apply_spk_mapping
# from nemo.collections.common.data.lhotse.dataloader import get_lhotse_dataloader_from_config
from nemo.collections.common.data.lhotse import get_lhotse_dataloader_from_config
from nemo.collections.common.metrics import GlobalAverageLossMetric
from nemo.collections.common.parts import transformer_weights_init
from nemo.collections.common.parts.preprocessing.manifest import get_full_path
from nemo.core.classes.common import typecheck
from nemo.core.neural_types import (
    AudioSignal,
    ChannelType,
    LabelsType,
    StringType,
    LengthsType,
    LogprobsType,
    MaskType,
    NeuralType,
    SpectrogramType,
)
from nemo.utils import logging, model_utils

__all__ = ['EncDecMultiTaskModel', 'MSEncDecMultiTaskModel']


def lens_to_mask(lens, max_length):
    batch_size = lens.shape[0]
    mask = torch.arange(max_length).repeat(batch_size, 1).to(lens.device) < lens[:, None]
    return mask


def _config_check(cfg):
    if 'tokenizer' not in cfg:
        raise ValueError("`cfg` must have `tokenizer` config to create a tokenizer !")
    # Assert config has "prompt_format"
    if "prompt_format" not in cfg:
        raise ValueError("`cfg` must have `prompt_format` config to create a multi task model !")
    # Assert config has `model_defaults`
    if 'model_defaults' not in cfg:
        raise ValueError("`cfg` must have `model_defaults` config to create a model !")
    if "asr_enc_hidden" not in cfg.model_defaults:
        raise ValueError("`cfg.model_defaults` must have `asr_enc_hidden` key !")
    if "lm_enc_hidden" not in cfg.model_defaults:
        raise ValueError("`cfg.model_defaults` must have `lm_enc_hidden` key !")
    if "lm_dec_hidden" not in cfg.model_defaults:
        raise ValueError("`cfg.model_defaults` must have `lm_dec_hidden` key !")


@dataclass
class MultiTaskTranscriptionInternalConfig(InternalTranscribeConfig):
    """
    Configuration for Multi Task Transcription
    """

    manifest_filepath: Optional[str] = None
    primary_language: Optional[str] = None


@dataclass
class MultiTaskTranscriptionConfig(TranscribeConfig):
    """
    Configuration for Multi Task Transcription
    """
    task: Optional[str] = None
    pnc: Optional[bool] = None
    source_lang: Optional[str] = None
    target_lang: Optional[str] = None
    text_field: str = "answer"
    lang_field: str = "target_lang"

    # Multispeaker configs
    shuffle_spk_mapping: bool = False
    spk_supervision_strategy: str = 'diar'

    _internal: Optional[MultiTaskTranscriptionInternalConfig] = field(
        default_factory=lambda: MultiTaskTranscriptionInternalConfig()
    )

    def __post_init__(self):
        required_fields = ['task', 'pnc', 'source_lang', 'target_lang', 'text_field', 'lang_field']
        for field in required_fields:
            if not hasattr(self, field):
                raise ValueError(f"`{field}` must be present in the transcription config: {self}")


class EncDecMultiTaskModel(ASRModel, ExportableEncDecModel, ASRBPEMixin, ASRTranscriptionMixin):
    """Base class for AED multi-task models"""

    def __init__(self, cfg: DictConfig, trainer: Trainer = None):

        # Convert to Hydra 1.0 compatible DictConfig
        cfg = model_utils.convert_model_config_to_dict_config(cfg)
        cfg = model_utils.maybe_update_config_version(cfg)
        _config_check(cfg)

        self.prompt_format = cfg.prompt_format
        self.sample_rate = cfg.sample_rate
        self._setup_tokenizer(cfg.tokenizer)

        super().__init__(cfg=cfg, trainer=trainer)

        # Setup audio preprocessor
        self.preprocessor = EncDecMultiTaskModel.from_config_dict(self.cfg.preprocessor)
        # Setup audio encoder
        self.encoder = EncDecMultiTaskModel.from_config_dict(self.cfg.encoder)

        # Add projection layer if encoder and decoder differ in hidden size
        asr_enc_hidden_size = self.cfg.model_defaults.asr_enc_hidden
        decoder_hidden_size = self.cfg.model_defaults.lm_dec_hidden
        if asr_enc_hidden_size != decoder_hidden_size:
            self.encoder_decoder_proj = torch.nn.Linear(asr_enc_hidden_size, decoder_hidden_size)
        else:
            self.encoder_decoder_proj = torch.nn.Identity()

        transf_encoder_cfg_dict = self.cfg.get('transf_encoder', None)

        # Whether to add Transformer Encoder block between Conformer and Transformer Decoder
        self.use_transf_encoder = False
        if transf_encoder_cfg_dict is not None and transf_encoder_cfg_dict['num_layers'] > 0:
            self.use_transf_encoder = True

            self.transf_encoder = EncDecMultiTaskModel.from_config_dict(transf_encoder_cfg_dict)

            # Initialize weights
            std_init_range = 1 / self.cfg.model_defaults.lm_enc_hidden ** 0.5
            self.transf_encoder.apply(lambda module: transformer_weights_init(module, std_init_range))

        transf_decoder_cfg_dict = cfg.transf_decoder

        # Transformer decoder
        vocab_size = 8 * ceil(self.tokenizer.vocab_size / 8)

        # Auto inject vocab size for `get_transformer`
        with open_dict(transf_decoder_cfg_dict):
            if 'config_dict' in transf_decoder_cfg_dict:
                transf_decoder_cfg_dict['config_dict']['vocab_size'] = vocab_size

        self.transf_decoder = EncDecMultiTaskModel.from_config_dict(transf_decoder_cfg_dict)

        # Setup token classifier
        with open_dict(self.cfg.head):
            self.cfg.head.num_classes = vocab_size

        self.log_softmax = EncDecMultiTaskModel.from_config_dict(self.cfg.head)

        # Weight tying - if using TokenClassifier only
        if isinstance(self.log_softmax, TokenClassifier):
            self.log_softmax.mlp.layer0.weight = self.transf_decoder.embedding.token_embedding.weight

        # Initialize weights
        std_init_range = 1 / self.cfg.model_defaults.lm_dec_hidden ** 0.5
        self.transf_decoder.apply(lambda module: transformer_weights_init(module, std_init_range))
        self.log_softmax.apply(lambda module: transformer_weights_init(module, std_init_range))

        # Setup decoding objects
        decoding_cfg = self.cfg.get('decoding', None)

        # In case decoding config not found, use default config
        if decoding_cfg is None:
            decoding_cfg = OmegaConf.structured(MultiTaskDecodingConfig)
            with open_dict(self.cfg):
                self.cfg.decoding = decoding_cfg

        self.decoding = MultiTaskDecoding(
            decoding_cfg=self.cfg.decoding,
            transformer_decoder=self.transf_decoder,
            log_softmax_module=self.log_softmax,
            tokenizer=self.tokenizer,
        )

        # Define autoregressive CE loss
        with open_dict(self.cfg.loss):
            self.cfg.loss.pad_id = self.tokenizer.pad_id

        self.loss = EncDecMultiTaskModel.from_config_dict(self.cfg.loss)

        if hasattr(self.cfg, 'spec_augment') and self.cfg.spec_augment is not None:
            self.spec_augmentation = EncDecMultiTaskModel.from_config_dict(self.cfg.spec_augment)
        else:
            self.spec_augmentation = None

        self.val_loss = GlobalAverageLossMetric(dist_sync_on_step=False, take_avg_loss=True)

        # TODO: PytorchMetrics lets you join two metrics together to save compute. But need to make wer and bleu have same outputs first
        self.wer = WER(self.decoding, log_prediction=self.cfg.get("log_prediction"))
        self.bleu = BLEU(
            self.decoding, tokenize=self.cfg.get('bleu_tokenizer', "13a"), log_prediction=False
        )  # Wer is handling logging

    def change_decoding_strategy(self, decoding_cfg: DictConfig):
        """
        Changes decoding strategy used during Multi Task decoding process.

        Args:
            decoding_cfg: A config for the decoder, which is optional. If the decoding type
                needs to be changed (from say Greedy to Beam decoding etc), the config can be passed here.
        """
        if decoding_cfg is None:
            # Assume same decoding config as before
            logging.info("No `decoding_cfg` passed when changing decoding strategy, using internal config")
            decoding_cfg = self.cfg.decoding

        # Assert the decoding config with all hyper parameters
        decoding_cls = OmegaConf.structured(MultiTaskDecodingConfig)
        decoding_cls = OmegaConf.create(OmegaConf.to_container(decoding_cls))
        decoding_cfg = OmegaConf.merge(decoding_cls, decoding_cfg)

        self.decoding = MultiTaskDecoding(
            decoding_cfg=decoding_cfg,
            transformer_decoder=self.transf_decoder,
            log_softmax_module=self.log_softmax,
            tokenizer=self.tokenizer,
        )

        # Update config
        with open_dict(self.cfg.decoding):
            self.cfg.decoding = decoding_cfg

        logging.info(f"Changed decoding strategy to \n{OmegaConf.to_yaml(self.cfg.decoding)}")

    def change_vocabulary(
        self,
        new_tokenizer_dir: Union[str, DictConfig],
        new_tokenizer_type: str,
        decoding_cfg: Optional[DictConfig] = None,
        prompt_format: Optional[str] = None,
    ):
        """
        Changes vocabulary used during AED decoding process. Use this method when fine-tuning on from pre-trained model.
        This method changes only decoder and leaves encoder and pre-processing modules unchanged. For example, you would
        use it if you want to use pretrained encoder when fine-tuning on data in another language, or when you'd need
        model to learn capitalization, punctuation and/or special characters.

        Args:
            new_tokenizer_dir: Directory path to tokenizer or a config for a new tokenizer (if the tokenizer type is `agg`)
            new_tokenizer_type: Type of tokenizer. Can be either `agg`, `bpe` or `wpe`.
            decoding_cfg: A config for the decoding, which is optional. If the decoding type
                needs to be changed (from say Greedy to Beam decoding etc), the config can be passed here.
            prompt_format: A string alias of the object that represents the prompt structure.
                If not None, it will be used to update the prompt format.
        """
        if isinstance(new_tokenizer_dir, (dict, DictConfig)):
            if new_tokenizer_type == 'agg':
                if not isinstance(new_tokenizer_dir, DictConfig):
                    new_tokenizer_dir = OmegaConf.create(new_tokenizer_dir)

                new_tokenizer_cfg = new_tokenizer_dir
            else:
                raise ValueError(
                    f'New tokenizer dir should be a string unless the tokenizer is `agg`, but this tokenizer type is: {new_tokenizer_type}'
                )
        else:
            new_tokenizer_cfg = None

        if new_tokenizer_cfg is not None:
            tokenizer_cfg = new_tokenizer_cfg
        else:
            if not os.path.isdir(new_tokenizer_dir):
                raise NotADirectoryError(
                    f'New tokenizer dir must be non-empty path to a directory. But instead got: {new_tokenizer_dir}'
                )

            if new_tokenizer_type.lower() not in ('bpe', 'wpe'):
                raise ValueError(f'New tokenizer type must be either `bpe` or `wpe`')

            tokenizer_cfg = OmegaConf.create({'dir': new_tokenizer_dir, 'type': new_tokenizer_type})

        if prompt_format is None:
            prompt_format = self.cfg.prompt_format

        # Setup the tokenizer
        self._setup_tokenizer(tokenizer_cfg)

        # Initialize a dummy vocabulary
        vocabulary = self.tokenizer.tokenizer.get_vocab()

        # Setup Decoder
        transf_decoder_cfg_dict = self.transf_decoder.to_config_dict()

        vocab_size = 8 * ceil(self.tokenizer.vocab_size / 8)

        # Auto inject vocab size for `get_transformer`
        with open_dict(transf_decoder_cfg_dict):
            if 'config_dict' in transf_decoder_cfg_dict:
                transf_decoder_cfg_dict['config_dict']['vocab_size'] = vocab_size

        original_decoder_state_dict = self.transf_decoder.state_dict()
        self.transf_decoder = EncDecMultiTaskModel.from_config_dict(transf_decoder_cfg_dict)

        # Partially load the original state dict into the new decoder
        decoder_state_dict = self.transf_decoder.state_dict()
        for og_key, og_value in original_decoder_state_dict.items():
            if og_key in decoder_state_dict and og_value.shape == decoder_state_dict[og_key].shape:
                decoder_state_dict[og_key] = og_value
            else:
                logging.warning(
                    f"Skipping key `{og_key}` in the `transf_decoder` module from original state dict due "
                    f"to shape mismatch after change in vocabulary.\n"
                    f"Original shape: {og_value.shape}, New shape: {decoder_state_dict[og_key].shape}"
                )

        self.transf_decoder.load_state_dict(decoder_state_dict)

        # Setup token classifier
        with open_dict(self.cfg.head):
            self.cfg.head.num_classes = vocab_size

        del self.log_softmax
        self.log_softmax = EncDecMultiTaskModel.from_config_dict(self.cfg.head)

        # Weight tying - if using TokenClassifier only
        if isinstance(self.log_softmax, TokenClassifier):
            self.log_softmax.mlp.layer0.weight = self.transf_decoder.embedding.token_embedding.weight

        # Initialize weights of token classifier
        std_init_range = 1 / self.cfg.model_defaults.lm_dec_hidden ** 0.5
        self.log_softmax.apply(lambda module: transformer_weights_init(module, std_init_range))

        # Setup Decoding class
        if decoding_cfg is None:
            # Assume same decoding config as before
            decoding_cfg = self.cfg.decoding

        # Assert the decoding config with all hyper parameters
        decoding_cls = OmegaConf.structured(MultiTaskDecodingConfig)
        decoding_cls = OmegaConf.create(OmegaConf.to_container(decoding_cls))
        decoding_cfg = OmegaConf.merge(decoding_cls, decoding_cfg)

        del self.decoding
        self.decoding = MultiTaskDecoding(
            decoding_cfg=decoding_cfg,
            transformer_decoder=self.transf_decoder,
            log_softmax_module=self.log_softmax,
            tokenizer=self.tokenizer,
        )

        with open_dict(self.cfg.decoding):
            self.cfg.decoding = decoding_cfg

        # Setup loss
        with open_dict(self.cfg.loss):
            self.cfg.loss.pad_id = self.tokenizer.pad_id

        del self.loss
        self.loss = EncDecMultiTaskModel.from_config_dict(self.cfg.loss)

        # Update config
        with open_dict(self.cfg):
            self.cfg.prompt_format = prompt_format

        logging.info(f"Changed decoder to output to {vocabulary} vocabulary.")

    @torch.no_grad()
    def transcribe(
        self,
        audio: Union[List[str], str],
        batch_size: int = 4,
        return_hypotheses: bool = False,
        task: Optional[str] = None,
        pnc: Optional[bool] = None,
        source_lang: Optional[str] = None,
        target_lang: Optional[str] = None,
        num_workers: int = 0,
        channel_selector: Optional[ChannelSelectorType] = None,
        augmentor: DictConfig = None,
        verbose: bool = True,
        override_config: Optional[MultiTaskTranscriptionConfig] = None,
    ) -> Union[List[str], List[Hypothesis]]:
        """
        Uses greedy decoding to transcribe audio files. Use this method for debugging and prototyping.
        Args:
            audio: (a list) of paths to audio files. \
                Recommended length per file is between 5 and 25 seconds. \
                But it is possible to pass a few hours long file if enough GPU memory is available.
            batch_size: (int) batch size to use during inference.
                Bigger will result in better throughput performance but would use more memory.
            return_hypotheses: (bool) Either return hypotheses or text
                With hypotheses can do some postprocessing like getting timestamp or rescoring
            task: (str) task name. Defaults to `asr`.
            pnc: (bool) whether to apply punctuation and capitalization or not. Defaults to True.
            source_lang: (str) source language. Defaults to `en`.
            target_lang: (str) target language. Defaults to `en`.
            num_workers: (int) number of workers for DataLoader
            channel_selector (int | Iterable[int] | str): select a single channel or a subset of channels from multi-channel audio. If set to `'average'`, it performs averaging across channels. Disabled if set to `None`. Defaults to `None`.
            augmentor: (DictConfig): Augment audio samples during transcription if augmentor is applied.
            verbose: (bool) whether to display tqdm progress bar
            override_config: (Optional[MultiTaskTranscriptionConfig]) A config to override the default config.

        Returns:
            A list of transcriptions (or raw log probabilities if logprobs is True) in the same order as paths2audio_files
        """
        if override_config is None:
            trcfg = MultiTaskTranscriptionConfig(
                batch_size=batch_size,
                return_hypotheses=return_hypotheses,
                num_workers=num_workers,
                channel_selector=channel_selector,
                augmentor=augmentor,
                verbose=verbose,
                task=task,
                pnc=pnc,
                source_lang=source_lang,
                target_lang=target_lang,
            )
        else:
            if not isinstance(override_config, MultiTaskTranscriptionConfig):
                raise ValueError(
                    f"override_config must be of type {MultiTaskTranscriptionConfig}, "
                    f"but got {type(override_config)}"
                )
            trcfg = override_config

        return super().transcribe(audio=audio, override_config=trcfg)

    def _setup_dataloader_from_config(self, config: Optional[Dict], inference: bool = False):
        assert config.get("use_lhotse", False), (
            "Multi-task model only supports dataloading with Lhotse. "
            "Please set config.{train,validation,test}_ds.use_lhotse=True"
        )
        return get_lhotse_dataloader_from_config(
            config,
            global_rank=self.global_rank,
            world_size=self.world_size,
            dataset=PromptedAudioToTextLhotseDataset(
                tokenizer=self.tokenizer,
                prompt_format_fn=get_prompt_format_fn(self.prompt_format),
                inference=inference,
            ),
        )

    def setup_training_data(self, train_data_config: Optional[DictConfig]):

        # create audio-only data loader
        self._update_dataset_config(dataset_name='train', config=train_data_config)
        self._train_dl = self._setup_dataloader_from_config(config=train_data_config)

        # Need to set this because if using an IterableDataset, the length of the
        # dataloader is the total number of samples rather than the number of batches,
        # and this messes up the tqdm progress bar. So we set the number of steps manually
        # (to the correct number) to fix this.
        if 'is_tarred' in train_data_config and train_data_config['is_tarred']:
            # We also need to check if limit_train_batches is already set.
            # If it's an int, we assume that the user has set it to something sane,
            # i.e. <= # training batches, and don't change it. Otherwise, adjust
            # batches accordingly if it's a float (including 1.0).
            if self._trainer is not None and isinstance(self._trainer.limit_train_batches, float):
                self._trainer.limit_train_batches = int(
                    self._trainer.limit_train_batches
                    * ceil((len(self._train_dl.dataset) / self.world_size) / train_data_config['batch_size'])
                )
            elif self._trainer is None:
                logging.warning(
                    "Model Trainer was not set before constructing the dataset, incorrect number of "
                    "training batches will be used. Please set the trainer and rebuild the dataset."
                )

    def setup_validation_data(self, val_data_config: Optional[Union[DictConfig, Dict]]):
        """
        Sets up the validation data loader via a Dict-like object.
        Args:
            val_data_config: A config that contains the information regarding construction
                of an ASR Training dataset.
        Supported Datasets:
            -   :class:`~nemo.collections.asr.data.audio_to_text_lhotse_prompted.PromptedAudioToTextLhotseDataset`
        """
        if 'shuffle' not in val_data_config:
            val_data_config['shuffle'] = False

        # preserve config
        self._update_dataset_config(dataset_name='validation', config=val_data_config)
        self._validation_dl = self._setup_dataloader_from_config(config=val_data_config, inference=True)

    def setup_test_data(self, test_data_config: Optional[Union[DictConfig, Dict]]):
        """
        Sets up the test data loader via a Dict-like object.
        Args:
            test_data_config: A config that contains the information regarding construction
                of an ASR Training dataset.
        Supported Datasets:
            -   :class:`~nemo.collections.asr.data.audio_to_text_lhotse_prompted.PromptedAudioToTextLhotseDataset`
        """
        if 'shuffle' not in test_data_config:
            test_data_config['shuffle'] = False

        # preserve config
        self._update_dataset_config(dataset_name='test', config=test_data_config)
        self._test_dl = self._setup_dataloader_from_config(config=test_data_config, inference=True)

    @property
    def input_types(self) -> Optional[Dict[str, NeuralType]]:
        if hasattr(self.preprocessor, '_sample_rate'):
            input_signal_eltype = AudioSignal(freq=self.preprocessor._sample_rate)
        else:
            input_signal_eltype = AudioSignal()
        return {
            "input_signal": NeuralType(('B', 'T'), input_signal_eltype, optional=True),
            "input_signal_length": NeuralType(tuple('B'), LengthsType(), optional=True),
            "processed_signal": NeuralType(('B', 'D', 'T'), SpectrogramType(), optional=True),
            "processed_signal_length": NeuralType(tuple('B'), LengthsType(), optional=True),
            "transcript": NeuralType(('B', 'T'), LabelsType(), optional=True),
            "transcript_length": NeuralType(tuple('B'), LengthsType(), optional=True),
            "prompt": NeuralType(('B', 'T'), LabelsType(), optional=True),
            "prompt_length": NeuralType(tuple('B'), LengthsType(), optional=True),
            "sample_id": NeuralType(tuple('B'), LengthsType(), optional=True),
        }

    @property
    def output_types(self) -> Optional[Dict[str, NeuralType]]:
        return {
            "transf_log_probs": NeuralType(('B', 'T', 'D'), LogprobsType()),
            "encoded_lengths": NeuralType(tuple('B'), LengthsType()),
            "encoder_states": NeuralType(('B', 'T', 'D'), ChannelType()),
            "encoder_mask": NeuralType(('B', 'T'), MaskType()),
        }

    @typecheck()
    def forward(
        self,
        input_signal=None,
        input_signal_length=None,
        processed_signal=None,
        processed_signal_length=None,
        transcript=None,
        transcript_length=None,
    ):
        """
        Forward pass of the model.
        Args:
            input_signal: Tensor that represents a batch of raw audio signals,
                of shape [B, T]. T here represents timesteps, with 1 second of audio represented as
                `self.sample_rate` number of floating point values.
            input_signal_length: Vector of length B, that contains the individual lengths of the audio
                sequences.
            processed_signal: Tensor that represents a batch of processed audio signals,
                of shape (B, D, T).
            processed_signal_length: Vector of length B, that contains the individual lengths of the
                processed audio sequences.
            # TODO: Add support for `transcript` and `transcript_length` in the docstring

        Returns:
            A tuple of 3 elements -
            1) The log probabilities tensor of shape [B, T, D].
            2) The lengths of the acoustic sequence after propagation through the encoder, of shape [B].
            3) The greedy token predictions of the model of shape [B, T] (via argmax)
        """
        has_input_signal = input_signal is not None and input_signal_length is not None
        has_processed_signal = processed_signal is not None and processed_signal_length is not None
        if (has_input_signal ^ has_processed_signal) == False:
            raise ValueError(
                f"{self} Arguments ``input_signal`` and ``input_signal_length`` are mutually exclusive "
                " with ``processed_signal`` and ``processed_signal_len`` arguments."
            )

        if not has_processed_signal:
            processed_signal, processed_signal_length = self.preprocessor(
                input_signal=input_signal, length=input_signal_length
            )

        if self.spec_augmentation is not None and self.training:
            processed_signal = self.spec_augmentation(input_spec=processed_signal, length=processed_signal_length)

        encoded, encoded_len = self.encoder(audio_signal=processed_signal, length=processed_signal_length)

        enc_states = encoded.permute(0, 2, 1)
        enc_states = self.encoder_decoder_proj(enc_states)
        enc_mask = lens_to_mask(encoded_len, enc_states.shape[1]).to(enc_states.dtype)
        if self.use_transf_encoder:
            enc_states = self.transf_encoder(encoder_states=enc_states, encoder_mask=enc_mask)

        transf_log_probs = None
        if transcript is not None:
            dec_mask = lens_to_mask(transcript_length, transcript.shape[1]).to(transcript.dtype)
            dec_states = self.transf_decoder(
                input_ids=transcript, decoder_mask=dec_mask, encoder_embeddings=enc_states, encoder_mask=enc_mask
            )
            transf_log_probs = self.log_softmax(hidden_states=dec_states)

        return transf_log_probs, encoded_len, enc_states, enc_mask

    # PTL-specific methods
    def training_step(self, batch, batch_nb):

        if batch is None:
            return torch.tensor([0.0])

        # During training prompt and prompt_len are null, ignore.
        signal, signal_len, transcript, transcript_len, prompt, prompt_len = batch
        input_ids, labels = transcript[:, :-1], transcript[:, 1:]

        transf_log_probs, encoded_len, enc_states, enc_mask = self.forward(
            input_signal=signal,
            input_signal_length=signal_len,
            transcript=input_ids,
            transcript_length=transcript_len,
        )

        audio_loss = self.loss(log_probs=transf_log_probs, labels=labels)

        tensorboard_logs = {
            'train_loss': audio_loss,
            'learning_rate': self._optimizer.param_groups[0]['lr'],
        }

        return {'loss': audio_loss, 'log': tensorboard_logs}

    def validation_pass(self, batch, batch_idx, dataloader_idx=0, eval_mode="val"):
        # During inference, dataloader passes pure prompt without transcript text.
        signal, signal_len, transcript, transcript_len, prompt, prompt_len = batch
        input_ids, labels = transcript[:, :-1], transcript[:, 1:]

        transf_log_probs, encoded_len, enc_states, enc_mask = self.forward(
            input_signal=signal,
            input_signal_length=signal_len,
            transcript=input_ids,
            transcript_length=transcript_len,
        )

        transf_loss = self.loss(log_probs=transf_log_probs, labels=labels)
        self.val_loss(loss=transf_loss, num_measurements=transf_log_probs.shape[0] * transf_log_probs.shape[1])
        output_dict = {
            f'{eval_mode}_loss': transf_loss,
        }

        self.wer.update(
            predictions=enc_states,
            predictions_lengths=encoded_len,
            targets=transcript,
            targets_lengths=transcript_len,
            predictions_mask=enc_mask,
            input_ids=prompt,
        )
        wer, wer_num, wer_denom = self.wer.compute()
        output_dict.update({"val_wer": wer, "val_wer_num": wer_num, "val_wer_denom": wer_denom})
        self.wer.reset()

        self.bleu.update(
            predictions=enc_states,
            predictions_lengths=encoded_len,
            targets=transcript,
            targets_lengths=transcript_len,
            predictions_mask=enc_mask,
            input_ids=prompt,
        )
        bleu_metrics = self.bleu.compute(prefix=f"{eval_mode}_")
        output_dict.update(bleu_metrics)
        self.bleu.reset()

        return output_dict

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        metrics = self.validation_pass(batch, batch_idx, dataloader_idx, eval_mode="val")
        if type(self.trainer.val_dataloaders) == list and len(self.trainer.val_dataloaders) > 1:
            self.validation_step_outputs[dataloader_idx].append(metrics)
        else:
            self.validation_step_outputs.append(metrics)
        return metrics

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        metrics = self.validation_pass(batch, batch_idx, dataloader_idx, eval_mode="test")
        if type(self.trainer.val_dataloaders) == list and len(self.trainer.val_dataloaders) > 1:
            self.validation_step_outputs[dataloader_idx].append(metrics)
        else:
            self.validation_step_outputs.append(metrics)
        return metrics

    def test_dataloader(self):
        if self._test_dl is not None:
            return self._test_dl

    """ Transcription methods """

    def _transcribe_on_begin(self, audio, trcfg: MultiTaskTranscriptionConfig):
        """
        Transcription setup method.
        Args:
            audio: A list of paths to audio files or a path to a manifest file.
            trcfg: A config for the transcription, which is optional. If the decoding type
                needs to be changed (from say Greedy to Beam decoding etc), the config can be passed here.
        """
        super()._transcribe_on_begin(audio, trcfg)

        # Switch model to evaluation mode
        self.transf_decoder.freeze()

        if isinstance(audio, list):
            logging.debug(f"Found 'audio' to be a list of {len(audio)} items.")
            logging.debug(f"Assuming each item in 'audio' is a path to audio file.")

            if isinstance(self.tokenizer, tokenizers.AggregateTokenizer):
                if hasattr(trcfg, '_internal') and hasattr(trcfg._internal, 'primary_language'):
                    trcfg._internal.primary_language = self.tokenizer.langs[0]
                    logging.debug(f"Transcribing with default setting of {trcfg._internal.primary_language}.")

        elif isinstance(audio, str):
            logging.debug(f"Found 'audio' to be a string. Assuming it is a path to manifest file.")
            assert os.path.exists(audio), f"File {audio} doesn't exist"
            # assert audio.endswith('.json') or audio.endswith('.jsonl'), f"File {audio} must be a json or jsonl file"

            # load json lines
            manifest_path = audio  # need to save this as we are overwriting paths2audio_files in nextline
            if audio.endswith('.json') or audio.endswith('.jsonl'):
                if hasattr(trcfg, '_internal') and hasattr(trcfg._internal, 'manifest_path'):
                    trcfg._internal.manifest_filepath = manifest_path

        elif isinstance(audio, (np.ndarray, torch.Tensor)):
            raise NotImplementedError("Transcribing from numpy or torch tensors is not supported yet.")

    def _transcribe_input_manifest_processing(
        self, audio_files: List[str], temp_dir: str, trcfg: MultiTaskTranscriptionConfig
    ) -> Dict[str, Any]:
        """
        Internal function to process the input audio filepaths and return a config dict for the dataloader.
        This implementation adds support for dictionaries as manifest items.

        Args:
            audio_files: A list of string filepaths for audio files, or a single string filepath for a manifest file.
            temp_dir: A temporary directory to store intermediate files.
            trcfg: The transcription config dataclass. Subclasses can change this to a different dataclass if needed.

        Returns:
            A config dict that is used to setup the dataloader for transcription.
        """
        manifest_filepath = None
        if len(audio_files) == 1 and isinstance(audio_files[0], str):
            # Check if manifest file is provided
            if (
                hasattr(trcfg._internal, 'manifest_filepath')
                and getattr(trcfg._internal, 'manifest_filepath') is not None
            ):
                manifest_filepath = trcfg._internal.manifest_filepath

            elif audio_files[0].endswith('.json') or audio_files[0].endswith('.jsonl'):
                # Assume it is a path to a manifest file
                manifest_filepath = audio_files[0]

            if manifest_filepath is not None:
                audio_files = manifest_utils.read_manifest(audio_files[0])

        audio_files = self._may_be_make_dict_and_fix_paths(audio_files, manifest_filepath, trcfg)

        return super()._transcribe_input_manifest_processing(audio_files, temp_dir, trcfg)

    def _transcribe_forward(self, batch: Any, trcfg: MultiTaskTranscriptionConfig):
        """
        Internal function to perform the model's custom forward pass to return outputs that are processed by
        `_transcribe_output_processing()`.
        This function is called by `transcribe()` and `transcribe_generator()` to perform the model's forward pass.

        Args:
            batch: A batch of input data from the data loader that is used to perform the model's forward pass.
            trcfg: The transcription config dataclass. Subclasses can change this to a different dataclass if needed.

        Returns:
            The model's outputs that are processed by `_transcribe_output_processing()`.
        """
        log_probs, encoded_len, enc_states, enc_mask = self.forward(
            input_signal=batch[0], input_signal_length=batch[1]
        )
        decoder_input_ids = batch[-2].to(trcfg._internal.device)
        output = dict(
            log_probs=log_probs,
            encoded_lengths=encoded_len,
            encoder_states=enc_states,
            encoder_mask=enc_mask,
            decoder_input_ids=decoder_input_ids,
        )
        return output

    def _transcribe_output_processing(self, outputs, trcfg: MultiTaskTranscriptionConfig) -> GenericTranscriptionType:
        """
        Internal function to process the model's outputs to return the results to the user. This function is called by
        `transcribe()` and `transcribe_generator()` to process the model's outputs.

        Args:
            outputs: The model's outputs that are processed by `_transcribe_forward()`.
            trcfg: The transcription config dataclass. Subclasses can change this to a different dataclass if needed.

        Returns:
            The output can be a list of
            objects, list of list of objects, tuple of objects, tuple of list of objects, or a dict of list of objects.
            Its type is defined in `TranscriptionReturnType`.
        """
        log_probs = outputs.pop('log_probs')
        encoded_len = outputs.pop('encoded_lengths')
        enc_states = outputs.pop('encoder_states')
        enc_mask = outputs.pop('encoder_mask')
        decoder_input_ids = outputs.pop('decoder_input_ids')

        del log_probs, encoded_len

        best_hypotheses, all_hypotheses = self.decoding.decode_predictions_tensor(
            encoder_hidden_states=enc_states,
            encoder_input_mask=enc_mask,
            decoder_input_ids=decoder_input_ids,
            return_hypotheses=trcfg.return_hypotheses,
        )

        if trcfg.return_hypotheses:
            for hyp in best_hypotheses:
                hyp.text = self.decoding.strip_special_tokens(hyp.text)
            if all_hypotheses is not None:
                for i in range(len(all_hypotheses)):
                    for j in range(len(all_hypotheses[i])):
                        all_hypotheses[i][j].text = self.decoding.strip_special_tokens(all_hypotheses[i][j].text)
        else:
            best_hypotheses = [self.decoding.strip_special_tokens(text) for text in best_hypotheses]
            if all_hypotheses is not None:
                for i in range(len(all_hypotheses)):
                    all_hypotheses[i] = [self.decoding.strip_special_tokens(text) for text in all_hypotheses[i]]

        del enc_states, enc_mask, decoder_input_ids
        if all_hypotheses is None:
            return best_hypotheses
        return best_hypotheses, all_hypotheses

    def _setup_transcribe_dataloader(self, config: Dict) -> 'torch.utils.data.DataLoader':
        """
        Setup function for a temporary data loader which wraps the provided audio file.
        Args:
            config: A python dictionary which contains the following keys:
            paths2audio_files: (a list) of paths to audio files. The files should be relatively short fragments. \
                Recommended length per file is between 5 and 25 seconds.
            batch_size: (int) batch size to use during inference. \
                Bigger will result in better throughput performance but would use more memory.
            temp_dir: (str) A temporary directory where the audio manifest is temporarily
                stored.
        Returns:
            A pytorch DataLoader for the given audio file(s).
        """
        batch_size = min(config['batch_size'], len(config['paths2audio_files']))
        dl_config = {
            'manifest_filepath': os.path.join(config['temp_dir'], 'manifest.json'),
            'sample_rate': self.preprocessor._sample_rate,
            'batch_size': batch_size,
            'trim_silence': False,
            'shuffle': False,
            'num_workers': min(batch_size, os.cpu_count() - 1),
            'pin_memory': True,
            'use_lhotse': True,
            'use_bucketing': False,
            'drop_last': False,
            'text_field': config.get('text_field', 'answer'),
            'lang_field': config.get('lang_field', 'target_lang'),
        }

        temporary_datalayer = self._setup_dataloader_from_config(config=DictConfig(dl_config), inference=True)
        return temporary_datalayer

    def _transcribe_on_end(self, trcfg: MultiTaskTranscriptionConfig):
        """
        Internal function to teardown the model after transcription. Perform all teardown and post-checks here.

        Args:
            trcfg: The transcription config dataclass. Subclasses can change this to a different dataclass if needed.
        """
        super()._transcribe_on_end(trcfg)

        self.transf_decoder.unfreeze()

    def _may_be_make_dict_and_fix_paths(self, json_items, manifest_path, trcfg: MultiTaskTranscriptionConfig):
        """
        Utility method to convert a list of strings to a list of dictionaries.

        Args:
            json_items: A list of strings or dictionaries.
            manifest_path: A path to a manifest file.
            trcfg: The transcription config dataclass. Subclasses can change this to a different dataclass if needed.

        Returns:
            A list of dictionaries with the audio file paths fixed.
        """
        out_json_items = []
        for item in json_items:
            if isinstance(item, str):
                # assume it is a path to audio file
                entry = {
                    'audio_filepath': item,
                    'duration': 100000,
                    'source_lang': 'en' if trcfg.source_lang is None else trcfg.source_lang,
                    'taskname': 'asr' if trcfg.task is None else trcfg.task,
                    'target_lang': 'en' if trcfg.target_lang is None else trcfg.target_lang,
                    'pnc': 'yes' if trcfg.pnc is None else 'yes' if trcfg.pnc else 'no',
                    trcfg.text_field: 'nothing',
                }
            elif isinstance(item, dict):
                entry = item
                entry['audio_filepath'] = get_full_path(entry['audio_filepath'], manifest_file=manifest_path)

                if 'source_lang' not in entry:
                    entry['source_lang'] = 'en' if trcfg.source_lang is None else trcfg.source_lang
                if 'taskname' not in entry:
                    entry['taskname'] = 'asr' if trcfg.task is None else trcfg.task
                if 'target_lang' not in entry:
                    entry['target_lang'] = 'en' if trcfg.target_lang is None else trcfg.target_lang
                if 'pnc' not in entry:
                    entry['pnc'] = 'yes' if trcfg.pnc is None else 'yes' if trcfg.pnc else 'no'
                if trcfg.text_field not in entry:
                    entry[trcfg.text_field] = 'nothing'
            else:
                raise ValueError(f"Expected str or dict, got {type(item)}")
            out_json_items.append(entry)
        return out_json_items

    @classmethod
    def get_transcribe_config(cls) -> MultiTaskTranscriptionConfig:
        """
        Utility method that returns the default config for transcribe() function.

        Returns:
            A dataclass
        """
        return MultiTaskTranscriptionConfig()

    def predict_step(self, batch, batch_idx=0, dataloader_idx=0, has_processed_signal=False):
        signal, signal_len, _, _, prompt, prompt_len = batch

        processed_signal = None
        processed_signal_length = None
        if has_processed_signal:
            processed_signal = signal
            processed_signal_length = signal_len
            signal = None
            signal_len = None

        transf_log_probs, encoded_len, enc_states, enc_mask = self.forward(
            input_signal=signal,
            input_signal_length=signal_len,
            processed_signal=processed_signal,
            processed_signal_length=processed_signal_length,
            transcript=prompt,
            transcript_length=prompt_len,
        )

        text = self.decoding.decode_predictions_tensor(
            encoder_hidden_states=enc_states,
            encoder_input_mask=enc_mask,
            decoder_input_ids=prompt,
            return_hypotheses=False,
        )[0]

        text = [self.decoding.strip_special_tokens(t) for t in text]
        return text

class MSEncDecMultiTaskModel(EncDecMultiTaskModel):
    """
    A Multi-Task model that uses a Masked Sequence-to-Sequence model for the encoder-decoder architecture.
    """

    def __init__(self, cfg: DictConfig, trainer: Trainer = None):
        super().__init__(cfg)
        # Initialize the asr branch
        self._init_asr_model()
        if 'diar_model_path' in self.cfg:
            self.diar = True
            # Initialize the speaker branch
            self._init_diar_model()
            
            if 'max_num_speakers' in cfg:
                self.max_num_speakers = cfg.max_num_speakers
            else:
                self.max_num_speakers = 4

            # layer normalization, ln, l2, or None
            if 'norm' in cfg:
                if cfg.norm == 'ln':
                    self.asr_norm = torch.nn.LayerNorm(cfg.model_defaults.asr_enc_hidden)
                    self.diar_norm = torch.nn.LayerNorm(4)
                self.norm = cfg.norm
            else:
                self.norm = None

            if 'kernel_norm' in cfg:
                self.kernel_norm = cfg.kernel_norm
            else:
                self.kernel_norm = None

            # projection layer
            proj_in_size = 4 + cfg.model_defaults.asr_enc_hidden
            proj_out_size = cfg.model_defaults.lm_dec_hidden
            self.joint_proj = torch.nn.Sequential(
                torch.nn.Linear(proj_in_size, proj_out_size*2),
                torch.nn.ReLU(),
                torch.nn.Linear(proj_out_size*2, proj_out_size)
            )

            # segment length and shift
            if 'segment_length' in cfg:
                self.segment_length = cfg.segment_length
            else:
                self.segment_length = 16
            if 'segment_shift' in cfg:
                self.segment_shift = cfg.segment_shift
            else:
                self.segment_shift = 8

            if 'diar_kernel_type' in cfg:
                if cfg.diar_kernel_type == 'sinusoidal':
                    self.diar_kernel_type = cfg.diar_kernel_type
                    self.diar_kernel = self.get_sinusoid_position_encoding(self.max_num_speakers, cfg.model_defaults.asr_enc_hidden)
            else:
                self.diar_kernel_type = 'projection'
                self.diar_kernel = self.joint_proj
        else:
            self.diar = False
        self.spk_token_pattern = r'<\|spltoken\d+\|>'
        
    @property
    def input_types(self) -> Optional[Dict[str, NeuralType]]:
        if hasattr(self.preprocessor, '_sample_rate'):
            input_signal_eltype = AudioSignal(freq=self.preprocessor._sample_rate)
        else:
            input_signal_eltype = AudioSignal()
        return {
            "input_signal": NeuralType(('B', 'T'), input_signal_eltype, optional=True),
            "input_signal_length": NeuralType(tuple('B'), LengthsType(), optional=True),
            "processed_signal": NeuralType(('B', 'D', 'T'), SpectrogramType(), optional=True),
            "processed_signal_length": NeuralType(tuple('B'), LengthsType(), optional=True),
            "transcript": NeuralType(('B', 'T'), LabelsType(), optional=True),
            "transcript_length": NeuralType(tuple('B'), LengthsType(), optional=True),
            "prompt": NeuralType(('B', 'T'), LabelsType(), optional=True),
            "prompt_length": NeuralType(tuple('B'), LengthsType(), optional=True),
            "sample_id": NeuralType(tuple('B'), LengthsType(), optional=True),
            "spk_targets": NeuralType(('B', 'T', 'D'), LabelsType(), optional=True),
            "spk_mappings": NeuralType(('B', 'D'), LabelsType(), optional=True),
            "spk_supervision_strategy": NeuralType(elements_type=StringType(), optional=True),
        }

    def _setup_dataloader_from_config(self, config: Optional[Dict], inference: bool = False):
        assert config.get("use_lhotse", False), (
            "Multi-task model only supports dataloading with Lhotse. "
            "Please set config.{train,validation,test}_ds.use_lhotse=True"
        )
        return get_lhotse_dataloader_from_config(
            config,
            global_rank=self.global_rank,
            world_size=self.world_size,
            dataset=PromptedAudioToTextSpkLhotseDataset(
                cfg=config,
                tokenizer=self.tokenizer,
                prompt_format_fn=get_prompt_format_fn(self.prompt_format),
                inference=inference,
            ),
        )

    def _transcribe_forward(self, batch: Any, trcfg: MultiTaskTranscriptionConfig):
        """
        Internal function to perform the model's custom forward pass to return outputs that are processed by
        `_transcribe_output_processing()`.
        This function is called by `transcribe()` and `transcribe_generator()` to perform the model's forward pass.

        Args:
            batch: A batch of input data from the data loader that is used to perform the model's forward pass.
            trcfg: The transcription config dataclass. Subclasses can change this to a different dataclass if needed.

        Returns:
            The model's outputs that are processed by `_transcribe_output_processing()`.
        """

        spk_mappings = batch[-1] if trcfg.shuffle_spk_mapping else None

        log_probs, encoded_len, enc_states, enc_mask = self.forward(
            input_signal=batch[0],
            input_signal_length=batch[1],
            spk_targets=batch[-2],
            spk_mappings=spk_mappings,
            spk_supervision_strategy = trcfg.spk_supervision_strategy
        )
        decoder_input_ids = batch[-4].to(trcfg._internal.device)
        output = dict(
            log_probs=log_probs,
            encoded_lengths=encoded_len,
            encoder_states=enc_states,
            encoder_mask=enc_mask,
            decoder_input_ids=decoder_input_ids,
        )
        return output

    def _init_diar_model(self):
        """
        Initialize the speaker model.
        """

        model_path = self.cfg.diar_model_path

        if model_path.endswith('.nemo'):
            pretrained_diar_model = SortformerEncLabelModel.restore_from(model_path, map_location="cpu")
            logging.info("Diarization Model restored locally from {}".format(model_path))
        elif model_path.endswith('.ckpt'):
            pretrained_diar_model = SortformerEncLabelModel.load_from_checkpoint(model_path, map_location="cpu")
            logging.info("Diarization Model restored locally from {}".format(model_path))
        else:
            pretrained_diar_model = None
            logging.info("Model path incorrect")

        self.diarization_model = pretrained_diar_model

        if self.cfg.freeze_diar:
           self.diarization_model.eval()

    def _init_asr_model(self):
        """
        Initialize the ASR model. Assuming that the model is initialized with super().__init__().
        """
        model_path = self.cfg.asr_model_path
        
        if model_path is not None and model_path.endswith('.nemo'):
            pretrained_asr_model = EncDecMultiTaskModel.restore_from(model_path, map_location="cpu")
            logging.info("ASR Model restored locally from {}".format(model_path))
        elif model_path.endswith('.ckpt'):
            pretrained_asr_model = EncDecMultiTaskModel.load_from_checkpoint(model_path, map_location="cpu")
            logging.info("ASR Model restored locally from {}".format(model_path))
        else:
            pretrained_asr_model = None
        
        if pretrained_asr_model is not None:
            logging.info("Restoring ASR model parameters from pretrained model.")
            self.encoder.load_state_dict(pretrained_asr_model.encoder.state_dict(), strict=True)
            self.encoder_decoder_proj.load_state_dict(pretrained_asr_model.encoder_decoder_proj.state_dict(), strict=True)
            if self.use_transf_encoder:
                self.transf_encoder.load_state_dict(pretrained_asr_model.transf_encoder.state_dict(), strict=True)
            self.transf_decoder.load_state_dict(pretrained_asr_model.transf_decoder.state_dict(), strict=True) 
        
        if self.cfg.freeze_asr:
            self.encoder.eval()
            self.encoder_decoder_proj.eval()
            if self.use_transf_encoder:
                self.transf_encoder.eval()
                
    def forward_asr(
        self,
        input_signal=None,
        input_signal_length=None,
        processed_signal=None,
        processed_signal_length=None
    ):
        """"""
        has_input_signal = input_signal is not None and input_signal_length is not None
        has_processed_signal = processed_signal is not None and processed_signal_length is not None
        if (has_input_signal ^ has_processed_signal) == False:
            raise ValueError(
                f"{self} Arguments ``input_signal`` and ``input_signal_length`` are mutually exclusive "
                " with ``processed_signal`` and ``processed_signal_len`` arguments."
            )

        if not has_processed_signal:
            processed_signal, processed_signal_length = self.preprocessor(
                input_signal=input_signal, length=input_signal_length
            )

        if self.spec_augmentation is not None and self.training:
            processed_signal = self.spec_augmentation(input_spec=processed_signal, length=processed_signal_length)

        encoded, encoded_len = self.encoder(audio_signal=processed_signal, length=processed_signal_length)

        enc_states = encoded.permute(0, 2, 1)
        enc_states = self.encoder_decoder_proj(enc_states)
        enc_mask = lens_to_mask(encoded_len, enc_states.shape[1]).to(enc_states.dtype)
        if self.use_transf_encoder:
            enc_states = self.transf_encoder(encoder_states=enc_states, encoder_mask=enc_mask)

        return enc_states, encoded_len, enc_mask

    def segmentation(self, feat=None, segment_length=16, segment_shift=8):
        """
        given a feature sequence, segment it into segments with `segment_length` and `segment_shift along the time axis`
        this is like single-scale speaker embedding compared with multi-scale (MSDD)
        Args:
            feat: B x D x T
            segment_length: int, the number of frames in each segment
            segment_shift: int, the number of frames to shift between segments
        Return:
            seg_feats: B x D x n_segments x segment_length
        """
        res = feat.shape[-1] % segment_shift
        if res == 0:
            pad = segment_length - segment_shift
        else:
            pad = segment_length - res
        pad_l, pad_r = pad // 2, pad - pad // 2
        
        feat = torch.nn.functional.pad(input=feat, pad=(pad_l, pad_r), mode='constant', value=0)
        B, D, T = feat.shape
        n_segments = (T - segment_length) // segment_shift + 1
        seg_feats = torch.zeros(B, D, n_segments, segment_length).to(feat.device)
        for i in range(n_segments):
            seg_feats[:, :, i, :] = feat[:, :, i * segment_shift : i * segment_shift + segment_length]
        return seg_feats

    def random_zero_mask(self, 
                         feat=None, 
                         prob=0.5, 
                         mask_value=0, 
                         min_mask_len=0.2,
                         mask_range=(0.1, 0.9),
                         ):
        """
        randomly mask some frames in the feature sequence
        Args:
            feat: B x D x T
            prob: float, the probability to mask a sample in the batch
            mask_value: float, the value to fill in the masked frame
            mask_range: tuple, only mask the frames in the range of mask_range[0] * T to mask_range[1] * T
        Return:
            masked_feat: B x D x T
        """
        if prob <= 0:
            return None, feat
        B, T, D = feat.shape

        mask = torch.ones_like(feat).to(feat.device)
        selected_sample_idx = torch.where(torch.rand(B) < prob)[0]
        n_masked_samples = len(selected_sample_idx)
        start_range = (int(mask_range[0] * T), int((mask_range[1] - min_mask_len) * T))
        mask_starts = torch.randint(start_range[0], start_range[1], (n_masked_samples,))
        for i in range(n_masked_samples):
            mask_len = torch.randint(int(min_mask_len * T), int(T * mask_range[1] - mask_starts[i]), (1, ))[0]
            mask[selected_sample_idx[i], mask_starts[i]:mask_starts[i] + mask_len] = mask_value

        return mask, feat * mask
       
    def training_step(self, batch, batch_nb):

        if batch is None:
            return torch.tensor([0.0])

        # During training prompt and prompt_len are null, ignore.
        signal, signal_len, transcript, transcript_len, prompt, prompt_len, spk_targets, spk_mappings = batch
        input_ids, labels = transcript[:, :-1], transcript[:, 1:]
        
        transf_log_probs, encoded_len, enc_states, enc_mask = self.forward(
            input_signal=signal,
            input_signal_length=signal_len,
            transcript=input_ids,
            transcript_length=transcript_len,
            spk_targets=spk_targets,
            spk_mappings=spk_mappings,
            spk_supervision_strategy=self.cfg.train_ds.get("spk_supervision_strategy", self.cfg.spk_supervision_strategy)
        )

        audio_loss = self.loss(log_probs=transf_log_probs, labels=labels)

        tensorboard_logs = {
            'train_loss': audio_loss,
            'learning_rate': self._optimizer.param_groups[0]['lr'],
        }

        return {'loss': audio_loss, 'log': tensorboard_logs}

    def validation_pass(self, batch, batch_idx, dataloader_idx=0, eval_mode="val"):
        # During inference, dataloader passes pure prompt without transcript text.
        signal, signal_len, transcript, transcript_len, prompt, prompt_len, spk_targets, spk_mappings = batch
        input_ids, labels = transcript[:, :-1], transcript[:, 1:]
        transf_log_probs, encoded_len, enc_states, enc_mask = self.forward(
            input_signal=signal,
            input_signal_length=signal_len,
            transcript=input_ids,
            transcript_length=transcript_len,
            spk_targets=spk_targets,
            spk_mappings=spk_mappings,
            spk_supervision_strategy=self.cfg.validation_ds.get("spk_supervision_strategy", "diar"),
        )

        transf_loss = self.loss(log_probs=transf_log_probs, labels=labels)
        self.val_loss(loss=transf_loss, num_measurements=transf_log_probs.shape[0] * transf_log_probs.shape[1])
        output_dict = {
            f'{eval_mode}_loss': transf_loss,
        }

        self.wer.update(
            predictions=enc_states,
            predictions_lengths=encoded_len,
            targets=transcript,
            targets_lengths=transcript_len,
            predictions_mask=enc_mask,
            input_ids=prompt,
        )
        wer, wer_num, wer_denom = self.wer.compute()
        output_dict.update({"val_wer": wer, "val_wer_num": wer_num, "val_wer_denom": wer_denom})
        self.wer.reset()

        self.bleu.update(
            predictions=enc_states,
            predictions_lengths=encoded_len,
            targets=transcript,
            targets_lengths=transcript_len,
            predictions_mask=enc_mask,
            input_ids=prompt,
        )
        bleu_metrics = self.bleu.compute(prefix=f"{eval_mode}_")
        output_dict.update(bleu_metrics)
        self.bleu.reset()

        return output_dict

    def forward_spk(
        self,
        input_signal=None,
        input_signal_length=None,
        segment_length=16,
        segment_shift=8,
        time_resolution=8
    ):
        """"""
        processed_signal, processed_signal_len = self.spk_preprocessor(
            input_signal=input_signal, length=input_signal_length,
        )

        if self.spec_augmentation is not None and self.training:
            processed_signal = self.spec_augmentation(input_spec=processed_signal, length=processed_signal_len)
        
        encoded, length = self.spk_encoder(audio_signal=processed_signal, length=processed_signal_len) # B x D x T
        seg_encoded = self.segmentation(encoded, segment_length, segment_shift) # B x D x n_segments x segment_length
        B, D, n_segments, _ = seg_encoded.shape
        seg_encoded = seg_encoded.transpose(1, 2).reshape(B*n_segments, D, segment_length) # B * n_segments x D x segment_length
        _, embs = self.spk_decoder(encoder_output=seg_encoded, length=torch.ones((B*n_segments,)).to(length.device) * segment_length ) # B * n_segments x D_emb
        embs = embs.view(B, n_segments, -1)

        n_repeats = int(segment_shift / time_resolution)
        if n_repeats > 1:
            embs = embs.unsqueeze(2).repeat(1, 1, int(n_repeats), 1).view(B, n_segments*n_repeats, -1)

        return embs, length
    
    def forward_diar(
        self,
        input_signal=None,
        input_signal_length=None,
    ):
        preds, _preds, attn_score_stack, total_memory_list, encoder_states_list = self.diarization_model.forward(audio_signal=input_signal, audio_signal_length=input_signal_length)

        return preds, _preds, attn_score_stack, total_memory_list, encoder_states_list

    def fix_diar_output(
        self,
        diar_pred,
        asr_frame_count
    ):
        """
        Duct-tapeß solution for extending the speaker predictions 
        """
        # Extract the first and last embeddings along the second dimension
        # first_emb = diar_pred[:, 0, :].unsqueeze(1)
        last_emb = diar_pred[:, -1, :].unsqueeze(1)

        #number of repeatitions needed
        additional_frames = asr_frame_count - diar_pred.shape[1]

        # Create tensors of repeated first and last embeddings
        # first_repeats = first_emb.repeat(1, additional_frames // 2, 1)
        # last_repeats = last_emb.repeat(1, (additional_frames + 1) // 2, 1)
        last_repeats = last_emb.repeat(1, additional_frames, 1)

        # Concatenate the repeated tensors with the original embeddings
        # extended_diar_preds = torch.cat((first_repeats, diar_pred, last_repeats), dim=1)
        extended_diar_preds = torch.cat((diar_pred, last_repeats), dim=1)

        return extended_diar_preds

    def get_sinusoid_position_encoding(self, max_position, embedding_dim):
        """
        Generates a sinusoid position encoding matrix.
        
        Args:
            max_position (int): The maximum position to generate encodings for.
            embedding_dim (int): The dimension of the embeddings.
        
        Returns:
            position_encoding_tensor(torch.Tensor): A tensor of shape (max_position, embedding_dim) containing the sinusoid position encodings.
        """
        position = np.arange(max_position)[:, np.newaxis]
        div_term = np.exp(np.arange(0, embedding_dim, 2) * -(np.log(10000.0) / embedding_dim))
        
        position_encoding = np.zeros((max_position, embedding_dim))
        position_encoding[:, 0::2] = np.sin(position * div_term)
        position_encoding[:, 1::2] = np.cos(position * div_term)
        
        # Convert the numpy array to a PyTorch tensor
        position_encoding_tensor = torch.tensor(position_encoding, dtype=torch.float32)
        
        return position_encoding_tensor


    def _get_probablistic_mix(self, diar_preds, spk_targets, rttm_mix_prob:float=0.0):
        """ 
        Sample a probablistic mixture of speaker labels for each time step then apply it to the diarization predictions and the speaker targets.
        
        Args:
            diar_preds (Tensor): Tensor of shape [B, T, D] representing the diarization predictions.
            spk_targets (Tensor): Tensor of shape [B, T, D] representing the speaker targets.
            rttm_mix_prob (float): Probability of mixing RTTM-based ground-truth speaker supervision against diarization predictions.
                                    If rttm_mix_prob is 0.0, the diarization predictions are used for speaker supervision.
                                    If rttm_mix_prob is 1.0, the RTTM-based ground-truth speaker supervisions are used.
            
        Returns:
            mix_prob (float): Tensor of shape [B, T, D] representing the probablistic mixture of speaker labels for each time step.
        """
        batch_probs_raw = torch.distributions.categorical.Categorical(probs=torch.tensor([(1-rttm_mix_prob), rttm_mix_prob]).repeat(diar_preds.shape[0],1)).sample()
        batch_probs = (batch_probs_raw.view(diar_preds.shape[0], 1, 1).repeat(1, diar_preds.shape[1], diar_preds.shape[2])).to(diar_preds.device)
        batch_diar_preds = (1 - batch_probs) * diar_preds + batch_probs * spk_targets
        return batch_diar_preds 
    
    @typecheck()
    def forward(
        self,
        input_signal=None,
        input_signal_length=None,
        processed_signal=None,
        processed_signal_length=None,
        transcript=None,
        transcript_length=None,
        spk_targets=None,
        spk_mappings=None,
        spk_supervision_strategy='diar'
    ):
        """
        Forward pass of the model.
        Args:
            input_signal: Tensor that represents a batch of raw audio signals,
                of shape [B, T]. T here represents timesteps, with 1 second of audio represented as
                `self.sample_rate` number of floating point values.
            input_signal_length: Vector of length B, that contains the individual lengths of the audio
                sequences.
            processed_signal: Tensor that represents a batch of processed audio signals,
                of shape (B, D, T).
            processed_signal_length: Vector of length B, that contains the individual lengths of the
                processed audio sequences.
            transcript: Tensor that represents a batch of transcripts,
                of shape [B, T]. T here represents timesteps.
            transcript_length: Vector of length B, that contains the individual lengths of the
                transcript sequences.

        Returns:
            A tuple of 4 elements -
            1) The log probabilities tensor of shape [B, T, D].
            2) The lengths of the acoustic sequence after propagation through the encoder, of shape [B].
            3) The greedy token predictions of the model of shape [B, T] (via argmax)
            4) The speaker embeddings of shape [B, D]
        """
        #logging.info('.......................', self.training, self.cfg.zero_prob)

        # ASR branch: downsample rate is 8
        with torch.set_grad_enabled(not self.cfg.freeze_asr):
            asr_enc_states, asr_encoded_len, asr_enc_mask = self.forward_asr( # B x T x D
                input_signal, input_signal_length, processed_signal, processed_signal_length
            )

        if self.diar == True:
            with torch.set_grad_enabled(not self.cfg.freeze_diar):
                diar_preds, _preds, attn_score_stack, total_memory_list, encoder_states_list = self.forward_diar(input_signal, input_signal_length)
                # pred shape = B * T (-1 or -2) * D 
                # May 23 2024 -> sortformer produces 1 or 2 frames less than FC, FIX!!!

            # Speaker supervision strategy settings
            if spk_supervision_strategy == 'rttm':
                if spk_targets is not None:
                    diar_preds = spk_targets 
                else:
                    raise ValueError("`spk_targets` is required for speaker supervision strategy 'rttm'")
            elif spk_supervision_strategy == 'diar':
                if diar_preds is None:
                    raise ValueError("`diar_pred`is required for speaker supervision strategy 'diar'")
            elif spk_supervision_strategy == 'mix':
                diar_preds = self._get_probablistic_mix(diar_preds=diar_preds, spk_targets=spk_targets, rttm_mix_prob=float(self.cfg.rttm_mix_prob))
            else:
                raise ValueError(f"Invalid RTTM strategy {spk_supervision_strategy} is not supported.")
            
            # Speaker mapping shuffling to equalize the speaker label's distributions
            if self.cfg.shuffle_spk_mapping and spk_mappings is not None:
                diar_preds = apply_spk_mapping(diar_preds, spk_mappings)
            
            if(diar_preds.shape[1]!=asr_enc_states.shape[1]):
            # KD duct-tape solution for extending the speaker predictions 
                asr_frame_count = asr_enc_states.shape[1]
                diar_preds = self.fix_diar_output(diar_preds, asr_frame_count)

            # Normalize the features
            if self.norm == 'ln':
                diar_preds = self.diar_norm(diar_preds)
                asr_enc_states = self.asr_norm(asr_enc_states)
            elif self.norm == 'l2':
                diar_preds = torch.nn.functional.normalize(diar_preds, p=2, dim=-1)
                asr_enc_states = torch.nn.functional.normalize(asr_enc_states, p=2, dim=-1)
            
            if diar_preds.shape[1] > asr_enc_states.shape[1]:
                diar_preds = diar_preds[:, :asr_enc_states.shape[1], :]
              
            if self.diar_kernel_type == 'sinusoidal':
                speaker_infusion_asr = torch.matmul(diar_preds, self.diar_kernel.to(diar_preds.device))
                if self.kernel_norm == 'l2':
                    speaker_infusion_asr = torch.nn.functional.normalize(speaker_infusion_asr, p=2, dim=-1)
                
                enc_states = speaker_infusion_asr + asr_enc_states
            else:
                concat_enc_states = torch.cat([asr_enc_states, diar_preds], dim=-1)
                enc_states = self.joint_proj(concat_enc_states)
        else:
            enc_states = asr_enc_states
        
        # # Spk branch: downsample rate is 1 for encoder, but downsample rate is 8 for decoder
        # if self.spk == True:
        #     with torch.set_grad_enabled(not self.cfg.freeze_spk):
        #         spk_enc_states, spk_encoded_len = self.forward_spk( # B x T x D
        #             input_signal, input_signal_length, self.segment_length, self.segment_shift
        #         )
        #     # Normalize the features
        #     if self.norm == 'ln':
        #         spk_enc_states = self.spk_norm(spk_enc_states)
        #         asr_enc_states = self.asr_norm(asr_enc_states)
        #     elif self.norm == 'l2':
        #         spk_enc_states = torch.nn.functional.normalize(spk_enc_states, p=2, dim=-1)
        #         asr_enc_states = torch.nn.functional.normalize(asr_enc_states, p=2, dim=-1)
            
        #     if spk_enc_states.shape[1] > asr_enc_states.shape[1]:
        #         spk_enc_states = spk_enc_states[:, :asr_enc_states.shape[1], :]

        #     concat_enc_states = torch.cat([asr_enc_states, spk_enc_states], dim=-1)
        #     enc_states = self.joint_proj(concat_enc_states)
        # else:
        #     enc_states = asr_enc_states
            
        # merge two states
        transf_log_probs = None
        if transcript is not None:
            dec_mask = lens_to_mask(transcript_length, transcript.shape[1]).to(transcript.dtype)
            dec_states = self.transf_decoder(
                input_ids=transcript, decoder_mask=dec_mask, encoder_embeddings=enc_states, encoder_mask=asr_enc_mask
            )
            transf_log_probs = self.log_softmax(hidden_states=dec_states)

        return transf_log_probs, asr_encoded_len, enc_states, asr_enc_mask