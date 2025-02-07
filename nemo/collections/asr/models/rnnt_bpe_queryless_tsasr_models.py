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
#
import copy
import os
from typing import Dict, List, Optional, Union

import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import DictConfig, ListConfig, OmegaConf, open_dict
from pytorch_lightning import Trainer
from typing import Any, Dict, List, Optional, Tuple
from torch.utils.data import DataLoader

from nemo.collections.asr.data.audio_to_text_lhotse_speaker import LhotseSpeechToTextSpkBpeDataset

import nemo.collections.asr.models as asr_models
from nemo.core.classes.common import PretrainedModelInfo, typecheck
from nemo.core.classes.mixins import AccessMixin
from nemo.collections.asr.parts.preprocessing.segment import ChannelSelectorType
from nemo.collections.asr.parts.mixins.streaming import StreamingEncoder



from nemo.collections.asr.parts.mixins import (
    ASRModuleMixin,
    ASRTranscriptionMixin,
    TranscribeConfig,
    TranscriptionReturnType,
)

from nemo.collections.asr.models.rnnt_bpe_models import EncDecRNNTBPEModel
from nemo.collections.asr.models.sortformer_diar_models import SortformerEncLabelModel
from nemo.collections.common.data.lhotse import get_lhotse_dataloader_from_config
from nemo.core.classes.common import PretrainedModelInfo

from nemo.utils import logging

class MetaCatResidual(torch.nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.proj = torch.nn.Sequential(
            torch.nn.Linear(input_size, output_size*2),
            torch.nn.ReLU(),
            torch.nn.Linear(output_size*2, output_size)
        )

    def forward(self, x, mask):
        """
        x: (B, T, D)
        mask: (B, T)
        """
        if mask.shape[1] < x.shape[1]:
            mask = F.pad(mask, (x.shape[1] - mask.shape[1], 0), value=0)

        if mask.shape[1] > x.shape[1]:
            mask = mask[:, -x.shape[1]:]

        x = x * mask.unsqueeze(2)
        x = x + self.proj(x)
        return x

class EncDecRNNTBPEQLTSASRModel(EncDecRNNTBPEModel):
    """Base class for encoder decoder RNNT-based models with subword tokenization."""

    def __init__(self, cfg: DictConfig, trainer: Trainer = None):
        super().__init__(cfg=cfg, trainer=trainer)

        self.num_speakers = cfg.model_defaults.get('num_speakers', 4)
        if 'diar_model_path' in self.cfg:
            self._init_diar_model()

            self.pre_diar_kernel = cfg.get('pre_diar_kernel', None)
            self.fc_diar_kernel = cfg.get('fc_diar_kernel', None)
            self.diar_kernel_layer = cfg.get('diar_kernel_layer', [])
            self.binary_diar_preds = cfg.get('binary_diar_preds', True)
            self.spk_supervision = cfg.get('spk_supervision', 'rttm')
            
            if isinstance(self.diar_kernel_layer, int):
                self.diar_kernel_layer = list(range(1, 1 + self.diar_kernel_layer))
            elif isinstance(self.diar_kernel_layer, ListConfig):
                pass
            elif self.fc_diar_kernel is not None:
                raise ValueError(f"Invalid diar_kernel_layer {self.diar_kernel_layer}, should be int or list")

            in_size = cfg.model_defaults.enc_hidden
            out_size = cfg.model_defaults.enc_hidden
            self.spk_kernels = torch.nn.ModuleDict()

            # diar kernel for pre-encoder
            if 0 in self.diar_kernel_layer:
                if self.pre_diar_kernel == 'metacat_residule':
                    # projection layer
                    self.spk_kernels['0'] = MetaCatResidual(in_size, out_size)

                elif self.pre_diar_kernel == 'sinusoidal':
                    # self.pre_proj = self.get_sinusoid_position_encoding(self.num_speakers, cfg.model_defaults.enc_hidden)
                    pass

            # diar kernel for fast conformer encoder layers
            if self.fc_diar_kernel == 'metacat_residule':
                for l_i in self.diar_kernel_layer:
                    if l_i != 0:
                        self.spk_kernels[str(l_i)] = MetaCatResidual(in_size, out_size)
            elif self.fc_diar_kernel == 'sinusoidal':
                # self.post_proj = self.get_sinusoid_position_encoding(self.num_speakers, cfg.model_defaults.enc_hidden)
                pass
            
            logging.info(f"Registered speaker kernels to layers {list(self.spk_kernels.keys())}")
            # register the speaker injection model to each layer
            for k, spk_kernel in self.spk_kernels.items():
                hook_func = self.get_hook_function(k)
                idx = int(k)
                if idx == 0:
                    self.encoder.pre_encode.register_forward_hook(hook_func)
                else:
                    self.encoder.layers[idx-1].register_forward_hook(hook_func)
                

    def _init_diar_model(self):
        """
        Initialize the speaker model.
        """
        logging.info(f"Initializing diarization model from pretrained checkpoint {self.cfg.diar_model_path}")

        model_path = self.cfg.diar_model_path

        try:
            if model_path.endswith('.nemo'):
                pretrained_diar_model = SortformerEncLabelModel.restore_from(model_path, map_location="cpu")
                logging.info("Diarization Model restored locally from {}".format(model_path))
            elif model_path.endswith('.ckpt'):
                pretrained_diar_model = SortformerEncLabelModel.load_from_checkpoint(model_path, map_location="cpu")
                logging.info("Diarization Model restored locally from {}".format(model_path))
            else:
                pretrained_diar_model = SortformerEncLabelModel.from_pretrained(model_path)
                logging.info("Diarization Model restored from NGC")
        except:
            pretrained_diar_model = SortformerEncLabelModel.from_pretrained("nvidia/diar_sortformer_4spk-v1")
            logging.info("Diarization Model restored from NGC")
        
        self.diarization_model = pretrained_diar_model

        if self.cfg.freeze_diar:
            self.diarization_model.eval()

    def get_hook_function(self, index):
        """Returns a forward hook function that applies the metacat modules"""
        def hook(module, input, output):

            if isinstance(output, tuple): # pre-encode, B x T x D
                x, *cache = output
                x = self.spk_kernels[index](x, self.spk_targets)
                
                return (x, *cache)
            
            else:
                return self.spk_kernels[index](output, self.spk_targets)
        
        return hook

    def forward_diar(
        self,
        audio_signal=None,
        audio_signal_length=None
    ):
        with torch.no_grad():
            processed_signal, processed_signal_length = self.diarization_model.process_signal(audio_signal=audio_signal, audio_signal_length=audio_signal_length)
            processed_signal = processed_signal[:, :, :processed_signal_length.max()]
            emb_seq, _ = self.diarization_model.frontend_encoder(processed_signal=processed_signal, processed_signal_length=processed_signal_length)
            preds = self.diarization_model.forward_infer(emb_seq)

        return preds

    def forward_train_val(
        self,
        signal=None,
        signal_len=None,
        spk_targets=None
        ):
        """
        Forward pass of the model. Note that for RNNT Models, the forward pass of the model is a 3 step process,
        and this method only performs the first step - forward of the acoustic model.

        Please refer to the `training_step` in order to see the full `forward` step for training - which
        performs the forward of the acoustic model, the prediction network and then the joint network.
        Finally, it computes the loss and possibly compute the detokenized text via the `decoding` step.

        Please refer to the `validation_step` in order to see the full `forward` step for inference - which
        performs the forward of the acoustic model, the prediction network and then the joint network.
        Finally, it computes the decoded tokens via the `decoding` step and possibly compute the batch metrics.

        Args:
            input_signal: Tensor that represents a batch of raw audio signals,
                of shape [B, T]. T here represents timesteps, with 1 second of audio represented as
                `self.sample_rate` number of floating point values.
            input_signal_length: Vector of length B, that contains the individual lengths of the audio
                sequences.
            processed_signal: Tensor that represents a batch of processed audio signals,
                of shape (B, D, T) that has undergone processing via some DALI preprocessor.
            processed_signal_length: Vector of length B, that contains the individual lengths of the
                processed audio sequences.

        Returns:
            A tuple of 2 elements -
            1) The log probabilities tpensor of shape [B, T, D].
            2) The lengths of the acoustic sequence after propagation through the encoder, of shape [B].
        """

        processed_signal, processed_signal_length = self.preprocessor(
                input_signal=signal,
                length=signal_len,
            )

        # Spec augment is not applied during evaluation/testing
        if self.spec_augmentation is not None and self.training:
            processed_signal = self.spec_augmentation(input_spec=processed_signal, length=processed_signal_length)

        encoded, encoded_len = self.encoder(audio_signal=processed_signal, length=processed_signal_length)

        return encoded, encoded_len

    def get_sinusoid_position_encoding(self, max_position, embedding_dim):
        """
        Generates a sinusoid position encoding matrix.
        
        Args:
        - max_position (int): The maximum position to generate encodings for.
        - embedding_dim (int): The dimension of the embeddings.
        
        Returns:
        - torch.Tensor: A tensor of shape (max_position, embedding_dim) containing the sinusoid position encodings.
        """
        position = np.arange(max_position)[:, np.newaxis]
        div_term = np.exp(np.arange(0, embedding_dim, 2) * -(np.log(10000.0) / embedding_dim))
        
        position_encoding = np.zeros((max_position, embedding_dim))
        position_encoding[:, 0::2] = np.sin(position * div_term)
        position_encoding[:, 1::2] = np.cos(position * div_term)
        
        # Convert the numpy array to a PyTorch tensor
        position_encoding_tensor = torch.tensor(position_encoding, dtype=torch.float32)
        
        return position_encoding_tensor

    def _setup_dataloader_from_config(self, config: Optional[Dict]):
        if config.get("use_lhotse"):
            return get_lhotse_dataloader_from_config(
                config,
                global_rank=self.global_rank,
                world_size=self.world_size,
                dataset=LhotseSpeechToTextSpkBpeDataset(cfg = config, tokenizer=self.tokenizer,),
            )

    def training_step(self, batch, batch_nb):
        # Reset access registry
        if AccessMixin.is_access_enabled(self.model_guid):
            AccessMixin.reset_registry(self)

        signal, signal_len, transcript, transcript_len, spk_targets, spk_ids = batch

        if self.spk_supervision == 'diar':
            spk_targets = self.forward_diar(audio_signal=signal, audio_signal_length=signal_len)
        
        self.spk_targets = torch.stack([spk_targets[i, :, spk_ids[i]] for i in range(len(spk_ids))])

        encoded, encoded_len = self.forward_train_val(
            signal=signal, signal_len=signal_len, spk_targets=spk_targets
        )
        del signal

        # During training, loss must be computed, so decoder forward is necessary
        decoder, target_length, states = self.decoder(targets=transcript, target_length=transcript_len)

        if hasattr(self, '_trainer') and self._trainer is not None:
            log_every_n_steps = self._trainer.log_every_n_steps
            sample_id = self._trainer.global_step
        else:
            log_every_n_steps = 1
            sample_id = batch_nb

        # If experimental fused Joint-Loss-WER is not used
        if not self.joint.fuse_loss_wer:
            # Compute full joint and loss
            joint = self.joint(encoder_outputs=encoded, decoder_outputs=decoder)
            loss_value = self.loss(
                log_probs=joint, targets=transcript, input_lengths=encoded_len, target_lengths=target_length
            )

            # Add auxiliary losses, if registered
            loss_value = self.add_auxiliary_losses(loss_value)

            # Reset access registry
            if AccessMixin.is_access_enabled(self.model_guid):
                AccessMixin.reset_registry(self)

            tensorboard_logs = {
                'train_loss': loss_value,
                'learning_rate': self._optimizer.param_groups[0]['lr'],
                'global_step': torch.tensor(self.trainer.global_step, dtype=torch.float32),
            }

            if (sample_id + 1) % log_every_n_steps == 0:
                self.wer.update(
                    predictions=encoded,
                    predictions_lengths=encoded_len,
                    targets=transcript,
                    targets_lengths=transcript_len,
                )
                _, scores, words = self.wer.compute()
                self.wer.reset()
                tensorboard_logs.update({'training_batch_wer': scores.float() / words})

        else:
            # If experimental fused Joint-Loss-WER is used
            if (sample_id + 1) % log_every_n_steps == 0:
                compute_wer = True
            else:
                compute_wer = False

            # Fused joint step
            loss_value, wer, _, _ = self.joint(
                encoder_outputs=encoded,
                decoder_outputs=decoder,
                encoder_lengths=encoded_len,
                transcripts=transcript,
                transcript_lengths=transcript_len,
                compute_wer=compute_wer,
            )

            # Add auxiliary losses, if registered
            loss_value = self.add_auxiliary_losses(loss_value)

            # Reset access registry
            if AccessMixin.is_access_enabled(self.model_guid):
                AccessMixin.reset_registry(self)

            tensorboard_logs = {
                'train_loss': loss_value,
                'learning_rate': self._optimizer.param_groups[0]['lr'],
                'global_step': torch.tensor(self.trainer.global_step, dtype=torch.float32),
            }

            if compute_wer:
                tensorboard_logs.update({'training_batch_wer': wer})

        # Log items
        self.log_dict(tensorboard_logs)

        # Preserve batch acoustic model T and language model U parameters if normalizing
        if self._optim_normalize_joint_txu:
            self._optim_normalize_txu = [encoded_len.max(), transcript_len.max()]

        return {'loss': loss_value}


    def validation_pass(self, batch, batch_idx, dataloader_idx=0):

        signal, signal_len, transcript, transcript_len, spk_targets, spk_ids = batch

        if self.spk_supervision == 'diar':
            spk_targets = self.forward_diar(audio_signal=signal, audio_signal_length=signal_len)
        
        self.spk_targets = torch.stack([spk_targets[i, :, spk_ids[i]] for i in range(len(spk_ids))])

        encoded, encoded_len = self.forward_train_val(
            signal=signal, signal_len=signal_len, spk_targets=spk_targets
        )
        del signal

        tensorboard_logs = {}

        # If experimental fused Joint-Loss-WER is not used
        if not self.joint.fuse_loss_wer:
            if self.compute_eval_loss:
                decoder, target_length, states = self.decoder(targets=transcript, target_length=transcript_len)
                joint = self.joint(encoder_outputs=encoded, decoder_outputs=decoder)

                loss_value = self.loss(
                    log_probs=joint, targets=transcript, input_lengths=encoded_len, target_lengths=target_length
                )

                tensorboard_logs['val_loss'] = loss_value

            self.wer.update(
                predictions=encoded,
                predictions_lengths=encoded_len,
                targets=transcript,
                targets_lengths=transcript_len,
            )
            wer, wer_num, wer_denom = self.wer.compute()
            self.wer.reset()

            tensorboard_logs['val_wer_num'] = wer_num
            tensorboard_logs['val_wer_denom'] = wer_denom
            tensorboard_logs['val_wer'] = wer

        else:
            # If experimental fused Joint-Loss-WER is used
            compute_wer = True

            if self.compute_eval_loss:
                decoded, target_len, states = self.decoder(targets=transcript, target_length=transcript_len)
            else:
                decoded = None
                target_len = transcript_len

            # Fused joint step
            loss_value, wer, wer_num, wer_denom = self.joint(
                encoder_outputs=encoded,
                decoder_outputs=decoded,
                encoder_lengths=encoded_len,
                transcripts=transcript,
                transcript_lengths=target_len,
                compute_wer=compute_wer,
            )

            if loss_value is not None:
                tensorboard_logs['val_loss'] = loss_value

            tensorboard_logs['val_wer_num'] = wer_num
            tensorboard_logs['val_wer_denom'] = wer_denom
            tensorboard_logs['val_wer'] = wer

        self.log('global_step', torch.tensor(self.trainer.global_step, dtype=torch.float32))

        return tensorboard_logs

    def _transcribe_forward(self, batch: Any, trcfg: TranscribeConfig):

        signal, signal_len, transcript, transcript_len, spk_targets, spk_ids = batch

        if self.spk_supervision == 'diar':
            spk_targets = self.forward_diar(audio_signal=signal, audio_signal_length=signal_len) # B x T x N
        n_mix = trcfg.mix

        # Multi-instance inference
        self.spk_targets = spk_targets[:, :, :n_mix].transpose(1, 2).reshape(-1, spk_targets.size(1)) # B x T x N_mix -> BN_mix x T
        signal = signal.unsqueeze(1).repeat(1, n_mix, 1).reshape(-1, signal.size(1)) # B x T -> BN_mix x T
        signal_len = signal_len.unsqueeze(1).repeat(1, n_mix).reshape(-1) # B -> BN_mix

        encoded, encoded_len = self.forward_train_val(
            signal=signal, signal_len=signal_len, spk_targets=spk_targets
        )
        del signal

        output = dict(encoded=encoded, encoded_len=encoded_len, spk_targets=self.spk_targets)

        return output

    def _transcribe_output_processing(
        self, outputs, trcfg: TranscribeConfig
    ) -> Tuple[List['Hypothesis'], List['Hypothesis']]:
        encoded = outputs.pop('encoded')
        encoded_len = outputs.pop('encoded_len')
        spk_targets = outputs.pop('spk_targets')

        best_hyp, all_hyp = self.decoding.rnnt_decoder_predictions_tensor(
            encoded,
            encoded_len,
            return_hypotheses=trcfg.return_hypotheses,
            partial_hypotheses=trcfg.partial_hypothesis,
        )

        # cleanup memory
        del encoded, encoded_len

        hypotheses = []
        all_hypotheses = []

        hypotheses += best_hyp
        if all_hyp is not None:
            all_hypotheses += all_hyp
        else:
            all_hypotheses += best_hyp
        self.total_results.append(hypotheses)

        # import ipdb; ipdb.set_trace()

        return (hypotheses, all_hypotheses)

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
        if 'dataset_manifest' in config:
            manifest_filepath = config['dataset_manifest']
            batch_size = config['batch_size']
        else:
            manifest_filepath = os.path.join(config['temp_dir'], 'manifest.json')
            batch_size = min(config['batch_size'], len(config['paths2audio_files']))

        dl_config = {
            'manifest_filepath': manifest_filepath,
            'sample_rate': self.preprocessor._sample_rate,
            'batch_size': batch_size,
            'shuffle': False,
            'num_workers': config.get('num_workers', min(batch_size, os.cpu_count() - 1)),
            'pin_memory': True,
            'use_lhotse': True,
            'use_bucketing': False,
            'channel_selector': config.get('channel_selector', None),
            'inference_mode': self.cfg.test_ds.get('inference_mode', True),
            'fixed_spk_id': config.get('fixed_spk_id', None)
        }

        if config.get("augmentor"):
            dl_config['augmentor'] = config.get("augmentor")

        temporary_datalayer = self._setup_dataloader_from_config(config=DictConfig(dl_config))

        return temporary_datalayer

    @torch.no_grad()
    def transcribe(
        self,
        audio: Union[str, List[str], np.ndarray, DataLoader],
        batch_size: int = 4,
        return_hypotheses: bool = False,
        partial_hypothesis: Optional[List['Hypothesis']] = None,
        num_workers: int = 0,
        channel_selector: Optional[ChannelSelectorType] = None,
        augmentor: DictConfig = None,
        verbose: bool = True,
        override_config: Optional[TranscribeConfig] = None,
    ) -> TranscriptionReturnType:
        """
        Uses greedy decoding to transcribe audio files. Use this method for debugging and prototyping.

        Args:
            audio: (a single or list) of paths to audio files or a np.ndarray/tensor audio array or path to a manifest file.
                Can also be a dataloader object that provides values that can be consumed by the model.
                Recommended length per file is between 5 and 25 seconds. \
                But it is possible to pass a few hours long file if enough GPU memory is available.
            batch_size: (int) batch size to use during inference. \
                Bigger will result in better throughput performance but would use more memory.
            return_hypotheses: (bool) Either return hypotheses or text
                With hypotheses can do some postprocessing like getting timestamp or rescoring
            partial_hypothesis: Optional[List['Hypothesis']] - A list of partial hypotheses to be used during rnnt
                decoding. This is useful for streaming rnnt decoding. If this is not None, then the length of this
                list should be equal to the length of the audio list.
            num_workers: (int) number of workers for DataLoader
            channel_selector (int | Iterable[int] | str): select a single channel or a subset of channels from multi-channel audio. If set to `'average'`, it performs averaging across channels. Disabled if set to `None`. Defaults to `None`. Uses zero-based indexing.
            augmentor: (DictConfig): Augment audio samples during transcription if augmentor is applied.
            verbose: (bool) whether to display tqdm progress bar
            override_config: (Optional[TranscribeConfig]) override transcription config pre-defined by the user.
                **Note**: All other arguments in the function will be ignored if override_config is passed.
                You should call this argument as `model.transcribe(audio, override_config=TranscribeConfig(...))`.

        Returns:
            Returns a tuple of 2 items -
            * A list of greedy transcript texts / Hypothesis
            * An optional list of beam search transcript texts / Hypothesis / NBestHypothesis.
        """
        self.total_results = []
        self.preds_rttms = []
        if override_config.fixed_spk_id is not None:
            self.fixed_spk_id = override_config.fixed_spk_id

        super().transcribe(
            audio=audio,
            batch_size=batch_size,
            return_hypotheses=return_hypotheses,
            num_workers=num_workers,
            channel_selector=channel_selector,
            augmentor=augmentor,
            verbose=verbose,
            override_config=override_config,
            # Additional arguments
            partial_hypothesis=partial_hypothesis,
        )

        return self.total_results

    def conformer_stream_step(
        self,
        processed_signal: torch.Tensor,
        processed_signal_length: torch.Tensor = None,
        cache_last_channel: torch.Tensor = None,
        cache_last_time: torch.Tensor = None,
        cache_last_channel_len: torch.Tensor = None,
        keep_all_outputs: bool = True,
        previous_hypotheses: List['Hypothesis'] = None,
        previous_pred_out: torch.Tensor = None,
        drop_extra_pre_encoded: int = None,
        return_transcription: bool = True,
        return_log_probs: bool = False,
        spk_targets: torch.Tensor = None,
        n_mix = 1,
    ):
        """
        It simulates a forward step with caching for streaming purposes.
        It supports the ASR models where their encoder supports streaming like Conformer.
        Args:
            processed_signal: the input audio signals
            processed_signal_length: the length of the audios
            cache_last_channel: the cache tensor for last channel layers like MHA
            cache_last_channel_len: engths for cache_last_channel
            cache_last_time: the cache tensor for last time layers like convolutions
            keep_all_outputs: if set to True, would not drop the extra outputs specified by encoder.streaming_cfg.valid_out_len
            previous_hypotheses: the hypotheses from the previous step for RNNT models
            previous_pred_out: the predicted outputs from the previous step for CTC models
            drop_extra_pre_encoded: number of steps to drop from the beginning of the outputs after the downsampling module. This can be used if extra paddings are added on the left side of the input.
            return_transcription: whether to decode and return the transcriptions. It can not get disabled for Transducer models.
            return_log_probs: whether to return the log probs, only valid for ctc model

        Returns:
            greedy_predictions: the greedy predictions from the decoder
            all_hyp_or_transcribed_texts: the decoder hypotheses for Transducer models and the transcriptions for CTC models
            cache_last_channel_next: the updated tensor cache for last channel layers to be used for next streaming step
            cache_last_time_next: the updated tensor cache for last time layers to be used for next streaming step
            cache_last_channel_next_len: the updated lengths for cache_last_channel
            best_hyp: the best hypotheses for the Transducer models
            log_probs: the logits tensor of current streaming chunk, only returned when return_log_probs=True
            encoded_len: the length of the output log_probs + history chunk log_probs, only returned when return_log_probs=True
        """
        # Multi-instance inference
        if len(spk_targets.size()) == 3:
            # N: # speakers
            # spk_targets: (B, T, N) -> (BN, T)
            self.spk_targets = spk_targets[:, :, :n_mix].transpose(1, 2).reshape(-1, spk_targets.size(1))

            # processed_signal: (B, T, D) -> (BN, T, D)
            processed_signal = processed_signal.unsqueeze(1).repeat(1, n_mix, 1, 1).reshape(-1, processed_signal.size(1), processed_signal.size(2))
            processed_signal_length = processed_signal_length.unsqueeze(1).repeat(1, n_mix).reshape(-1)

        if cache_last_channel_len.shape[0] != processed_signal.shape[0]:
            cache_last_channel = cache_last_channel.unsqueeze(2).repeat(1, 1, n_mix, 1, 1).reshape(cache_last_channel.size(0), -1, cache_last_channel.size(2), cache_last_channel.size(3))
            cache_last_time = cache_last_time.unsqueeze(2).repeat(1, 1, n_mix, 1, 1).reshape(cache_last_time.size(0), -1, cache_last_time.size(2), cache_last_time.size(3))
            cache_last_channel_len = cache_last_channel_len.unsqueeze(1).repeat(1, n_mix).reshape(-1)

        return super().conformer_stream_step(
            processed_signal=processed_signal,
            processed_signal_length=processed_signal_length,
            cache_last_channel=cache_last_channel,
            cache_last_time=cache_last_time,
            cache_last_channel_len=cache_last_channel_len,
            keep_all_outputs=keep_all_outputs,
            previous_hypotheses=previous_hypotheses,
            previous_pred_out=previous_pred_out,
            drop_extra_pre_encoded=drop_extra_pre_encoded,
            return_transcription=return_transcription,
            return_log_probs=return_log_probs
        )
    