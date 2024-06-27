# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
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

import inspect
import copy
from collections import OrderedDict
from typing import Dict, List, Optional, Union
from operator import attrgetter
from math import ceil
from torch import Tensor
from einops import rearrange
import time
import torch
from torch import nn
import torch.nn.functional as F
from hydra.utils import instantiate
from omegaconf import DictConfig, open_dict
from pytorch_lightning import Trainer
from tqdm import tqdm
from typing import List, Optional, Union, Dict 
import itertools
from dataclasses import dataclass
from pathlib import Path

from pyannote.database.util import load_rttm
from pyannote.core import Annotation, Segment
from pyannote.metrics.diarization import DiarizationErrorRate
from nemo.collections.asr.parts.preprocessing.perturb import process_augmentations
from nemo.collections.asr.data.audio_to_eesd_label_heh import get_audio_to_eesd_label_dataset_from_config, EESDAudioRTTMBatch
from nemo.collections.asr.metrics.multi_binary_acc import MultiBinaryAccuracy
from nemo.collections.asr.models.asr_model import ExportableEncDecModel
from nemo.collections.asr.models.clustering_diarizer import (
    get_available_model_names,
)
from nemo.collections.asr.parts.mixins import ASRAdapterModelMixin
from pytorch_lightning.utilities import rank_zero_only

from nemo.collections.asr.losses.bce_loss import MaskedBCELoss
from nemo.collections.asr.parts.preprocessing.features import WaveformFeaturizer
from nemo.core.classes import ModelPT
from nemo.core.classes import Loss, Typing, typecheck
from nemo.core.classes.common import PretrainedModelInfo
from nemo.core.neural_types import AudioSignal, LengthsType, NeuralType
from nemo.core.neural_types.elements import ProbsType
from nemo.utils import logging
from nemo.utils.nemo_logging import LogMode
from nemo.collections.asr.modules.transformer.transformer_modules import FixedPositionalEncoding

try:
    from torch.cuda.amp import autocast
except ImportError:
    from contextlib import contextmanager

    @contextmanager
    def autocast(enabled=None):
        yield


torch.backends.cudnn.enabled = False 

__all__ = ['EncDecEESDModel']


class EncDecEESDModel(ModelPT, ExportableEncDecModel, ASRAdapterModelMixin):
    def __init__(self, cfg: DictConfig, trainer: Trainer = None):
        self.threshold = 0.5
        self.collar = 0.25
        self.skip_overlap = False
        self.output_dir = None
        self.use_der_metric = False
        self.use_rttm_for_der = False
        super().__init__(cfg=cfg, trainer=trainer)

        self.preprocessor = EncDecEESDModel.from_config_dict(self._cfg.preprocessor)
        self.encoder = EncDecEESDModel.from_config_dict(self._cfg.encoder)
        self.decoder = EncDecEESDModel.from_config_dict(self._cfg.decoder)
        self.bce_loss = EncDecEESDModel.from_config_dict(self._cfg.loss)

        if getattr(self._cfg, 'spec_augment', None) is not None:
            self.spec_augmentation = EncDecEESDModel.from_config_dict(self._cfg.spec_augment)
        else:
            self.spec_augmentation = None

        self._accuracy_train = MultiBinaryAccuracy()
        self._accuracy_val = MultiBinaryAccuracy()
        self._accuracy_test = MultiBinaryAccuracy()

        self._accuracy_train_vad= MultiBinaryAccuracy()
        self._accuracy_val_vad= MultiBinaryAccuracy()
        self._accuracy_test_vad= MultiBinaryAccuracy()

        self._accuracy_train_ovl= MultiBinaryAccuracy()
        self._accuracy_val_ovl= MultiBinaryAccuracy()
        self._accuracy_test_ovl= MultiBinaryAccuracy()

        speaker_inds = list(range(self._cfg.max_num_of_spks))
        self.spk_perm = nn.Parameter(torch.tensor(list(itertools.permutations(speaker_inds))), requires_grad=False) # Get all permutations
        self.max_num_speakers = self._cfg.max_num_of_spks
        self.frame_len_secs = self._cfg.frame_len_secs
        self.sample_rate = self._cfg.sample_rate
        self.use_pil = self._cfg.get('use_pil', True)

        
    def list_available_models(self):
        return []

    def _setup_dataloader_from_config(self, config: DictConfig):
        dataset = get_audio_to_eesd_label_dataset_from_config(
            config=config,
            local_rank=self.local_rank,
            global_rank=self.global_rank,
            world_size=self.world_size,
        )

        if dataset is None:
            return None

        shuffle = config['shuffle']
        if isinstance(dataset, torch.utils.data.IterableDataset):
            shuffle = False

        if hasattr(dataset, 'collate_fn'):
            collate_fn = dataset.collate_fn
        elif hasattr(dataset.datasets[0], 'collate_fn'):
            # support datasets that are lists of entries
            collate_fn = dataset.datasets[0].collate_fn
        else:
            # support datasets that are lists of lists
            collate_fn = dataset.datasets[0].datasets[0].collate_fn

        return torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=config['batch_size'],
            collate_fn=collate_fn,
            drop_last=config.get('drop_last', False),
            shuffle=shuffle,
            num_workers=config.get('num_workers', 0),
            pin_memory=config.get('pin_memory', False),
        )

    def setup_training_data(self, train_data_config: Optional[Union[DictConfig, Dict]]):
        """
        Sets up the training data loader via a Dict-like object.

        Args:
            train_data_config: A config that contains the information regarding construction
                of an ASR Training dataset.
        """
        if 'shuffle' not in train_data_config:
            train_data_config['shuffle'] = True

        # preserve config
        self._update_dataset_config(dataset_name='train', config=train_data_config)

        self._train_dl = self._setup_dataloader_from_config(config=train_data_config)

        # Need to set this because if using an IterableDataset, the length of the dataloader is the total number
        # of samples rather than the number of batches, and this messes up the tqdm progress bar.
        # So we set the number of steps manually (to the correct number) to fix this.
        if (
            self._train_dl is not None
            and hasattr(self._train_dl, 'dataset')
            and isinstance(self._train_dl.dataset, torch.utils.data.IterableDataset)
        ):
            # We also need to check if limit_train_batches is already set.
            # If it's an int, we assume that the user has set it to something sane, i.e. <= # training batches,
            # and don't change it. Otherwise, adjust batches accordingly if it's a float (including 1.0).
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
        if 'shuffle' not in val_data_config:
            val_data_config['shuffle'] = False

        # preserve config
        self._update_dataset_config(dataset_name='validation', config=val_data_config)
        self.use_der_metric = val_data_config.get('use_der_metric', getattr(self, 'use_der_metric', False))
        if self.use_der_metric:
            logging.info("DER metric will be used for validation.")
        self.use_rttm_for_der = val_data_config.get('use_rttm_for_der', getattr(self, 'use_rttm_for_der', False))
        if self.use_rttm_for_der:
            logging.info("Raw RTTM files will be used for DER calculation.")
        self._validation_dl = self._setup_dataloader_from_config(config=val_data_config)

    def setup_test_data(self, test_data_config: Optional[Union[DictConfig, Dict]]):
        if test_data_config.get("manifest_filepath", None) is None:
            return 

        if 'shuffle' not in test_data_config:
            test_data_config['shuffle'] = False

        # preserve config
        self._update_dataset_config(dataset_name='test', config=test_data_config)
        self.use_der_metric = test_data_config.get('use_der_metric', getattr(self, 'use_der_metric', False))
        self.use_rttm_for_der = test_data_config.get('use_rttm_for_der', getattr(self, 'use_rttm_for_der', False))
        self._test_dl = self._setup_dataloader_from_config(config=test_data_config)

    def test_dataloader(self):
        if self._test_dl is not None:
            return self._test_dl

    def get_best_label_perm_and_loss(self, *, preds, labels, lengths):
        """
        Sorts probs and labels in descending order of signal_lengths.
        """

        # create debug example
        # p1 = torch.zeros(1, 10, 4)
        # p1[0, :, 0] = 1
        # p2 = torch.zeros(1, 10, 4)
        # p2[0, :, 1] = 1
        # preds = torch.cat([p1, p2], dim=0).cuda()
        # labels = torch.cat([p2, p1], dim=0).cuda()
        # lengths = torch.tensor([10, 10]).cuda()

        batch_size = labels.size(0)
        num_frames = labels.size(1)
        num_speakers = labels.size(2)
        perm_size = self.spk_perm.shape[0] 
        permed_labels = labels[:, :, self.spk_perm]  # (batch_size, frame_len, perm_size, max_num_of_spks)
        permed_preds = torch.unsqueeze(preds, 2).repeat(1,1, self.spk_perm.shape[0],1)  # (batch_size, frame_len, perm_size, max_num_of_spks)

        flattened_permed_labels = rearrange(permed_labels, 'b t p s -> (b p) t s')
        flattened_permed_preds = rearrange(permed_preds, 'b t p s -> (b p) t s')
        flattened_lengths = lengths.unsqueeze(-1).repeat(1, perm_size).view(-1)  # (batch_size*perm_size)

        flattened_loss, _ = self.bce_loss(probs=flattened_permed_preds, labels=flattened_permed_labels, lengths=flattened_lengths)  # [batch_size * perm_size, frame_len, spk_num]
        
        permed_loss = flattened_loss.sum(dim=1) / flattened_lengths.unsqueeze(-1)  # average across time, (batch_size * perm_size, spk_num)
        permed_loss = permed_loss.mean(dim=1)  # average across speakers, (batch_size * perm_size)
        permed_loss = rearrange(permed_loss, '(b p)-> b p', b=batch_size, p=perm_size)

        best_perm_idx = torch.argmin(permed_loss, dim=1)  # (batch_size)
        best_loss = permed_loss[torch.arange(batch_size), best_perm_idx]  # (batch_size)
        best_perm = self.spk_perm[best_perm_idx]  # (batch_size, num_speakers)
        best_permed_labels = permed_labels[torch.arange(batch_size), :, best_perm_idx]  # (batch_size, frame_len, max_num_of_spks)

        return best_permed_labels, best_perm, best_loss

    def forward(self, input_signal=None, input_signal_length=None, processed_signal=None, processed_signal_length=None):
        has_input_signal = input_signal is not None and input_signal_length is not None
        has_processed_signal = processed_signal is not None and processed_signal_length is not None
        if (has_input_signal ^ has_processed_signal) == False:
            raise ValueError(
                f"{self} Arguments ``input_signal`` and ``input_signal_length`` are mutually exclusive "
                " with ``processed_signal`` and ``processed_signal_len`` arguments."
            )

        if not has_processed_signal:
            processed_signal, processed_signal_length = self.preprocessor(
                input_signal=input_signal, length=input_signal_length,
            )

        if self.spec_augmentation is not None and self.training:
            processed_signal = self.spec_augmentation(input_spec=processed_signal, length=processed_signal_length)

        # get all params from self.encoder
        encoder_params = dict(inspect.signature(self.encoder.forward).parameters).keys()
        if "length" in encoder_params:
            # Conformer encoder
            encoder_output = self.encoder(processed_signal, length=processed_signal_length)
            encoded = encoder_output[0]
            encoded_len = encoder_output[1]
            encoded = encoded.transpose(1,2)
        elif "encoder_mask" in encoder_params:
            # Transformer encoder
            processed_signal = processed_signal.transpose(1,2)  # (B, C, T) -> (B, T, C)
            encoder_mask = torch.arange(processed_signal.size(1)).unsqueeze(0).to(processed_signal.device) < processed_signal_length.unsqueeze(-1)
            encoded = self.encoder(processed_signal, encoder_mask)
            encoded_len = processed_signal_length
        else:
            # MLP encoder
            encoded = self.encoder(processed_signal)
            encoded_len = processed_signal_length

        logits = self.decoder(encoded)  # (B, T, C)
        probs = torch.sigmoid(logits)
        return probs, encoded_len
    
    def trim_preds_and_labels(self, preds, labels, encoded_len):
        # clip encoded length to the maximum length of the labels
        encoded_len = torch.clamp(encoded_len, max=labels.size(1))
        # truncate labels or preds based on the shorter one
        if labels.size(1) < preds.size(1):
            preds = preds[:, : labels.size(1), :]  # (batch_size, frame_len, max_num_of_spks)
        elif labels.size(1) > preds.size(1):
            labels = labels[:, : preds.size(1), :]
        return preds, labels, encoded_len

    def training_step(self, batch: EESDAudioRTTMBatch, batch_idx):

        if hasattr(self, '_trainer') and self._trainer is not None:
            log_every_n_steps = self._trainer.log_every_n_steps
        else:
            log_every_n_steps = 1

        probs, encoded_len = self.forward(
            input_signal=batch.audio_signal,
            input_signal_length=batch.audio_signal_length,
            processed_signal=getattr(batch, 'processed_signal', None),
            processed_signal_length=getattr(batch, 'processed_signal_length', None),
        )

        labels = batch.labels

        probs, labels, encoded_len = self.trim_preds_and_labels(probs, labels, encoded_len)

        if self.use_pil:
            # permed_labels: (batch_size, frame_len, max_num_of_spks)
            # best_perm: (batch_size, max_num_of_spks)
            # best_loss: (batch_size)
            permed_labels, best_perm, best_loss = self.get_best_label_perm_and_loss(preds=probs, labels=labels, lengths=encoded_len)
            # loss_value = best_loss.mean()
        else:
            permed_labels = labels

        loss, masks = self.bce_loss(probs=probs, labels=labels.detach(), lengths=encoded_len)
        loss = loss.sum(dim=1) / encoded_len.unsqueeze(-1)  # average across time, (batch_size, spk_num)
        loss = loss.mean(dim=1)  # average across speakers, (batch_size)
        loss_value = loss.mean()

        tensorboard_logs = {}
        tensorboard_logs.update(
            {
                'train_loss': loss_value,
                'learning_rate': self._optimizer.param_groups[0]['lr'],
                'global_step': torch.tensor(self.trainer.global_step, dtype=torch.float32),
            }
        )

        if (batch_idx + 1) % log_every_n_steps == 0:
            self._reset_train_f1_accs()
            preds_vad, preds_ovl, targets_vad, targets_ovl = self.compute_aux_f1(probs, permed_labels)
            self._accuracy_train_vad.update(preds_vad, targets_vad, encoded_len)
            self._accuracy_train_ovl.update(preds_ovl, targets_ovl, encoded_len)
            self._accuracy_train.update(probs, permed_labels, encoded_len)
            train_f1_vad = self._accuracy_train_vad.compute()
            train_f1_ovl = self._accuracy_train_ovl.compute()
            train_f1_acc = self._accuracy_train.compute()
            tensorboard_logs.update(
                {
                    'train_batch_f1': train_f1_acc,
                    'train_batch_f1_vad': train_f1_vad,
                    'train_batch_f1_ovl': train_f1_ovl,
                }
            )

        return {'loss': loss_value, 'log': tensorboard_logs}
    
    def evaluation_step(self, batch: EESDAudioRTTMBatch, batch_idx: int, dataloader_idx: int = 0, mode: str = 'val'):
        metrics = {}
        probs, encoded_len = self.forward(
            input_signal=batch.audio_signal,
            input_signal_length=batch.audio_signal_length,
            processed_signal=getattr(batch, 'processed_signal', None),
            processed_signal_length=getattr(batch, 'processed_signal_length', None),
        )

        labels = batch.labels

        if self.use_der_metric:
            uem_list = batch.uem
            rttm_files = batch.rttm_file if self.use_rttm_for_der else None
            results, pred_anno, label_anno = get_diarization_error_rate(probs, labels, encoded_len, frame_len_secs=self.frame_len_secs, uem_list=uem_list, threshold=self.threshold, collar=self.collar, skip_overlap=self.skip_overlap, detailed=True, rttm_files=rttm_files)
            der_list = []
            cer_list = []
            miss_list = []
            fa_list = []
            self.write_predictions(batch.sample_id, pred_anno, label_anno)
            for i,metric in enumerate(results):
                if metric['total'] == 0:
                    logging.warning(f"Total evaluation time is 0 for sample {batch.sample_id[i]}. Skipping.")
                    continue

                der = metric['diarization error rate']
                cer = metric['confusion'] / metric['total']
                fa = metric['false alarm'] / metric['total']
                miss = metric['missed detection'] / metric['total']
                der_list.append(der)
                cer_list.append(cer)
                fa_list.append(fa)
                miss_list.append(miss)
            der_list = torch.tensor(der_list)
            cer_list = torch.tensor(cer_list)
            miss_list = torch.tensor(miss_list)
            fa_list = torch.tensor(fa_list)
            metrics.update({
                f'{mode}_der': der_list,
                f'{mode}_cer': cer_list,
                f'{mode}_miss': miss_list,
                f'{mode}_fa': fa_list,
            })

        probs, labels, encoded_len = self.trim_preds_and_labels(probs, labels, encoded_len)

        if self.use_pil:
            # permed_labels: (batch_size, frame_len, max_num_of_spks)
            # best_perm: (batch_size, max_num_of_spks)
            # best_loss: (batch_size)
            permed_labels, best_perm, best_loss = self.get_best_label_perm_and_loss(preds=probs, labels=labels, lengths=encoded_len)
        else:
            permed_labels = labels

        loss, masks = self.bce_loss(probs=probs, labels=labels.detach(), lengths=encoded_len)
        loss = loss.sum(dim=1) / encoded_len.unsqueeze(-1)  # average across time, (batch_size, spk_num)
        loss = loss.mean(dim=1)  # average across speakers, (batch_size)
        loss = loss.mean() # average across batch, (1)

        preds_vad, preds_ovl, targets_vad, targets_ovl = self.compute_aux_f1(probs, permed_labels)
        getattr(self, f"_reset_{mode}_f1_accs")()
        getattr(self, f"_accuracy_{mode}").update(probs, permed_labels, encoded_len, cumulative=True)
        getattr(self, f"_accuracy_{mode}_vad").update(preds_vad, targets_vad, encoded_len, cumulative=True)
        getattr(self, f"_accuracy_{mode}_ovl").update(preds_ovl, targets_ovl, encoded_len, cumulative=True)
        acc = getattr(self, f"_accuracy_{mode}").compute()
        acc_vad = getattr(self, f"_accuracy_{mode}_vad").compute()
        acc_ovl = getattr(self, f"_accuracy_{mode}_ovl").compute()

        metrics.update({
            f'{mode}_loss': loss,
            f'{mode}_f1_acc': acc,
            f'{mode}_f1_acc_vad': acc_vad,
            f'{mode}_f1_acc_ovl': acc_ovl,
        })
        
        return metrics

    def evaluation_epoch_end(self, outputs: List[Dict], dataloader_idx: int = 0, mode: str = 'val'):
        if not outputs or any([x is None for x in outputs]):
            return {}

        loss_value = torch.stack([x[f'{mode}_loss'] for x in outputs]).mean()
        acc = torch.stack([x[f'{mode}_f1_acc'] for x in outputs]).mean()
        acc_vad = torch.stack([x[f'{mode}_f1_acc_vad'] for x in outputs]).mean()
        acc_ovl = torch.stack([x[f'{mode}_f1_acc_ovl'] for x in outputs]).mean()

        metrics = {
            f'{mode}_loss': loss_value,
            f'{mode}_f1_acc': acc,
            f'{mode}_f1_acc_vad': acc_vad,
            f'{mode}_f1_acc_ovl': acc_ovl,
        }

        if self.use_der_metric:
            der_all = torch.cat([x[f'{mode}_der'] for x in outputs]).mean()
            cer_all = torch.cat([x[f'{mode}_cer'] for x in outputs]).mean()
            miss_all = torch.cat([x[f'{mode}_miss'] for x in outputs]).mean()
            fa_all = torch.cat([x[f'{mode}_fa'] for x in outputs]).mean()
            metrics.update({
                f'{mode}_der': der_all,
                f'{mode}_cer': cer_all,
                f'{mode}_miss': miss_all,
                f'{mode}_fa': fa_all,
            })

        return {f'{mode}_loss': loss_value, 'log': metrics}

    def validation_step(self, batch: EESDAudioRTTMBatch, batch_idx, dataloader_idx: int = 0):
        metrics = self.evaluation_step(batch, batch_idx, dataloader_idx, mode='val')
        if type(self.trainer.val_dataloaders) == list and len(self.trainer.val_dataloaders) > 1:
            self.validation_step_outputs[dataloader_idx].append(metrics)
        else:
            self.validation_step_outputs.append(metrics)
        return metrics
    
    def test_step(self, batch: EESDAudioRTTMBatch, batch_idx, dataloader_idx: int = 0):
        metrics = self.evaluation_step(batch, batch_idx, dataloader_idx, mode='test')
        if type(self.trainer.test_dataloaders) == list and len(self.trainer.test_dataloaders) > 1:
            self.test_step_outputs[dataloader_idx].append(metrics)
        else:
            self.test_step_outputs.append(metrics)
        return metrics
        

    def multi_validation_epoch_end(self, outputs, dataloader_idx: int = 0):
        return self.evaluation_epoch_end(outputs, dataloader_idx, mode='val')
    
    def multi_test_epoch_end(self, outputs, dataloader_idx: int = 0):
        return self.evaluation_epoch_end(outputs, dataloader_idx, mode='test')

    def compute_aux_f1(self, preds, targets):
        """
        Args:
            preds: (batch_size, frame_len, max_num_of_spks)
            targets: (batch_size, frame_len, max_num_of_spks)
        Returns:
            preds_vad_mask_: (batch_size, frame_len)
            preds_ovl: (batch_size, frame_len, max_num_of_spks)
            targets_vad_mask_: (batch_size, frame_len)
            targets_ovl: (batch_size, frame_len, max_num_of_spks)
        """
        preds_bin = (preds > 0.5).to(torch.int64).detach()
        targets_ovl_mask = (targets.sum(dim=2) >= 2)
        preds_vad_mask = (preds_bin.sum(dim=2) > 0)
        targets_vad_mask = (targets.sum(dim=2) > 0)
        preds_ovl = preds[targets_ovl_mask, :].unsqueeze(0)
        targets_ovl = targets[targets_ovl_mask, :].unsqueeze(0)
        preds_vad_mask_ = preds_vad_mask.int().unsqueeze(0)
        targets_vad_mask_ = targets_vad_mask.int().unsqueeze(0) 
        return preds_vad_mask_, preds_ovl, targets_vad_mask_, targets_ovl
    
    def _reset_train_f1_accs(self):
        self._accuracy_train.reset() 
        self._accuracy_train_vad.reset()
        self._accuracy_train_ovl.reset()

    def _reset_val_f1_accs(self):
        self._accuracy_val.reset() 
        self._accuracy_val_vad.reset()
        self._accuracy_val_ovl.reset()
    
    def _reset_test_f1_accs(self):
        self._accuracy_test.reset() 
        self._accuracy_test_vad.reset()
        self._accuracy_test_ovl.reset()

    def change_predict_params(self, cfg: DictConfig):
        self.threshold = cfg.get('threshold', self.threshold)
        self.collar = cfg.get('collar', self.collar)
        self.skip_overlap = cfg.get('skip_overlap', self.skip_overlap)
        self.output_dir = cfg.get('output_dir', None)
        logging.info(f"Changed prediction parameters: threshold={self.threshold}, collar={self.collar}, skip_overlap={self.skip_overlap}")

    def get_inference_dataloader(self, cfg: DictConfig):
        data_cfg = {
            'manifest_filepath': cfg.manifest_filepath,
            'uem_filepath': cfg.get('uem_filepath', None),
            'batch_size': cfg.get('batch_size', 1),
            'num_workers': cfg.get('num_workers', 1),
            'sample_rate': self.sample_rate,
            'shuffle': False,
            'window_stride': self.frame_len_secs,
            'int_values': cfg.get('int_values', False),
            'trim': cfg.get('trim_silence', False),
            'max_speakers': self.max_num_speakers,
            'round_digits': cfg.get('round_digits', 2),
            'channel_selector': cfg.get('channel_selector', None),
            'normalize_db': cfg.get('normalize_db', None),
        }
        self.use_der_metric = cfg.get('use_der_metric', False)
        self.use_rttm_for_der = cfg.get('use_rttm_for_der', False)
        self.change_predict_params(cfg)
        dataloader = self._setup_dataloader_from_config(config=data_cfg)
        return dataloader
    
    @rank_zero_only
    def write_predictions(self, sample_idx, pred_annotations: List[Annotation], label_annotations = None):
        if not label_annotations:
            label_annotations = [None for _ in range(len(pred_annotations))]
        if not self.output_dir:
            return
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        logging.info(f"Writing predictions to {self.output_dir}", mode=LogMode.ONCE)
        for i, (pred_anno, label_anno) in enumerate(zip(pred_annotations, label_annotations)):
            with open(f"{self.output_dir}/pred_{sample_idx[i]}.rttm", 'w') as f:
                pred_anno.write_rttm(f)
            if label_anno:
                with open(f"{self.output_dir}/label_{sample_idx[i]}.rttm", 'w') as f:
                    label_anno.write_rttm(f)

def rttm_to_annotation(rttm_files):
    annotations = []
    for f in rttm_files:
        anno = load_rttm(f) if f else Annotation()
        if isinstance(anno, dict):
            if len(anno) > 1:
                logging.error(f"More than one audio clip in {f}. Please make sure there is only one audio clip per rttm file by using the same `uri`.")
            anno = list(anno.values())[0]
        annotations.append(anno)
    return annotations

def binary_labels_to_annotation(labels, label_lengths, frame_len_secs) -> List[Annotation]:
    """
    convert binary labels to pyannote annotation for calculating DER
    Args:
        labels: binary labels (batch_size, frame_len, max_num_of_spks)
        label_lengths: lengths of labels (batch_size)
        frame_len_secs: frame length in seconds
    """
    if isinstance(labels, torch.Tensor):
        labels = labels.cpu().numpy()
    if isinstance(label_lengths, torch.Tensor):
        label_lengths = label_lengths.cpu().numpy()
    annotations = []
    for i in range(labels.shape[0]):
        annotation = Annotation()
        max_seq_len = min(labels.shape[1], label_lengths[i])
        for s in range(labels.shape[2]):
            start = 0
            end = 0
            while start < max_seq_len:
                # find the start of the next segment
                while start < max_seq_len and labels[i, start, s] != 1:
                    start += 1
                end = start + 1
                # find the end of the segment
                while end < max_seq_len and labels[i, end, s] != 0:
                    end += 1
                # add the segment to the annotation if there is at least one frame
                if start < max_seq_len and labels[i, start, s] == 1:
                    annotation[Segment(start * frame_len_secs, end * frame_len_secs)] = f"spk_{s}"
                start = end
        annotations.append(annotation)
    return annotations


def get_diarization_error_rate(preds, labels, lengths, frame_len_secs, threshold = 0.5, collar = 0.25, skip_overlap = False, uem_list = None, detailed = True, rttm_files: Optional[List[str]] = None):
    """
    Args:
        preds: predictions (batch_size, frame_len, max_num_of_spks)
        labels: binary labels (batch_size, frame_len, max_num_of_spks)
        lengths: lengths of labels (batch_size)
        frame_len_secs: frame length in seconds
        threshold: threshold for prediction
    """
    metric = DiarizationErrorRate(collar=collar, skip_overlap=skip_overlap)
    preds = preds >= threshold
    preds = preds.cpu().long().numpy()
    labels = labels.cpu().long().numpy()
    lengths = lengths.cpu().numpy()
    results = []
    pred_annotation = binary_labels_to_annotation(preds, lengths, frame_len_secs)
    if rttm_files:
        label_annotation = rttm_to_annotation(rttm_files)
    else:
        label_annotation = binary_labels_to_annotation(labels, lengths, frame_len_secs)
    for i in range(preds.shape[0]):
        uem = uem_list[i] if uem_list else None
        res = metric(hypothesis=pred_annotation[i], reference=label_annotation[i], uem=uem, detailed=detailed)
        results.append(res)
        metric.reset()
    return results, pred_annotation, label_annotation
