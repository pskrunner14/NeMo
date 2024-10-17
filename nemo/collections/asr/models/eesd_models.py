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

from collections import OrderedDict
from typing import Dict, List, Optional, Union
import time
import torch
from torch import nn
from hydra.utils import instantiate
from omegaconf import DictConfig
from pytorch_lightning import Trainer
from tqdm import tqdm
import itertools
import random
from nemo.collections.asr.parts.preprocessing.perturb import process_augmentations
from nemo.collections.common.data.lhotse import get_lhotse_dataloader_from_config
from nemo.collections.asr.data.audio_to_eesd_label_lhotse import LhotseSpeechToDiarizationLabelDataset
from nemo.collections.asr.data.audio_to_eesd_label import AudioToSpeechMSDDTrainDataset
from nemo.collections.asr.metrics.multi_binary_acc import MultiBinaryAccuracy
from nemo.collections.asr.models.asr_model import ExportableEncDecModel
from nemo.collections.asr.models.label_models import EncDecSpeakerLabelModel
from nemo.collections.asr.parts.preprocessing.features import WaveformFeaturizer
from nemo.collections.asr.parts.utils.asr_multispeaker_utils import get_pil_target, get_ats_targets
from nemo.core.classes import ModelPT
from nemo.core.classes.common import PretrainedModelInfo
from nemo.core.neural_types import AudioSignal, LengthsType, NeuralType
from nemo.core.neural_types.elements import ProbsType
from nemo.utils import logging

try:
    from torch.cuda.amp import autocast
except ImportError:
    from contextlib import contextmanager

    @contextmanager
    def autocast(enabled=None):
        yield


torch.backends.cudnn.enabled = False 

__all__ = ['EncDecDiarLabelModel']

def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform(m.weight)
        m.bias.data.fill_(0.01)

class SortformerEncLabelModel(ModelPT, ExportableEncDecModel):
    """
    Encoder decoder class for multiscale diarization decoder (MSDD). Model class creates training, validation methods for setting
    up data performing model forward pass.

    This model class expects config dict for:
        * preprocessor
        * Transformer Encoder
        * FastConformer Encoder
    """

    @classmethod
    def list_available_models(cls) -> List[PretrainedModelInfo]:
        """
        This method returns a list of pre-trained model which can be instantiated directly from NVIDIA's NGC cloud.

        Returns:
            List of available pre-trained models.
        """
        result = []
        return result

    def __init__(self, cfg: DictConfig, trainer: Trainer = None):
        """
        Initialize an MSDD model and the specified speaker embedding model. 
        In this init function, training and validation datasets are prepared.
        """
        random.seed(42)
        self._trainer = trainer if trainer else None
        self._cfg = cfg
        
        if self._trainer:
            self.world_size = trainer.num_nodes * trainer.num_devices
        else:
            self.world_size = 1

        if self._trainer is not None and self._cfg.get('augmentor', None) is not None:
            self.augmentor = process_augmentations(self._cfg.augmentor)
        else:
            self.augmentor = None
        super().__init__(cfg=self._cfg, trainer=trainer)
        self.preprocessor = SortformerEncLabelModel.from_config_dict(self._cfg.preprocessor)

        if hasattr(self._cfg, 'spec_augment') and self._cfg.spec_augment is not None:
            self.spec_augmentation = SortformerEncLabelModel.from_config_dict(self._cfg.spec_augment)
        else:
            self.spec_augmentation = None

        self.encoder = SortformerEncLabelModel.from_config_dict(self._cfg.encoder)
        self.sortformer_modules = SortformerEncLabelModel.from_config_dict(self._cfg.sortformer_modules)
        self.transformer_encoder = SortformerEncLabelModel.from_config_dict(self._cfg.transformer_encoder)
        self._init_loss_weights()

        self.eps = 1e-3
        self.loss = instantiate(self._cfg.loss)

        self.streaming_mode = self._cfg.get("streaming_mode", False)
        self.save_hyperparameters("cfg")
        self._init_eval_metrics()
        
        speaker_inds = list(range(self._cfg.max_num_of_spks))
        self.speaker_permutations = torch.tensor(list(itertools.permutations(speaker_inds))) # Get all permutations
    
    def _init_loss_weights(self):
        pil_weight = self._cfg.get("pil_weight", 0.0)
        ats_weight = self._cfg.get("ats_weight", 1.0)
        if pil_weight + ats_weight == 0:
            raise ValueError(f"weights for PIL {pil_weight} and ATS {ats_weight} cannot sum to 0")
        self.pil_weight = pil_weight/(pil_weight + ats_weight)
        self.ats_weight = ats_weight/(pil_weight + ats_weight)
        logging.info(f"Normalized weights for PIL {self.pil_weight} and ATS {self.ats_weight}")
        
    def _init_eval_metrics(self):
        """ 
        If there is no label, then the evaluation metrics will be based on Permutation Invariant Loss (PIL).
        """
        self._accuracy_test = MultiBinaryAccuracy()
        self._accuracy_train = MultiBinaryAccuracy()
        self._accuracy_valid = MultiBinaryAccuracy()
        
        self._accuracy_test_ats = MultiBinaryAccuracy()
        self._accuracy_train_ats = MultiBinaryAccuracy()
        self._accuracy_valid_ats = MultiBinaryAccuracy()

    def _reset_train_metrics(self):
        self._accuracy_train.reset()
        self._accuracy_train_ats.reset()
        
    def _reset_valid_metrics(self):
        self._accuracy_valid.reset()
        self._accuracy_valid_ats.reset()
        
    def __setup_dataloader_from_config(self, config):
        # Switch to lhotse dataloader if specified in the config
        if config.get("use_lhotse"):
            return get_lhotse_dataloader_from_config(
                config,
                global_rank=self.global_rank,
                world_size=self.world_size,
                dataset=LhotseSpeechToDiarizationLabelDataset(cfg=config),
            )

        featurizer = WaveformFeaturizer(
            sample_rate=config['sample_rate'], int_values=config.get('int_values', False), augmentor=self.augmentor
        )

        if 'manifest_filepath' in config and config['manifest_filepath'] is None:
            logging.warning(f"Could not load dataset as `manifest_filepath` was None. Provided config : {config}")
            return None

        logging.info(f"Loading dataset from {config.manifest_filepath}")

        if self._trainer is not None:
            global_rank = self._trainer.global_rank
        else:
            global_rank = 0
        time_flag = time.time()
        logging.info("AAB: Starting Dataloader Instance loading... Step A")
        AudioToSpeechDiarTrainDataset = AudioToSpeechMSDDTrainDataset
        
        preprocessor = EncDecSpeakerLabelModel.from_config_dict(self._cfg.preprocessor)
        dataset = AudioToSpeechDiarTrainDataset(
            manifest_filepath=config.manifest_filepath,
            preprocessor=preprocessor,
            soft_label_thres=config.soft_label_thres,
            session_len_sec=config.session_len_sec,
            num_spks=config.num_spks,
            featurizer=featurizer,
            window_stride=self._cfg.preprocessor.window_stride,
            global_rank=global_rank,
            soft_targets=config.soft_targets if 'soft_targets' in config else False,
        )
        logging.info(f"AAB: Dataloader dataset is created, starting torch.utils.data.Dataloader step B: {time.time() - time_flag}")

        self.data_collection = dataset.collection
        self.collate_ds = dataset
         
        dataloader_instance = torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=config.batch_size,
            collate_fn=self.collate_ds.msdd_train_collate_fn,
            drop_last=config.get('drop_last', False),
            shuffle=False,
            num_workers=config.get('num_workers', 1),
            pin_memory=config.get('pin_memory', False),
        )
        logging.info(f"AAC: Dataloader Instance loading is done ETA Step B done: {time.time() - time_flag}")
        return dataloader_instance
    
    def setup_training_data(self, train_data_config: Optional[Union[DictConfig, Dict]]):
        self._train_dl = self.__setup_dataloader_from_config(config=train_data_config,)

    def setup_validation_data(self, val_data_layer_config: Optional[Union[DictConfig, Dict]]):
        self._validation_dl = self.__setup_dataloader_from_config(config=val_data_layer_config,)
    
    def setup_test_data(self, test_data_config: Optional[Union[DictConfig, Dict]]):
        self._test_dl = self.__setup_dataloader_from_config(config=test_data_config,)

    def test_dataloader(self):
        if self._test_dl is not None:
            return self._test_dl

    @property
    def input_types(self) -> Optional[Dict[str, NeuralType]]:
        if hasattr(self.preprocessor, '_sample_rate'):
            audio_eltype = AudioSignal(freq=self.preprocessor._sample_rate)
        else:
            audio_eltype = AudioSignal()
        return {
            "audio_signal": NeuralType(('B', 'T'), audio_eltype),
            "audio_signal_length": NeuralType(('B',), LengthsType()),
        }

    @property
    def output_types(self) -> Dict[str, NeuralType]:
        return OrderedDict(
            {
                "preds": NeuralType(('B', 'T', 'C'), ProbsType()),
            }
        )
    
    def frontend_encoder(self, processed_signal, processed_signal_length, pre_encode_input=False):
        """ 
        Generate encoder outputs from frontend encoder.
        
        Args:
            process_signal (torch.Tensor): tensor containing audio signal
            processed_signal_length (torch.Tensor): tensor containing lengths of audio signal

        Returns:
            emb_seq (torch.Tensor): tensor containing encoder outputs
            emb_seq_length (torch.Tensor): tensor containing lengths of encoder outputs
        """
        # Spec augment is not applied during evaluation/testing
        if self.spec_augmentation is not None and self.training:
            processed_signal = self.spec_augmentation(input_spec=processed_signal, length=processed_signal_length)
        self.encoder = self.encoder.to(self.device)
        emb_seq, emb_seq_length = self.encoder(audio_signal=processed_signal, length=processed_signal_length)
        emb_seq = emb_seq.transpose(1, 2)
        if self._cfg.encoder.d_model != self._cfg.tf_d_model:
            self.sortformer_modules.encoder_proj = self.sortformer_modules.encoder_proj.to(self.device)
            emb_seq = self.sortformer_modules.encoder_proj(emb_seq)   
        return emb_seq, emb_seq_length

    def forward_infer(self, emb_seq, start_pos=0):
        """
        The main forward pass for diarization inference.

        Args:
            emb_seq (torch.Tensor): tensor containing embeddings of multiscale embedding vectors
                Dimension: (batch_size, max_seg_count, msdd_scale_n, emb_dim)
        
        Returns:
            preds (torch.Tensor): Sorted tensor containing predicted speaker labels
                Dimension: (batch_size, max. diar frame count, num_speakers)
            encoder_states_list (list): List containing total speaker memory for each step for debugging purposes
                Dimension: [(batch_size, max. diar frame count, inner dim), ]
        """
        encoder_mask = self.sortformer_modules.length_to_mask(emb_seq)
        trans_emb_seq = self.transformer_encoder(encoder_states=emb_seq, encoder_mask=encoder_mask)
        preds = self.sortformer_modules.forward_speaker_sigmoids(trans_emb_seq)
        return preds
    
    def process_signal(self, audio_signal, audio_signal_length):
        audio_signal = audio_signal.to(self.device)
        audio_signal = (1/(audio_signal.max()+self.eps)) * audio_signal 
        processed_signal, processed_signal_length = self.preprocessor(input_signal=audio_signal, length=audio_signal_length) 
        return processed_signal, processed_signal_length
    
    def forward(
        self, 
        audio_signal, 
        audio_signal_length, 
    ):
        """
        Forward pass for training and inference.
        
        Args:
            audio_signal (torch.Tensor): tensor containing audio waveform
                Dimension: (batch_size, num_samples)
            audio_signal_length (torch.Tensor): tensor containing lengths of audio waveforms
                Dimension: (batch_size,)
            
        Returns:
            preds (torch.Tensor): Sorted tensor containing predicted speaker labels
                Dimension: (batch_size, max. diar frame count, num_speakers)
            encoder_states_list (list): List containing total speaker memory for each step for debugging purposes
                Dimension: [(batch_size, max. diar frame count, inner dim), ]
        """
        processed_signal, processed_signal_length = self.process_signal(audio_signal=audio_signal, audio_signal_length=audio_signal_length)
        processed_signal = processed_signal[:, :, :processed_signal_length.max()]
        if self._cfg.get("streaming_mode", False):
            raise NotImplementedError("Streaming mode is not implemented yet.")
        else:
            emb_seq, _ = self.frontend_encoder(processed_signal=processed_signal, processed_signal_length=processed_signal_length)
            preds = self.forward_infer(emb_seq)
        return preds
    
    def _get_aux_train_evaluations(self, preds, targets, target_lens):
        # Arrival-time sorted (ATS) targets
        targets_ats = get_ats_targets(targets.clone(), preds)
        # Optimally permuted targets for Permutation-Invariant Loss (PIL)
        targets_pil = get_pil_target(targets.clone(), preds, speaker_permutations=self.speaker_permutations)
        ats_loss = self.loss(probs=preds, labels=targets_ats, target_lens=target_lens)
        pil_loss = self.loss(probs=preds, labels=targets_pil, target_lens=target_lens)
        loss = self.ats_weight * ats_loss + self.pil_weight * pil_loss

        self._accuracy_train(preds, targets_pil, target_lens)
        train_f1_acc, train_precision, train_recall = self._accuracy_train.compute()

        self._accuracy_train_ats(preds, targets_ats, target_lens)
        train_f1_acc_ats, _, _ = self._accuracy_train_ats.compute()

        train_metrics = {
            'loss': loss,
            'ats_loss': ats_loss,
            'pil_loss': pil_loss,
            'learning_rate': self._optimizer.param_groups[0]['lr'],
            'train_f1_acc': train_f1_acc,
            'train_precision': train_precision,
            'train_recall': train_recall,
            'train_f1_acc_ats': train_f1_acc_ats,
        } 
        return train_metrics

    def training_step(self, batch: list, batch_idx: int):
        audio_signal, audio_signal_length, targets, target_lens = batch
        preds = self.forward(audio_signal=audio_signal, audio_signal_length=audio_signal_length)
        train_metrics = self._get_aux_train_evaluations(preds, targets, target_lens)
        self._reset_train_metrics()
        self.log_dict(train_metrics, sync_dist=True, on_step=True, on_epoch=False, logger=True)
        return {'loss': train_metrics['loss']}
        
    def _cumulative_test_set_eval(self, score_dict: Dict[str, float], batch_idx: int, sample_count: int):
        if batch_idx == 0:
            self.total_sample_counts = 0
            self.cumulative_f1_acc_sum = 0
            
        self.total_sample_counts += sample_count
        self.cumulative_f1_acc_sum += score_dict['f1_acc'] * sample_count
        
        cumulative_f1_acc = self.cumulative_f1_acc_sum / self.total_sample_counts
        return {"cum_test_f1_acc": cumulative_f1_acc}

    def _get_aux_validation_evaluations(self, preds, targets, target_lens):
        targets_ats = get_ats_targets(targets.clone(), preds, speaker_permutations=self.speaker_permutations)
        targets_pil = get_pil_target(targets.clone(), preds, speaker_permutations=self.speaker_permutations)

        val_ats_loss = self.loss(probs=preds, labels=targets_ats, target_lens=target_lens)
        val_pil_loss = self.loss(probs=preds, labels=targets_pil, target_lens=target_lens)
        val_loss = self.ats_weight * val_ats_loss + self.pil_weight * val_pil_loss

        self._accuracy_valid(preds, targets_pil, target_lens)
        val_f1_acc, val_precision, val_recall = self._accuracy_valid.compute()

        self._accuracy_valid_ats(preds, targets_ats, target_lens)
        valid_f1_acc_ats, _, _ = self._accuracy_valid_ats.compute()

        self._accuracy_valid.reset()
        self._accuracy_valid_ats.reset()

        val_metrics = {
            'val_loss': val_loss,
            'val_ats_loss': val_ats_loss,
            'val_pil_loss': val_pil_loss,
            'val_f1_acc': val_f1_acc,
            'val_precision': val_precision,
            'val_recall': val_recall,
            'val_f1_acc_ats': valid_f1_acc_ats,
        }
        return val_metrics

    def validation_step(self, batch: list, batch_idx: int, dataloader_idx: int = 0):
        audio_signal, audio_signal_length, targets, target_lens = batch
        preds = self.forward(
            audio_signal=audio_signal,
            audio_signal_length=audio_signal_length,
        )
        val_metrics = self._get_aux_validation_evaluations(preds, targets, target_lens)
        if isinstance(self.trainer.val_dataloaders, list) and len(self.trainer.val_dataloaders) > 1:
            self.validation_step_outputs[dataloader_idx].append(val_metrics)
        else:
            self.validation_step_outputs.append(val_metrics)
        return val_metrics

    def multi_validation_epoch_end(self, outputs: list, dataloader_idx: int = 0):
        if not outputs:
            logging.warning(f"`outputs` is None; empty outputs for dataloader={dataloader_idx}")
            return None
        val_loss_mean = torch.stack([x['val_loss'] for x in outputs]).mean()
        val_ats_loss_mean = torch.stack([x['val_ats_loss'] for x in outputs]).mean()
        val_pil_loss_mean = torch.stack([x['val_pil_loss'] for x in outputs]).mean()
        val_f1_acc_mean = torch.stack([x['val_f1_acc'] for x in outputs]).mean()
        val_precision_mean = torch.stack([x['val_precision'] for x in outputs]).mean()
        val_recall_mean = torch.stack([x['val_recall'] for x in outputs]).mean()
        val_f1_acc_ats_mean = torch.stack([x['val_f1_acc_ats'] for x in outputs]).mean()

        self._reset_valid_metrics()
        
        multi_val_metrics = {
            'val_loss': val_loss_mean,
            'val_ats_loss': val_ats_loss_mean,
            'val_pil_loss': val_pil_loss_mean,
            'val_f1_acc': val_f1_acc_mean,
            'val_precision': val_precision_mean,
            'val_recall': val_recall_mean,
            'val_f1_acc_ats': val_f1_acc_ats_mean,
        }
        return {'log': multi_val_metrics}
    
    def multi_test_epoch_end(self, outputs: List[Dict[str, torch.Tensor]], dataloader_idx: int = 0):
        test_loss_mean = torch.stack([x['test_loss'] for x in outputs]).mean()
        f1_acc, _, _ = self._accuracy_test.compute()
        self._accuracy_test.reset()
        multi_test_metrics = {
            'test_loss': test_loss_mean,
            'test_f1_acc': f1_acc,
        }
        self.log_dict(multi_test_metrics, sync_dist=True, on_step=True, on_epoch=False, logger=True)
        return multi_test_metrics
   
    def test_step(self, batch: list, batch_idx: int, dataloader_idx: int = 0):
        audio_signal, audio_signal_length, targets, target_lens = batch
        batch_size = audio_signal.shape[0]
        target_lens = self.target_lens.unsqueeze(0).repeat(batch_size, 1).to(audio_signal.device)
        preds = self.forward(
            audio_signal=audio_signal,
            audio_signal_length=audio_signal_length,
        )
        targets_pil = get_pil_target(targets.clone(), preds, speaker_permutations=self.speaker_permutations)
        self._accuracy_test(preds, targets_pil, target_lens, cumulative=True)
        f1_acc, _, _ = self._accuracy_test.compute()
        batch_score_dict = {"f1_acc": f1_acc}
        cum_score_dict = self._cumulative_test_set_eval(score_dict=batch_score_dict, batch_idx=batch_idx, sample_count=len(sequence_lengths))
        return self.preds_all

    def _get_aux_test_batch_evaluations(self, batch_idx, preds, targets, target_lens):
        targets_ats = get_ats_targets(targets.clone(), preds, speaker_permutations=self.speaker_permutations)
        targets_pil = get_pil_target(targets.clone(), preds, speaker_permutations=self.speaker_permutations)
        self._accuracy_test(preds, targets_pil, target_lens)
        f1_acc, precision, recall = self._accuracy_test.compute()
        self.batch_f1_accs_list.append(f1_acc)
        self.batch_precision_list.append(precision)
        self.batch_recall_list.append(recall)
        logging.info(f"batch {batch_idx}: f1_acc={f1_acc}, precision={precision}, recall={recall}")

        self._accuracy_test_ats(preds, targets_ats, target_lens)
        f1_acc_ats, precision_ats, recall_ats = self._accuracy_test_ats.compute()
        self.batch_f1_accs_ats_list.append(f1_acc_ats)
        logging.info(f"batch {batch_idx}: f1_acc_ats={f1_acc_ats}, precision_ats={precision_ats}, recall_ats={recall_ats}")

        self._accuracy_test.reset()
        self._accuracy_test_ats.reset()

    def test_batch(self,):
        self.preds_total_list, self.batch_f1_accs_list, self.batch_precision_list, self.batch_recall_list, self.batch_f1_accs_ats_list = [], [], [], [], []

        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(self._test_dl)):
                audio_signal, audio_signal_length, targets, target_lens = batch
                audio_signal = audio_signal.to(self.device)
                audio_signal_length = audio_signal_length.to(self.device)
                preds = self.forward(
                    audio_signal=audio_signal,
                    audio_signal_length=audio_signal_length,
                )
                preds = preds.detach().to('cpu')
                if preds.shape[0] == 1: # batch size = 1
                    self.preds_total_list.append(preds)
                else:
                    self.preds_total_list.extend(torch.split(preds, [1] * preds.shape[0]))
                torch.cuda.empty_cache()
                self._get_aux_test_batch_evaluations(batch_idx, preds, targets, target_lens)

        logging.info(f"Batch F1Acc. MEAN: {torch.mean(torch.tensor(self.batch_f1_accs_list))}")
        logging.info(f"Batch Precision MEAN: {torch.mean(torch.tensor(self.batch_precision_list))}")
        logging.info(f"Batch Recall MEAN: {torch.mean(torch.tensor(self.batch_recall_list))}")
        logging.info(f"Batch ATS F1Acc. MEAN: {torch.mean(torch.tensor(self.batch_f1_accs_ats_list))}")
        logging.info(f"Batch ATS Precision MEAN: {torch.mean(torch.tensor(self.batch_precision_ats_list))}")
        logging.info(f"Batch ATS Recall MEAN: {torch.mean(torch.tensor(self.batch_recall_ats_list))}")


class CompressiveSortformerEncLabelModel(SortformerEncLabelModel):
    def __init__(self, cfg: DictConfig, trainer: Trainer = None):
        super().__init__(cfg=cfg, trainer=trainer)
        self.automatic_optimization = False
        self.split_size = self._cfg.step_size        
        self.truncated_bptt_size = self._cfg.truncated_bptt_steps
        self.compressive_transformer_encoder = CompressiveSortformerEncLabelModel.from_config_dict(cfg.compressive_transformer_encoder)
        pretrained_sortformer_model = SortformerEncLabelModel.restore_from('/home/weiqingw/workspace/projects/diasr/clean_diarization_compressive_sortformer/im303a-ft7_epoch6-19.nemo', map_location="cpu")
        self.encoder.load_state_dict(pretrained_sortformer_model.encoder.state_dict(), strict=True)

    def split_batch(self, data, split_size):
        """
        data: B x T x D
        """

        splits = []
        for t in range(0, data.shape[1], split_size):
            data_split = data[:, t:t + split_size]
            splits.append(data_split)    

        return splits
        
    def forward(
        self, 
        audio_signal, 
        audio_signal_length, 
    ):
        """
        Forward pass for training.
        
        Args:
            audio_signal (torch.Tensor): tensor containing audio waveform
                Dimension: (batch_size, num_samples)
            audio_signal_length (torch.Tensor): tensor containing lengths of audio waveforms
                Dimension: (batch_size,)
            
        Returns:
            preds (torch.Tensor): Sorted tensor containing predicted speaker labels
                Dimension: (batch_size, max. diar frame count, num_speakers)
            encoder_states_list (list): List containing total speaker memory for each step for debugging purposes
                Dimension: [(batch_size, max. diar frame count, inner dim), ]
        """
        processed_signal, processed_signal_length = self.process_signal(audio_signal=audio_signal, audio_signal_length=audio_signal_length)
        processed_signal = processed_signal[:, :, :processed_signal_length.max()]
        emb_seq, _ = self.frontend_encoder(processed_signal=processed_signal, processed_signal_length=processed_signal_length)

        preds, encoder_states_list = self.forward_infer(emb_seq, streaming_mode=self.streaming_mode)
        return preds, encoder_states_list
    
    @torch.no_grad()
    def get_stats(self, preds, targets_pil, targets_ats, target_lens, suffix=""):
        self._reset_train_f1_accs()
        preds_vad, preds_ovl, targets_vad, targets_ovl = self.compute_aux_f1(preds, targets_pil)
        self._accuracy_train_vad(preds_vad, targets_vad, target_lens)
        self._accuracy_train_ovl(preds_ovl, targets_ovl, target_lens)
        train_f1_vad = self._accuracy_train_vad.compute()
        train_f1_ovl = self._accuracy_train_ovl.compute()
        self._accuracy_train(preds, targets_pil, target_lens)
        f1_acc = self._accuracy_train.compute()
        precision, recall = self._accuracy_train.compute_pr()

        self.log('train_f1_acc' + suffix, f1_acc, sync_dist=True)
        self.log('train_precision' + suffix, precision, sync_dist=True)
        self.log('train_recall' + suffix, recall, sync_dist=True)
        self.log('train_f1_vad_acc' + suffix, train_f1_vad, sync_dist=True)
        self.log('train_f1_ovl_acc' + suffix, train_f1_ovl, sync_dist=True)

        self._reset_train_f1_accs()
        preds_vad, preds_ovl, targets_vad, targets_ovl = self.compute_aux_f1(preds, targets_ats)
        self._accuracy_train_vad(preds_vad, targets_vad, target_lens)
        self._accuracy_train_ovl(preds_ovl, targets_ovl, target_lens)
        train_f1_vad = self._accuracy_train_vad.compute()
        train_f1_ovl = self._accuracy_train_ovl.compute()
        self._accuracy_train(preds, targets_ats, target_lens)
        f1_acc = self._accuracy_train.compute()

        self.log('train_f1_acc_ats' + suffix, f1_acc, sync_dist=True)
        self.log('train_f1_vad_acc_ats' + suffix, train_f1_vad, sync_dist=True)
        self.log('train_f1_ovl_acc_ats' + suffix, train_f1_ovl, sync_dist=True)
        self._accuracy_train.reset()
    
    def training_step(self, batch: List, batch_idx: int):
        opt = self.optimizers()

        audio_signal, audio_signal_length, targets, target_lens = batch
        # 0. get the targets 
        targets_ats = get_ats_targets(targets.clone(), discrete=False, accum_frames=self.sort_accum_frames)

        # 1. Split the batch for truncated BPTT
        audio_signal_splits = self.split_batch(audio_signal, self.split_size*8*160)
        targets_ats_splits = self.split_batch(targets_ats, self.split_size)
        targets_splits = self.split_batch(targets, self.split_size)
        split_batches = list(zip(audio_signal_splits, targets_ats_splits, targets_splits))[:-1]
                
        memories, mask = None, None
        # total_preds = torch.empty(processed_signal.shape[0], 0, 2).to(processed_signal.device)
        # 2. start the training loop for BPTT
        grad_accum_every = len(split_batches)
        losses = []
        for i, split_batches in tqdm(enumerate(split_batches), total=len(split_batches), desc="BPTT loop, lr %.8lf" % opt.param_groups[0]['lr'], leave=False):
            audio_signal_splits, targets_ats_split, targets_split = split_batches
            audio_signal_length_split = (torch.ones(audio_signal_splits.shape[0], ).to(audio_signal_splits.device) * audio_signal_splits.shape[1]).int()
            processed_signal_split, processed_signal_length_split = self.process_signal(audio_signal=audio_signal_splits, audio_signal_length=audio_signal_length_split)
            processed_signal_length_split -= 1
            processed_signal_split = processed_signal_split[:, :, :processed_signal_length_split.max()] # B x D x T
            target_lens_split = (torch.ones(targets_ats.shape[0], 1).to(targets_ats.device) * targets_ats_split.shape[1]).int()
            
            # preds,  = self.forward(emb_seq_split, memories=memories, mask=mask)
            emb_seq_split, _ = self.frontend_encoder(processed_signal=processed_signal_split, processed_signal_length=processed_signal_length_split)
            trans_emb_seq, memories = self.compressive_transformer_encoder(encoder_states=emb_seq_split, encoder_targets=targets_ats_split, encoder_mask=mask, encoder_mems_list=memories, return_mems=True)
            preds = self.sortformer_diarizer.forward_speaker_sigmoids(trans_emb_seq)
            targets_pil_split = self.sort_targets_with_preds(targets_split.clone(), preds)
            ats_loss = self.loss(probs=preds, labels=targets_ats_split, target_lens=target_lens_split)
            pil_loss = self.loss(probs=preds, labels=targets_pil_split, target_lens=target_lens_split)
            loss = self.ats_weight * ats_loss + self.pil_weight * pil_loss # + 0.1*aux_loss
            losses.append(loss)

            if (i+1) % self.truncated_bptt_size == 0 or i == grad_accum_every - 1:
                opt.zero_grad()
                losses = torch.mean(torch.stack(losses))
                self.manual_backward(losses)
                opt.step()

                sch = self.lr_schedulers()
                sch.step()
                losses = []
            # loss = ats_loss
            self.log('loss', loss, sync_dist=True)
            self.log('ats_loss', ats_loss, sync_dist=True)
            self.log('pil_loss', pil_loss, sync_dist=True)
            # self.log('aux_loss', aux_loss, sync_dist=True)
            self.log('learning_rate', self._optimizer.param_groups[0]['lr'], sync_dist=True)

            self.get_stats(preds, targets_pil_split, targets_ats_split, target_lens_split, suffix="_split")
            # total_preds = torch.cat([total_preds, preds], dim=1)

            # opt.zero_grad()
            # self.manual_backward(loss / grad_accum_every)
            # opt.step()

            # sch = self.lr_schedulers()
            # sch.step()
        # return {'loss': loss / grad_accum_every}
        # targets_pil = self.sort_targets_with_preds(targets.clone(), total_preds)
        # self.get_stats(total_preds, targets_pil, targets_ats, target_lens, suffix="_total")

        # self.clip_gradients(opt, gradient_clip_val=0.5, gradient_clip_algorithm="norm")

    def sort_probs_and_labels(self, labels, discrete=True, thres=0.5, return_inds=False, accum_frames=1):
        max_cap_val = labels.shape[1] + 1
        labels_discrete = labels.clone()
        if not discrete:
            labels_discrete[labels_discrete < thres] = 0
            labels_discrete[labels_discrete >= thres] = 1
        m=torch.ones(labels.shape[1],labels.shape[1]).triu().to(labels.device)
        labels_accum = torch.matmul(labels_discrete.permute(0,2,1),m).permute(0,2,1)
        labels_accum[labels_accum < accum_frames] = 0
        label_fz = self.find_first_nonzero(labels_accum, max_cap_val)
        label_fz[label_fz == -1] = max_cap_val
        sorted_inds = torch.sort(label_fz)[1]
        sorted_labels = labels.transpose(0,1)[:, torch.arange(labels.shape[0]).unsqueeze(1), sorted_inds].transpose(0, 1)
        if return_inds:
            return sorted_labels, sorted_inds
        else:
            return sorted_labels

    def validation_step(self, batch: list, batch_idx: int, dataloader_idx: int = 0):
        audio_signal, audio_signal_length, targets, target_lens = batch
        targets_ats = self.sort_probs_and_labels(targets.clone(), discrete=False, accum_frames=self.sort_accum_frames)

        # 1. Split the batch for truncated BPTT
        audio_signal_splits = self.split_batch(audio_signal, self.split_size*8*160)
        targets_ats_splits = self.split_batch(targets_ats, self.split_size)
        split_batches = list(zip(audio_signal_splits, targets_ats_splits))[:-1]
        
        memories, mask = None, None
        loss, ats_loss, pil_loss, f1_acc, precision, recall, valid_f1_vad, valid_f1_ovl, f1_acc_ats, valid_f1_ovl_ats = 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
        total_preds = torch.empty(audio_signal.shape[0], 0, 4).to(audio_signal.device)
        # 3. start the training loop for BPTT
        for i, sbatch in tqdm(enumerate(split_batches), total=len(split_batches), desc="BPTT loop", leave=False):
            audio_signal_splits, targets_split = sbatch
            audio_signal_length_split = (torch.ones(audio_signal_splits.shape[0], ).to(audio_signal_splits.device) * audio_signal_splits.shape[1]).int()
            processed_signal_split, processed_signal_length_split = self.process_signal(audio_signal=audio_signal_splits, audio_signal_length=audio_signal_length_split)
            processed_signal_length_split -= 1
            processed_signal_split = processed_signal_split[:, :, :processed_signal_length_split.max()] # B x D x T
            target_lens = (torch.ones(targets_split.shape[0], 1).to(targets_split.device) * targets_split.shape[1]).int()
            
            emb_seq_split, _ = self.frontend_encoder(processed_signal=processed_signal_split, processed_signal_length=processed_signal_length_split)
            encoder_mask = self.length_to_mask(emb_seq_split)
            trans_emb_seq, memories = self.compressive_transformer_encoder(encoder_states=emb_seq_split, encoder_targets=targets_split, encoder_mask=mask, encoder_mems_list=memories, return_mems=True)
            preds = self.sortformer_diarizer.forward_speaker_sigmoids(trans_emb_seq)

            torch.save(preds, 'test/preds%d.pt'%i)
            torch.save(targets_split, 'test/targets%d.pt'%i)

            total_preds = torch.cat([total_preds, preds], dim=1)
            # preds = total_preds

            # Arrival-time sorted (ATS) targets
            targets_ats = targets_split
            # Optimally permuted targets for Permutation-Invariant Loss (PIL)
            targets_pil = self.get_pil_target(targets_split.clone(), preds)

            ats_loss += self.loss(probs=preds, labels=targets_ats, target_lens=target_lens)
            pil_loss += self.loss(probs=preds, labels=targets_pil, target_lens=target_lens)
            loss += self.ats_weight * ats_loss + self.pil_weight * pil_loss

            self._reset_valid_f1_accs()
            preds_vad, preds_ovl, targets_vad, targets_ovl = self.compute_aux_f1(preds, targets_pil)
            self._accuracy_valid_vad(preds_vad, targets_vad, target_lens)
            valid_f1_vad += self._accuracy_valid_vad.compute()
            self._accuracy_valid_ovl(preds_ovl, targets_ovl, target_lens)
            valid_f1_ovl += self._accuracy_valid_ovl.compute()
            self._accuracy_valid(preds, targets_pil, target_lens)
            f1_acc += self._accuracy_valid.compute()
            tprecision, trecall = self._accuracy_valid.compute_pr()
            precision += tprecision
            recall += trecall

            self._reset_valid_f1_accs()
            preds_vad, preds_ovl, targets_vad, targets_ovl = self.compute_aux_f1(preds, targets_ats)
            self._accuracy_valid_vad(preds_vad, targets_vad, target_lens)
            self._accuracy_valid_ovl(preds_ovl, targets_ovl, target_lens)
            valid_f1_ovl_ats += self._accuracy_valid_ovl.compute()
            self._accuracy_valid(preds, targets_ats, target_lens)
            f1_acc_ats += self._accuracy_valid.compute()

        n = len(targets_ats_splits)
        metrics = {
            'val_loss': loss/n,
            'val_ats_loss': ats_loss/n,
            'val_pil_loss': pil_loss/n,
            'val_f1_acc': f1_acc/n,
            'val_precision': precision/n,
            'val_recall': recall/n,
            'val_f1_vad_acc': valid_f1_vad/n,
            'val_f1_ovl_acc': valid_f1_ovl/n,
            'val_f1_acc_ats': f1_acc_ats/n,
            'val_f1_ovl_acc_ats': valid_f1_ovl_ats/n
        }

        if isinstance(self.trainer.val_dataloaders, list) and len(self.trainer.val_dataloaders) > 1:
            self.validation_step_outputs[dataloader_idx].append(metrics)
        else:
            self.validation_step_outputs.append(metrics)
        return metrics
    
    def test_batch(self,):
        self.preds_total_list, self.batch_f1_accs_list, self.batch_precision_list, self.batch_recall_list = [], [], [], []
        self.batch_f1_accs_ats_list, self.batch_precision_ats_list, self.batch_recall_ats_list = [], [], []

        with torch.no_grad():
            
            for batch_idx, batch in enumerate(tqdm(self._test_dl)):
                audio_signal, audio_signal_length, targets, target_lens = batch
                audio_signal = audio_signal.to(self.device)
                audio_signal_length = audio_signal_length.to(self.device)
                # targets_ats = self.sort_probs_and_labels(targets.clone(), discrete=False, accum_frames=self.sort_accum_frames)
                targets_ats = get_ats_targets(targets.clone(), discrete=False, accum_frames=self.sort_accum_frames)

                audio_signal_splits = self.split_batch(audio_signal, self.split_size*8*160)
                targets_ats_splits = self.split_batch(targets_ats, self.split_size)
                split_batches = list(zip(audio_signal_splits, targets_ats_splits))[:-1]

                memories, mask, preds = None, None, None
                total_preds = torch.empty(audio_signal.shape[0], 0, 4).to(audio_signal.device)
                # 2. start the training loop for BPTT
                for i, split_batches in tqdm(enumerate(split_batches), total=len(split_batches), desc="BPTT loop", leave=False):
                    audio_signal_splits, targets_ats_split = split_batches
                    audio_signal_length_split = (torch.ones(audio_signal_splits.shape[0], ).to(audio_signal_splits.device) * audio_signal_splits.shape[1]).int()
                    processed_signal_split, processed_signal_length_split = self.process_signal(audio_signal=audio_signal_splits, audio_signal_length=audio_signal_length_split)
                    processed_signal_length_split -= 1
                    processed_signal_split = processed_signal_split[:, :, :processed_signal_length_split.max()] # B x D x T
                    targets_ats_split = targets_ats_split.to(audio_signal_splits.device)
                    target_lens_split = (torch.ones(targets_ats.shape[0], 1).to(targets_ats.device) * targets_ats_split.shape[1]).int()
                    
                    emb_seq_split, _ = self.frontend_encoder(processed_signal=processed_signal_split, processed_signal_length=processed_signal_length_split)
                    if i == 0:
                        trans_emb_seq, memories = self.compressive_transformer_encoder(encoder_states=emb_seq_split, encoder_targets=None, encoder_mask=mask, encoder_mems_list=memories, return_mems=True)
                    else:
                        trans_emb_seq, memories = self.compressive_transformer_encoder(encoder_states=emb_seq_split, encoder_targets=preds, encoder_mask=mask, encoder_mems_list=memories, return_mems=True)

                    preds = self.sortformer_diarizer.forward_speaker_sigmoids(trans_emb_seq)

                    total_preds = torch.cat([total_preds, preds], dim=1)

                    preds = (preds > 0.5).float()
                preds = total_preds
                torch.save(preds, 'test/preds_infer_%d.pt'%batch_idx)
                torch.save(targets_ats, 'test/targets_infer_%d.pt'%batch_idx)
                # preds, encoder_states_list = self.forward(
                #     audio_signal=audio_signal,
                #     audio_signal_length=audio_signal_length,
                # )
                targets = targets[:, :preds.shape[1]]
                preds = preds.detach().to('cpu')
                if preds.shape[0] == 1: # Batch size = 1
                    self.preds_total_list.append(preds)
                else:
                    self.preds_total_list.extend(torch.split(preds, [1] * preds.shape[0]))
                torch.cuda.empty_cache()
                # Batch-wise evaluation: Arrival-time sorted (ATS) targets
                targets_ats = self.sort_probs_and_labels(targets.clone(), discrete=False, accum_frames=self.sort_accum_frames)
                # Optimally permuted targets for Permutation-Invariant Loss (PIL)
                if self.use_new_pil:
                    targets_pil = self.sort_targets_with_preds_new(targets.clone(), preds, target_lens)
                else:
                    targets_pil = self.sort_targets_with_preds(targets.clone(), preds)
                preds_vad, preds_ovl, targets_vad, targets_ovl = self.compute_aux_f1(preds, targets_pil)
                self._accuracy_valid(preds, targets_pil, target_lens)
                f1_acc = self._accuracy_valid.compute()
                precision, recall = self._accuracy_valid.compute_pr()
                self.batch_f1_accs_list.append(f1_acc)
                self.batch_precision_list.append(precision)
                self.batch_recall_list.append(recall)
                logging.info(f"batch {batch_idx}: f1_acc={f1_acc}, precision={precision}, recall={recall}")

                self._reset_valid_f1_accs()
                self._accuracy_valid(preds, targets_ats, target_lens)
                f1_acc = self._accuracy_valid.compute()
                precision, recall = self._accuracy_valid.compute_pr()
                self.batch_f1_accs_ats_list.append(f1_acc)
                self.batch_precision_ats_list.append(precision)
                self.batch_recall_ats_list.append(recall)
                logging.info(f"batch {batch_idx}: f1_acc_ats={f1_acc}, precision_ats={precision}, recall_ats={recall}")
                
        logging.info(f"Batch F1Acc. MEAN: {torch.mean(torch.tensor(self.batch_f1_accs_list))}")
        logging.info(f"Batch Precision MEAN: {torch.mean(torch.tensor(self.batch_precision_list))}")
        logging.info(f"Batch Recall MEAN: {torch.mean(torch.tensor(self.batch_recall_list))}")
        logging.info(f"Batch ATS F1Acc. MEAN: {torch.mean(torch.tensor(self.batch_f1_accs_ats_list))}")
        logging.info(f"Batch ATS Precision MEAN: {torch.mean(torch.tensor(self.batch_precision_ats_list))}")
        logging.info(f"Batch ATS Recall MEAN: {torch.mean(torch.tensor(self.batch_recall_ats_list))}")
    
