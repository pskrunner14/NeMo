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

import torch
import numpy as np
from nemo.collections.asr.parts.utils.speaker_utils import parse_scale_configs
from nemo.collections.asr.parts.utils.speaker_utils import convert_rttm_line, prepare_split_data, get_subsegments
from nemo.collections.asr.parts.utils.offline_clustering import get_argmin_mat
from typing import Dict, List, Tuple, Optional
try:
    from torch.cuda.amp import autocast
except ImportError:
    from contextlib import contextmanager

    @contextmanager
    def autocast(enabled=None):
        yield

torch.backends.cudnn.enabled = False 

class MultiScaleLayer:
    def __init__(self,cfg_e2e_diarizer_model, preprocessor_cfg, speaker_model, dtype=torch.float32):
        self.cfg_e2e_diarizer_model = cfg_e2e_diarizer_model
        self._init_segmentation_info()
        self.preprocessor_cfg = preprocessor_cfg
        self._speaker_model = speaker_model
        window_length_in_sec = self.cfg_e2e_diarizer_model.diarizer.speaker_embeddings.parameters.window_length_in_sec
        self.msdd_multiscale_args_dict = self.multiscale_args_dict
        self.model_spk_num = self.cfg_e2e_diarizer_model.max_num_of_spks
        if self.cfg_e2e_diarizer_model.get('interpolated_scale', None) is not None:
            self.cfg_e2e_diarizer_model.scale_n = len(window_length_in_sec) + 1 # Adding one interpolated scale
            self.emb_scale_n = len(window_length_in_sec) # Scales that are extracted from the audio
            self.msdd_multiscale_args_dict['scale_dict'][self.emb_scale_n] = (self.cfg_e2e_diarizer_model.interpolated_scale, self.cfg_e2e_diarizer_model.interpolated_scale/2)
            self.msdd_multiscale_args_dict['multiscale_weights'] = [1.0] * (self.emb_scale_n+1)
            self.msdd_scale_n = int(self.emb_scale_n+1) if self.cfg_e2e_diarizer_model.interpolated_scale is not None else int(self.emb_scale_n)
        else:
            # Only use the scales in window_length_in_sec
            self.cfg_e2e_diarizer_model.scale_n = len(window_length_in_sec)
            self.emb_scale_n = self.cfg_e2e_diarizer_model.scale_n
            self.msdd_scale_n = self.cfg_e2e_diarizer_model.scale_n
        self.scale_n = len(self.multiscale_args_dict['scale_dict'])
        self.frame_hop = self.cfg_e2e_diarizer_model.interpolated_scale/2 
        self.scale_dict = {int(k): v for k, v in self.multiscale_args_dict['scale_dict'].items()}
        self.longest_scale = self.msdd_multiscale_args_dict['scale_dict'][0][0]
        self.frame_per_sec = (1 / self.preprocessor_cfg.window_stride)
        self.feat_per_sec = self.frame_per_sec
        self.feat_dim = self.preprocessor_cfg.features
        self.max_feat_frame_count = int(self.msdd_multiscale_args_dict["scale_dict"][0][0] * self.frame_per_sec) # 0-th scale, window length
        self.interpolated_scale =  self.cfg_e2e_diarizer_model.get('interpolated_scale', None) 
        self.dtype = dtype
    
    def _init_segmentation_info(self):
        """Initialize segmentation settings: window, shift and multiscale weights.
        """
        self._diarizer_params = self.cfg_e2e_diarizer_model.diarizer
        self.multiscale_args_dict = parse_scale_configs(
            self._diarizer_params.speaker_embeddings.parameters.window_length_in_sec,
            self._diarizer_params.speaker_embeddings.parameters.shift_length_in_sec,
            self._diarizer_params.speaker_embeddings.parameters.multiscale_weights,
        )
        
    def get_subsegments_to_scale_timestamps(self, subsegments: List[Tuple[float, float]], decimals=2):
        """
        Convert subsegment timestamps to scale timestamps.

        Args:
            subsegments (List[Tuple[float, float]]):
                List of subsegment timestamps.

        Returns:
            scale_ts (torch.tensor):
                Tensor containing scale timestamps.
        """
        seg_ts = (torch.tensor(subsegments) * self.feat_per_sec).float()
        scale_ts_round = torch.round(seg_ts, decimals=decimals)
        scale_ts = scale_ts_round.long()
        scale_ts[:, 1] = scale_ts[:, 0] + scale_ts[:, 1]
        return scale_ts 

    def get_scale_dist_mat_t(self, source_scale_idx, ms_seg_timestamps, msdd_scale_n, deci=1):
        """
        Get distance matrix between anchors of the source scale and base scale.

        Args:
            source_scale_idx (int): Source scale index
            ms_seg_timestamps (Tensor): Multi-scale segment timestamps

        Returns:
            abs_dist_mat (Tensor): Distance matrix between anchors of the source scale and base scale
        """
        source_scale_anchor_zeros = torch.mean(ms_seg_timestamps[source_scale_idx, :, :], dim=1) / deci
        base_scale_anchor_zeros = torch.mean(ms_seg_timestamps[msdd_scale_n-1, :, :], dim=1) / deci
        # Get only non-zero timestamps (= Remove zero-padding)
        source_scale_anchor = source_scale_anchor_zeros[source_scale_anchor_zeros.nonzero()].t()
        base_scale_anchor = base_scale_anchor_zeros[base_scale_anchor_zeros.nonzero()].t()
        # Calculate absolute distance matrix
        curr_mat = torch.tile(source_scale_anchor, (base_scale_anchor.shape[0], 1))
        base_mat = torch.tile(base_scale_anchor, (source_scale_anchor.shape[0], 1)).t()
        abs_dist_mat = torch.abs(curr_mat - base_mat)
        return abs_dist_mat
    
    def get_ms_seg_timestamps(
        self, 
        duration: float, 
        min_subsegment_duration: float=0.03
        ):
        """
        Get start and end time of segments in each scale.

        Args:
            sample:
                `DiarizationSpeechLabel` instance from preprocessing.collections
        Returns:
            ms_seg_timestamps (torch.tensor):
                Tensor containing Multiscale segment timestamps.
            ms_seg_counts (torch.tensor):
                Number of segments for each scale. This information is used for reshaping embedding batch
                during forward propagation.
        """
        if duration < 0:
            raise ValueError(f"duration {duration} cannot be negative")
        ms_seg_timestamps_list = []
        total_steps = None
        ms_seg_counts = [0 for _ in range(self.scale_n)]
        for scale_idx in reversed(range(self.scale_n)):
            subsegments = get_subsegments(offset=0, 
                                          window=self.multiscale_args_dict['scale_dict'][scale_idx][0],
                                          shift=self.multiscale_args_dict['scale_dict'][scale_idx][1],
                                          duration=duration, 
                                          min_subsegment_duration=min_subsegment_duration)
            scale_ts_tensor = self.get_subsegments_to_scale_timestamps(subsegments)
            if scale_idx == self.scale_n - 1:
                total_steps = scale_ts_tensor.shape[0]
            ms_seg_counts[scale_idx] = scale_ts_tensor.shape[0]
            scale_ts_padded = torch.cat([scale_ts_tensor, torch.zeros(total_steps - scale_ts_tensor.shape[0], 2, dtype=scale_ts_tensor.dtype)], dim=0)
            ms_seg_timestamps_list.append(scale_ts_padded.detach())
        ms_seg_timestamps_list = ms_seg_timestamps_list[::-1]
        ms_seg_timestamps = torch.stack(ms_seg_timestamps_list).type(self.dtype)
        ms_seg_counts = torch.tensor(ms_seg_counts)
        return ms_seg_timestamps, ms_seg_counts

    def get_interpolate_weights(
        self,
        ms_seg_timestamps: torch.Tensor, 
        base_seq_len: int, 
        msdd_multiscale_args_dict: dict, 
        emb_scale_n: int, 
        msdd_scale_n: int, 
        is_integer_ts=False
        ):
        """
        Interpolate embeddings to a finer scale.

        Args:
            emb_fix (torch.Tensor): embeddings of the base scale
            ms_seg_timestamps (torch.Tensor): timestamps of the base scale
            base_seq_len (int): length of the base scale
        
        Returns:
            emb_fix (torch.Tensor): interpolated embeddings
        """
        deci = self.feat_per_sec if is_integer_ts else 1.0
        half_scale = msdd_multiscale_args_dict['scale_dict'][emb_scale_n-1][1]
        session_scale_dist_mat = self.get_scale_dist_mat_t(source_scale_idx=emb_scale_n-1, 
                                                    ms_seg_timestamps=ms_seg_timestamps[:, :base_seq_len, :], 
                                                    msdd_scale_n=msdd_scale_n, deci=deci)
        target_bool = (session_scale_dist_mat < half_scale)
        session_scale_dist_mat.flatten()[target_bool.flatten() == False] = half_scale
        dist_delta = (half_scale - session_scale_dist_mat.flatten()).reshape(base_seq_len, target_bool.shape[1])
        interpolated_weights = ((dist_delta ** 2).t() / torch.sum(dist_delta ** 2, dim=1).t()).t()  
        return interpolated_weights 
 
    def get_ms_emb_fixed(self, embs: torch.Tensor):
        batch_size = self.scale_mapping.shape[0]
        split_emb_tup = torch.split(embs, self.ms_seg_counts[:, :self.emb_scale_n].flatten().tolist(), dim=0)

        base_seq_len = self.ms_seg_counts[0][self.msdd_scale_n-1].item()
        target_embs = torch.vstack(split_emb_tup).reshape(batch_size, -1, embs.shape[-1])
        intp_w = self.get_interpolate_weights(self.ms_seg_timestamps[0], 
                                         base_seq_len, 
                                         self.msdd_multiscale_args_dict, 
                                         self.emb_scale_n, 
                                         self.msdd_scale_n, 
                                         is_integer_ts=True)
        
        # To make offset values such as, [10, 20, 60, x] -> [0, 10, 30, 90]
        ms_emb_seq = self.add_interpolated_embs(target_embs=target_embs, 
                                                intp_w=intp_w, 
                                                scale_mapping=self.scale_mapping,
                                                ms_seg_counts=self.ms_seg_counts, 
                                                embs=embs, 
        )
        return ms_emb_seq
    
    def repeat_and_align(
        self, 
        ms_seg_timestamps, 
        scale_mapping, 
        all_seq_len, 
        batch_size
        ):
        device = ms_seg_timestamps.device
        repeat_mats_all = scale_mapping[0].to(device)
        ms_ts = ms_seg_timestamps.reshape(batch_size, -1, 2)
        offsets_for_batch = (all_seq_len * torch.arange(self.msdd_scale_n).to(device)).unsqueeze(1).repeat(1, all_seq_len).to(device)
        repeat_mats_all = repeat_mats_all + offsets_for_batch
        ms_ts_rep = ms_ts[:, repeat_mats_all.flatten(), :].reshape(batch_size, self.msdd_scale_n, -1, 2)
        return ms_ts_rep

    def add_interpolated_embs(
        self, 
        target_embs, 
        intp_w,
        scale_mapping,
        ms_seg_counts, 
        embs, 
        ):
        batch_size = scale_mapping.shape[0]
        repeat_mats_ext = scale_mapping[0][:self.emb_scale_n].to(embs.device)
        all_seq_len = ms_seg_counts[0][-1].to(embs.device) 
        scale_count_offsets = torch.tensor([0] + torch.cumsum(ms_seg_counts[0][:self.emb_scale_n-1], dim=0).tolist())
        repeat_mats_ext = repeat_mats_ext + (scale_count_offsets.to(embs.device)).unsqueeze(1).repeat(1, all_seq_len).to(embs.device)
        extracted_embs = target_embs[:, repeat_mats_ext.flatten(), :].reshape(batch_size, self.emb_scale_n, -1, embs.shape[-1])
        finest_extracted_start = ms_seg_counts[0][:self.emb_scale_n-1].sum()
        interpolated_embs = torch.bmm(intp_w.repeat(batch_size, 1, 1), target_embs[:, finest_extracted_start:, :]).unsqueeze(1)
        ms_emb_seq = torch.cat((extracted_embs, interpolated_embs), dim=1).transpose(2, 1) 
        return ms_emb_seq
    
    def get_feature_index_map(
        self, 
        emb_scale_n,
        processed_signal, 
        ms_seg_timestamps, 
        ms_seg_counts,
        device: torch.device,
        ):
        batch_size = processed_signal.shape[0]
        ms_seg_counts_embs = ms_seg_counts[:, :emb_scale_n] 

        total_seg_count = torch.sum(ms_seg_counts_embs)
        ms_seg_counts_embs_flatten =  ms_seg_counts_embs.flatten()

        # The following index-tensors are needed for matrix reshaping without nested for-loops.
        batch_index_range = torch.repeat_interleave(torch.arange(batch_size).to(device), ms_seg_counts_embs.sum(dim=1), dim=0)
        scale_index_range = torch.repeat_interleave(torch.arange(emb_scale_n).repeat(batch_size).to(device) , ms_seg_counts_embs_flatten)

        # Pre-compute sequence indices for faster assigning: 
        seq_index_range = torch.arange(ms_seg_counts_embs_flatten.max())
        segment_index_range = torch.concat([seq_index_range[:seq_len] for seq_len in ms_seg_counts_embs_flatten]).to(device)
        target_timestamps = ms_seg_timestamps[batch_index_range, scale_index_range, segment_index_range, :].to(torch.int64)
        feature_count_range = target_timestamps[:, 1] - target_timestamps[:, 0]
        
        # Pre-compute feature indices for faster assigning:
        feature_frame_length_range, feature_frame_interval_range= self.get_feat_range_mats(max_feat_len=processed_signal.shape[2], 
                                                                                             feature_count_range=feature_count_range, 
                                                                                             target_timestamps=target_timestamps, 
                                                                                             device=processed_signal.device)
        # Assign frame-by-frame indices for one-pass assignment without nested for-loops
        ms_seg_count_frame_range = torch.repeat_interleave(torch.arange(total_seg_count).to(device), feature_count_range)       
        batch_frame_range = torch.repeat_interleave(batch_index_range, feature_count_range)
        return total_seg_count, ms_seg_count_frame_range, feature_frame_length_range, batch_frame_range, feature_frame_interval_range, feature_count_range
    
    def forward_multiscale(
        self, 
        processed_signal, 
        processed_signal_len, 
        ):
        bs = processed_signal.shape[0]
        frame_count = torch.floor(torch.tensor((processed_signal_len.max().item() / self.frame_per_sec)/self.frame_hop)).to(int).item()
        ms_seg_timestamps, ms_seg_counts = self.get_ms_seg_timestamps(duration=(frame_count * self.frame_hop),
                                                                      min_subsegment_duration=self.interpolated_scale)
        scale_mapping = torch.stack(get_argmin_mat(ms_seg_timestamps))
        self.ms_seg_timestamps = ms_seg_timestamps.unsqueeze(0).repeat(bs, 1, 1, 1).to(processed_signal.device)
        self.ms_seg_counts = ms_seg_counts.unsqueeze(0).repeat(bs, 1).to(processed_signal.device)
        self.scale_mapping = scale_mapping.unsqueeze(0).repeat(bs, 1, 1).to(processed_signal.device)
        tsc, mscfr, fflr, bfr, ffir, fcr = self.get_feature_index_map(emb_scale_n=self.emb_scale_n,
                                                                      processed_signal=processed_signal, 
                                                                      ms_seg_timestamps=self.ms_seg_timestamps, 
                                                                      ms_seg_counts=self.ms_seg_counts, 
                                                                      device=processed_signal.device)

        _embs, pools = self.forward_multi_decoder(processed_signal=processed_signal, 
                                            processed_signal_len=processed_signal_len, 
                                            total_seg_count=tsc,
                                            ms_seg_count_frame_range=mscfr, 
                                            feature_frame_length_range=fflr, 
                                            batch_frame_range=bfr, 
                                            feature_frame_interval_range=ffir,
                                            feature_count_range=fcr,
                                            device=processed_signal.device,
                                            )
        # Reshape the embedding vectors into multi-scale inputs
        ms_emb_seq = self.get_ms_emb_fixed(embs=_embs)
        return ms_emb_seq

    def forward_multi_decoder(
        self,
        processed_signal, 
        processed_signal_len, 
        total_seg_count,
        ms_seg_count_frame_range, 
        feature_frame_length_range, 
        batch_frame_range, 
        feature_frame_interval_range,
        feature_count_range,
        device,
        ):
        # Assign the acoustic feature values in processed_signal at once
        encoded, _ = self._speaker_model.encoder(audio_signal=processed_signal, length=processed_signal_len)
        encoded_segments = torch.zeros(total_seg_count, encoded.shape[1], self.max_feat_frame_count).to(torch.float32).to(device)
        encoded_segments[ms_seg_count_frame_range, :, feature_frame_length_range] = encoded[batch_frame_range, :, feature_frame_interval_range]
        pools, embs = self._speaker_model.decoder(encoder_output=encoded_segments, length=feature_count_range) 
        return embs, pools
    
    def get_feat_range_mats(self, max_feat_len, feature_count_range, target_timestamps, device):
        """ 
        """
        feat_index_range = torch.arange(0, max_feat_len).to(device) 
        feature_frame_offsets = torch.repeat_interleave(target_timestamps[:, 0], feature_count_range)
        feature_frame_interval_range = torch.concat([feat_index_range[stt:end] for (stt, end) in target_timestamps]).to(device)
        feature_frame_length_range = feature_frame_interval_range - feature_frame_offsets
        return feature_frame_length_range, feature_frame_interval_range 