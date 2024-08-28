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

"""
This script can be used to simulate cache-aware streaming for ASR models. The ASR model to be used with this script need to get trained in streaming mode. Currently only Conformer models supports this streaming mode.
You may find examples of streaming models under 'NeMo/example/asr/conf/conformer/streaming/'.

It works both on a manifest of audio files or a single audio file. It can perform streaming for a single stream (audio) or perform the evalution in multi-stream model (batch_size>1).
The manifest file must conform to standard ASR definition - containing `audio_filepath` and `text` as the ground truth.

# Usage

## To evaluate a model in cache-aware streaming mode on a single audio file:

python speech_to_text_streaming_infer.py \
    --asr_model=asr_model.nemo \
    --audio_file=audio_file.wav \
    --compare_vs_offline \
    --use_amp \
    --debug_mode

## To evaluate a model in cache-aware streaming mode on a manifest file:

python speech_to_text_streaming_infer.py \
    --asr_model=asr_model.nemo \
    --manifest_file=manifest_file.json \
    --batch_size=16 \
    --compare_vs_offline \
    --use_amp \
    --debug_mode

You may drop the '--debug_mode' and '--compare_vs_offline' to speedup the streaming evaluation.
If compare_vs_offline is not used, then significantly larger batch_size can be used.
Setting `--pad_and_drop_preencoded` would perform the caching for all steps including the first step.
It may result in slightly different outputs from the sub-sampling module compared to offline mode for some techniques like striding and sw_striding.
Enabling it would make it easier to export the model to ONNX.

## Hybrid ASR models
For Hybrid ASR models which have two decoders, you may select the decoder by --set_decoder DECODER_TYPE, where DECODER_TYPE can be "ctc" or "rnnt".
If decoder is not set, then the default decoder would be used which is the RNNT decoder for Hybrid ASR models.

## Multi-lookahead models
For models which support multiple lookaheads, the default is the first one in the list of model.encoder.att_context_size. To change it, you may use --att_context_size, for example --att_context_size [70,1].


## Evaluate a model trained with full context for offline mode

You may try the cache-aware streaming with a model trained with full context in offline mode.
But the accuracy would not be very good with small chunks as there is inconsistency between how the model is trained and how the streaming inference is done.
The accuracy of the model on the borders of chunks would not be very good.

To use a model trained with full context, you need to pass the chunk_size and shift_size arguments.
If shift_size is not passed, chunk_size would be used as the shift_size too.
Also argument online_normalization should be enabled to simulate a realistic streaming.
The following command would simulate cache-aware streaming on a pretrained model from NGC with chunk_size of 100, shift_size of 50 and 2 left chunks as left context.
The chunk_size of 100 would be 100*4*10=4000ms for a model with 4x downsampling and 10ms shift in feature extraction.

python speech_to_text_streaming_infer.py \
    --asr_model=stt_en_conformer_ctc_large \
    --chunk_size=100 \
    --shift_size=50 \
    --left_chunks=2 \
    --online_normalization \
    --manifest_file=manifest_file.json \
    --batch_size=16 \
    --compare_vs_offline \
    --use_amp \
    --debug_mode

"""


import contextlib
import json
import os
import time
from argparse import ArgumentParser

import torch
from omegaconf import open_dict, OmegaConf

import nemo.collections.asr as nemo_asr
from nemo.collections.asr.metrics.wer import word_error_rate
from nemo.collections.asr.parts.utils.rnnt_utils import Hypothesis
from nemo.collections.asr.parts.utils.streaming_utils import CacheAwareStreamingAudioBuffer
from nemo.collections.asr.parts.utils.streaming_spk_utils import CacheAwareStreamingAudioSpkBuffer, CacheAwareStreamingAudioTgtSpkBuffer
from nemo.utils import logging, model_utils

from nemo.collections.asr.models import ASRModel

from lhotse.cut import MixedCut, MonoCut
from lhotse.dataset import AudioSamples
from lhotse.dataset.collation import collate_vectors, collate_matrices
from lhotse.utils import compute_num_samples
from lhotse import SupervisionSet, SupervisionSegment, MonoCut, Recording, CutSet

from lhotse.utils import compute_num_samples


import numpy as np


def extract_transcriptions(hyps):
    """
        The transcribed_texts returned by CTC and RNNT models are different.
        This method would extract and return the text section of the hypothesis.
    """
    if isinstance(hyps[0], Hypothesis):
        transcriptions = []
        for hyp in hyps:
            transcriptions.append(hyp.text)
    else:
        transcriptions = hyps
    return transcriptions


def calc_drop_extra_pre_encoded(asr_model, step_num, pad_and_drop_preencoded):
    # for the first step there is no need to drop any tokens after the downsampling as no caching is being used
    if step_num == 0 and not pad_and_drop_preencoded:
        return 0
    else:
        return asr_model.encoder.streaming_cfg.drop_extra_pre_encoded


def perform_streaming(
    asr_model, audio_info, streaming_buffer, rttm_left, rttm_right, compare_vs_offline=False, debug_mode=False, pad_and_drop_preencoded=False
):
    batch_size = len(streaming_buffer.streams_length)
    if compare_vs_offline:
        # would pass the whole audio at once through the model like offline mode in order to compare the results with the stremaing mode
        # the output of the model in the offline and streaming mode should be exactly the same
        with torch.inference_mode():
            with autocast():
                processed_signal, processed_signal_length = streaming_buffer.get_all_audios()
                with torch.no_grad():
                    (
                        pred_out_offline,
                        transcribed_texts,
                        cache_last_channel_next,
                        cache_last_time_next,
                        cache_last_channel_len,
                        best_hyp,
                    ) = asr_model.conformer_stream_step(
                        processed_signal=processed_signal,
                        processed_signal_length=processed_signal_length,
                        return_transcription=True,
                    )
        final_offline_tran = extract_transcriptions(transcribed_texts)
        logging.info(f" Final offline transcriptions:   {final_offline_tran}")
    else:
        final_offline_tran = None

    cache_last_channel, cache_last_time, cache_last_channel_len = asr_model.encoder.get_initial_cache_state(
        batch_size=batch_size
    )


    previous_hypotheses = None
    streaming_buffer_iter = iter(streaming_buffer)
    pred_out_stream = None

    rttms = SupervisionSet.from_rttm(audio_info['rttm_filepath'])
    #get speaker_mapping
    spk_id_mapping = {}
    idx = 0
    for i in range(len(rttms)):
        if rttms[i].start >= audio_info['offset']:
            if rttms[i].speaker not in spk_id_mapping:
                spk_id_mapping[rttms[i].speaker] = idx
                idx += 1
    

    text_idx = 0
    token_idx = 0
    raw_texts = ''
    prev_spk = ''

    for step_num, (chunk_audio, chunk_lengths, chunk_raw_audio, chunk_lengths_raw_audio, start_sample, end_sample) in enumerate(streaming_buffer_iter):

        cut_rec = Recording.from_file(audio_info['audio_filepath'])
        cut_sups = [SupervisionSegment(id=cut_rec.id, recording_id = cut_rec.id, start = 0, duration = cut_rec.duration)]
        cut = MonoCut(id = cut_rec.id, start = start_sample / 16000, duration = (end_sample - start_sample) / 16000, channel = 0, recording = cut_rec, supervisions = cut_sups)
        segments_iterator = rttms.find(recording_id=cut.recording_id, start_after=cut.start-rttm_left, end_before=cut.end+rttm_right, adjust_offset=True)


        segments = [s for s in segments_iterator]
        new_segments = []

        # for segment in segments:
        #     print('*'*10)
        #     print(segment)
        # import pdb; pdb.set_trace()

        for segment in segments:
            if segment.end > rttm_left:
                new_segments.append(segment)
        segments = new_segments
        #generate speaker id according to arrival time
        segments.sort(key = lambda rttm_sup: rttm_sup.start)
        seen = set()
        seen_add = seen.add
        speaker_ats = [s.speaker for s in segments if not (s.speaker in seen or seen_add(s.speaker))]
        speaker_to_idx_map = {
                spk: idx
                for idx, spk in enumerate(speaker_ats)
        }
        #initialize mask matrices (num_speaker, encoder_hidden)
        mask = np.zeros((4, int(np.ceil(np.ceil((cut.num_samples + 1 + 16000*(rttm_left+rttm_right)) / 160) / 8))))

        for rttm_sup in segments:
            speaker_idx = speaker_to_idx_map[rttm_sup.speaker]
            #only consider the first <num_speakers> speakers
            if speaker_idx < 4:
                st = (
                            compute_num_samples(rttm_sup.start, cut.sampling_rate)
                            if rttm_sup.start > 0
                            else 0
                        )
                et = (
                            compute_num_samples(rttm_sup.end, cut.sampling_rate)
                            if rttm_sup.end < cut.duration + rttm_left + rttm_right
                            else compute_num_samples(cut.duration, cut.sampling_rate)
                        )         
                mask[speaker_idx, int(np.ceil(np.ceil((st + 1) / 160) / 8)):int(np.ceil(np.ceil((et + 1) / 160) / 8))] = 1
        #cutoff mask first 1 sec and last 1 sec
        left_hidden_len = int(np.ceil(np.ceil((16000*rttm_left + 1) / 160) / 8))
        right_hidden_len = int(np.ceil(np.ceil((16000*rttm_right + 1) / 160) / 8))
        mask = mask[:,left_hidden_len-1 : -right_hidden_len+1]
        #rematch spk_id_mapping and speaker_to_idx_map: important
        global_mapping = {}

        for key, value in speaker_to_idx_map.items():
            global_mapping[value] = spk_id_mapping[key]
        mask = torch.tensor(mask, dtype=torch.float32).unsqueeze(0)
        mask = torch.transpose(mask, 1, 2)
        print(f'Step {step_num}')
        with torch.inference_mode():
            with autocast():
                # keep_all_outputs needs to be True for the last step of streaming when model is trained with att_context_style=regular
                # otherwise the last outputs would get dropped
                with torch.no_grad():
                    (
                        pred_out_stream,
                        transcribed_texts,
                        raw_texts,
                        cache_last_channel,
                        cache_last_time,
                        cache_last_channel_len,
                        previous_hypotheses,
                        text_idx,
                        token_idx,
                        prev_spk,
                    ) = asr_model.conformer_stream_step(
                        audio_signal=chunk_raw_audio,
                        audio_signal_lengths=chunk_lengths_raw_audio,
                        processed_signal=chunk_audio,
                        processed_signal_length=chunk_lengths,
                        spk_targets=mask.to(chunk_raw_audio.device),
                        global_mapping=global_mapping,
                        raw_texts = raw_texts,
                        text_idx = text_idx,
                        token_idx = token_idx,
                        prev_spk = prev_spk,
                        cache_last_channel=cache_last_channel,
                        cache_last_time=cache_last_time,
                        cache_last_channel_len=cache_last_channel_len,
                        keep_all_outputs=streaming_buffer.is_buffer_empty(),
                        previous_hypotheses=previous_hypotheses,
                        previous_pred_out=pred_out_stream,
                        drop_extra_pre_encoded=calc_drop_extra_pre_encoded(
                            asr_model, step_num, pad_and_drop_preencoded
                        ),
                        return_transcription=True,
                    )
        if debug_mode:
            logging.info(f"Streaming transcriptions: {extract_transcriptions(transcribed_texts)}")
    # final_streaming_tran = extract_transcriptions(transcribed_texts)
    final_streaming_tran = [raw_texts]
    logging.info(f"Final streaming transcriptions: {final_streaming_tran}")

    if compare_vs_offline:
        # calculates and report the differences between the predictions of the model in offline mode vs streaming mode
        # Normally they should be exactly the same predictions for streaming models
        pred_out_stream_cat = torch.cat(pred_out_stream)
        pred_out_offline_cat = torch.cat(pred_out_offline)
        if pred_out_stream_cat.size() == pred_out_offline_cat.size():
            diff_num = torch.sum(pred_out_stream_cat != pred_out_offline_cat).cpu().numpy()
            logging.info(
                f"Found {diff_num} differences in the outputs of the model in streaming mode vs offline mode."
            )
        else:
            logging.info(
                f"The shape of the outputs of the model in streaming mode ({pred_out_stream_cat.size()}) is different from offline mode ({pred_out_offline_cat.size()})."
            )
    return final_streaming_tran, final_offline_tran


def perform_streaming_tgt(
    asr_model, audio_info, streaming_buffer, rttm_left, rttm_right, compare_vs_offline=False, debug_mode=False, pad_and_drop_preencoded=False
):
    batch_size = len(streaming_buffer.streams_length)
    if compare_vs_offline:
        # would pass the whole audio at once through the model like offline mode in order to compare the results with the stremaing mode
        # the output of the model in the offline and streaming mode should be exactly the same
        with torch.inference_mode():
            with autocast():
                processed_signal, processed_signal_length = streaming_buffer.get_all_audios()
                with torch.no_grad():
                    (
                        pred_out_offline,
                        transcribed_texts,
                        cache_last_channel_next,
                        cache_last_time_next,
                        cache_last_channel_len,
                        best_hyp,
                    ) = asr_model.conformer_stream_step(
                        processed_signal=processed_signal,
                        processed_signal_length=processed_signal_length,
                        return_transcription=True,
                    )
        final_offline_tran = extract_transcriptions(transcribed_texts)
        logging.info(f" Final offline transcriptions:   {final_offline_tran}")
    else:
        final_offline_tran = None

    cache_last_channel, cache_last_time, cache_last_channel_len = asr_model.encoder.get_initial_cache_state(
        batch_size=batch_size
    )


    previous_hypotheses = None
    streaming_buffer_iter = iter(streaming_buffer)
    pred_out_stream = None

    rttms = SupervisionSet.from_rttm(audio_info['rttm_filepath'])
    #get speaker_mapping
    spk_id_mapping = {}
    idx = 0
    for i in range(len(rttms)):
        if rttms[i].start >= audio_info['offset']:
            if rttms[i].speaker not in spk_id_mapping:
                spk_id_mapping[rttms[i].speaker] = idx
                idx += 1
    target_spk = audio_info['query_speaker_id']

    text_idx = 0
    token_idx = 0
    raw_texts = ''
    prev_spk = ''
    global_mapping = {}

    for step_num, (chunk_audio, chunk_lengths, chunk_raw_audio, chunk_lengths_raw_audio, start_sample, end_sample) in enumerate(streaming_buffer_iter):
        if step_num == 0:

            query_duration = audio_info['query_duration']
            separater_duration = 1
            query_num_sample = int(16000 * query_duration)
            separater_num_sample = int(separater_duration * 16000)
            mask_prepend = np.zeros((4, int(np.ceil(np.ceil((query_num_sample + separater_num_sample + 1) / 160) / 8))))
            mask_prepend[0,:query_num_sample] = 1
            residule = chunk_raw_audio.shape[1] - query_num_sample - separater_num_sample
        
            if residule > 0:
                cut_rec = Recording.from_file(audio_info['audio_filepath'])
                cut_sups = [SupervisionSegment(id=cut_rec.id, recording_id = cut_rec.id, start = audio_info['offset'], duration = audio_info['duration'])]
                cut = MonoCut(id = cut_rec.id, start = audio_info['offset'], duration = residule / 16000, channel = 0, recording = cut_rec, supervisions = cut_sups)
        else:

            cut_rec = Recording.from_file(audio_info['audio_filepath'])
            cut_sups = [SupervisionSegment(id=cut_rec.id, recording_id = cut_rec.id, start = audio_info['offset'], duration = audio_info['duration'])]
            cut = MonoCut(id = cut_rec.id, start = (start_sample - query_num_sample - separater_num_sample)  / 16000, duration = (end_sample - start_sample) / 16000, channel = 0, recording = cut_rec, supervisions = cut_sups)

        
        segments_iterator = rttms.find(recording_id=cut.recording_id, start_after=cut.start-rttm_left, end_before=cut.end+rttm_right, adjust_offset=True)


        segments = [s for s in segments_iterator]
        new_segments = []

        for segment in segments:
            if segment.end > rttm_left:
                new_segments.append(segment)
        segments = new_segments
        #generate speaker id according to arrival time
        segments.sort(key = lambda rttm_sup: rttm_sup.start)
        seen = set()
        seen_add = seen.add
        speaker_lst = [target_spk] + [s.speaker for s in segments]
        speaker_ats = [s for s in speaker_lst if not (s in seen or seen_add(s))]
        speaker_to_idx_map = {
                spk: idx
                for idx, spk in enumerate(speaker_ats)
        }
        #initialize mask matrices (num_speaker, encoder_hidden)
        mask = np.zeros((4, int(np.ceil(np.ceil((cut.num_samples + 1 + 16000*(rttm_left+rttm_right)) / 160) / 8))))

        for rttm_sup in segments:
            speaker_idx = speaker_to_idx_map[rttm_sup.speaker]
            #only consider the first <num_speakers> speakers
            if speaker_idx < 4:
                st = (
                            compute_num_samples(rttm_sup.start, cut.sampling_rate)
                            if rttm_sup.start > 0
                            else 0
                        )
                et = (
                            compute_num_samples(rttm_sup.end, cut.sampling_rate)
                            if rttm_sup.end < cut.duration + rttm_left + rttm_right
                            else compute_num_samples(cut.duration, cut.sampling_rate)
                        )         
                mask[speaker_idx, int(np.ceil(np.ceil((st + 1) / 160) / 8)):int(np.ceil(np.ceil((et + 1) / 160) / 8))] = 1
        #cutoff mask first 1 sec and last 1 sec
        left_hidden_len = int(np.ceil(np.ceil((16000*rttm_left + 1) / 160) / 8))
        right_hidden_len = int(np.ceil(np.ceil((16000*rttm_right + 1) / 160) / 8))
        mask = mask[:,left_hidden_len-1 : -right_hidden_len+1]
        #rematch spk_id_mapping and speaker_to_idx_map: important
        if step_num == 0:
            mask = np.concatenate([mask_prepend,mask], axis = 1)

        for key, value in speaker_to_idx_map.items():
            global_mapping[value] = spk_id_mapping[key]


        mask = torch.tensor(mask, dtype=torch.float32).unsqueeze(0)
        mask = torch.transpose(mask, 1, 2)
        print(f'Step {step_num}')
        with torch.inference_mode():
            with autocast():
                # keep_all_outputs needs to be True for the last step of streaming when model is trained with att_context_style=regular
                # otherwise the last outputs would get dropped
                with torch.no_grad():
                    (
                        pred_out_stream,
                        transcribed_texts,
                        raw_texts,
                        cache_last_channel,
                        cache_last_time,
                        cache_last_channel_len,
                        previous_hypotheses,
                        text_idx,
                        token_idx,
                        prev_spk,
                    ) = asr_model.conformer_stream_step(
                        audio_signal=chunk_raw_audio,
                        audio_signal_lengths=chunk_lengths_raw_audio,
                        processed_signal=chunk_audio,
                        processed_signal_length=chunk_lengths,
                        spk_targets=mask.to(chunk_raw_audio.device),
                        global_mapping=global_mapping,
                        raw_texts = raw_texts,
                        text_idx = text_idx,
                        token_idx = token_idx,
                        prev_spk = prev_spk,
                        cache_last_channel=cache_last_channel,
                        cache_last_time=cache_last_time,
                        cache_last_channel_len=cache_last_channel_len,
                        keep_all_outputs=streaming_buffer.is_buffer_empty(),
                        previous_hypotheses=previous_hypotheses,
                        previous_pred_out=pred_out_stream,
                        drop_extra_pre_encoded=calc_drop_extra_pre_encoded(
                            asr_model, step_num, pad_and_drop_preencoded
                        ),
                        return_transcription=True,
                    )
        if debug_mode:
            logging.info(f"Streaming transcriptions: {extract_transcriptions(transcribed_texts)}")
    final_streaming_tran = extract_transcriptions(transcribed_texts)
    # final_streaming_tran = [raw_texts]
    logging.info(f"Final streaming transcriptions: {final_streaming_tran}")

    if compare_vs_offline:
        # calculates and report the differences between the predictions of the model in offline mode vs streaming mode
        # Normally they should be exactly the same predictions for streaming models
        pred_out_stream_cat = torch.cat(pred_out_stream)
        pred_out_offline_cat = torch.cat(pred_out_offline)
        if pred_out_stream_cat.size() == pred_out_offline_cat.size():
            diff_num = torch.sum(pred_out_stream_cat != pred_out_offline_cat).cpu().numpy()
            logging.info(
                f"Found {diff_num} differences in the outputs of the model in streaming mode vs offline mode."
            )
        else:
            logging.info(
                f"The shape of the outputs of the model in streaming mode ({pred_out_stream_cat.size()}) is different from offline mode ({pred_out_offline_cat.size()})."
            )
    return final_streaming_tran, final_offline_tran


def main():
    parser = ArgumentParser()
    parser.add_argument(
        "--asr_model", type=str, required=True, help="Path to an ASR model .nemo file or name of a pretrained model.",
    )
    parser.add_argument(
        "--device", type=str, help="The device to load the model onto and perform the streaming", default="cuda"
    )
    parser.add_argument("--audio_file", type=str, help="Path to an audio file to perform streaming", default=None)
    parser.add_argument(
        "--manifest_file",
        type=str,
        help="Path to a manifest file containing audio files to perform streaming",
        default=None,
    )
    parser.add_argument("--use_amp", action="store_true", help="Whether to use AMP")
    parser.add_argument("--debug_mode", action="store_true", help="Whether to print more detail in the output.")
    parser.add_argument(
        "--compare_vs_offline",
        action="store_true",
        help="Whether to compare the output of the model with the offline mode.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="The batch size to be used to perform streaming in batch mode with multiple streams",
    )
    parser.add_argument(
        "--chunk_size",
        type=int,
        default=-1,
        help="The chunk_size to be used for models trained with full context and offline models",
    )
    parser.add_argument(
        "--shift_size",
        type=int,
        default=-1,
        help="The shift_size to be used for models trained with full context and offline models",
    )
    parser.add_argument(
        "--left_chunks",
        type=int,
        default=2,
        help="The number of left chunks to be used as left context via caching for offline models",
    )

    parser.add_argument(
        "--online_normalization",
        default=False,
        action='store_true',
        help="Perform normalization on the run per chunk.",
    )
    parser.add_argument(
        "--output_path", type=str, help="path to output file when manifest is used as input", default=None
    )
    parser.add_argument(
        "--pad_and_drop_preencoded",
        action="store_true",
        help="Enables padding the audio input and then dropping the extra steps after the pre-encoding for all the steps including the the first step. It may make the outputs of the downsampling slightly different from offline mode for some techniques like striding or sw_striding.",
    )

    parser.add_argument(
        "--set_decoder",
        choices=["ctc", "rnnt"],
        default=None,
        help="Selects the decoder for Hybrid ASR models which has both the CTC and RNNT decoder. Supported decoders are ['ctc', 'rnnt']",
    )

    parser.add_argument(
        "--att_context_size",
        type=str,
        default=None,
        help="Sets the att_context_size for the models which support multiple lookaheads",
    )

    parser.add_argument(
        "--rttm_left",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--rttm_right",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--override",
        default=None,
        help="Sets the att_context_size for the models which support multiple lookaheads",
    )

    parser.add_argument(
        "--diar_model_path",
        default=None,
    )

    parser.add_argument(
        "--rttm_mix_prob",
        type=int,
        default=-1,
    )

    parser.add_argument(
        "--preserve_alignments",
        default=False,
    )

    parser.add_argument(
        "--pos_emb_max_len",
        default=5000,
        type=int,
    )

    parser.add_argument(
        "--initial_chunk",
        default=105,
        type=int,
    )

    parser.add_argument(
        "--initial_shift",
        default=105,
        type=int,
    )

    parser.add_argument(
        "--mode",
        type=str,
        default="multi",
        help="set either multi or tgt, default multi",
    )

    args = parser.parse_args()

    if (args.audio_file is None and args.manifest_file is None) or (
        args.audio_file is not None and args.manifest_file is not None
    ):
        raise ValueError("One of the audio_file and manifest_file should be non-empty!")

    if args.asr_model.endswith('.nemo'):
        logging.info(f"Using local ASR model from {args.asr_model}")
        # restore model from .nemo file path
        model_cfg = ASRModel.restore_from(restore_path=args.asr_model, return_config=True)
        classpath = model_cfg.target  # original class path
        imported_class = model_utils.import_class_by_path(classpath)  # type: ASRModel
        logging.info(f"Restoring model : {imported_class.__name__}")
        if args.override:
            orig_config = imported_class.restore_from(
                restore_path=args.asr_model,
                return_config=True
            )
            if args.rttm_mix_prob != -1:
                orig_config.rttm_mix_prob = args.rttm_mix_prob
                if args.rttm_mix_prob == 1:
                    orig_config.spk_supervision_strategy = 'rttm'
                elif args.rttm_mix_prob == 0:
                    orig_config.spk_supervision_strategy = 'diar'
            if args.preserve_alignments:
                OmegaConf.update(orig_config.decoding, 'preserve_alignments', True)
            if args.pos_emb_max_len:
                orig_config.encoder.pos_emb_max_len = args.pos_emb_max_len

            orig_config.diar_model_path = args.diar_model_path
            new_config = orig_config
            asr_model = imported_class.restore_from(
                restore_path=args.asr_model, override_config_path=new_config
            )
        else:
            asr_model = nemo_asr.models.ASRModel.restore_from(restore_path=args.asr_model)
    else:
        logging.info(f"Using NGC cloud ASR model {args.asr_model}")
        asr_model = nemo_asr.models.ASRModel.from_pretrained(model_name=args.asr_model)
    logging.info(asr_model.encoder.streaming_cfg)
    if args.set_decoder is not None:
        if hasattr(asr_model, "cur_decoder"):
            asr_model.change_decoding_strategy(decoder_type=args.set_decoder)
        else:
            raise ValueError("Decoder cannot get changed for non-Hybrid ASR models.")

    if args.att_context_size is not None:
        if hasattr(asr_model.encoder, "set_default_att_context_size"):
            asr_model.encoder.set_default_att_context_size(att_context_size=json.loads(args.att_context_size))
        else:
            raise ValueError("Model does not support multiple lookaheads.")

    global autocast
    if (
        args.use_amp
        and torch.cuda.is_available()
        and hasattr(torch.cuda, 'amp')
        and hasattr(torch.cuda.amp, 'autocast')
    ):
        logging.info("AMP enabled!\n")
        autocast = torch.cuda.amp.autocast
    else:

        @contextlib.contextmanager
        def autocast():
            yield

    # configure the decoding config
    decoding_cfg = asr_model.cfg.decoding
    with open_dict(decoding_cfg):
        decoding_cfg.strategy = "greedy"
        # decoding_cfg.preserve_alignments = False
        decoding_cfg.preserve_alignments = asr_model.decoding.cfg.get('preserve_alignments',False)
        if hasattr(asr_model, 'joint'):  # if an RNNT model
            decoding_cfg.greedy.max_symbols = 10
            decoding_cfg.fused_batch_size = -1
        asr_model.change_decoding_strategy(decoding_cfg)

    asr_model = asr_model.to(args.device)
    asr_model.eval()

    # chunk_size is set automatically for models trained for streaming. For models trained for offline mode with full context, we need to pass the chunk_size explicitly.

    if args.chunk_size > 0:
        if args.shift_size < 0:
            shift_size = args.chunk_size
        else:
            shift_size = args.shift_size
        asr_model.encoder.setup_streaming_params(
            chunk_size=args.chunk_size, left_chunks=args.left_chunks, shift_size=shift_size
        )

    # In streaming, offline normalization is not feasible as we don't have access to the whole audio at the beginning
    # When online_normalization is enabled, the normalization of the input features (mel-spectrograms) are done per step
    # It is suggested to train the streaming models without any normalization in the input features.
    if args.online_normalization:
        if asr_model.cfg.preprocessor.normalize not in ["per_feature", "all_feature"]:
            logging.warning(
                "online_normalization is enabled but the model has no normalization in the feature extration part, so it is ignored."
            )
            online_normalization = False
        else:
            online_normalization = True

    else:
        online_normalization = False
    
    if args.audio_file is not None:
        # stream a single audio file
        processed_signal, processed_signal_length, stream_id = streaming_buffer.append_audio_file(
            args.audio_file, stream_id=-1
        )
        perform_streaming(
            asr_model=asr_model,
            streaming_buffer=streaming_buffer,
            compare_vs_offline=args.compare_vs_offline,
            pad_and_drop_preencoded=args.pad_and_drop_preencoded,
        )
    else:
        # stream audio files in a manifest file in batched mode
        samples = []
        all_streaming_tran = []
        all_offline_tran = []
        all_refs_text = []

        with open(args.manifest_file, 'r') as f:
            for line in f:
                item = json.loads(line)
                samples.append(item)

        logging.info(f"Loaded {len(samples)} from the manifest at {args.manifest_file}.")

        start_time = time.time()

        if args.mode == 'multi':

            for sample_idx, sample in enumerate(samples):

                streaming_buffer= CacheAwareStreamingAudioSpkBuffer(
                                                    model=asr_model,
                                                    online_normalization=online_normalization,
                                                    pad_and_drop_preencoded=args.pad_and_drop_preencoded,
                                                )
                processed_signal, processed_signal_length, stream_id = streaming_buffer.append_audio_file(
                    sample['audio_filepath'], stream_id=-1
                )
                if "text" in sample:
                    all_refs_text.append(sample["text"])
                logging.info(f'Added this sample to the buffer: {sample["audio_filepath"]}')
                if (sample_idx + 1) % args.batch_size == 0 or sample_idx == len(samples) - 1:
                    logging.info(f"Starting to stream samples {sample_idx - len(streaming_buffer) + 1} to {sample_idx}...")
                    streaming_tran, offline_tran = perform_streaming(
                        asr_model=asr_model,
                        audio_info=sample,
                        streaming_buffer=streaming_buffer,
                        rttm_left=args.rttm_left,
                        rttm_right=args.rttm_right,
                        compare_vs_offline=args.compare_vs_offline,
                        debug_mode=args.debug_mode,
                        pad_and_drop_preencoded=args. pad_and_drop_preencoded,
                    )
                    all_streaming_tran.extend(streaming_tran)
                    if args.compare_vs_offline:
                        all_offline_tran.extend(offline_tran)
                    streaming_buffer.reset_buffer()

        elif args.mode == 'tgt':
            if args.initial_chunk:
                asr_model.encoder.streaming_cfg.chunk_size[0] = args.initial_chunk
                asr_model.encoder.streaming_cfg.shift_size[0] = args.initial_shift
            for sample_idx, sample in enumerate(samples):
                streaming_buffer= CacheAwareStreamingAudioTgtSpkBuffer(
                                                    model=asr_model,
                                                    online_normalization=online_normalization,
                                                    pad_and_drop_preencoded=args.pad_and_drop_preencoded,
                                                )
                processed_signal, processed_signal_length, stream_id = streaming_buffer.append_audio_file(
                    sample, stream_id=-1
                )
                if "text" in sample:
                    all_refs_text.append(sample["text"])
                logging.info(f'Added this sample to the buffer: {sample["audio_filepath"]}')
                if (sample_idx + 1) % args.batch_size == 0 or sample_idx == len(samples) - 1:
                    logging.info(f"Starting to stream samples {sample_idx - len(streaming_buffer) + 1} to {sample_idx}...")
                    streaming_tran, offline_tran = perform_streaming_tgt(
                        asr_model=asr_model,
                        audio_info=sample,
                        streaming_buffer=streaming_buffer,
                        rttm_left=args.rttm_left,
                        rttm_right=args.rttm_right,
                        compare_vs_offline=args.compare_vs_offline,
                        debug_mode=args.debug_mode,
                        pad_and_drop_preencoded=args. pad_and_drop_preencoded,
                    )
                    all_streaming_tran.extend(streaming_tran)
                    if args.compare_vs_offline:
                        all_offline_tran.extend(offline_tran)
                    streaming_buffer.reset_buffer()



        if args.compare_vs_offline and len(all_refs_text) == len(all_offline_tran):
            offline_wer = word_error_rate(hypotheses=all_offline_tran, references=all_refs_text)
            logging.info(f"WER% of offline mode: {round(offline_wer * 100, 2)}")
        if len(all_refs_text) == len(all_streaming_tran):
            streaming_wer = word_error_rate(hypotheses=all_streaming_tran, references=all_refs_text)
            logging.info(f"WER% of streaming mode: {round(streaming_wer*100, 2)}")

        end_time = time.time()
        logging.info(f"The whole streaming process took: {round(end_time - start_time, 2)}s")

        # stores the results including the transcriptions of the streaming inference in a json file
        if args.output_path is not None and len(all_refs_text) == len(all_streaming_tran):
            # fname = (
            #     "streaming_out_"
            #     + os.path.splitext(os.path.basename(args.asr_model))[0]
            #     + "_"
            #     + os.path.splitext(os.path.basename(args.manifest_file))[0]
            #     + ".json"
            # )

            hyp_json = args.output_path
            # os.makedirs(args.output_path, exist_ok=True)
            with open(hyp_json, "w") as out_f:
                for i, hyp in enumerate(all_streaming_tran):
                    record = {
                    'audio_filepath': samples[i]['audio_filepath'], 
                    'offset':samples[i]['offset'],
                    'duration': samples[i]['duration'],
                    "pred_text": all_streaming_tran[i],
                    "text": all_refs_text[i],
                    # "wer": round(word_error_rate(hypotheses=[hyp], references=[all_refs_text[i]]) * 100, 2),
                    }
                    out_f.write(json.dumps(record) + '\n')


if __name__ == '__main__':
    main()
