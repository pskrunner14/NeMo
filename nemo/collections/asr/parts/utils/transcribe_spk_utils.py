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
import glob
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple, Union

import torch
from omegaconf import DictConfig
from tqdm.auto import tqdm

import nemo.collections.asr as nemo_asr
from nemo.collections.asr.metrics.wer import word_error_rate
from nemo.collections.asr.models import ASRModel, EncDecHybridRNNTCTCModel, EncDecMultiTaskModel
from nemo.collections.asr.parts.utils import manifest_utils, rnnt_utils
from nemo.collections.asr.parts.utils.streaming_tgt_spk_utils import FrameBatchASR_tgt_spk, FeatureFrameBatchASR_tgt_spk
from nemo.collections.common.metrics.punct_er import OccurancePunctuationErrorRate
from nemo.collections.common.parts.preprocessing.manifest import get_full_path
from nemo.utils import logging, model_utils

from omegaconf import open_dict, OmegaConf

from nemo.collections.asr.parts.utils.asr_multispeaker_utils import apply_spk_mapping

from nemo.collections.asr.parts.utils.transcribe_utils import wrap_transcription

def get_buffered_pred_feat_tgt_spk(
    asr: Union[FrameBatchASR_tgt_spk, FeatureFrameBatchASR_tgt_spk],
    frame_len: float,
    tokens_per_chunk: int,
    delay: int,
    preprocessor_cfg: DictConfig,
    model_stride_in_secs: int,
    device: Union[List[int], int],
    manifest: str = None,
    filepaths: List[list] = None,
) -> List[rnnt_utils.Hypothesis]:
    """
    Moved from examples/asr/asr_chunked_inference/ctc/speech_to_text_buffered_infer_ctc.py
    Write all information presented in input manifest to output manifest and removed WER calculation.
    """
    # Create a preprocessor to convert audio samples into raw features,
    # Normalization will be done per buffer in frame_bufferer
    # Do not normalize whatever the model's preprocessor setting is
    preprocessor_cfg.normalize = "None"
    preprocessor = nemo_asr.models.EncDecCTCModelBPE.from_config_dict(preprocessor_cfg)
    preprocessor.to(device)
    hyps = []
    refs = []

    if filepaths and manifest:
        raise ValueError("Please select either filepaths or manifest")
    if filepaths is None and manifest is None:
        raise ValueError("Either filepaths or manifest shoud not be None")
    if filepaths:
        for l in tqdm(filepaths, desc="Sample:"):
            asr.reset()
            asr.read_audio_file(l, delay, model_stride_in_secs)
            hyp = asr.transcribe(tokens_per_chunk, delay)
            hyps.append(hyp)
    else:
        with open(manifest, "r", encoding='utf_8') as mfst_f:
            for l in tqdm(mfst_f, desc="Sample:"):
                asr.reset()
                row = json.loads(l.strip())
                if 'text' in row:
                    refs.append(row['text'])
                audio_file = get_full_path(audio_file=row['audio_filepath'], manifest_file=manifest)
                offset = row['offset']
                duration = row['duration']
                #query info
                query_audio_file = row['query_audio_filepath']
                query_offset = row['query_offset']
                query_duration = row['query_duration']
                #separater info
                separater_freq = asr.asr_model.cfg.test_ds.separater_freq
                separater_duration = asr.asr_model.cfg.test_ds.separater_duration
                separater_unvoice_ratio = asr.asr_model.cfg.test_ds.separater_unvoice_ratio
                # do not support partial audio
                asr.read_audio_file(audio_file, offset, duration, query_audio_file, query_offset, query_duration, separater_freq, separater_duration, separater_unvoice_ratio, delay, model_stride_in_secs)
                hyp = asr.transcribe(tokens_per_chunk, delay)
                hyps.append(hyp)

    if os.environ.get('DEBUG', '0') in ('1', 'y', 't'):
        if len(refs) == 0:
            print("ground-truth text does not present!")
            for hyp in hyps:
                print("hyp:", hyp)
        else:
            for hyp, ref in zip(hyps, refs):
                print("hyp:", hyp)
                print("ref:", ref)

    wrapped_hyps = wrap_transcription(hyps)
    return wrapped_hyps

def setup_model(cfg: DictConfig, map_location: torch.device) -> Tuple[ASRModel, str]:
    """ Setup model from cfg and return model and model name for next step """
    if cfg.model_path is not None and cfg.model_path != "None":
        # restore model from .nemo file path
        model_cfg = ASRModel.restore_from(restore_path=cfg.model_path, return_config=True)
        classpath = model_cfg.target  # original class path
        imported_class = model_utils.import_class_by_path(classpath)  # type: ASRModel
        logging.info(f"Restoring model : {imported_class.__name__}")
        if cfg.override:
            orig_config = imported_class.restore_from(
                restore_path=cfg.model_path, map_location=map_location,
                return_config=True
            )
            if 'rttm_mix_prob' in cfg:
                orig_config.rttm_mix_prob = cfg.rttm_mix_prob
                if cfg.rttm_mix_prob == 1:
                    orig_config.spk_supervision_strategy = 'rttm'
                elif cfg.rttm_mix_prob == 0:
                    orig_config.spk_supervision_strategy = 'diar'
            if 'test_shuffle_spk_mapping' in cfg:
                orig_config.test_ds.shuffle_spk_mapping = cfg.test_shuffle_spk_mapping
                orig_config.shuffle_spk_mapping = cfg.test_shuffle_spk_mapping
            if "<|beep|>" not in orig_config.labels:
                OmegaConf.update(orig_config.test_ds,'special_token', '<|spltoken0|>')
            if 'inference_mode' in cfg:
                orig_config.test_ds.inference_mode = cfg.inference_mode
                if cfg.inference_mode:
                    orig_config.spk_supervision_strategy = 'diar'
            if 'binarize_diar_preds_threshold' in cfg:
                if cfg.binarize_diar_preds_threshold:
                    if cfg.binarize_diar_preds_threshold == -1:
                        #deactivate binarization
                        orig_config.binarize_diar_preds_threshold = False
                    else:
                        #override binarize_diar_preds_threshold
                        orig_config.binarize_diar_preds_threshold = cfg.binarize_diar_preds_threshold
            if 'diar_model_path' in cfg:
                orig_config.diar_model_path = cfg.diar_model_path
            if 'diar_model_cfg_path' in cfg:
                orig_config.diar_model_cfg = OmegaConf.load(cfg.diar_model_cfg_path)
            if 'disable_preprocessor_norm' in cfg:
                if cfg.disable_preprocessor_norm:
                    orig_config.preprocessor.normalize = 'NA'
            new_config = orig_config
            #set strict to False if model is trained with old diarization model, otherwise set to True
            asr_model = imported_class.restore_from(
                restore_path=cfg.model_path, strict = True, map_location=map_location, override_config_path=new_config
            )
            asr_model.diarization_model.to(asr_model.device)
        else:
            asr_model = imported_class.restore_from(
                restore_path=cfg.model_path,map_location=map_location,
            )

        model_name = os.path.splitext(os.path.basename(cfg.model_path))[0]
    else:
        # restore model by name
        asr_model = ASRModel.from_pretrained(
            model_name=cfg.pretrained_name, map_location=map_location,
        )  # type: ASRModel
        model_name = cfg.pretrained_name

    if hasattr(cfg, "model_change") and hasattr(asr_model, "change_attention_model"):
        asr_model.change_attention_model(
            self_attention_model=cfg.model_change.conformer.get("self_attention_model", None),
            att_context_size=cfg.model_change.conformer.get("att_context_size", None),
        )

    return asr_model, model_name


def transcribe_partial_audio(
        
    asr_model, 
    path2manifest: str = None,
    batch_size: int = 4,
    logprobs: bool = False,
    return_hypotheses: bool = False,
    num_workers: int = 0,
    channel_selector: Optional[int] = None,
    augmentor: DictConfig = None,
    decoder_type: Optional[str] = None,
    cfg: DictConfig = None,
) -> List[str]:
    """
    See description of this function in trancribe() in nemo/collections/asr/models/ctc_models.py and nemo/collections/asr/models/rnnt_models.py
    """

    if return_hypotheses and logprobs:
        raise ValueError(
            "Either `return_hypotheses` or `logprobs` can be True at any given time."
            "Returned hypotheses will contain the logprobs."
        )
    if num_workers is None:
        num_workers = min(batch_size, os.cpu_count() - 1)

    # We will store transcriptions here
    hypotheses = []
    # store spk mapping here
    spk_mappings = []
    # Model's mode and device
    mode = asr_model.training
    device = next(asr_model.parameters()).device
    dither_value = asr_model.preprocessor.featurizer.dither
    pad_to_value = asr_model.preprocessor.featurizer.pad_to
    if decoder_type is not None:  # Hybrid model
        decode_function = (
            asr_model.decoding.rnnt_decoder_predictions_tensor
            if decoder_type == 'rnnt'
            else asr_model.ctc_decoding.ctc_decoder_predictions_tensor
        )
    elif hasattr(asr_model, 'joint'):  # RNNT model
        decode_function = asr_model.decoding.rnnt_decoder_predictions_tensor
    else:  # CTC model
        decode_function = asr_model.decoding.ctc_decoder_predictions_tensor

    try:
        asr_model.preprocessor.featurizer.dither = 0.0
        asr_model.preprocessor.featurizer.pad_to = 0
        # Switch model to evaluation mode
        asr_model.eval()
        # Freeze the encoder and decoder modules
        asr_model.encoder.freeze()
        asr_model.decoder.freeze()
        logging_level = logging.get_verbosity()
        logging.set_verbosity(logging.WARNING)

        config = {
            'manifest_filepath': path2manifest,
            'batch_size': batch_size,
            'num_workers': num_workers,
            'channel_selector': channel_selector,
        }
        if augmentor:
            config['augmentor'] = augmentor

        temporary_datalayer = asr_model._setup_transcribe_dataloader(config)
        for test_batch in tqdm(temporary_datalayer, desc="Transcribing"):
            logits, logits_len, transcript, transcript_len = asr_model.train_val_forward(
                [x.to(device) for x in test_batch], 0
            )

            if isinstance(asr_model, EncDecHybridRNNTCTCModel) and decoder_type == "ctc":
                logits = asr_model.ctc_decoder(encoder_output=logits)

            if logprobs:
                logits = logits.numpy()
                # dump log probs per file
                for idx in range(logits.shape[0]):
                    lg = logits[idx][: logits_len[idx]]
                    hypotheses.append(lg)
            else:
                current_hypotheses, _ = decode_function(logits, logits_len, return_hypotheses=return_hypotheses,)

                if return_hypotheses:
                    # dump log probs per file
                    for idx in range(logits.shape[0]):
                        current_hypotheses[idx].y_sequence = logits[idx][: logits_len[idx]]
                        if current_hypotheses[idx].alignments is None:
                            current_hypotheses[idx].alignments = current_hypotheses[idx].y_sequence

                hypotheses += current_hypotheses

                spk_mappings += test_batch[5]
                

            del logits
            del test_batch

    finally:
        # set mode back to its original value
        asr_model.train(mode=mode)
        asr_model.preprocessor.featurizer.dither = dither_value
        asr_model.preprocessor.featurizer.pad_to = pad_to_value
        if mode is True:
            asr_model.encoder.unfreeze()
            asr_model.decoder.unfreeze()
        logging.set_verbosity(logging_level)

    return hypotheses, spk_mappings


def write_transcription(
    transcriptions: Union[List[rnnt_utils.Hypothesis], List[List[rnnt_utils.Hypothesis]], List[str]],
    spk_mappings,
    cfg: DictConfig,
    model_name: str,
    filepaths: List[str] = None,
    compute_langs: bool = False,
    compute_timestamps: bool = False,
) -> Tuple[str, str]:
    """ Write generated transcription to output file. """
    if cfg.append_pred:
        logging.info(f'Transcripts will be written in "{cfg.output_filename}" file')
        if cfg.pred_name_postfix is not None:
            pred_by_model_name = cfg.pred_name_postfix
        else:
            pred_by_model_name = model_name
        pred_text_attr_name = 'pred_text_' + pred_by_model_name
    else:
        pred_text_attr_name = 'pred_text'

    return_hypotheses = True
    if isinstance(transcriptions[0], str):  # List[str]:
        best_hyps = transcriptions
        return_hypotheses = False
    elif isinstance(transcriptions[0], rnnt_utils.Hypothesis):  # List[rnnt_utils.Hypothesis]
        best_hyps = transcriptions
        assert cfg.decoding.beam.return_best_hypothesis, "Works only with return_best_hypothesis=true"
    elif isinstance(transcriptions[0], list) and isinstance(
        transcriptions[0][0], rnnt_utils.Hypothesis
    ):  # List[List[rnnt_utils.Hypothesis]] NBestHypothesis
        best_hyps, beams = [], []
        for hyps in transcriptions:
            best_hyps.append(hyps[0])
            if not cfg.decoding.beam.return_best_hypothesis:
                beam = []
                for hyp in hyps:
                    score = hyp.score.numpy().item() if isinstance(hyp.score, torch.Tensor) else hyp.score
                    beam.append((hyp.text, score))
                beams.append(beam)
    else:
        raise TypeError

    # create output dir if not exists
    Path(cfg.output_filename).parent.mkdir(parents=True, exist_ok=True)
    with open(cfg.output_filename, 'w', encoding='utf-8', newline='\n') as f:
        if cfg.audio_dir is not None:
            for idx, transcription in enumerate(best_hyps):  # type: rnnt_utils.Hypothesis or str
                if not return_hypotheses:  # transcription is str
                    item = {'audio_filepath': filepaths[idx], pred_text_attr_name: transcription}
                else:  # transcription is Hypothesis
                    item = {'audio_filepath': filepaths[idx], pred_text_attr_name: transcription.text}

                    if compute_timestamps:
                        timestamps = transcription.timestep
                        if timestamps is not None and isinstance(timestamps, dict):
                            timestamps.pop(
                                'timestep', None
                            )  # Pytorch tensor calculating index of each token, not needed.
                            for key in timestamps.keys():
                                values = normalize_timestamp_output(timestamps[key])
                                item[f'timestamps_{key}'] = values

                    if compute_langs:
                        item['pred_lang'] = transcription.langs
                        item['pred_lang_chars'] = transcription.langs_chars
                    if not cfg.decoding.beam.return_best_hypothesis:
                        item['beams'] = beams[idx]
                f.write(json.dumps(item) + "\n")
        else:
            with open(cfg.dataset_manifest, 'r', encoding='utf-8') as fr:
                for idx, line in enumerate(fr):
                    item = json.loads(line)
                    if not return_hypotheses:  # transcription is str
                        item[pred_text_attr_name] = best_hyps[idx]
                    else:  # transcription is Hypothesis
                        item[pred_text_attr_name] = best_hyps[idx].text

                        if compute_timestamps:
                            timestamps = best_hyps[idx].timestep
                            if timestamps is not None and isinstance(timestamps, dict):
                                timestamps.pop(
                                    'timestep', None
                                )  # Pytorch tensor calculating index of each token, not needed.
                                for key in timestamps.keys():
                                    values = normalize_timestamp_output(timestamps[key])
                                    item[f'timestamps_{key}'] = values

                        if compute_langs:
                            item['pred_lang'] = best_hyps[idx].langs
                            item['pred_lang_chars'] = best_hyps[idx].langs_chars

                        if not cfg.decoding.beam.return_best_hypothesis:
                            item['beams'] = beams[idx]
                        item['spk_mapping'] = str(spk_mappings[idx].numpy())
                    f.write(json.dumps(item) + "\n")

    return cfg.output_filename, pred_text_attr_name