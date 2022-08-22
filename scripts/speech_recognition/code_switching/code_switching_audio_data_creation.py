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

import argparse
import json
import logging
import os
from tqdm import tqdm
import librosa
from scipy.io import wavfile
import numpy as np

# Checks - 
# Audio data is 16 KHz, if not please use ffmpeg to convert them to 16 KHz


parser = argparse.ArgumentParser(description='Create synthetic code-switching data audio data from monolingual data')
parser.add_argument("--manifest_path", default=None, type=str, help='Path to CS indermediate manifest')
parser.add_argument("--audio_save_folder_path", default=None, type=str, help='Path to directory where created synthetic set would be saved')
parser.add_argument("--manifest_save_path", default=None, type=str, help='Path to save the created manifest')
parser.add_argument("--audio_normalized_amplitude", default=15000, type=int, help='Normalized amplitdue of audio samples')
parser.add_argument("--sample_beginning_pause_msec", default=20, type=int, help='Pause to be added at the beginning of the sample (msec)')
parser.add_argument("--sample_joining_pause_msec", default=100, type=int, help='Pause to be added between different phrases of the sample (msec)')
parser.add_argument("--sample_end_pause_msec", default=20, type=int, help='Pause to be added at the end of the sample (msec)')

args = parser.parse_args()

def __read_manifest(
    manifest_path: str
):
    """
    Args:
        manifest_path: absolute path of the manifest file

    Returns:
        List with manifest entires as elements

    """
    data = []

    for line in open(manifest_path, 'r'):
        data.append(json.loads(line))

    return data

def __create_cs_data(
    intermediate_cs_manifest_list: list,
    audio_save_folder: str,
    manfest_save_path: str,
    audio_amplitude_normalization: int,
    pause_beg_msec: int,
    pause_join_msec: int,
    pause_end_msec: int

):

    """
    Args:
        intermediate_cs_manifest_list: the intermediate cs manifest obtained from code_switching_manifest_creation.py as a list
        audio_save_folder: Absolute path to save the generated audio samples
        manfest_save_path: Absolute path to save the corresponding manifest
        audio_amplitude_normalization: The amplitude to scale to after normalization
        pause_beg_msec: Pause to be added at the beginning of the sample (msec)
        pause_join_msec: Pause to be added between different phrases of the sample (msec)
        pause_end_msec: Pause to be added at the end of the sample (msec)

    Returns:

    """

    sample_id = 0
    fs = 16000
    incorrect_sample_flag = 0

    with open(manfest_save_path, 'w') as outfile:
        for data in tqdm(intermediate_cs_manifest_list):

            combined_audio = []

            staring_pause = np.zeros(int(pause_beg_msec*fs/1000))
            combined_audio += list(staring_pause)

            for index in range(len(data['lang_ids'])):

                data_sample, fs_sample = librosa.load(data['paths'][index], sr=fs)
                #Alternative-  fs_sample, data_sample = wavfile.read(data['paths'][index])

                if(fs_sample != 16000):
                    print('!!ERROR!!!')

                #Remove leading and trailing zeros
                data_sample = np.trim_zeros(data_sample)

                #take care of empty arrays: rare
                if (data_sample.size == 0):
                    incorrect_sample_flag = 1
                    continue

                #normalizing data
                data_sample_norm = data_sample / np.maximum(np.abs(data_sample.max()), np.abs(data_sample.min())) * audio_amplitude_normalization

                combined_audio += list(data_sample_norm)

                #adding small pause between gemgments
                if(index != (len(data['lang_ids']) - 1)):
                    pause = np.zeros(int(pause_join_msec*fs/1000))
                    combined_audio += list(pause)

            if (incorrect_sample_flag == 1):
                incorrect_sample_flag = 0
                continue

            ending_pause = np.zeros(int(pause_end_msec*fs/1000))
            combined_audio += list(ending_pause)

            audio_file_path = audio_save_folder + '/' + str(sample_id) + ".wav"

            # saving audio
            wavfile.write(audio_file_path, fs, np.array(combined_audio).astype(np.int16))
            #Alternative-  librosa.output.write_wav(audio_file_path, combined_audio, fs)
            sample_id += 1

            metadata_json = {}
            metadata_json['audio_filepath'] = audio_file_path
            metadata_json['duration'] = float(len(combined_audio)/fs)
            metadata_json['text'] = ' '.join(data['texts'])

            metadata_json['language_ids'] = data['lang_ids']
            metadata_json['original_texts'] = data['texts']
            metadata_json['original_paths'] = data['paths']
            metadata_json['original_durations'] = data['durations']

            s = json.dumps(metadata_json)
            outfile.write(s + '\n')

def main():

    cs_intermediate_manifest_path = args.manifest_path
    audio_save_folder = args.audio_save_folder_path
    manifest_save_path = args.manifest_save_path
    audio_amplitude_normalization = args.audio_normalized_amplitude
    pause_beg_msec = args.sample_beginning_pause_msec
    pause_join_msec = args.sample_joining_pause_msec
    pause_end_msec = args.sample_end_pause_msec

    # Reading data
    logging.info('Reading manifests')
    intermediate_cs_manifest = __read_manifest(cs_intermediate_manifest_path)

    # Creating Audio data
    logging.info('Creating synthetic audio data')
    __create_cs_data(intermediate_cs_manifest, audio_save_folder, manifest_save_path, audio_amplitude_normalization, pause_beg_msec, pause_join_msec, pause_end_msec)

    print("Synthetic CS audio data saved at :", audio_save_folder)
    print("Synthetic CS manifest saved at :", manifest_save_path)

    logging.info('Done!')


if __name__ == "__main__":
    main()


