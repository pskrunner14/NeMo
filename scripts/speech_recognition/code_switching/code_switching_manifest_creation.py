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
import random

# Checks -
# Please normalize the text for each language so that we have only the alphabets (no numbers, special characters, punctuation)
# Please ensure that the audio_fielpaths are absolute locations


parser = argparse.ArgumentParser(description='Create synthetic code-switching data manifest from monolingual data')

parser.add_argument("--manifest_language1", default=None, type=str, help='Manifest file for language 1', required=True)
parser.add_argument("--manifest_language2", default=None, type=str, help='Manifest file for language 2', required=True)
parser.add_argument(
    "--manifest_save_path", default=None, type=str, help='Path to save created CS indermediate manifest', required=True
)
parser.add_argument(
    "--id_language1", default=None, type=str, help='Identifier for language 1, eg: en, es, hi', required=True
)
parser.add_argument(
    "--id_language2", default=None, type=str, help='Identifier for language 2, eg: en, es, hi', required=True
)
parser.add_argument("--max_sample_duration_sec", default=19, type=int, help='Maximum duration of sample (sec)')
parser.add_argument("--min_sample_duration_sec", default=16, type=int, help='Minimum duration of sample (sec)')
parser.add_argument("--dataset_size_required_hrs", default=10, type=int, help='Duration of dataset required (hrs)')

args = parser.parse_args()


def __read_manifest(manifest_path: str):
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


def __write_manifest(manifest_path: str, data: list):
    """
    Args:
        manifest_path: absolute path for where to save the manifest file
        data: list consisting of entries for the manifest

    Returns:

    """

    with open(manifest_path, 'w') as outfile:
        for elem in data:
            s = json.dumps(elem)
            outfile.write(s + '\n')


def __create_cs_manifest(
    data_lang_0: list,
    data_lang_1: list,
    lid_lang_0: str,
    lid_lang_1: str,
    max_sample_duration_sec: int,
    min_sample_duration_sec: int,
    data_requirement_hrs: int,
):
    """
    Args:
        data_lang_0: Manifest entries from first langauge
        data_lang_1: Manifest entries from second langauge
        lid_lang_0: Language ID marker for first langauge
        lid_lang_1: Language ID marker for second langauge
        max_sample_duration_sec: Maximum permissible duration of generated CS sample in sec
        min_sample_duration_sec: Minimum permissible duration of generated CS sample in sec
        data_requirement_hrs: Required size of generated corpus

    Returns:
        Created synthetic CS manifest as list

    """

    total_duration = 0
    constructed_data = []

    num_samples_lang0 = len(data_lang_0)
    num_samples_lang1 = len(data_lang_1)

    while total_duration < (data_requirement_hrs * 3600):

        created_sample_duration_sec = 0
        created_sample_dict = {}
        created_sample_dict['lang_ids'] = []
        created_sample_dict['texts'] = []
        created_sample_dict['paths'] = []
        created_sample_dict['durations'] = []

        while created_sample_duration_sec < min_sample_duration_sec:

            lang_selection = random.randint(0, 1)

            if lang_selection == 0:
                index = random.randint(0, num_samples_lang0 - 1)
                sample = data_lang_0[index]
                lang_id = lid_lang_0
            else:
                index = random.randint(0, num_samples_lang1 - 1)
                sample = data_lang_1[index]
                lang_id = lid_lang_1

            if (created_sample_duration_sec + sample['duration']) > max_sample_duration_sec:
                continue
            else:
                created_sample_duration_sec += sample['duration']
                created_sample_dict['lang_ids'].append(lang_id)
                created_sample_dict['texts'].append(sample['text'])
                created_sample_dict['paths'].append(sample['audio_filepath'])
                created_sample_dict['durations'].append(sample['duration'])

        created_sample_dict['total_duration'] = created_sample_duration_sec
        constructed_data.append(created_sample_dict)
        total_duration += created_sample_duration_sec

    return constructed_data


def main():

    manifest0 = args.manifest_language1
    manifest1 = args.manifest_language2
    lid0 = args.id_language1
    lid1 = args.id_language2
    min_sample_duration = args.min_sample_duration_sec
    max_sample_duration = args.max_sample_duration_sec
    dataset_requirement = args.dataset_size_required_hrs
    manifest_save_path = args.manifest_save_path

    # Reading data
    logging.info('Reading manifests')
    data_language0 = __read_manifest(manifest0)
    data_language1 = __read_manifest(manifest1)

    # Creating the CS data Manifest
    logging.info('Creating CS manifest')
    constructed_data = __create_cs_manifest(
        data_language0, data_language1, lid0, lid1, max_sample_duration, min_sample_duration, dataset_requirement
    )

    # Saving Manifest
    logging.info('saving manifest')
    __write_manifest(manifest_save_path, constructed_data)

    print("Synthetic CS manifest saved at :", manifest_save_path)

    logging.info('Done!')


if __name__ == "__main__":
    main()
