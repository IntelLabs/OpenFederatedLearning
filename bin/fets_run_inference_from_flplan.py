#!/usr/bin/env python3

# Copyright (C) 2020 Intel Corporation
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
import sys
import os
import logging

from openfl.flplan import create_data_object_with_explicit_data_path, parse_fl_plan, create_model_object
from setup_logging import setup_logging




def main(plan, fetsai_model_weights_filename, fetsai_native_model_weights_filepath, data_dir, logging_config_path, logging_default_level, logging_directory, model_device):
    """Runs the inference according to the flplan, data-dir and weights file. The assumption here is that the model instantiation
    includes population of model weights (thus a model weights file must be indicated)
    The inference output format is determined by the data object in the flplan

    Args:
        plan (string)                           : The filename for the federation (FL) plan YAML file
        fetsai_model_weights_filename (string)  : A .pbuf filename in the common weights directory (mutually exclusive with native_model_weights_filepath). NOTE: these must be uncompressed weights!!
        fetsai_native_model_weights_filepath    : A framework-specific filepath. Path will be relative to the working directory. (mutually exclusive with model_weights_filename)
        data_dir (string)                       : The directory path for the parent directory containing the data. Path will be relative to the working directory.
        logging_config_fname (string)           : The log file
        logging_default_level (string)          : The log level

    """

    if fetsai_model_weights_filename is not None and fetsai_native_model_weights_filepath is not None:
        sys.exit("Parameters fetsai_model_weights_filename and fetsai_native_model_weights_filepath are mutually exclusive.\nmodel_weights_file was set to {}\native_model_weights_filepath was set to {}".format(fetsai_model_weights_filename, fetsai_native_model_weights_filepath))

    # FIXME: consistent filesystem (#15)
    script_dir = os.path.dirname(os.path.realpath(__file__))
    base_dir = os.path.join(script_dir, 'federations')
    plan_dir = os.path.join(base_dir, 'plans')
    logging_directory = os.path.join(script_dir, logging_directory)

    setup_logging(path=logging_config_path, default_level=logging_default_level, logging_directory=logging_directory)

    flplan = parse_fl_plan(os.path.join(plan_dir, plan))

    # providing guidence on the arguments
    if 'inference' not in flplan:
        sys.exit("FL Plan does not contain a top-level 'inference' entry. By default, inference is disabled.")
    
    if 'allowed' not in flplan['inference'] or flplan['inference']['allowed'] != True:        
        sys.exit("FL Plan must contain a {'inference: {'allowed': True}} entry in order for inference to be allowed.")

    if fets-ai_model_weights_filename is not None and fets-ai_native_model_weights_filepath is not None:
        sys.exit('Arguments: fets-ai_model_weights_filename (provided as {}) and fets-ai_native_model_weights_filepath (provided as {}) are mutually exclusive'.format(fets-ai_model_weights_filename, fets-ai_native_model_weights_filepath))

    # This script expects the model to populate its weights upon initialization.
    # Supplementing the flplan model init kwargs to include model weights file info.
    if fets-ai_model_weights_filename is not None:
        flplan['model_object_init']['init_kwargs'].update({'model_weights_filename': fetsai_model_weights_filename})
    elif fets-ai_native_model_weights_filepath is not None:
        flplan['model_object_init']['init_kwargs'].update({'native_model_weights_filepath': fets-ai_native_model_weights_filepath})
    else:
        sys.exit('One of the arguments: fets-ai_model_weights_filename or fets-ai_native_model_weights_filepath must be provided.')

    # create the data object
    data = create_data_object_with_explicit_data_path(flplan=flplan, data_path=data_dir)

    # create the model object
    model = create_model_object(flplan, data, model_device=model_device)

    # finally, call the model object's run_inference_and_store_results with the kwargs from the inference block
    inference_kwargs = flplan['inference'].get('kwargs') or {}
    model.run_inference_and_store_results(**inference_kwargs)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--plan', '-p', type=str, required=True)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--model_weights_filename', '-mwf', type=str, default=None)
    group.add_argument('--native_model_weights_filepath', '-nmwf', type=str, default=None)
    # FIXME: data_dir should be data_path
    parser.add_argument('--data_dir', '-d', type=str, default=None, required=True)
    parser.add_argument('--logging_config_path', '-lc', type=str, default="logging.yaml")
    parser.add_argument('--logging_default_level', '-l', type=str, default="info")
    parser.add_argument('--logging_directory', '-ld', type=str, default="logs")
    # FIXME: this kind of commandline configuration needs to be done in a consistent way
    parser.add_argument('--model_device', '-md', type=str, default='cpu')
    args = parser.parse_args()
    main(**vars(args))