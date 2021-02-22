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

import time
import os
import logging
import time
import hashlib

import numpy as np
import yaml
from threading import Lock

from .. import check_equal, check_not_equal, check_is_in, check_not_in, load_yaml
from ..proto.collaborator_aggregator_interface_pb2 import MessageHeader
from ..proto.collaborator_aggregator_interface_pb2 import Job, JobRequest, JobReply
from ..proto.collaborator_aggregator_interface_pb2 import JOB_DOWNLOAD_MODEL, JOB_QUIT, JOB_TRAIN, JOB_VALIDATE, JOB_YIELD
from ..proto.collaborator_aggregator_interface_pb2 import ModelProto, ModelHeader, TensorProto
from ..proto.collaborator_aggregator_interface_pb2 import ModelDownloadRequest, GlobalModelUpdate
from ..proto.collaborator_aggregator_interface_pb2 import LocalModelUpdate, LocalValidationResults, LocalModelUpdateAck, LocalValidationResultsAck


from openfl.proto.protoutils import dump_proto, load_proto, construct_proto, deconstruct_proto
from openfl.tensor_transformation_pipelines import NoCompressionPipeline

# FIXME: simpler "stats" collection/handling
# FIXME: remove the round tracking/job-result tracking stuff from this?
# Feels like it conflates model aggregation with round management
# FIXME: persistence of the trained weights.
class Aggregator(object):
    """An Aggregator is the central node in federated learning.

    Args:
        aggregator_uuid (string)                : UUID of this object.
        federation_uuid (string)                : Federation UUID.
        collaborator_common_names (list of str) : The list of approved collaborator IDs. These IDs should match the common_names in the collaborator certificates, unless "single_col_cert_common_name" is specified.
        init_model_fpath (string)               : The filepath of the initial weights file.
        latest_model_fpath (string)             : The filepath to store the latest aggregated weights
        best_model_fpath (string)               : The filepath to store the weight of the best model.
        rounds_to_train (int)                   : Number of rounds to train (default: 256)
        minimum_reporting (int)                 : Aggregator will not end a round early until this number of clients has reported in. (default: -1)
        straggler_cutoff_time (scalar)          : Aggregator will not end a round early until this number of seconds has passed. (default: np.inf)
        single_col_cert_common_name (string)    : (default: None)
        compression_pipeline                    : (default: None)
        end_of_round_metadata (list of str)     : (default: None)
        init_metadata_fname (string)            : (default: None)
        latest_metadata_fname (string)          : (default: None)
        send_metadata_to_clients (bool)         : (default: False)
        kwargs                                  : Currently unused
    """
    # FIXME: no selector logic is in place
    def __init__(self,
                 aggregator_uuid,
                 federation_uuid,
                 collaborator_common_names,
                 init_model_fpath,
                 latest_model_fpath,
                 best_model_fpath,
                 rounds_to_train=256,
                 minimum_reporting=-1,
                 straggler_cutoff_time=np.inf,
                 single_col_cert_common_name=None,
                 compression_pipeline=None,
                 end_of_round_metadata=None,
                 init_metadata_fname=None,
                 latest_metadata_fname=None,
                 send_metadata_to_clients=False,
                 save_all_models_path=None,
                 runtime_aggregator_config_dir=None,
                 runtime_configurable_params=None,
                 num_models=1,
                 **kwargs):
        self.logger = logging.getLogger(__name__)
        self.uuid = aggregator_uuid
        self.federation_uuid = federation_uuid

        self.latest_model_fpath = latest_model_fpath
        self.best_model_fpath = best_model_fpath
        self.collaborator_common_names = collaborator_common_names
        self.rounds_to_train = rounds_to_train
        self.quit_job_sent_to = []
        self.minimum_reporting = minimum_reporting
        self.straggler_cutoff_time = straggler_cutoff_time
        self.round_start_time = None
        self.single_col_cert_common_name = single_col_cert_common_name

        self.save_all_models_path = save_all_models_path
        self.runtime_aggregator_config_dir = runtime_aggregator_config_dir
        self.runtime_configurable_params = runtime_configurable_params

        if self.runtime_aggregator_config_dir is not None:
            self.update_config_from_filesystem()

        if self.single_col_cert_common_name is not None:
            self.log_big_warning()
        else:
            self.single_col_cert_common_name = '' # FIXME: '' instead of None is just for protobuf compatibility. Cleaner solution?

        self.num_models = num_models
        self.models = []
        for _ in range(self.num_models):
            self.models.append(load_proto(init_model_fpath))
        self.model = self.models[0]
        self.logger.info("Loaded initial model from {}".format(init_model_fpath))
        self.logger.info("Initial model version is {}".format(self.model.header.version))

        self.model_update_in_progress = [None for _ in range(self.num_models)]

        self.round_num = self.model.header.version + 1

        self.init_per_col_round_stats()
        self.best_model_score = None
        self.aggregated_model_is_global_best = True
        self.mutex = Lock()

        self.compression_pipeline = compression_pipeline or NoCompressionPipeline()

        self.end_of_round_metadata      = end_of_round_metadata
        self.init_metadata_fname        = init_metadata_fname
        self.latest_metadata_fname      = latest_metadata_fname
        self.send_metadata_to_clients   = send_metadata_to_clients

        if self.init_metadata_fname is not None:
            self.metadata = load_yaml(init_metadata_fname)
        else:
            self.metadata = {}
        self.metadata['aggregator_uuid'] = aggregator_uuid
        self.metadata['federation_uuid'] = federation_uuid
        self.metadata_for_round = {}

    def update_config_from_filesystem(self):
        if self.runtime_aggregator_config_dir is None:
            return

        if self.runtime_configurable_params is None:
            return

        # make the directory for convenience
        config_dir = os.path.join(self.runtime_aggregator_config_dir, self.federation_uuid)
        os.makedirs(config_dir, exist_ok=True)

        config_file = os.path.join(config_dir, 'agg_control.yaml')

        if os.path.exists(config_file):
            config = load_yaml(config_file)
            self.logger.info("Updating aggregator config from {}".format(config_file))
        else:
            self.logger.info("Aggregator did not find config file: {}".format(config_file))
            return

        for k, v in config.items():
            if k not in self.runtime_configurable_params:
                self.logger.warning("Aggregator config file contains {}. This is not allowed by the flplan.".format(k))
            elif not hasattr(self, k):
                self.logger.warning("Aggregator config file contains {}. This is not a valid aggregator parameter".format(k))
            else:
                setattr(self, k, v)
                self.logger.info("Aggregator config {} updated to {}".format(k, v))

    def ensure_save_all_path_exists(self):
        if self.save_all_models_path is None:
            return

        dir_path = os.path.join(self.save_all_models_path, str(self.federation_uuid), str(self.round_num))
        os.makedirs(dir_path, exist_ok=True)
        return dir_path

    def save_aggregated_model(self):
        if self.save_all_models_path is None:
            return

        dir_path = self.ensure_save_all_path_exists()

        dump_proto(self.model, os.path.join(dir_path, "aggregated.pbuf"))

    def save_local_update(self, collaborator, update):
        if self.save_all_models_path is None:
            return

        # FIXME: better user experience would be good
        # hash the collaborator name so we can ensure directory names are legal
        md5 = hashlib.md5()
        md5.update(collaborator.encode())
        hashed_col = md5.hexdigest()[:8]

        dir_path = self.ensure_save_all_path_exists()

        dump_proto(update, os.path.join(dir_path, "{}.pbuf".format(hashed_col)))

    def valid_collaborator_CN_and_id(self, cert_common_name, collaborator_common_name):
        """Determine if the collaborator certificate and ID are valid for this federation.

        Args:
            cert_common_name: Common name for security certificate
            collaborator_common_name: Common name for collaborator

        Returns:
            bool: True means the collaborator common name matches the name in the security certificate.

        """
        # if self.test_mode_whitelist is None, then the common_name must match collaborator_common_name and be in collaborator_common_names
        if self.single_col_cert_common_name == '':  # FIXME: '' instead of None is just for protobuf compatibility. Cleaner solution?
            return cert_common_name == collaborator_common_name and collaborator_common_name in self.collaborator_common_names
        # otherwise, common_name must be in whitelist and collaborator_common_name must be in collaborator_common_names
        else:
            return cert_common_name == self.single_col_cert_common_name and collaborator_common_name in self.collaborator_common_names

    def all_quit_jobs_sent(self):
        """Determines if all collaborators have been sent the QUIT command.

        Returns:
            bool: True if all collaborators have been sent the QUIT command.

        """
        return sorted(self.quit_job_sent_to) == sorted(self.collaborator_common_names)

    def validate_header(self, message):
        """Validates the message is from valid collaborator in this federation.

        Returns:
            bool: True if the message is from a valid collaborator in this federation.

        """

        # validate that the message is for me
        check_equal(message.header.recipient, self.uuid, self.logger)

        # validate that the message is for my federation
        check_equal(message.header.federation_id, self.federation_uuid, self.logger)

        # validate that the sender is one of my collaborators
        check_is_in(message.header.sender, self.collaborator_common_names, self.logger)

        # check that we agree on single_col_cert_common_name
        check_equal(message.header.single_col_cert_common_name, self.single_col_cert_common_name, self.logger)

    def init_per_col_round_stats(self):
        """Initalize the metrics from collaborators for each round of aggregation. """
        keys = ["loss_results", "collaborator_training_sizes", "agg_validation_results", "preagg_validation_results", "collaborator_validation_sizes"]
        values = [{} for i in range(len(keys))]
        self.per_col_round_stats = dict(zip(keys, values))

    def collaborator_is_done(self, c):
        """Determines if a collaborator is finished a round.

        Args:
            c: Collaborator name

        Returns:
            bool: True if collaborator c is done.

        """
        assert c in self.collaborator_common_names

        # FIXME: this only works because we have fixed roles each round
        return (c in self.per_col_round_stats["loss_results"] and
                c in self.per_col_round_stats["collaborator_training_sizes"] and
                c in self.per_col_round_stats["agg_validation_results"] and
                c in self.per_col_round_stats["preagg_validation_results"] and
                c in self.per_col_round_stats["collaborator_validation_sizes"])

    def num_collaborators_done(self):
        """Returns the number of collaborators that have finished the training round.

        Returns:
            int: The number of collaborators that have finished this round of training.

        """
        return sum([self.collaborator_is_done(c) for c in self.collaborator_common_names])

    def straggler_time_expired(self):
        """Determines if there are still collaborators that have not returned past the expected round time.
        Returns:
            bool: True if training round limit has expired (i.e. there are straggler collaborators that have not returned in the expected time)

        """
        return self.round_start_time is not None and ((time.time() - self.round_start_time) > self.straggler_cutoff_time)

    def minimum_collaborators_reported(self):
        """Determines if enough collaborators have returned to do the aggregation.

        Returns:
            bool: True if the number of collaborators that have finished is greater than the minimum threshold set.

        """
        return self.num_collaborators_done() >= self.minimum_reporting

    def straggler_cutoff_check(self):
        """Determines if we will end the round now, cutting off any remaining stragglers.

        Returns:
            bool: True if collaborator c is done.

        """
        cutoff = self.straggler_time_expired() and self.minimum_collaborators_reported()
        if cutoff:
            collaborators_done = [c for c in self.collaborator_common_names if self.collaborator_is_done(c)]
            self.logger.info('\tEnding round early due to straggler cutoff. Collaborators done: {}'.format(collaborators_done))
        return cutoff

    def end_of_round_check(self):
        """Determines if it is the end of a training round, if so calling method to perform end of round operations.

        Returns:
            None

        """
        # FIXME: find a nice, clean way to manage these values without having to manually ensure
        # the keys are in sync

        # assert our dictionary keys are in sync
        check_equal(self.per_col_round_stats["loss_results"].keys(), self.per_col_round_stats["collaborator_training_sizes"].keys(), self.logger)
        check_equal(self.per_col_round_stats["agg_validation_results"].keys(), self.per_col_round_stats["collaborator_validation_sizes"].keys(), self.logger)

        # if everyone is done OR our straggler policy calls for an early round end
        if self.num_collaborators_done() == len(self.collaborator_common_names) or self.straggler_cutoff_check():
            self.end_of_round()

    def get_weighted_average_of_collaborators(self, value_dict, weight_dict):
        """Calculate the weighted average of data from the collaborators.

        Args:
            value_dict: A dictionary of the values (values can be dictionaries)
            weight_dict: A dictionary of the collaborator weights (percentage of total data size)

        Returns:
            Dictionary containing weighted averages across collaborators

        """
        cols = [k for k in value_dict.keys() if k in self.collaborator_common_names]
        # detect if our value dictionary has one or two levels,and if so get the second level keys
        example_value = value_dict[cols[0]]
        if isinstance(example_value, float):
            averages = float(np.average([value_dict[c] for c in cols], weights=[weight_dict[c] for c in cols]))
        else:
            averages = {}
            for key in example_value.keys():
                averages[key] = float(np.average([value_dict[c][key] for c in cols], weights=[weight_dict[c] for c in cols])) 
            
        return averages

    def end_of_round(self):
        """Runs required tasks when the training round has ended.
        """
        # FIXME: what all should we do to track results/metrics? It should really be an easy, extensible solution

        # compute the weighted loss average
        round_loss = self.get_weighted_average_of_collaborators(self.per_col_round_stats["loss_results"],
                                                                self.per_col_round_stats["collaborator_training_sizes"])

        # compute the weighted validation average
        round_val = self.get_weighted_average_of_collaborators(self.per_col_round_stats["agg_validation_results"],
                                                                self.per_col_round_stats["collaborator_validation_sizes"])

        # FIXME: is it correct to put this in the metadata?
        self.metadata_for_round.update({'loss': round_loss, 'round_{}_validation'.format(self.round_num-1): round_val})

        # FIXME: proper logging
        self.logger.info('round results for model id/version {}/{}'.format(self.model.header.id, self.model.header.version))
        self.logger.info('\tvalidation: {}'.format(round_val))
        self.logger.info('\tloss: {}'.format(round_loss))

        # construct the model protobuf from in progress tensors (with incremented version number)
        # self.model = construct_proto(tensor_dict=self.model_update_in_progress["tensor_dict"],
        #                              model_id=self.model.header.id,
        #                              model_version=self.model.header.version + 1,
        #                              is_delta=self.model_update_in_progress["is_delta"],
        #                              delta_from_version=self.model_update_in_progress["delta_from_version"],
        #                              compression_pipeline=self.compression_pipeline)

        # add end of round metadata
        self.metadata_for_round.update(self.get_end_of_round_metadata())

        # add the metadata for this round to the total metadata file 
        self.metadata['round {}'.format(self.round_num)] = self.metadata_for_round
        self.metadata_for_round = {}

        self.logger.info("Metadata:\n{}".format(yaml.dump(self.metadata)))

        if self.latest_metadata_fname is not None:
            with open(self.latest_metadata_fname, 'w') as f:
                f.write(yaml.dump(self.metadata))
            self.logger.info("Wrote metadata to {}".format(self.latest_metadata_fname))

        # Save the new model as latest model.
        dump_proto(self.model, self.latest_model_fpath)

        # if configured, also save to the backup location
        if self.save_all_models_path is not None:
            self.save_aggregated_model()

        # in case that round_val is a dictionary (asuming one level only), basing best model on average of inner value
        if isinstance(round_val, dict):
            model_score = np.average(list(round_val.values()))
        else:
            model_score = round_val
        if self.best_model_score is None or self.best_model_score < model_score:
            self.logger.info("Saved the best model with score {:f}.".format(model_score))
            self.best_model_score = model_score
            # Save a model proto version to file as current best model.
            dump_proto(self.model, self.best_model_fpath)
            self.aggregated_model_is_global_best = True
        else:
            self.aggregated_model_is_global_best = False

        # clear the update pointer
        # self.model_update_in_progress = self.model

        # if we have enabled runtime configuration updates, do that now
        if self.runtime_aggregator_config_dir is not None:
            self.update_config_from_filesystem()

        self.init_per_col_round_stats()

        self.round_num += 1
        self.logger.debug("Start a new round %d." % self.round_num)
        self.round_start_time = None


    def UploadLocalModelUpdate(self, message):
        """Parses the collaborator reply message to get the collaborator model update

        Args:
            message: Message from the collaborator

        Returns:
            The reply to the message (usually just the acknowledgement to the collaborator)

        """

        self.mutex.acquire(blocking=True)
        try:
            t = time.time()
            self.validate_header(message)

            # self.logger.info("Receive model update from %s " % message.header.sender)

            # Get the model parameters from the model proto and additional model info
            model_tensors = deconstruct_proto(model_proto=message.model, compression_pipeline=self.compression_pipeline)
            is_delta = message.model.header.is_delta
            delta_from_version = message.model.header.delta_from_version

            # validate this model header
            check_equal(message.model.header.id, self.model.header.id, self.logger)
            check_equal(message.model.header.version, self.model.header.version, self.logger)

            # ensure we haven't received an update from this collaborator already
            # check_not_in(message.header.sender, self.per_col_round_stats["loss_results"], self.logger)
            # check_not_in(message.header.sender, self.per_col_round_stats["collaborator_training_sizes"], self.logger)

            # dump the local update, if necessary
            if self.save_all_models_path is not None:
                self.save_local_update(message.header.sender, message.model)

            m = message.model_num

            # if this is our very first update for the round, we take these model tensors as-is
            # FIXME: move to model deltas, add with original to reconstruct
            # FIXME: this really only works with a trusted collaborator. Sanity check this against self.model
            if self.model_update_in_progress[m] is None:
                self.model_update_in_progress[m] = {"tensor_dict": model_tensors,
                                                    "is_delta": is_delta,
                                                    "delta_from_version": delta_from_version}

            # otherwise, we compute the streaming weighted average
            else:
                # get the current update size total
                total_update_size = np.sum(list(self.per_col_round_stats["collaborator_training_sizes"].values()))

                # compute the weights for the global vs local tensors for our streaming average
                weight_g = total_update_size / (message.data_size + total_update_size)
                weight_l = message.data_size / (message.data_size + total_update_size)

                # The model parameters are represented in float32 and will be transmitted in byte stream.
                weight_g = weight_g.astype(np.float32)
                weight_l = weight_l.astype(np.float32)

                # FIXME: right now we're really using names just to sanity check consistent ordering

                # check that the models include the same number of tensors, and that whether or not
                # it is a delta and from what version is the same
                # check_equal(len(self.model_update_in_progress["tensor_dict"]), len(model_tensors), self.logger)
                # check_equal(self.model_update_in_progress["is_delta"], is_delta, self.logger)
                # check_equal(self.model_update_in_progress["delta_from_version"], delta_from_version, self.logger)

                # aggregate all the model tensors in the tensor_dict
                # (weighted average of local update l and global tensor g for all l, g)
                for name, l in model_tensors.items():
                    g = self.model_update_in_progress[m]["tensor_dict"][name]
                    # check that g and l have the same shape
                    check_equal(g.shape, l.shape, self.logger)

                    # now store a weighted average into the update in progress
                    self.model_update_in_progress[m]["tensor_dict"][name] = np.average([g, l], weights=[weight_g, weight_l], axis=0)

            # store the loss results and training update size
            self.per_col_round_stats["loss_results"][message.header.sender] = message.loss
            self.per_col_round_stats["collaborator_training_sizes"][message.header.sender] = message.data_size

            # return LocalModelUpdateAck
            # self.logger.debug("Complete model update from %s " % message.header.sender)
            reply = LocalModelUpdateAck(header=self.create_reply_header(message))

            self.end_of_round_check()

            # self.logger.debug('aggregator handled UploadLocalModelUpdate in time {}'.format(time.time() - t))
            if m == self.num_models - 1:
                self.logger.info("Received model last model update from %s " % message.header.sender)

        finally:
            self.mutex.release()

        return reply

    def UploadLocalMetricsUpdate(self, message):
        """Parses the collaborator reply message to get the collaborator metrics (usually the local validation score)

        Args:
            message: Message from the collaborator

        Returns:
            The reply to the message (usually just the acknowledgement to the collaborator)

        """

        self.mutex.acquire(blocking=True)
        try:
            t = time.time()
            self.validate_header(message)

            self.logger.debug("Receive local validation results from %s " % message.header.sender)
            model_header = message.model_header

            # validate this model header
            check_equal(model_header.id, self.model.header.id, self.logger)
            check_equal(model_header.version, self.model.header.version, self.logger)

            sender = message.header.sender

            if sender not in self.per_col_round_stats["agg_validation_results"]:
                # Pre-train validation
                # ensure we haven't received an update from this collaborator already
                # FIXME: is this an error case that should be handled?
                check_not_in(message.header.sender, self.per_col_round_stats["agg_validation_results"], self.logger)
                check_not_in(message.header.sender, self.per_col_round_stats["collaborator_validation_sizes"], self.logger)

                # store the validation results and validation size
                self.per_col_round_stats["agg_validation_results"][message.header.sender] = message.results
                self.per_col_round_stats["collaborator_validation_sizes"][message.header.sender] = message.data_size
            elif sender not in self.per_col_round_stats["preagg_validation_results"]:
                # Post-train validation
                check_not_in(message.header.sender, self.per_col_round_stats["preagg_validation_results"], self.logger)
                self.per_col_round_stats["preagg_validation_results"][message.header.sender] = message.results

            reply = LocalValidationResultsAck(header=self.create_reply_header(message))

            self.end_of_round_check()

            self.logger.debug('aggregator handled UploadLocalMetricsUpdate in time {}'.format(time.time() - t))
        finally:
            self.mutex.release()

        self.logger.debug('aggregator handled UploadLocalMetricsUpdate in time {}'.format(time.time() - t))

        return reply

    def RequestJob(self, message):
        """Parse message for job request and act accordingly.

        Args:
            message: Message from the collaborator

        Returns:
            The reply to the message (usually just the acknowledgement to the collaborator)

        """

        t = time.time()
        self.validate_header(message)

        # FIXME: we should really have each collaborator validate one last time
        # check if we are done
        if self.round_num > self.rounds_to_train:
            job = JOB_QUIT

            # we may need to send this again if the collaborator process has been restarted for some reason
            # thus we may not need to add it to the list again
            if message.header.sender not in self.quit_job_sent_to:
                self.quit_job_sent_to.append(message.header.sender)
        # FIXME: this flow needs to depend on a job selection output for the round
        # for now, all jobs require an in-sync model, so it is the first check
        # check if the sender model is out of date
        elif self.collaborator_out_of_date(message.model_header):
            job = JOB_DOWNLOAD_MODEL
        # else, check if this collaborator has not sent validation results
        elif message.header.sender not in self.per_col_round_stats["agg_validation_results"]:
            job = JOB_VALIDATE
        # else, check if this collaborator has not sent training results
        elif message.header.sender not in self.per_col_round_stats["collaborator_training_sizes"]:
            job = JOB_TRAIN
        elif message.header.sender not in self.per_col_round_stats["preagg_validation_results"]:
            job = JOB_VALIDATE
        # else this collaborator is done for the round
        else:
            job = JOB_YIELD

        self.logger.debug("Receive job request from %s and assign with %s" % (message.header.sender, job))

        reply = JobReply(header=self.create_reply_header(message), job=job)

        if reply.job is not JOB_YIELD:
            # check to see if we need to set our round start time
            self.mutex.acquire(blocking=True)
            try:
                if self.round_start_time is None:
                    self.round_start_time = time.time()
            finally:
                self.mutex.release()

            self.logger.debug('aggregator handled RequestJob in time {}'.format(time.time() - t))
        elif self.straggler_cutoff_time != np.inf:
            # we have an idle collaborator and a straggler cutoff time, so we should check for early round end
            self.mutex.acquire(blocking=True)
            try:
                self.end_of_round_check()
            finally:
                self.mutex.release()

        return reply

    def DownloadModel(self, message):
        """Sends a model to the collaborator

        Args:
            message: Message from the collaborator

        Returns:
            The reply to the message (usually just the acknowledgement to the collaborator)

        """

        t = time.time()
        self.validate_header(message)

        # self.logger.info("Received model download request from %s " % message.header.sender)

        # ensure the models don't match
        # if not(self.collaborator_out_of_date(message.model_header)):
        #     statement = "Collaborator asking for download when not out of date."
        #     self.logger.exception(statement)
        #     raise RuntimeError(statement)

        # check whether there is an issue related to the sending of deltas or non-deltas
        # if message.model_header.version == -1:
        #     if self.model.header.is_delta:
        #         raise RuntimeError('First collaborator model download, and we only have a delta.')
        # elif message.model_header.is_delta != self.model.header.is_delta:
        #     raise RuntimeError('Collaborator requesting non-initial download should hold a model with the same is_delta as aggregated model.')
        # elif message.model_header.is_delta and (message.model_header.delta_from_version != self.model.header.delta_from_version):
        #     # TODO: In the future we could send non-delta model here to restore base model.
        #     raise NotImplementedError('Base of download model delta does not match current collaborator base, and aggregator restoration of base model not implemented.')

        metadata_yaml = None
        if self.send_metadata_to_clients:
            metadata_yaml = yaml.dump(self.metadata)

        model = self.models[message.model_num]

        reply = GlobalModelUpdate(header=self.create_reply_header(message), model=self.model, is_global_best=self.aggregated_model_is_global_best, metadata_yaml=metadata_yaml)

        if message.model_num == self.num_models - 1:
            self.logger.info("Completed last model download request from %s " % message.header.sender)

        return reply

    def create_reply_header(self, message):
        """Creates a header for the reply to the message

        Args:
            message: Message from the collaborator

        Returns:
            The message header.

        """
        return MessageHeader(sender=self.uuid, recipient=message.header.sender, federation_id=self.federation_uuid, counter=message.header.counter, single_col_cert_common_name=self.single_col_cert_common_name)

    def collaborator_out_of_date(self, model_header):
        """Determines if the collaborator has the wrong version of the model (aka out of date)

        Args:
            model_header: Header for the model

        Returns:
            The reply to the message (usually just the acknowledgement to the collaborator)

        """
        # validate that this is the right model to be checking
        check_equal(model_header.id, self.model.header.id, self.logger)

        return model_header.version != self.model.header.version

    def log_big_warning(self):
        self.logger.warning("\n{}\nYOU ARE RUNNING IN SINGLE COLLABORATOR CERT MODE! THIS IS NOT PROPER PKI AND SHOULD ONLY BE USED IN DEVELOPMENT SETTINGS!!!! YE HAVE BEEN WARNED!!!".format(the_dragon))

    def get_end_of_round_metadata(self):
        metadata = {}
        
        # if no metadata to get, return empty dict
        if self.end_of_round_metadata is None:
            return metadata
        
        # otherwise, iterate through list of function calls
        for func_name in self.end_of_round_metadata:
            try:
                func = getattr(self, func_name)
                metadata[func_name] = func()
            except AttributeError as e:
                self.logger.critical("Aggregator unable to compute metadata: no Aggregator function called {}. Check the flplan/aggregator_object_init/init_kwargs/end_of_round_metadata list for this name. It may be misspelled".format(func_name))
        return metadata

    def aggregation_time(self):
        return time.time()

    def participant_training_data_size(self):
        return sum(list(self.per_col_round_stats["collaborator_training_sizes"].values()))


the_dragon = """

 ,@@.@@+@@##@,@@@@.`@@#@+  *@@@@ #@##@  `@@#@# @@@@@   @@    @@@@` #@@@ :@@ `@#`@@@#.@
  @@ #@ ,@ +. @@.@* #@ :`   @+*@ .@`+.   @@ *@::@`@@   @@#  @@  #`;@`.@@ @@@`@`#@* +:@`
  @@@@@ ,@@@  @@@@  +@@+    @@@@ .@@@    @@ .@+:@@@:  .;+@` @@ ,;,#@` @@ @@@@@ ,@@@* @
  @@ #@ ,@`*. @@.@@ #@ ,;  `@+,@#.@.*`   @@ ,@::@`@@` @@@@# @@`:@;*@+ @@ @`:@@`@ *@@ `
 .@@`@@,+@+;@.@@ @@`@@;*@  ;@@#@:*@+;@  `@@;@@ #@**@+;@ `@@:`@@@@  @@@@.`@+ .@ +@+@*,@
  `` ``     ` ``  .     `     `      `     `    `  .` `  ``   ``    ``   `       .   `



                                            .**
                                      ;`  `****:
                                     @**`*******
                         ***        +***********;
                        ,@***;` .*:,;************
                        ;***********@@***********
                        ;************************,
                        `*************************
                         *************************
                         ,************************
                          **#*********************
                          *@****`     :**********;
                          +**;          .********.
                          ;*;            `*******#:                       `,:
                                          ****@@@++::                ,,;***.
                                          *@@@**;#;:         +:      **++*,
                                          @***#@@@:          +*;     ,****
                                          @*@+****           ***`     ****,
                                         ,@#******.  ,       ****     **;,**.
                                         * ******** :,       ;*:*+    **  :,**
                                        #  ********::      *,.*:**`   *      ,*;
                                        .  *********:      .+,*:;*:   :      `:**
                                       ;   :********:       ***::**   `       ` **
                                       +   :****::***  ,    *;;::**`             :*
                                      ``   .****::;**:::    *;::::*;              ;*
                                      *     *****::***:.    **::::**               ;:
                                      #     *****;:****     ;*::;***               ,*`
                                      ;     ************`  ,**:****;               ::*
                                      :     *************;:;*;*++:                   *.
                                      :     *****************;*                      `*
                                     `.    `*****************;  :                     *.
                                     .`    .*+************+****;:                     :*
                                     `.    :;+***********+******;`    :              .,*
                                      ;    ::*+*******************. `::              .`:.
                                      +    :::**********************;;:`                *
                                      +    ,::;*************;:::*******.                *
                                      #    `:::+*************:::;********  :,           *
                                      @     :::***************;:;*********;:,           *
                                      @     ::::******:*********************:         ,:*
                                      @     .:::******:;*********************,         :*
                                      #      :::******::******###@*******;;****        *,
                                      #      .::;*****::*****#****@*****;:::***;  ``  **
                                      *       ::;***********+*****+#******::*****,,,,**
                                      :        :;***********#******#******************
                                      .`       `;***********#******+****+************
                                      `,        ***#**@**+***+*****+**************;`
                                       ;         *++**#******#+****+`      `.,..
                                       +         `@***#*******#****#
                                       +          +***@********+**+:
                                       *         .+**+;**;;;**;#**#
                                      ,`         ****@         +*+:
                                      #          +**+         :+**
                                      @         ;**+,       ,***+
                                      #      #@+****      *#****+
                                     `;     @+***+@      `#**+#++
                                     #      #*#@##,      .++:.,#
                                    `*      @#            +.
                                  @@@
                                 #`@
                                  ,                                                        """
