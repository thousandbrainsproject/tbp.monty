# Copyright 2025 Thousand Brains Project
# Copyright 2024 Numenta Inc.
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

import logging
import os
import shutil

import numpy as np
from tqdm import tqdm

from tbp.monty.frameworks.models.evidence_matching import EvidenceGraphLM


class LoggerSDR:
    """A simple logger that saves the data passed to it.

    This logger maintains an episode counter and logs
    the data it receives under different files named
    by the episode counter.

    *See more information about what data is being logged
    under the `log_episode` function.*
    """

    # TODO: Needs to be removed. The logger should be part
    # of Monty loggers. Fix issue #328 first.
    def __init__(self, path):
        if path is None:
            logging.warning("EvidenceSDR log path is set to None.")
            return

        path = os.path.expanduser(path)

        # overwrite existing logs
        if os.path.exists(path):
            shutil.rmtree(path)
        os.makedirs(path)

        self.path = path
        self.episode = 0

    def log_episode(self, data):
        """Receives data dictionary and saves it as a pth file.

        This function will save all the data passed to it. Here is
        a breakdown of the data to be logged by this function.

        The data dictionary contains these key-value pairs:
            - mask: 2d tensor of the available overlap targets after this
                    episode
            - target_overlap: 2d tensor of the target overlap at the end of
                    this episode
            - training: Dictionary of training statistics for every epoch.
                    Includes overlap_error, training_summed_distance,
                    dense representations, and sdrs
            - obj2id: Objects to ids dictionary mapping
            - id2obj: Ids to objects dictionary mapping
        """
        if hasattr(self, "path"):
            np.save(
                os.path.join(self.path, f"episode_{str(self.episode).zfill(3)}.npy"),
                data,
            )
            self.episode += 1


class EncoderSDR:
    """The SDR Encoder class.

    This class keeps track of the dense representations, and trains them to output SDRs
    when binarized.  This class also contains its own optimizer and function to add more
    objects/representations.

    The representations are stored as dense vectors and binarized using
    top-k to convert them to SDRs. During training, the pairwise overlaps between the
    sdrs are compared to the target overlaps. This error signal trains the dense
    representations.

    Refer to the `self.train_sdrs` function for more information on the training details

    Attributes:
        sdr_length: The size of the SDRs (total number of bits).
        sdr_on_bits: The number of on bits in the SDRs. Controls sparsity.
        lr: The learning rate of the encoding algorithm.
        n_epochs: The number of training epochs per episode
        stability: The stability parameter controls by how much old SDRs
            change relative to new SDRs.  Value range is [0.0, 1.0], where 0.0 is no
            stability constraint applied and 1.0 is fixed SDRs. Values in between are
            for partial stability.
        log_flag: Flag to activate the logger.
    """

    def __init__(
        self,
        sdr_length=2048,
        sdr_on_bits=41,
        lr=1e-2,
        n_epochs=1000,
        stability=0.0,
        log_flag=False,
    ):
        if sdr_on_bits >= sdr_length or sdr_on_bits <= 0:
            logging.warning(
                f"Invalid sparsity: sdr_on_bits set to 2% ({round(sdr_length*0.02)})"
            )
            sdr_on_bits = round(sdr_length * 0.02)

        self.sdr_length, self.sdr_on_bits = sdr_length, sdr_on_bits
        self.lr = lr
        self.n_epochs = n_epochs
        self.stability = stability
        if self.stability > 1.0 or self.stability < 0.0:
            self.stability = np.clip(self.stability, 0.0, 1.0)
            logging.warning(
                f"Invalid stability parameter: stability clamped to {self.stability}"
            )
        self.log_flag = log_flag

        # Initialize obj SDR array with arbitrary values
        self.obj_sdrs = np.zeros((0, self.sdr_length))

    @property
    def n_objects(self):
        """Return the available number of objects."""
        return self.obj_sdrs.shape[0]

    @property
    def sdrs(self):
        """Return the available SDRs."""
        return self.binarize(self.obj_sdrs)

    def get_sdr(self, index):
        """Return the SDR at a specific index.

        This index refers to the object index in the SDRs dictionary
        (i.e., self.obj_sdrs)
        """
        return self.sdrs[index]

    def optimize(self, overlap_error, mask):
        """Compute and apply local gradient descent.

        Compute based on the overlap error and mask.

        Note there is no use of the chain rule, i.e. each SDR is optimized based on the
        derivative of its clustering error with respect to its values, with no
        intermediary functions.

        The overlap error helps correct the sign and also provides a magnitude for the
        representation updates.

        Args:
            overlap_error: The difference between target and predicted overlaps.
            mask: Mask indicating valid entries in the overlap matrix.

        Note:
            num_objects = self.n_objects

        Note:
            A vectorized version of the algorithm is provided below, although it
            would need to be modified to avoid repeated creation of arrays in order to
            be more efficient. Leaving for now as this algorithm is not a bottle-neck
            (circa 10-20 seconds to learn 60 object SDRs).:

            # Initialize gradients
            grad = np.zeros_like(self.obj_sdrs)

            # Compute the pairwise differences between SDRs
            diff_matrix = (
                self.obj_sdrs[:, np.newaxis, :] - self.obj_sdrs[np.newaxis, :, :]
            )

            # Compute the absolute differences for each pair
            abs_diff = np.sum(np.abs(diff_matrix), axis=2)

            # Create a mask for non-zero differences
            non_zero_mask = abs_diff > 0

            # Apply the mask to the original mask
            valid_mask = mask & non_zero_mask

            # Calculate the summed distance and gradient contributions where the mask
            # is valid for logging
            summed_distance = np.sum(overlap_error * valid_mask * abs_diff)

            # Calculate the gradients
            grad_contrib = overlap_error[:, :, np.newaxis] * 2 * diff_matrix
            grad += np.sum(grad_contrib * valid_mask[:, :, np.newaxis], axis=1)
            grad -= np.sum(grad_contrib * valid_mask[:, :, np.newaxis], axis=0)

            # Update the SDRs using the gradient
            self.obj_sdrs -= self.lr * grad

        Returns:
            The summed distance for logging.
        """
        # Initialize the gradient array
        grad = np.zeros_like(self.obj_sdrs)

        # Track the summed distance for logging, i.e. to be able to visualize that
        # it is decreasing during each update
        summed_distance = 0

        # Calculate the gradient for each pair of objects
        for i in range(self.n_objects):
            for j in range(self.n_objects):
                if mask[i, j]:
                    # As we're optimizing the L2 norm of the difference between
                    # the two SDRs, the gradient is the difference between the
                    # two non-binarized (dense) representations.
                    diff = self.obj_sdrs[i] - self.obj_sdrs[j]
                    if np.sum(np.abs(diff)) > 0:
                        summed_distance += overlap_error[i, j] * np.sum(np.abs(diff))
                        # Multiply the gradient by 2 to exactly match the
                        # derivative of the L2 norm
                        grad[i] += overlap_error[i, j] * 2 * diff
                        grad[j] -= overlap_error[i, j] * 2 * diff

        # Update the SDRs using the gradient
        self.obj_sdrs -= self.lr * grad

        return summed_distance

    def add_objects(self, n_objects):
        """Adds more objects to the available objects and re-initializes the optimizer.

        We keep track of the stable representation ids (old objects) when
        adding new objects.

        Args:
            n_objects: Number of objects to add

        """
        if n_objects == 0:
            return

        # store stable data and ids
        stable_data = self.obj_sdrs.copy()
        self.stable_ids = np.arange(stable_data.shape[0])

        new_obj_sdrs = np.random.randn(
            stable_data.shape[0] + n_objects, self.sdr_length
        )

        new_obj_sdrs[: stable_data.shape[0]] = stable_data
        self.obj_sdrs = new_obj_sdrs

    def train_sdrs(self, target_overlaps, log_epoch_every=10):
        """Main SDR training function.

        This function receives a copy of the average target overlap 2D tensor
        and trains the sdr representations for `n_epochs` to achieve these target
        overlap scores.

        We use the overlap target as a learning signal to move the dense representations
        towards or away from each other. The magnitude of the overlap error controls the
        strength of moving dense representations. Also the sign of the overlap error
        controls whether the representations will be moving towards or away from each
        other.

        We want to limit the amount by which trained representation change relative
        to untrained object representations such that higher-level LMs would not suffer
        from significant changes in lower-level representations that were used to build
        higher-level graphs.

        When adding new representations, we keep track of the ids of the older
        representations (i.e., `self.stable_ids`). This allows us to control by how much
        the older representations move relative to the newer ones during training. This
        behavior is controlled by the stability value. During each training iteration,
        we update these older representations with an average of the optimizer output
        and the original representation (weighted by the stability value). Note that
        too much stability restricts the SDRs from adapting to desired changes in the
        target overlaps caused by normalization or distribution shift, affecting the
        overall encoding performance.

        Consider two dense representations, A_dense and B_dense. We apply top-k
        operation on both to convert them to A_sdr and B_sdr, then calculate their
        overlaps. If the overlap is less than the target overlap, we move dense
        representations (A_dense and B_dense) closer to eachother with strength
        proportional to the error in overlaps. We move them apart if they have more
        overlap than the target.

        Note:
            The `distance_matrix` variable is calculated using the cdist function
            and it denotes the pairwise euclidean distances between *dense*
            representations. The term "overlap" always refers to the overlap in bits
            between SDRs.

        Note:
            The overlap_error is only used to weight the distance_matrix for
            each pair of objects, and gradients *do not* flow through the sparse
            overlap calculations.

        Returns:
            The stats dictionary for logging.
        """
        # return if no target provided
        stats = {}
        if np.all(np.isnan(target_overlaps)):
            logging.warning("Empty overlap targets. No training needed.")
            return stats

        if np.all(np.array(target_overlaps.shape) > self.n_objects):
            logging.warning(
                "Overlap targets have larger size than "
                + f"{(self.n_objects, self.n_objects)}"
            )
            target_overlaps = target_overlaps[: self.n_objects, : self.n_objects]

        # Calculate the training mask and target
        # The mask determines the valid entries with value:
        #     - 1: where overlap target exists
        #     - 0: where overlap target does not exist as of this episode.
        mask = ~np.isnan(target_overlaps)
        overlaps = np.nan_to_num(target_overlaps, nan=0)

        # logging details
        if self.log_flag:
            stats["mask"] = mask
            stats["target_overlap"] = overlaps
            stats["training"] = {}

        for epoch in tqdm(range(self.n_epochs)):
            # These values are used to pull back the representations from moving
            # too far during training. Notice this is only applied on self.stable_ids.
            sdrs_stable_before = self.obj_sdrs[self.stable_ids].copy()

            # calculate predicted overlaps from existing representations
            reps = self.obj_sdrs
            bins = self.binarize(reps)
            pred_overlaps = bins @ bins.T

            # calculate error and optimize
            overlap_error = overlaps - pred_overlaps

            summed_distance = self.optimize(overlap_error, mask)

            # stabilize the SDRs at `self.stable_ids` by pulling them back towards
            # sdrs_stable_before
            sdrs_stable_after = self.obj_sdrs[self.stable_ids].copy()
            self.obj_sdrs[self.stable_ids] = (self.stability * sdrs_stable_before) + (
                (1 - self.stability) * sdrs_stable_after
            )

            # logging details
            if self.log_flag and epoch % log_epoch_every == 0:
                stats["training"][epoch] = {}
                stats["training"][epoch]["obj_dense"] = self.obj_sdrs.copy()
                stats["training"][epoch]["obj_sdr"] = bins.copy()
                stats["training"][epoch]["overlap_error"] = overlap_error
                stats["training"][epoch]["summed_distance"] = summed_distance

        # Reset stable ids.
        # Stability training only used after adding new objects
        self.stable_ids = np.array([]).astype(int)
        return stats

    def binarize(self, emb):
        """Convert dense representations to SDRs (0s and 1s) using Top-k function.

        Returns:
            The SDRs.
        """
        topk_indices = np.argsort(emb, axis=1)[:, -self.sdr_on_bits :]
        mask = np.zeros_like(emb)
        np.put_along_axis(mask, topk_indices, 1, axis=1)
        return mask


class EvidenceSDRTargetOverlaps:
    """Keep track of the running average of target overlaps for each episode.

    The target overlaps is implemented as a 2D tensor where the indices
    of the tensor represent the ids of the objects, and the values of the
    tensor represent a running average of the overlap target.

    To achieve this, we implement functions for expanding the size of the
    overlap target tensor, linear mapping for normalization, and updating
    the overlap tensor (i.e., running average).

    Note:
        We are averaging over multiple overlap targets.
        Multiple targets can happen for different reasons:
            - Asymmetric evidences: the target overlap for object 1 w.r.t object 2
                ([2,1]) is averaged with object 2 w.r.t object 1 ([1,2]). This is
                possible because we sort the ids when we add the evidences to
                overlaps. Both evidences get added to the location [1,2].
            - Additional episodes with similar MLO (most-likely object): More episodes
                can accumulate additional evidences on to the same key if the MLO
                is similar to previous MLO of another episode.
    """

    def __init__(self):
        """Initialize with overlap tensor.

        Initialize the class with overlap tensor to store the running average of the
        target scores. Additionally we store the counts to easily calculate the running
        average.
        """
        self._overlaps = np.full((0, 0), np.nan)
        self._counts = np.zeros_like(self._overlaps)

    @property
    def overlaps(self):
        """Returns the target overlap values rounded to the nearest integer."""
        # TODO: Experiment without rounding. Shouldn't make much of a difference
        # since this is only used to weight the encoding distances.
        return np.round(self._overlaps)

    def add_objects(self, new_size):
        """Expands the overlaps and the counts 2D tensors to accomodate new objects."""
        # expand the overlaps tensor to the new size
        new_overlaps = np.full((new_size, new_size), np.nan)
        new_overlaps[: self._overlaps.shape[0], : self._overlaps.shape[0]] = (
            self._overlaps
        )
        self._overlaps = new_overlaps

        # expand the counts tensor to the new size
        new_counts = np.zeros((new_size, new_size))
        new_counts[: self._counts.shape[0], : self._counts.shape[0]] = self._counts
        self._counts = new_counts

    def map_to_overlaps(self, evidence, output_range):
        """Linear mapping of values from input range to output range.

        Only applies to real values (i.e., ignores nan values).

        Returns:
            ?
        """
        valid_ix = ~np.isnan(evidence)
        min_evidence = np.nanmin(evidence[valid_ix])
        max_evidence = np.nanmax(evidence[valid_ix])
        input_range = [min_evidence, max_evidence]

        output_range_diff = output_range[1] - output_range[0]
        input_range_diff = input_range[1] - input_range[0]

        evidence[valid_ix] = (evidence[valid_ix] - input_range[0]) * (
            output_range_diff
        ) / input_range_diff + output_range[0]

        return evidence

    def add_overlaps(self, mapped_overlaps):
        """Main function for updating the running average with overlaps.

        The running average equation we use is:
        new_average = ((old_average * counts) + (new_val * 1))/ (counts + 1)

        This calculates equally-weighted average, assuming that we keep track
        of the counts and increment them every time we add a new value to the
        average.
        """
        # calculate the mask of indices for existing avg overlaps and
        # new overlaps. The mask should be True where both values are
        # not nan
        mask_avg = np.logical_and(~np.isnan(self._overlaps), ~np.isnan(mapped_overlaps))

        # apply the running average equation explained in the docstring
        self._overlaps[mask_avg] = (
            (self._overlaps[mask_avg] * self._counts[mask_avg])
            + mapped_overlaps[mask_avg]
        ) / (self._counts[mask_avg] + 1)

        # calculate the mask of indices with True values where existing overlaps
        # are nan and new overlaps are not nan.
        mask_overwrite = np.logical_and(
            np.isnan(self._overlaps), ~np.isnan(mapped_overlaps)
        )

        # overlap existing nan values in `self._overlaps` with new overlaps values
        # in `mapped_overlaps`
        self._overlaps[mask_overwrite] = mapped_overlaps[mask_overwrite]

        # update counts of all new entries
        self._counts[np.logical_or(mask_avg, mask_overwrite)] += 1

    def add_evidence(self, evidence, mapping_output_range):
        """Main function for updating the running average with evidence.

        This function receives as input the relative evidence scores and
        maps them to overlaps in the `mapping_output_range`. The mapped
        overlaps are added to the running average in the function `add_overlaps`.
        """
        # map relative evidences of the current episode to the output range
        mapped_overlaps = self.map_to_overlaps(evidence, mapping_output_range)

        # add overlaps to running average
        self.add_overlaps(mapped_overlaps)


class EvidenceSDRLMMixin:
    """This Mixin adds training of SDR representations to the EvidenceGraphLM.

    It overrides the __init__ and post_episode functions of the LM

    To use this Mixin, pass the EvidenceSDRGraphLM class as the `learning_module_class`
    in the `learning_module_configs`.

    Additionally pass the `sdr_args` dictionary as an additional key
    in the `learning_module_args`.

    The sdr_args dictionary should contain:
        - `log_path` (string): A string that points to a temporary location for saving
                experiment logs. "None" means don't save to file
        - `sdr_length` (int): The size of the SDR to be used for encoding
        - `sdr_on_bits` (int): The number of active bits to be used with these SDRs
        - `sdr_lr` (float): The learning rate of the encoding algorithm
        - `n_sdr_epochs` (int): The number of epochs to train the encoding algorithm
        - `stability` (float): Stability of older object SDRs.
                Value range is [0.0, 1.0], where 0.0 is no stability
                applied and 1.0 is fixed SDRs.
        - `sdr_log_flag` (bool): Flag indicating whether to log the results or not

    See the `monty_lab` repo for reference. Specifically,
    `experiments/configs/evidence_sdr_evaluation.py`

    """

    def __init__(self, *args, **kwargs):
        """The mixin overrides the `__init__` function of the Learning Module.

        The encoding algorithm is initialized here and it stores the actual SDRs.
        Also, a temporary logging function is initialized here.

        """
        self.sdr_args = kwargs.pop("sdr_args")
        super().__init__(*args, **kwargs)

        # keeps track of the Graph objects and their ids
        self.obj2id = {}
        self.id2obj = {}

        # keeps track of overlap running average values
        self.target_overlaps = EvidenceSDRTargetOverlaps()

        # initialize the encoding algorithm
        self.sdr_encoder = EncoderSDR(
            sdr_length=self.sdr_args["sdr_length"],
            sdr_on_bits=self.sdr_args["sdr_on_bits"],
            lr=self.sdr_args["sdr_lr"],
            n_epochs=self.sdr_args["n_sdr_epochs"],
            log_flag=self.sdr_args["sdr_log_flag"],
        )

        # TODO: remove this logger and merge with the Monty Loggers after
        # issue #328 is fixed.
        if self.sdr_args["sdr_log_flag"]:
            self.tmp_logger = LoggerSDR(self.sdr_args["log_path"])

    def collect_evidences(self):
        """Collect evidence scores from the Learning Module.

        We do this in three steps:
            - Step 1: We use the number of objects in the LM
                to update the sdr_encoder and id <-> obj tracking
                dictionaries, as well as the target overlap tensor
            - Step 2: We collect evidences relative to the current
                most likely hypothesis (mlh). Evidences are stored
                in a 2d tensor.
            - Step 3: We use the stored evidences to update the target overlap
                which stores the running average.
                Refer to `EvidenceSDRTargetOverlaps` for more details.

        **Note:** We sort the ids in step 2 because the overlap values are
        suppossed to be symmetric (e.g., "2,5" = "5,2"). This way the target
        overlaps for the ids "x,y" and "y,x" will be averaged together in the
        `EvidenceSDRTargetOverlaps` class.

        """
        # TODO: add more sophisticated logic to sync the SDR representations
        # with available objects in the graph memory. This should facilitate
        # merging or removing objects. The SDR representations should always
        # be in sync with graphs in memory

        # Step 1: add new objects if needed. Useful in learning from scratch experiments
        available_objects = self.get_all_known_object_ids()

        for obj in available_objects:
            if obj not in self.obj2id:
                self.obj2id[obj] = len(self.obj2id)
                self.id2obj[len(self.id2obj)] = obj
        self.sdr_encoder.add_objects(len(self.id2obj) - self.sdr_encoder.n_objects)
        self.target_overlaps.add_objects(len(self.id2obj))

        # Step 2: collect evidences
        mlh_object = self.get_current_mlh()["graph_id"]
        if mlh_object == "no_observations_yet" or self.sdr_encoder.n_objects == 1:
            return
        mlh_object_id = self.obj2id[mlh_object]
        mlh_evidence = np.max(self.evidence[mlh_object])

        relative_evidences = np.full_like(self.target_overlaps.overlaps, np.nan)
        for obj in self.evidence.keys():
            ids = sorted([mlh_object_id, self.obj2id[obj]])
            ev = np.max(self.evidence[obj]) - mlh_evidence
            relative_evidences[ids[0], ids[1]] = ev

        # Step 3: update running average with new evidence scores
        self.target_overlaps.add_evidence(
            relative_evidences, [0, self.sdr_args["sdr_on_bits"]]
        )

    def post_episode(self, *args, **kwargs):
        """Overrides the LM post_episode function.

        This function collects evidences, trains SDRs and logs the output.
        """
        super().post_episode(*args, **kwargs)

        # collect the evidences from Learning Module
        self.collect_evidences()

        # Train the SDR Encoder based on overlap targets
        stats = self.sdr_encoder.train_sdrs(self.target_overlaps.overlaps)

        # logging episode information if flag set to True
        if self.sdr_args["sdr_log_flag"]:
            stats.update(
                {
                    "obj2id": self.obj2id,
                    "id2obj": self.id2obj,
                }
            )
            self.tmp_logger.log_episode(stats)

    def _check_use_features_for_matching(self):
        """Check if features should be used for matching.

        EvidenceGraphLM bypasses comparing object ID by checking the number
        of features on the input channel. In this Mixin we want to use all
        features for matching.

        Returns:
            A dictionary indicating whether to use features for each input channel.
        """
        use_features = {}
        for input_channel in self.tolerances.keys():
            if input_channel not in self.feature_weights.keys():
                use_features[input_channel] = False
            elif self.feature_evidence_increment <= 0:
                use_features[input_channel] = False
            else:
                use_features[input_channel] = True
        return use_features

    def _object_id_to_features(self, object_id):
        """Retrieves the trained SDR corresponding to the object ID.

        Returns:
            The trained SDR corresponding to the object ID.
        """
        if object_id in self.obj2id:
            return self.sdr_encoder.get_sdr(self.obj2id[object_id])
        else:
            return np.zeros(self.sdr_args["sdr_length"])

    def _calculate_feature_evidence_sdr_for_all_nodes(
        self, query_features, input_channel, graph_id
    ):
        """Calculate overlap between stored and query SDR features.

        Calculates the overlap between the SDR features stored at every location in
        the graph and the query SDR feature. This overlap is then compared to the
        tolerance value and the result is used for adjusting the evidence score.

        We use the tolerance (in overlap bits) for generalization. If two objects are
        close enough, their overlap in bits should be higher that the set tolerance
        value.

        The tolerance sets the lowest overlap for adding evidence, the range
        [tolerance, sdr_on_bits] is mapped to [0,1] evidence points. Any overlap less
        then tolerance will not add any evidence. These evidence scores are then
        multiplied by the feature weight of object_ids which scales all of the
        evidence points to the range [0, feature_weights[input_channel]["object_id"]].

        The below variables have the following shapes:
            - feature_array: (n, sdr_length)
            - query_features[input_channel]["object_id"]: (sdr_length)
            - query_feat: (sdr_length, 1)
            - np.matmul(feature_array, query_feat): (n, 1)
            - overlaps: (n)

        Returns:
            The normalized overlaps.
        """
        feature_array = self.graph_memory.get_feature_array(graph_id)[input_channel]

        query_feat = np.expand_dims(query_features[input_channel]["object_id"], 1)
        tolerance = self.tolerances[input_channel]["object_id"]
        sdr_on_bits = query_feat.sum(axis=0)

        overlaps = feature_array @ query_feat.squeeze(-1)
        normalized_overlaps = (overlaps - tolerance) / (sdr_on_bits - tolerance)
        normalized_overlaps[normalized_overlaps < 0] = 0.0

        normalized_overlaps *= self.feature_weights[input_channel]["object_id"]
        return normalized_overlaps

    def _calculate_feature_evidence_for_all_nodes(
        self, query_features, input_channel, graph_id
    ):
        """Calculates feature evidence for all nodes stored in a graph.

        This override method tests if the input_channel is a learning_module. If so,
        a different function is used for feature comparison.

        Note: This assumes that learning modules always outputs 1 feature, object_id.
        If the learning modules output more than object_id features, we need to
        compare these according to their weights.

        Returns:
            The feature evidence for all nodes.
        """
        if input_channel.startswith("learning_module"):
            return self._calculate_feature_evidence_sdr_for_all_nodes(
                query_features, input_channel, graph_id
            )

        return super()._calculate_feature_evidence_for_all_nodes(
            query_features, input_channel, graph_id
        )


class EvidenceSDRGraphLM(EvidenceSDRLMMixin, EvidenceGraphLM):
    """Class that incorporates the EvidenceSDR Mixin with the EvidenceGraphLM."""

    pass
