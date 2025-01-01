# Copyright 2025 Thousand Brains Project
# Copyright 2022-2024 Numenta Inc.
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

import abc
import copy
import json
import logging
import time

import numpy as np
import quaternion
import torch
from scipy.spatial.transform import Rotation

from tbp.monty.frameworks.actions.actions import Action, ActionJSONEncoder


class BaseBuffer:
    @abc.abstractclassmethod
    def __len__(self):
        pass

    @abc.abstractclassmethod
    def append(self):
        pass

    @abc.abstractclassmethod
    def __getitem__(self):
        pass

    @abc.abstractclassmethod
    def reset(self):
        pass


class FeatureAtLocationBuffer(BaseBuffer):
    """Buffer which stores features at locations coming into one LM. Also stores stats.

    Used for building graph models and logging detailed stats about an episode. The
    location buffer is also used to calculate displacements.
    """

    def __init__(self):
        """Initialize buffer dicts for locations, features, displacements and stats."""
        self.locations = dict()
        self.features = dict()
        self.on_object = []
        self.input_states = []

        self.displacements = dict()

        self.stats = {
            "detected_path": None,
            "detected_location_on_model": [None, None, None],  # model ref frame
            "detected_location_rel_body": [
                None,
                None,
                None,
            ],  # body ref frame
            "detected_rotation": None,
            "detected_rotation_quat": None,
            "detected_scale": None,
            "symmetric_rotations": None,
            "symmetric_locations": None,
            "individual_ts_reached_at_step": None,
            "individual_ts_object": None,
            "individual_ts_pose": None,
            "symmetric_rotations_ts": None,
            "time": [],
            "lm_processed_steps": [],  # List of booleans that records whether a step
            # of the Monty model is associated with processing by the learning
            # module (i.e. information was actually passed from the SM to the LM); note
            # this is incremented in a way that assumes a 1:1 mapping between SMs and
            # LMs
        }
        self.start_time = time.time()

    def __getitem__(self, idx):
        """Get features observed at time step idx.

        Returns:
            The features observed at time step idx.
        """
        features_at_idx = dict()
        for input_channel in self.features.keys():
            features_at_idx[input_channel] = {
                attribute: self.features[input_channel][attribute][idx]
                for attribute in self.features[input_channel].keys()
            }
        return features_at_idx

    def __len__(self):
        """Return the number of observations stored in the buffer."""
        return len(self.on_object)

    def get_buffer_len_by_channel(self, input_channel):
        """Return the number of observations stored for that input channel."""
        if input_channel not in self.locations.keys():
            return 0
        return np.count_nonzero(~np.isnan(self.locations[input_channel][:, 0]))

    def append(self, list_of_data):
        """Add an observation to the buffer. Must be features at locations.

        TODO S: Store state objects instead of list of data?
        A provisional version of this is implemented below, as the GSG uses State
        objects for computations.
        """
        any_obs_on_obj = False
        for state in list_of_data:
            input_channel = state.sender_id
            self._add_loc_to_location_buffer(input_channel, state.location)
            if input_channel not in self.features.keys():
                self.features[input_channel] = dict()
            for attr in state.morphological_features.keys():
                attr_val = state.morphological_features[attr]
                self._add_attr_to_feature_buffer(input_channel, attr, attr_val)
            # TODO S: separate non-morphological features from morphological features?
            # May cause problems with graph.x array representation. Could be added when
            # using separate models for features and morphology
            for attr in state.non_morphological_features.keys():
                attr_val = state.non_morphological_features[attr]
                self._add_attr_to_feature_buffer(input_channel, attr, attr_val)
            for attr in state.displacement.keys():
                attr_val = state.displacement[attr]
                self._add_disp_to_displacement_buffer(input_channel, attr, attr_val)
            on_obj = state.get_on_object()
            self._add_attr_to_feature_buffer(input_channel, "on_object", on_obj)
            if on_obj:
                any_obs_on_obj = True
        self.on_object.append(any_obs_on_obj)  # TODO S: remove?

    def append_input_states(self, input_state):
        self.input_states.append(input_state)

    def update_stats(self, stats, update_time=True, append=True, init_list=True):
        """Update statistics for this step in the episode."""
        for stat in stats.keys():
            if stat in self.stats.keys() and append:
                self.stats[stat].append(copy.deepcopy(stats[stat]))
            else:
                if init_list:
                    self.stats[stat] = [copy.deepcopy(stats[stat])]
                else:
                    self.stats[stat] = copy.deepcopy(stats[stat])
        if update_time:
            self.stats["time"].append(time.time() - self.start_time)

    def add_overall_stats(self, stats):
        """Add overall episode stats to self.stats."""
        self.update_stats(stats, update_time=False, append=False, init_list=False)

    def update_last_stats_entry(self, stats):
        """Use this to overwrite last entry (for example after voting)."""
        for stat in stats.keys():
            if stat in self.stats.keys():
                self.stats[stat][-1] = copy.deepcopy(stats[stat])

    def reset(self):
        """Reset the buffer."""
        self.__init__()

    def get_current_location(self, input_channel):
        """Get the current location.

        Note:
            May have to add on_object check at some point.

        Returns:
            The current location.
        """
        if input_channel == "first":
            input_channel = self.get_first_sensory_input_channel()
        if len(self) > 0 and input_channel is not None:
            return self.locations[input_channel][-1]
        else:
            return None

    def get_current_features(self, keys):
        """Get the current value of a specific feature.

        Returns:
            The current features.
        """
        current_features = {}
        for input_channel in self.features.keys():
            current_features[input_channel] = {}
            for key in keys:
                current_features[input_channel][key] = self.features[input_channel][
                    key
                ][-1]
        return current_features

    def get_current_pose(self, input_channel):
        """Get currently sensed location and orientation.

        Returns:
            The currently sensed location and orientation.
        """
        if input_channel == "first":
            input_channel = self.get_first_sensory_input_channel()
        sensed_pose_features = self.get_current_features(["pose_vectors"])
        sensed_location = self.get_current_location(input_channel)
        channel_pose = sensed_pose_features[input_channel]["pose_vectors"].reshape(
            (3, 3)
        )
        sensed_pose = np.vstack(
            [
                sensed_location,
                channel_pose,
            ]
        )
        return sensed_pose

    def get_last_obs_processed(self):
        """Check whether last sensation was processed by LM.

        Returns:
            Whether the last sensation was processed by the LM.
        """
        if len(self) > 0:
            return self.stats["lm_processed_steps"][-1]
        else:
            return False

    def get_currently_on_object(self):
        """Check whether last sensation was on object.

        Returns:
            Whether the last sensation was on object.
        """
        if len(self) > 0:
            return self.on_object[-1]
        else:
            return False

    def get_all_locations_on_object(self, input_channel=None):
        """Get all observed locations that were on the object.

        Returns:
            All observed locations that were on the object.
        """
        if input_channel is None:
            return self.locations
        if input_channel == "first":
            input_channel = self.get_first_sensory_input_channel()
        on_object_ids = np.where(self.features[input_channel]["on_object"])[0]
        return np.array(self.locations[input_channel])[on_object_ids]

    def get_all_input_states(self):
        """Get all the input states that the buffer's parent LM has observed.

        Returns:
            All the input states that the buffer's parent LM has observed.
        """
        return self.input_states

    def get_previous_input_states(self):
        """Get previous State inputs received by the buffer's parent LM.

        i.e. in the last time step.

        Returns:
            The previous input states.
        """
        if len(self.input_states) > 1:
            return self.input_states[-2]
        else:
            return None

    def get_nth_displacement(self, n, input_channel):
        """Get the nth displacement.

        Returns:
            The nth displacement.
        """
        if input_channel == "first":
            input_channel = self.get_first_sensory_input_channel()
        return self.displacements[input_channel]["displacement"][n]

    def get_current_displacement(self, input_channel):
        """Get the current displacement.

        Returns:
            The current displacement.
        """
        if input_channel == "all" or input_channel is None:
            all_disps = dict()
            for input_channel in self.displacements.keys():
                all_disps[input_channel] = self.get_current_displacement(input_channel)
            return all_disps
        else:
            return self.get_nth_displacement(-1, input_channel)

    def get_current_ppf(self, input_channel):
        """Get the current ppf.

        Returns:
            The current ppf.
        """
        if input_channel == "first":
            input_channel = self.get_first_sensory_input_channel()
        return copy.deepcopy(self.displacements[input_channel]["ppf"][-1])

    def get_first_displacement_len(self, input_channel):
        """Get length of first observed displacement.

        Use for scale in DisplacementLM.

        Returns:
            The length of the first observed displacement.
        """
        if input_channel == "first":
            input_channel = self.get_first_sensory_input_channel()
        if "ppf" in self.displacements[input_channel].keys():
            return self.displacements[input_channel]["ppf"][1][0]
        else:
            return np.linalg.norm(self.displacements[input_channel]["displacement"][1])

    def get_all_features_on_object(self):
        """Get all observed features that were on the object.

        Like in get_all_locations_on_object the feature arrays should have
        the length of np.where(self.on_object) and contain all features that
        were observed when at least one of the input channels was on the object.
        However, we also check for each feature whether its input channel indicated
        on_object=True and if not set its value to nan.

        Note:
            Since all inputs to an LM should have overlapping receptive fields,
            there should be no difference between the two at the moment (except due to
            noisy estimates of on_object).

        Returns:
            All observed features that were on the object.
        """
        all_features_on_object = dict()
        # Number of steps where at least one input was on the object
        global_on_object_ids = np.where(self.on_object)[0]
        logging.debug(
            f"Observed {np.array(self.locations).shape} locations, "
            f"{len(global_on_object_ids)} global on object"
        )
        for input_channel in self.features.keys():
            # Here we want to make sure the input specific obs was on the object
            channel_off_object_ids = np.where(
                self.features[input_channel]["on_object"] is False
            )[0]
            logging.debug(
                f"{input_channel} has "
                f"{len(self.locations) - len(channel_off_object_ids)} "
                "on object observations"
            )

            channel_features_on_object = dict()
            for feature in self.features[input_channel].keys():
                # Pad end of array with 0s if last steps of episode were off object
                # for this channel
                if self.features[input_channel][feature].shape[0] < len(self):
                    self.features[input_channel][feature].resize(
                        (
                            len(self),
                            self.features[input_channel][feature].shape[1],
                        ),
                        refcheck=False,
                    )
                logging.debug(
                    f"{input_channel} observations for feature {feature} have "
                    f"shape {np.array(self.features[input_channel][feature]).shape}"
                )
                # Get feature at each time step where the global on_object flag was
                # True (at least one input was on object, not nescessarily this
                # features input channel).
                channel_features_on_object[feature] = np.array(
                    self.features[input_channel][feature]
                )[global_on_object_ids]

            all_features_on_object[input_channel] = channel_features_on_object
        return all_features_on_object

    def get_num_observations_on_object(self):
        """Get the number of steps where at least 1 input was on the object.

        Returns:
            The number of steps where at least 1 input was on the object.
        """
        return len(self.on_object)

    def get_num_matching_steps(self):
        """Return number of matching steps performed by this LM since episode start.

        Returns:
            The number of matching steps performed by this LM since episode start.
        """
        # TODO: rename, this is including exploration steps
        return np.sum(self.stats["lm_processed_steps"])

    def get_num_goal_states_generated(self):
        """Return number of goal states generated by the LM's GSG since episode start.

        Note use of length not sum.
        """
        return len(self.stats["goal_state_achieved"])

    def get_matching_step_when_output_goal_set(self):
        """Return matching step when last goal-state was generated.

        Return the LM matching step associated with the last time a goal-state was
        generated.
        """
        return self.stats["matching_step_when_output_goal_set"][-1]

    def get_num_steps_post_output_goal_generated(self):
        """Return number of steps since last output goal-state.

        Return the number of Monty-matching steps that have elapsed since the last
        time an output goal-state was generated.
        """
        if self.get_num_goal_states_generated() == 0:
            # No previous jumps attempted, so this is just the number
            # of Monty matching steps that have taken place in the episode
            return self.get_num_matching_steps()

        else:
            return (
                self.get_num_matching_steps()
                - self.get_matching_step_when_output_goal_set()
            )

    def get_infos_for_graph_update(self):
        """Return all stored infos require to update a graph in memory."""
        infos = dict(
            locations=self.get_all_locations_on_object(),
            features=self.get_all_features_on_object(),
            object_location_rel_body=self.stats["detected_location_rel_body"],
            location_rel_model=self.stats["detected_location_on_model"],
            object_scale=self.stats["detected_scale"],
        )
        return infos

    def get_first_sensory_input_channel(self):
        """Get name of first sensory (coming from SM) input channel in buffer.

        Returns:
            The name of the first sensory (coming from SM) input channel in buffer.

        Raises:
            ValueError: If no sensor channels are found in the buffer
        """
        all_channels = list(self.locations.keys())
        if len(all_channels) > 0:
            for channel in all_channels:
                # TODO: better way of checking this that doesn't rely on naming. Maybe
                # store sensory_type together with channel when adding state to buffer?
                if "patch" in channel:
                    return channel
            raise ValueError(
                "No sensor channel found in buffer. "
                "get_first_sensory_input_channel assumes we have at least one"
                f" sensor channel but channels are {all_channels}."
            )
        else:
            return None

    def set_individual_ts(self, object_id, pose):
        """Update self.stats with the individual LMs terminal state."""
        # Only log first time terminal condition is met
        if self.stats["individual_ts_reached_at_step"] is None:
            self.stats["individual_ts_reached_at_step"] = self.get_num_matching_steps()
            self.stats["individual_ts_object"] = object_id
            self.stats["individual_ts_pose"] = pose
            self.stats["individual_ts_rot"] = self.stats["detected_rotation_quat"]
            # If no symmetry was detected, this will be None.
            self.stats["symmetric_rotations_ts"] = self.stats["symmetric_rotations"]

    def _add_attr_to_feature_buffer(self, input_channel, attr_name, attr_value):
        """Add attribute to feature buffer.

        Args:
            input_channel: Input channel from which the feature was received.
            attr_name: Name of the feature.
            attr_value: Value of the feature.
        """
        if isinstance(attr_value, (int, float, bool)):
            attr_shape = 1
        else:
            if isinstance(attr_value, np.ndarray):
                # Store all features as flat list so we can easily concatenate them and
                # perform matrix operations on them.
                attr_value = attr_value.flatten()
            attr_shape = len(attr_value)

        if attr_name not in self.features[input_channel].keys():
            # If the feature is not stored in buffer yet (i.e. when
            # an LM sends an object ID to a higher level LM for the first
            # time) we fill the array for this feature with nans up to
            # this time step and then add the sensed feature. This makes
            # sure the same index in different feature array corresponds to
            # the same time step and location.
            self.features[input_channel][attr_name] = (
                np.empty((len(self.locations), attr_shape)) * np.nan
            )
        else:
            padded_feat = self._fill_old_values_with_nans(
                existing_vals=self.features[input_channel][attr_name],
                new_val_len=attr_shape,
            )
            self.features[input_channel][attr_name] = padded_feat
        self.features[input_channel][attr_name][-1] = attr_value

    def _add_loc_to_location_buffer(self, input_channel, location):
        """Add location to location buffer.

        Args:
            input_channel: Input channel from which the location was received.
            location: Location to add to buffer.
        """
        if input_channel not in self.locations.keys():
            self.locations[input_channel] = np.empty((0, 0))

        padded_locs = self._fill_old_values_with_nans(
            existing_vals=self.locations[input_channel],
            new_val_len=location.shape[0],
        )
        self.locations[input_channel] = padded_locs
        self.locations[input_channel][-1] = location

    def _add_disp_to_displacement_buffer(self, input_channel, disp_name, disp_val):
        """Add displacement to displacement buffer.

        Args:
            input_channel: Input channel from which the displacement was received.
            disp_name: Name of the displacement. Currently in ["displacement", "ppf"]
            disp_val: Value of the displacement.
        """
        if input_channel not in self.displacements.keys():
            self.displacements[input_channel] = dict()
        if disp_name not in self.displacements[input_channel].keys():
            self.displacements[input_channel][disp_name] = (
                np.empty((len(self.locations), len(disp_val))) * np.nan
            )

        padded_vals = self._fill_old_values_with_nans(
            existing_vals=self.displacements[input_channel][disp_name],
            new_val_len=disp_val.shape[0],
        )
        self.displacements[input_channel][disp_name] = padded_vals
        self.displacements[input_channel][disp_name][-1] = disp_val

    def _fill_old_values_with_nans(self, existing_vals, new_val_len):
        """Pad existing values with nans to make sure indices align.

        If len(existing_vals)== len(buffer) this operation has no effect (besides adding
        a nan at the end which will be filled with the feature value in the next line
        making it equivalent to calling append).

        TODO O investigate whether pre-allocating large arrays (rather than
        repeatedly creating new ones) might be faster, despite a cost in memory

        Args:
            existing_vals: Values already stored in buffer.
            new_val_len: Shape (along first dim) of new values to be added.

        Returns:
            The padded values.
        """
        # create new np array filled with nans of size (current_step, new_val_len)
        new_vals = np.empty((len(self) + 1, new_val_len)) * np.nan
        # Replace nans with stored values for this feature.
        # existing_feat has shape (last_stored_step, attr_shape)
        new_vals[: existing_vals.shape[0], : existing_vals.shape[1]] = existing_vals
        return new_vals


class BufferEncoder(json.JSONEncoder):
    """Encoder to turn the buffer into a JSON compliant format."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.action_encoder = ActionJSONEncoder(**kwargs)

    def default(self, obj):
        """Turn non compliant types into right format.

        Args:
            obj: The object to turn into a JSON compliant format.

        Returns:
            The object in a JSON compliant format.
        """
        if isinstance(obj, torch.Tensor):
            return obj.cpu().numpy()
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, quaternion.quaternion):
            return quaternion.as_float_array(obj)
        if isinstance(obj, Rotation):
            return obj.as_euler("xyz", degrees=True)
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, Action):
            return self.action_encoder.default(obj)
        # if isinstance(obj, magnum.Vector3):
        #     return list(obj)
        return json.JSONEncoder.default(self, obj)
