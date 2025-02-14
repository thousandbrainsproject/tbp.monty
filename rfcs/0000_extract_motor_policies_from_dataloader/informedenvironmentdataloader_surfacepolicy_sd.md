```mermaid
sequenceDiagram
    participant E as entrypoint
    participant SP as SurfacePolicy

    activate E

    participant EDL as EnvironmentDataLoader
    participant EDLO as EnvironmentDataLoaderPerObject

    create participant IEDL as InformedEnvironmentDataLoader
    E ->>+ IEDL : __init__(dataset, motor_system, ...)
    IEDL ->>+ EDLO : __init__(dataset, motor_system, ...)
    EDLO ->>+ EDL : __init__(dataset, motor_system, ...)
    participant OIS as object_init_sampler
    participant DS as Dataset
    participant ENV as Environment
    EDL ->>+ DS : reset
    DS -->>- EDL : observation, state
    EDL ->> EDL : self._observation = observation
    EDL ->> SP : self.state = state
    EDL -->>- EDLO : ...
    EDLO ->>+ EDLO : create_semantic_mapping
    deactivate EDLO
    EDLO ->>+ OIS : __call__
    OIS -->>- EDLO : object_params
    EDLO -->>- IEDL : ...
    IEDL -->>- E : ...

    E ->>+ IEDL : pre_episode
    IEDL ->>+ EDLO : pre_episode
    EDLO ->>+ EDL : pre_episode
    EDL ->>+ SP : pre_episode
    SP -->>- EDL : ...
    EDL -->>- EDLO : ...
    EDLO ->>+ EDLO : reset_agent
    EDLO ->>+ DS : reset
    DS -->>- EDLO : observation, state
    EDLO ->> EDLO : self._observation = observation
    EDLO ->> SP : self.state = state
    EDLO ->>+ SP : agent_id
    SP -->>- EDLO : agent_id
    EDLO ->> SP : self.state[agent_id]["motor_only_step"] = False
    deactivate EDLO
    EDLO -->>- IEDL : ...
    IEDL ->>+ DS : env
    DS -->>- IEDL : env
    IEDL ->>+ ENV : _agents[0].action_space_type
    ENV -->>- IEDL : action_space_type
    opt action_space_type == "surface_agent"
        IEDL ->>+ IEDL : get_good_view_with_patch_refinement
            IEDL ->>+ IEDL : get_good_view("view_finder")
            NOTE right of IEDL : Bookmark get_good_view
                opt multiple_objects_present
                    IEDL ->>+ SP : is_on_target_object
                    SP -->>- IEDL : on_target_object
                    opt not on_target_object
                        IEDL ->>+ SP : orient_to_object
                        SP -->>- IEDL : actions
                        loop action in actions
                            IEDL ->>+ DS : __getitem__(action)
                            DS -->>- IEDL : observation, state
                            IEDL ->> IEDL : self._observation = observation
                            IEDL ->> SP : self.state = state
                        end
                    end
                end
                opt allow_translation
                    IEDL ->>+ SP : move_close_enough
                    SP -->>- IEDL : action, close_enough
                    loop not close_enough
                        IEDL ->>+ DS : __getitem__(action)
                        DS -->>- IEDL : observation, state
                        IEDL ->> IEDL : self._observation = observation
                        IEDL ->> SP : self.state = state
                        IEDL ->>+ SP : move_close_enough
                        SP -->>- IEDL : action, close_enough
                    end
                end
                IEDL ->>+ SP : is_on_target_object
                SP -->>- IEDL : on_target_object
                loop not on_target_object and num_attempts < max_orientation_attempts
                    IEDL ->>+ SP : orient_to_object
                    SP -->>- IEDL : actions
                    loop action in actions
                        IEDL ->>+ DS : __getitem__(action)
                        DS -->>- IEDL : observation, state
                        IEDL ->> IEDL : self._observation = observation
                        IEDL ->> SP : self.state = state
                    end
                    IEDL ->>+ SP : is_on_target_object
                    SP -->>- IEDL : on_target_object
                end
            deactivate IEDL
            loop patch_id in ("patch", "patch_0")
                opt patch_id in self._observation["agent_id_0"].keys()
                    IEDL ->>+ IEDL : get_good_view
                    NOTE right of IEDL : See get_good_view above
                    deactivate IEDL
                end
            end
        deactivate IEDL
    end
    IEDL -->>- E : ...

    E ->>+ IEDL : __iter__
    IEDL -->>- E : ...

    E ->>+ IEDL : __next__
    opt self._counter == 0
        IEDL ->>+ IEDL : first_step
            IEDL ->>+ SP : agent_id
            SP -->>- IEDL : agent_id
            IEDL ->> SP : self.state[agent_id]["motor_only_step"] = True
        deactivate IEDL
        break early return
            IEDL -->> E : self._observation
        end
    end
    IEDL ->>+ SP : use_goal_state_driven_actions
    SP -->>- IEDL : use_goal_state_driven_actions
    IEDL ->>+ SP : driving_goal_state
    SP -->>- IEDL : driving_goal_state
    opt use_goal_state_driven_actions and driving_goal_state is not None
        IEDL ->>+ IEDL : execute_jump_attempt
            IEDL ->>+ SP : agent_id
            SP -->>- IEDL : agent_id
            IEDL ->>+ SP : self.state[agend_id]
            SP -->>- IEDL : pre_jump_state
            IEDL ->>+ SP : derive_habitat_goal_state
            SP -->>- IEDL : target_loc, target_np_quat
            IEDL ->>+ DS : __getitem__(set_agent_pose)
            DS -->>- IEDL : observation, state
            IEDL ->> IEDL : self._observation = observation
            IEDL ->> SP : self.state = state
            IEDL ->>+ DS : __getitem__(set_agent_rotation)
            DS -->>- IEDL : observation, state
            IEDL ->> IEDL : self._observation = observation
            IEDL ->> SP : self.state = state
            IEDL ->>+ SP : get_depth_at_center
            SP -->>- IEDL : depth_at_center
            alt depth_at_center < 1.0
                IEDL ->>+ IEDL : handle_successful_jump
                    IEDL ->>+ SP : agent_id
                    SP -->>- IEDL : agent_id
                    IEDL ->> SP : self.action = MoveTangentially
                    IEDL ->>+ SP : action_details
                    SP -->>- IEDL : action_details
                    IEDL ->> SP : self.action_details["pc_heading"].append("jump")
                    IEDL ->> SP : self.action_details["avoidance_heading"].append(False)
                    IEDL ->> SP : self.action_details["z_defined_pc"].append(None)
                deactivate IEDL
            else
                IEDL ->>+ IEDL : handle_failed_jump
                    IEDL ->>+ DS : __getitem__(set_agent_pose)
                    DS -->>- IEDL : observation, state
                    IEDL ->> IEDL : self._observation = observation
                    IEDL ->> SP : self.state = state
                    IEDL ->>+ DS : __getitem__(set_sensor_rotation)
                    DS -->>- IEDL : observation, state
                    IEDL ->> IEDL : self._observation = observation
                    IEDL ->> SP : self.state = state
                    IEDL ->>+ SP : agent_id
                    SP -->>- IEDL : agent_id
                    IEDL ->>+ SP : self.state[agent_id]["position"]
                    SP -->>- IEDL : position
                    IEDL ->>+ SP : self.state[agent_id]["rotation"]
                    SP -->>- IEDL : rotation
                    IEDL ->>+ SP : self.state[agent_id]["sensors"]
                    SP -->>- IEDL : sensors
                deactivate IEDL
            end
            IEDL ->>+ SP : agent_id
            SP -->>- IEDL : agent_id
            IEDL ->> SP : self.state[agent_id]["motor_only_step"] = True
            IEDL ->>+ SP : action
            SP -->>- IEDL : action
            IEDL ->>+ SP : post_action(action)
            SP -->>- IEDL : ...
        deactivate IEDL
        break early return
            IEDL -->> E : self._observation
        end
    end
    IEDL ->>+ SP : __call__
    opt self._action is None
        IEDL ->>+ SP : touch_object
        SP -->>- IEDL : action
        IEDL ->>+ SP : agent_id
        SP -->>- IEDL : agent_id
        IEDL ->> SP : self.state[agent_id]["motor_only_step"] = True
    end
    IEDL ->>+ DS : __getitem__(self._action)
    DS -->>- IEDL : observation, state
    IEDL ->> IEDL : self._observation = observation
    IEDL ->> SP : self.state = state
    alt self._action.name != "orient_vertical"
        IEDL ->>+ SP : agent_id
        SP -->>- IEDL : agent_id
        IEDL ->> SP : self.state[agent_id]["motor_only_step"] = True
    else
        IEDL ->>+ SP : agent_id
        SP -->>- IEDL : agent_id
        IEDL ->> SP : self.state[agent_id]["motor_only_step"] = False
    end
    IEDL -->> E : self._observation
    deactivate IEDL

    deactivate E
```