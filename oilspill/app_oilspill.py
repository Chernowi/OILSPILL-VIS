import streamlit as st
import os
import json
import shutil
import time
from typing import Optional, List, Dict, Tuple, Any

# Import from the newly created simulation bundle for oil spill
from simulation_bundle_oilspill import (
    AppSimConfig, WorldSimConfig, SACSimConfig, PPOSimConfig,
    MapperSimConfig, VisualizationSimConfig,
    LocationSimConfig, RandomizationRangeSimConfig,
    run_simulation_for_streamlit
)

# Import original DefaultConfig to help parse existing experiment configs
from configs import DefaultConfig

EXPERIMENTS_DIR = "oilspill/experiments"  # Standard directory for trained models
TEMP_VIS_DIR_BASE = "oilspill/streamlit_temp_vis_oilspill" # Unique temp dir

BASE_AGENT_TYPES = {
    "SAC_MLP": "SAC MLP (Most Recent)",
    "SAC_RNN": "SAC RNN (Most Recent)",
    "PPO_MLP": "PPO MLP (Most Recent)",
    "PPO_RNN": "PPO RNN (Most Recent)",
}

st.set_page_config(layout="wide")

def get_experiment_details() -> List[Dict[str, Any]]:
    experiment_details_list = []
    if not os.path.isdir(EXPERIMENTS_DIR):
        return []

    for exp_dir_name in os.listdir(EXPERIMENTS_DIR):
        exp_path = os.path.join(EXPERIMENTS_DIR, exp_dir_name)
        if not os.path.isdir(exp_path):
            continue

        config_json_path = os.path.join(exp_path, "config.json")
        
        # --- MODIFIED MODEL PATH LOGIC ---
        # Look for model.pt directly in the experiment folder
        model_path_to_use = None
        pt_files_in_exp_dir = [f for f in os.listdir(exp_path) if f.endswith(".pt")]
        
        if pt_files_in_exp_dir:
            if "model.pt" in pt_files_in_exp_dir: # Prioritize "model.pt" if it exists
                model_path_to_use = os.path.join(exp_path, "model.pt")
            else:
                # If "model.pt" is not present, take the most recent .pt file
                pt_files_in_exp_dir.sort(key=lambda f: os.path.getmtime(os.path.join(exp_path, f)), reverse=True)
                model_path_to_use = os.path.join(exp_path, pt_files_in_exp_dir[0])
        # --- END MODIFIED MODEL PATH LOGIC ---
        
        if os.path.exists(config_json_path) and model_path_to_use:
            try:
                with open(config_json_path, 'r') as f:
                    config_data_dict = json.load(f)
                
                # Use original DefaultConfig to parse the full structure easily
                parsed_config = DefaultConfig(**config_data_dict)
                
                algo = parsed_config.algorithm.lower()
                use_rnn = False
                if algo == "sac": use_rnn = parsed_config.sac.use_rnn
                elif algo == "ppo": use_rnn = parsed_config.ppo.use_rnn
                else: print(f"Warning: Unknown algorithm '{algo}' in {exp_dir_name}")

                agent_key = f"{algo.upper()}_{'RNN' if use_rnn else 'MLP'}"
                
                experiment_details_list.append({
                    "dir_name": exp_dir_name,
                    "full_path": exp_path,
                    "config_path": config_json_path, 
                    "config_data": parsed_config,    
                    "model_path": model_path_to_use,
                    "algorithm": algo,
                    "use_rnn": use_rnn,
                    "agent_key": agent_key,
                    "timestamp": os.path.getmtime(exp_path) 
                })
            except Exception as e:
                print(f"Error processing experiment {exp_dir_name}: {e}")
    
    experiment_details_list.sort(key=lambda x: x["timestamp"], reverse=True)
    return experiment_details_list


def find_representative_experiment(
    all_experiments: List[Dict[str, Any]], 
    target_agent_key: str
) -> Optional[Dict[str, Any]]:
    for exp_detail in all_experiments:
        if exp_detail["agent_key"] == target_agent_key:
            return exp_detail 
    return None


def main():
    st.title("Oil Spill Mapping RL Agent Simulation")

    all_exp_details = get_experiment_details()
    if not all_exp_details:
        st.error(f"No valid experiments (with config.json and a .pt model file) found in '{EXPERIMENTS_DIR}'.")
        return

    available_agent_display_names = []
    display_name_to_key_map = {} 
    
    for key, display_name in BASE_AGENT_TYPES.items():
        if any(exp["agent_key"] == key for exp in all_exp_details):
            available_agent_display_names.append(display_name)
            display_name_to_key_map[display_name] = key

    if not available_agent_display_names:
        st.error("No experiments found matching the predefined base agent types (SAC/PPO, MLP/RNN).")
        return

    st.sidebar.header("Simulation Setup")
    selected_display_name = st.sidebar.selectbox("Select Agent Type", available_agent_display_names)

    if not selected_display_name:
        st.info("Please select an agent type to configure the simulation.")
        return
        
    selected_agent_key = display_name_to_key_map[selected_display_name]
    representative_exp = find_representative_experiment(all_exp_details, selected_agent_key)

    if not representative_exp:
        st.sidebar.error(f"Could not find a representative experiment for {selected_display_name}.")
        return

    original_exp_config: DefaultConfig = representative_exp["config_data"]
    model_file_path = representative_exp["model_path"]
    
    st.sidebar.info(f"Selected: {selected_display_name}")
    st.sidebar.caption(f"Using experiment: {representative_exp['dir_name']}")
    st.sidebar.caption(f"Model: {os.path.relpath(model_file_path, EXPERIMENTS_DIR)}")

    st.sidebar.subheader("Initial Conditions")
    
    seed_options = ["Random Seed"] + [str(s) for s in original_exp_config.world.seeds]
    if original_exp_config.world.seeds: 
        seed_options.append("Cycle From List")
    
    selected_seed_option = st.sidebar.selectbox("Environment Seed", seed_options, index=0)
    
    specific_seed_input: Optional[int] = None
    if selected_seed_option == "Random Seed":
        specific_seed_input = None 
    elif selected_seed_option == "Cycle From List":
        specific_seed_input = -1 
    else: 
        try: specific_seed_input = int(selected_seed_option)
        except ValueError: st.sidebar.warning("Invalid seed selected, defaulting to random."); specific_seed_input = None

    use_config_initials = st.sidebar.checkbox("Use Initial Positions from Config", value=True)

    agent_x_init_override, agent_y_init_override = None, None
    oil_center_x_override, oil_center_y_override = None, None
    oil_std_dev_override = None

    if not use_config_initials:
        st.sidebar.markdown("---")
        st.sidebar.markdown("**Agent Initial Position (Override):**")
        agent_x_init_override = st.sidebar.number_input("Agent X", value=original_exp_config.world.agent_initial_location.x, step=1.0, format="%.1f")
        agent_y_init_override = st.sidebar.number_input("Agent Y", value=original_exp_config.world.agent_initial_location.y, step=1.0, format="%.1f")
        st.sidebar.markdown("**Oil Spill Initial Setup (Override):**")
        oil_center_x_override = st.sidebar.number_input("Oil Center X", value=original_exp_config.world.initial_oil_center.x, step=1.0, format="%.1f")
        oil_center_y_override = st.sidebar.number_input("Oil Center Y", value=original_exp_config.world.initial_oil_center.y, step=1.0, format="%.1f")
        oil_std_dev_override = st.sidebar.number_input("Oil Std Dev", value=original_exp_config.world.initial_oil_std_dev, step=0.1, format="%.1f")
        st.sidebar.markdown("---")

    default_steps = original_exp_config.evaluation.max_steps or 200
    num_steps = st.sidebar.slider("Number of Simulation Steps", 50, 500, default_steps)
    
    world_sim_cfg = WorldSimConfig(
        CORE_STATE_DIM=original_exp_config.world.CORE_STATE_DIM,
        CORE_ACTION_DIM=original_exp_config.world.CORE_ACTION_DIM,
        TRAJECTORY_REWARD_DIM=original_exp_config.world.TRAJECTORY_REWARD_DIM,
        dt=original_exp_config.world.dt,
        world_size=original_exp_config.world.world_size,
        normalize_coords=original_exp_config.world.normalize_coords,
        agent_speed=original_exp_config.world.agent_speed,
        yaw_angle_range=original_exp_config.world.yaw_angle_range,
        num_sensors=original_exp_config.world.num_sensors,
        sensor_distance=original_exp_config.world.sensor_distance,
        sensor_radius=original_exp_config.world.sensor_radius,
        agent_initial_location=LocationSimConfig(**original_exp_config.world.agent_initial_location.model_dump()),
        randomize_agent_initial_location=use_config_initials and original_exp_config.world.randomize_agent_initial_location,
        agent_randomization_ranges=RandomizationRangeSimConfig(**original_exp_config.world.agent_randomization_ranges.model_dump()),
        num_oil_points=original_exp_config.world.num_oil_points,
        num_water_points=original_exp_config.world.num_water_points,
        oil_cluster_std_dev_range=original_exp_config.world.oil_cluster_std_dev_range,
        randomize_oil_cluster=use_config_initials and original_exp_config.world.randomize_oil_cluster,
        oil_center_randomization_range=RandomizationRangeSimConfig(**original_exp_config.world.oil_center_randomization_range.model_dump()),
        initial_oil_center=LocationSimConfig(**original_exp_config.world.initial_oil_center.model_dump()),
        initial_oil_std_dev=original_exp_config.world.initial_oil_std_dev,
        min_initial_separation_distance=original_exp_config.world.min_initial_separation_distance,
        trajectory_length=original_exp_config.world.trajectory_length,
        trajectory_feature_dim=original_exp_config.world.trajectory_feature_dim,
        success_metric_threshold=original_exp_config.world.success_metric_threshold,
        terminate_on_success=original_exp_config.world.terminate_on_success,
        terminate_out_of_bounds=original_exp_config.world.terminate_out_of_bounds,
        metric_improvement_scale=original_exp_config.world.metric_improvement_scale,
        step_penalty=original_exp_config.world.step_penalty,
        new_oil_detection_bonus=original_exp_config.world.new_oil_detection_bonus,
        out_of_bounds_penalty=original_exp_config.world.out_of_bounds_penalty,
        success_bonus=original_exp_config.world.success_bonus,
        uninitialized_mapper_penalty=original_exp_config.world.uninitialized_mapper_penalty,
        mapper_config=MapperSimConfig(**original_exp_config.mapper.model_dump()), 
        seeds=original_exp_config.world.seeds if selected_seed_option == "Cycle From List" else [] 
    )

    sac_sim_cfg = None; ppo_sim_cfg = None
    if original_exp_config.algorithm.lower() == "sac":
        sac_orig = original_exp_config.sac
        sac_sim_cfg = SACSimConfig(
            state_dim=sac_orig.state_dim, action_dim=sac_orig.action_dim,
            hidden_dims=sac_orig.hidden_dims, log_std_min=sac_orig.log_std_min,
            log_std_max=sac_orig.log_std_max, use_rnn=sac_orig.use_rnn,
            rnn_type=sac_orig.rnn_type, rnn_hidden_size=sac_orig.rnn_hidden_size,
            rnn_num_layers=sac_orig.rnn_num_layers
        )
    elif original_exp_config.algorithm.lower() == "ppo":
        ppo_orig = original_exp_config.ppo
        ppo_sim_cfg = PPOSimConfig(
            state_dim=ppo_orig.state_dim, action_dim=ppo_orig.action_dim,
            hidden_dim=ppo_orig.hidden_dim, log_std_min=ppo_orig.log_std_min,
            log_std_max=ppo_orig.log_std_max, use_rnn=ppo_orig.use_rnn,
            rnn_type=ppo_orig.rnn_type, rnn_hidden_size=ppo_orig.rnn_hidden_size,
            rnn_num_layers=ppo_orig.rnn_num_layers
        )

    app_sim_config = AppSimConfig(
        sac=sac_sim_cfg, ppo=ppo_sim_cfg, world=world_sim_cfg,
        visualization=VisualizationSimConfig(**original_exp_config.visualization.model_dump()),
        cuda_device="cpu", 
        algorithm=original_exp_config.algorithm
    )
    
    if st.sidebar.button("Run Simulation"):
        run_timestamp = time.strftime("%Y%m%d-%H%M%S")
        sane_exp_dir_name = "".join(c if c.isalnum() else "_" for c in representative_exp['dir_name'])
        current_run_vis_dir = os.path.join(TEMP_VIS_DIR_BASE, f"{sane_exp_dir_name}_{selected_agent_key}_{run_timestamp}")
        os.makedirs(current_run_vis_dir, exist_ok=True)

        st.markdown("---"); st.subheader("Simulation Results")
        status_text = st.empty(); gif_display = st.empty()
        metrics_display = st.empty(); reward_plot_display = st.empty()
        status_text.info("Running simulation, please wait...")

        agent_loc_override_tuple = (agent_x_init_override, agent_y_init_override) if not use_config_initials and agent_x_init_override is not None else None
        oil_center_override_tuple = (oil_center_x_override, oil_center_y_override) if not use_config_initials and oil_center_x_override is not None else None

        try:
            gif_path, final_metric, total_reward, episode_rewards = run_simulation_for_streamlit(
                app_sim_config=app_sim_config, model_path=model_file_path,
                initial_agent_loc_data=agent_loc_override_tuple,
                initial_oil_center_data=oil_center_override_tuple,
                initial_oil_std_dev_data=oil_std_dev_override if not use_config_initials else None,
                seed_data=specific_seed_input if selected_seed_option != "Cycle From List" else None, 
                num_simulation_steps=num_steps,
                visualization_output_dir=current_run_vis_dir
            )
            status_text.success("Simulation complete!")
            if gif_path and os.path.exists(gif_path): gif_display.image(gif_path, caption="Simulation Animation")
            else: gif_display.warning("GIF could not be generated or found.")
            
            metrics_col1, metrics_col2 = st.columns(2)
            metrics_col1.metric(label="Final Performance Metric (Point Inclusion %)", value=f"{final_metric*100:.2f}%")
            metrics_col2.metric(label="Total Accumulated Reward", value=f"{total_reward:.2f}")

            if episode_rewards:
                import pandas as pd
                reward_df = pd.DataFrame({'Step': range(len(episode_rewards)), 'Step Reward': episode_rewards})
                st.markdown("#### Step Rewards Over Time")
                st.line_chart(reward_df.set_index('Step'))
        except Exception as e:
            status_text.error(f"Simulation error: {e}"); import traceback; st.error(traceback.format_exc())
        finally:
            if os.path.exists(current_run_vis_dir):
                try: shutil.rmtree(current_run_vis_dir)
                except Exception as e_clean: print(f"Warning: Cleanup failed for {current_run_vis_dir}: {e_clean}")

if __name__ == "__main__":
    if not os.path.exists(TEMP_VIS_DIR_BASE): os.makedirs(TEMP_VIS_DIR_BASE)
    main()