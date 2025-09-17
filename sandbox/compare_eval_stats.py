"""File that compares the eval_stats.csv of two experiments.

The two experiments are:
- base_77obj_surf_agent_hyp-1
- base_77obj_surf_agent_hyp-100

For each experiment, it prints a summary of the misclassified episodes.

It should also note whether the "hyp-100" experiment has any additional
misclassified episodes compared to the "hyp-1" experiment.
"""

from pathlib import Path
import pandas as pd
import numpy as np

RESULTS_DIR = Path("~/tbp/results/monty/projects/evidence_eval_runs/logs").expanduser()


def get_misclassified_episodes(stats: pd.DataFrame) -> pd.DataFrame:
    """Get the misclassified episodes from the stats dataframe.

    Args:
        stats: The stats dataframe.

    Returns:
        A dataframe of misclassified episodes with the following columns:
        - episode_id: The episode ID.
        - primary_performance: The primary performance of the episode.
        - primary_target_object: The primary target object of the episode.
        - result: Monty's detected object.
        - monty_steps
        - monty_matching_steps
        - primary_target_position
        - primary_target_rotation_euler
    """
    stats["episode_id"] = stats.index
    subset = stats[~stats["primary_performance"].isin(["correct", "correct_mlh"])]

    columns = [
        "episode_id",
        "primary_performance",
        "primary_target_object",
        "result",
        "monty_steps",
        "monty_matching_steps",
        "primary_target_position",
        "primary_target_rotation_euler",
    ]
    return subset[columns]


def get_accuracy(stats: pd.DataFrame) -> float:
    """Get the accuracy of the stats dataframe.

    Args:
        stats: The stats dataframe.
    """

    return len(
        stats[stats["primary_performance"].isin(["correct", "correct_mlh"])]
    ) / len(stats)

def get_average_monty_steps(stats: pd.DataFrame) -> float:
    """Get the average monty steps of the stats dataframe.

    Args:
        stats: The stats dataframe.
    """
    return np.mean(stats["monty_steps"])

def get_average_monty_matching_steps(stats: pd.DataFrame) -> float:
    """Get the average monty matching steps of the stats dataframe.

    Args:
        stats: The stats dataframe.
    """
    return np.mean(stats["monty_matching_steps"])

if __name__ == "__main__":
    # exp_1_dir = RESULTS_DIR / "77_base_surf_agent" / "base_77obj_surf_agent_hyp1_rerun"
    # exp_2_dir = RESULTS_DIR / "77_base_surf_agent" / "base_77obj_surf_agent_hyp100_rerun"

    exp_1_dir = RESULTS_DIR / "77_randrot_noise_surf_agent" / "randrot_noise_77obj_surf_agent_hyp-1"
    exp_2_dir = RESULTS_DIR / "77_randrot_noise_surf_agent" / "randrot_noise_77obj_surf_agent_hyp-100"
    
    exp_1_stats = pd.read_csv(exp_1_dir / "eval_stats.csv")
    exp_2_stats = pd.read_csv(exp_2_dir / "eval_stats.csv")

    exp_1_misclassified = get_misclassified_episodes(exp_1_stats)
    exp_2_misclassified = get_misclassified_episodes(exp_2_stats)
    exp_1_accuracy = get_accuracy(exp_1_stats)
    exp_2_accuracy = get_accuracy(exp_2_stats)
    exp_1_average_monty_steps = get_average_monty_steps(exp_1_stats)
    exp_2_average_monty_steps = get_average_monty_steps(exp_2_stats)
    exp_1_average_monty_matching_steps = get_average_monty_matching_steps(exp_1_stats)
    exp_2_average_monty_matching_steps = get_average_monty_matching_steps(exp_2_stats)

    print("------------------------------------------------------")
    print("Misclassified episodes for base_77obj_surf_agent_hyp1_rerun:")
    print("------------------------------------------------------")
    print(f"Accuracy: {exp_1_accuracy * 100:.2f}%")
    print(f"Average monty steps: {exp_1_average_monty_steps:.2f}")
    print(f"Average monty matching steps: {exp_1_average_monty_matching_steps:.2f}")
    print(exp_1_misclassified)

    print("\n")
    print("-------------------------------------------------------")
    print("Misclassified episodes for base_77obj_surf_agent_hyp100_rerun:")
    print("-------------------------------------------------------")
    print(f"Accuracy: {exp_2_accuracy * 100:.2f}%")
    print(f"Average monty steps: {exp_2_average_monty_steps:.2f}")
    print(f"Average monty matching steps: {exp_2_average_monty_matching_steps:.2f}")
    print(exp_2_misclassified)
