import numpy as np
import matplotlib.pyplot as plt

def plot_results(rewards: list, actions_done: list, areas_scratched: list,
                 max_rewards: list, min_actions_done: list, min_areas_scratched: list, 
                 EPISODES: int, file_name: str):
    """
    Plot the results of the training process. Includes 3 graphics: Rewards, Actions Done and Area Scratched.
    
    Each graphic highlights the best and worst values obtained during the training process according to the metric.
    """
    plt.figure(figsize=(18, 12))

    # --- Graphic 1: Rewards ---
    plt.subplot(3, 1, 1)
    plt.plot(rewards, label='Rewards')
    plt.plot(max_rewards, label='Max Rewards')
    max_reward_idx = np.argmax(max_rewards)
    min_reward_idx = np.argmin(rewards)

    # vertical lines
    plt.axvline(x=max_reward_idx, color='green', linestyle='--', alpha=0.7)
    plt.axvline(x=min_reward_idx, color='red', linestyle='--', alpha=0.7)

    # optimal values
    plt.text(max_reward_idx, plt.ylim()[0] - (plt.ylim()[1] - plt.ylim()[0]) * 0.1, 
            f'{max_rewards[max_reward_idx]} (X={max_reward_idx})', 
            color='green', fontsize=10, fontweight='bold', ha='center')
    plt.text(min_reward_idx, plt.ylim()[0] - (plt.ylim()[1] - plt.ylim()[0]) * 0.1, 
            f'{max_rewards[min_reward_idx]} (X={min_reward_idx})', 
            color='red', fontsize=10, fontweight='bold', ha='center')
    plt.xlabel('Episodes')
    plt.ylabel('Rewards')
    plt.title('Rewards vs Episodes')
    plt.legend()

    # --- Graphic 2: Actions Done ---
    plt.subplot(3, 1, 2)
    plt.plot(actions_done, label='Actions Done')
    plt.plot(min_actions_done, label='Min Actions Done')
    max_actions_idx = np.argmax(actions_done)
    min_actions_idx = np.argmin(min_actions_done)

    # vertical lines
    plt.axvline(x=max_actions_idx, color='red', linestyle='--', alpha=0.7)
    plt.axvline(x=min_actions_idx, color='green', linestyle='--', alpha=0.7)

    # optimal values
    plt.text(max_actions_idx, plt.ylim()[0] - (plt.ylim()[1] - plt.ylim()[0]) * 0.1, 
            f'{min_actions_done[max_actions_idx]} (X={max_actions_idx})', 
            color='red', fontsize=10, fontweight='bold', ha='center')
    plt.text(min_actions_idx, plt.ylim()[0] - (plt.ylim()[1] - plt.ylim()[0]) * 0.1, 
            f'{min_actions_done[min_actions_idx]} (X={min_actions_idx})', 
            color='green', fontsize=10, fontweight='bold', ha='center')
    plt.xlabel('Episodes')
    plt.ylabel('Actions Done')
    plt.title('Actions Done vs Episodes')
    plt.legend()

    # --- Graphic 3: Area Scratched ---
    plt.subplot(3, 1, 3)
    plt.plot(areas_scratched, label='Area Scratched')
    plt.plot(min_areas_scratched, label='Min Area Scratched')
    max_area_idx = np.argmax(areas_scratched)
    min_area_idx = np.argmin(min_areas_scratched)

    # vertical lines
    plt.axvline(x=max_area_idx, color='red', linestyle='--', alpha=0.7)
    plt.axvline(x=min_area_idx, color='green', linestyle='--', alpha=0.7)

    # optimal values
    plt.text(max_area_idx, plt.ylim()[0] - (plt.ylim()[1] - plt.ylim()[0]) * 0.1, 
            f'{min_areas_scratched[max_area_idx]:.2f}% (X={max_area_idx})', 
            color='red', fontsize=10, fontweight='bold', ha='center')
    plt.text(min_area_idx, plt.ylim()[0] - (plt.ylim()[1] - plt.ylim()[0]) * 0.1, 
            f'{min_areas_scratched[min_area_idx]:.2f}% (X={min_area_idx})', 
            color='green', fontsize=10, fontweight='bold', ha='center')
    plt.xlabel('Episodes')
    plt.ylabel('Area Scratched')
    plt.title('Area Scratched vs Episodes')
    plt.legend()

    plt.tight_layout()
    plt.savefig(file_name) # in folder results
    plt.close()