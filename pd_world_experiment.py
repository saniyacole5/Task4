"""
Task 4: Two-Agent Reinforcement Learning in PD-World
Experiment 1: Q-Learning with different policies
"""

import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import copy
import random

class PDWorld:
    """
    Pickup-Dropoff World Environment
    5x5 grid with two agents (M and F) that transport blocks
    """
    
    def __init__(self, pickup_locations=None, dropoff_locations=None):
        self.grid_size = 5
        
        # Default locations based on Figure 1 in the document
        if pickup_locations is None:
            self.pickup_locations = [(0, 1), (3, 1)]  # Two pickup locations
        else:
            self.pickup_locations = pickup_locations
            
        if dropoff_locations is None:
            self.dropoff_locations = [(0, 4), (2, 4), (4, 2), (4, 4)]  # Four dropoff locations
        else:
            self.dropoff_locations = dropoff_locations
        
        # Initialize dropoff capacities (each can hold 5 blocks)
        self.dropoff_capacity = {loc: 5 for loc in self.dropoff_locations}
        
        # Agent states: [row, col, has_block]
        self.agent_m = [0, 0, False]  # Male agent
        self.agent_f = [0, 0, False]  # Female agent
        
        self.step_count = 0
        self.terminal_count = 0
        
    def reset(self):
        """Reset the world to initial state (but don't reset Q-table)"""
        self.agent_m = [0, 0, False]
        self.agent_f = [0, 0, False]
        self.dropoff_capacity = {loc: 5 for loc in self.dropoff_locations}
        return self.get_state()
    
    def get_state(self):
        """
        Return current state representation
        State: (agent_m_pos, agent_m_has_block, agent_f_pos, agent_f_has_block)
        """
        return (
            (self.agent_m[0], self.agent_m[1]),
            self.agent_m[2],
            (self.agent_f[0], self.agent_f[1]),
            self.agent_f[2]
        )
    
    def get_applicable_actions(self, agent_name):
        """
        Get list of applicable actions for an agent
        Actions: 'up', 'down', 'left', 'right', 'pickup', 'dropoff'
        """
        agent = self.agent_m if agent_name == 'M' else self.agent_f
        other_agent = self.agent_f if agent_name == 'M' else self.agent_m
        
        actions = []
        row, col, has_block = agent
        
        # Movement actions (check boundaries and other agent position)
        if row > 0 and (row-1, col) != (other_agent[0], other_agent[1]):
            actions.append('up')
        if row < self.grid_size - 1 and (row+1, col) != (other_agent[0], other_agent[1]):
            actions.append('down')
        if col > 0 and (row, col-1) != (other_agent[0], other_agent[1]):
            actions.append('left')
        if col < self.grid_size - 1 and (row, col+1) != (other_agent[0], other_agent[1]):
            actions.append('right')
        
        # Pickup action
        if not has_block and (row, col) in self.pickup_locations:
            actions.append('pickup')
        
        # Dropoff action
        if has_block and (row, col) in self.dropoff_locations:
            if self.dropoff_capacity[(row, col)] > 0:
                actions.append('dropoff')
        
        return actions
    
    def execute_action(self, agent_name, action):
        """
        Execute an action for an agent and return reward
        """
        agent = self.agent_m if agent_name == 'M' else self.agent_f
        
        reward = -1  # Default step cost
        
        if action == 'up':
            agent[0] -= 1
        elif action == 'down':
            agent[0] += 1
        elif action == 'left':
            agent[1] -= 1
        elif action == 'right':
            agent[1] += 1
        elif action == 'pickup':
            agent[2] = True
            reward = -1
        elif action == 'dropoff':
            agent[2] = False
            loc = (agent[0], agent[1])
            self.dropoff_capacity[loc] -= 1
            reward = 13  # Positive reward for successful delivery
        
        return reward
    
    def is_terminal(self):
        """Check if terminal state is reached (all dropoffs filled)"""
        return all(capacity == 0 for capacity in self.dropoff_capacity.values())
    
    def get_manhattan_distance(self):
        """Calculate Manhattan distance between two agents"""
        return abs(self.agent_m[0] - self.agent_f[0]) + abs(self.agent_m[1] - self.agent_f[1])


class TwoAgentQLearning:
    """
    Q-Learning implementation for two-agent system
    Using approach (b): Single Q-table that moves both agents
    """
    
    def __init__(self, alpha=0.3, gamma=0.5):
        self.alpha = alpha  # Learning rate
        self.gamma = gamma  # Discount factor
        self.q_table = defaultdict(lambda: defaultdict(float))
        
        # Performance tracking
        self.rewards_history = []
        self.cumulative_rewards = []
        self.steps_per_terminal = []
        self.manhattan_distances = []
        
    def get_q_value(self, state, agent, action):
        """Get Q-value for state-agent-action"""
        return self.q_table[state][(agent, action)]
    
    def update_q_value(self, state, agent, action, reward, next_state, next_agent, applicable_next_actions):
        """
        Update Q-value using Q-learning formula
        Q(s,a) = Q(s,a) + α[r + γ * max(Q(s',a')) - Q(s,a)]
        """
        current_q = self.get_q_value(state, agent, action)
        
        # Find max Q-value for next state
        if applicable_next_actions:
            max_next_q = max([self.get_q_value(next_state, next_agent, a) 
                             for a in applicable_next_actions])
        else:
            max_next_q = 0
        
        # Q-learning update
        new_q = current_q + self.alpha * (reward + self.gamma * max_next_q - current_q)
        self.q_table[state][(agent, action)] = new_q
    
    def select_action_random(self, applicable_actions):
        """PRANDOM policy: Choose randomly from applicable actions"""
        if 'pickup' in applicable_actions:
            return 'pickup'
        if 'dropoff' in applicable_actions:
            return 'dropoff'
        return random.choice(applicable_actions)
    
    def select_action_greedy(self, state, agent, applicable_actions):
        """PGREEDY policy: Choose action with highest Q-value"""
        if 'pickup' in applicable_actions:
            return 'pickup'
        if 'dropoff' in applicable_actions:
            return 'dropoff'
        
        # Get Q-values for all applicable actions
        q_values = [(action, self.get_q_value(state, agent, action)) 
                   for action in applicable_actions]
        
        # Find maximum Q-value
        max_q = max(q_values, key=lambda x: x[1])[1]
        
        # Get all actions with max Q-value (for tie breaking)
        best_actions = [action for action, q in q_values if q == max_q]
        
        return random.choice(best_actions)
    
    def select_action_exploit(self, state, agent, applicable_actions):
        """PEXPLOIT policy: Choose best action with 0.8 prob, random with 0.2 prob"""
        if 'pickup' in applicable_actions:
            return 'pickup'
        if 'dropoff' in applicable_actions:
            return 'dropoff'
        
        if random.random() < 0.8:
            # Exploit: choose best action
            return self.select_action_greedy(state, agent, applicable_actions)
        else:
            # Explore: choose random action
            return random.choice(applicable_actions)
    
    def run_experiment(self, world, total_steps, policy_sequence, seed=42):
        """
        Run experiment with specified policy sequence
        policy_sequence: list of tuples (policy_name, duration)
        """
        random.seed(seed)
        np.random.seed(seed)
        
        print(f"\n{'='*60}")
        print(f"Running experiment with seed {seed}")
        print(f"Policy sequence: {policy_sequence}")
        print(f"{'='*60}\n")
        
        world.reset()
        step = 0
        terminal_states_reached = 0
        cumulative_reward = 0
        episode_rewards = []
        current_episode_reward = 0
        steps_in_episode = 0
        
        # Track which policy is currently active
        policy_idx = 0
        current_policy, policy_steps_remaining = policy_sequence[policy_idx]
        steps_in_current_policy = 0
        
        while step < total_steps:
            # Check if we need to switch policy
            if steps_in_current_policy >= policy_steps_remaining and policy_idx < len(policy_sequence) - 1:
                policy_idx += 1
                current_policy, policy_steps_remaining = policy_sequence[policy_idx]
                steps_in_current_policy = 0
                print(f"Step {step}: Switching to policy {current_policy}")
            
            # Agent F acts first (then M)
            for agent_name in ['F', 'M']:
                if step >= total_steps:
                    break
                
                state = world.get_state()
                applicable_actions = world.get_applicable_actions(agent_name)
                
                if not applicable_actions:
                    step += 1
                    continue
                
                # Select action based on current policy
                if current_policy == 'PRANDOM':
                    action = self.select_action_random(applicable_actions)
                elif current_policy == 'PGREEDY':
                    action = self.select_action_greedy(state, agent_name, applicable_actions)
                elif current_policy == 'PEXPLOIT':
                    action = self.select_action_exploit(state, agent_name, applicable_actions)
                
                # Execute action and get reward
                reward = world.execute_action(agent_name, action)
                current_episode_reward += reward
                cumulative_reward += reward
                
                # Get next state
                next_state = world.get_state()
                next_agent = 'M' if agent_name == 'F' else 'F'
                next_applicable_actions = world.get_applicable_actions(next_agent)
                
                # Update Q-table (Q-learning)
                self.update_q_value(state, agent_name, action, reward, 
                                   next_state, next_agent, next_applicable_actions)
                
                # Track metrics
                self.rewards_history.append(reward)
                self.cumulative_rewards.append(cumulative_reward)
                self.manhattan_distances.append(world.get_manhattan_distance())
                
                step += 1
                steps_in_current_policy += 1
                steps_in_episode += 1
                
                # Check for terminal state
                if world.is_terminal():
                    terminal_states_reached += 1
                    episode_rewards.append(current_episode_reward)
                    self.steps_per_terminal.append(steps_in_episode)
                    
                    print(f"Terminal state {terminal_states_reached} reached at step {step}")
                    print(f"  Episode reward: {current_episode_reward}")
                    print(f"  Steps in episode: {steps_in_episode}")
                    print(f"  Average Manhattan distance: {np.mean(self.manhattan_distances[-steps_in_episode:]):.2f}")
                    
                    world.reset()
                    current_episode_reward = 0
                    steps_in_episode = 0
        
        print(f"\n{'='*60}")
        print(f"Experiment completed!")
        print(f"Total steps: {step}")
        print(f"Terminal states reached: {terminal_states_reached}")
        print(f"Total cumulative reward: {cumulative_reward}")
        print(f"Average reward per step: {cumulative_reward/step:.3f}")
        if episode_rewards:
            print(f"Average episode reward: {np.mean(episode_rewards):.2f}")
            print(f"Average steps per terminal: {np.mean(self.steps_per_terminal):.2f}")
        print(f"Average Manhattan distance: {np.mean(self.manhattan_distances):.2f}")
        print(f"{'='*60}\n")
        
        return {
            'terminal_states': terminal_states_reached,
            'cumulative_reward': cumulative_reward,
            'episode_rewards': episode_rewards,
            'steps_per_terminal': self.steps_per_terminal.copy(),
            'avg_manhattan_distance': np.mean(self.manhattan_distances)
        }


def visualize_results(results_dict, title):
    """Create visualizations for experiment results"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(title, fontsize=16, fontweight='bold')
    
    # Plot 1: Cumulative Rewards
    ax1 = axes[0, 0]
    ax1.plot(results_dict['cumulative_rewards'], linewidth=1)
    ax1.set_xlabel('Steps')
    ax1.set_ylabel('Cumulative Reward')
    ax1.set_title('Cumulative Reward over Time')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Rewards per step (smoothed)
    ax2 = axes[0, 1]
    window_size = 100
    rewards = results_dict['rewards_history']
    if len(rewards) >= window_size:
        smoothed = np.convolve(rewards, np.ones(window_size)/window_size, mode='valid')
        ax2.plot(smoothed, linewidth=1)
    else:
        ax2.plot(rewards, linewidth=1)
    ax2.set_xlabel('Steps')
    ax2.set_ylabel('Average Reward')
    ax2.set_title(f'Reward per Step (smoothed, window={window_size})')
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Manhattan Distance
    ax3 = axes[1, 0]
    window_size = 100
    distances = results_dict['manhattan_distances']
    if len(distances) >= window_size:
        smoothed = np.convolve(distances, np.ones(window_size)/window_size, mode='valid')
        ax3.plot(smoothed, linewidth=1)
    else:
        ax3.plot(distances, linewidth=1)
    ax3.set_xlabel('Steps')
    ax3.set_ylabel('Manhattan Distance')
    ax3.set_title(f'Agent Coordination (Manhattan Distance, smoothed)')
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Steps per Terminal State
    ax4 = axes[1, 1]
    if results_dict['steps_per_terminal']:
        ax4.bar(range(len(results_dict['steps_per_terminal'])), 
               results_dict['steps_per_terminal'])
        ax4.set_xlabel('Terminal State Number')
        ax4.set_ylabel('Steps to Complete')
        ax4.set_title('Steps to Reach Each Terminal State')
        ax4.grid(True, alpha=0.3, axis='y')
    else:
        ax4.text(0.5, 0.5, 'No terminal states reached', 
                ha='center', va='center', transform=ax4.transAxes)
    
    plt.tight_layout()
    return fig


def print_q_table_summary(q_learner, world, filename=None):
    """Print summary of Q-table"""
    output = []
    output.append("\n" + "="*80)
    output.append("Q-TABLE SUMMARY")
    output.append("="*80)
    output.append(f"Total states in Q-table: {len(q_learner.q_table)}")
    output.append(f"Learning rate (α): {q_learner.alpha}")
    output.append(f"Discount factor (γ): {q_learner.gamma}")
    
    # Find states with highest Q-values
    output.append("\nTop 10 State-Action pairs by Q-value:")
    all_q_values = []
    for state, actions in q_learner.q_table.items():
        for (agent, action), q_val in actions.items():
            all_q_values.append((state, agent, action, q_val))
    
    all_q_values.sort(key=lambda x: x[3], reverse=True)
    
    for i, (state, agent, action, q_val) in enumerate(all_q_values[:10]):
        output.append(f"{i+1}. Agent {agent}, Action {action:8s}, Q={q_val:7.3f}")
        output.append(f"   State: M@{state[0]} {'(block)' if state[1] else '       '}, "
                     f"F@{state[2]} {'(block)' if state[3] else '       '}")
    
    output.append("="*80 + "\n")
    
    result = "\n".join(output)
    print(result)
    
    if filename:
        with open(filename, 'w', encoding='utf-8') as f:  # ← ADD encoding='utf-8'
            f.write(result)
    
    return result


def main():
    """Main function to run all Experiment 1 variations"""

    # Create output directory if it doesn't exist
    import os
    output_dir = 'outputs'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    print("="*80)
    print("TASK 4: TWO-AGENT REINFORCEMENT LEARNING")
    print("EXPERIMENT 1: Q-Learning with Different Policies")
    print("="*80)
    
    # Experiment parameters
    alpha = 0.3
    gamma = 0.5
    total_steps = 8000
    initial_random_steps = 500
    
    # Dictionary to store all results
    all_results = {}
    
    # ========================================================================
    # EXPERIMENT 1a: PRANDOM for all steps
    # ========================================================================
    print("\n" + "="*80)
    print("EXPERIMENT 1a: PRANDOM policy for all 8000 steps")
    print("="*80)
    
    policy_sequence_1a = [
        ('PRANDOM', 8000)
    ]
    
    for run in [1, 2]:
        print(f"\n--- Run {run} ---")
        world = PDWorld()
        q_learner = TwoAgentQLearning(alpha=alpha, gamma=gamma)
        
        results = q_learner.run_experiment(
            world, 
            total_steps, 
            policy_sequence_1a,
            seed=42 + run
        )
        
        results['learner'] = q_learner
        results['rewards_history'] = q_learner.rewards_history
        results['cumulative_rewards'] = q_learner.cumulative_rewards
        results['manhattan_distances'] = q_learner.manhattan_distances
        results['steps_per_terminal'] = q_learner.steps_per_terminal
        
        all_results[f'1a_run{run}'] = results
        
        # Visualize
        fig = visualize_results(results, f'Experiment 1a - Run {run}: PRANDOM')
        plt.savefig(f'outputs/exp1a_run{run}_results.png', dpi=150, bbox_inches='tight')
        plt.close()
    
    # ========================================================================
    # EXPERIMENT 1b: PRANDOM then PGREEDY
    # ========================================================================
    print("\n" + "="*80)
    print("EXPERIMENT 1b: PRANDOM (500 steps) then PGREEDY (7500 steps)")
    print("="*80)
    
    policy_sequence_1b = [
        ('PRANDOM', 500),
        ('PGREEDY', 7500)
    ]
    
    for run in [1, 2]:
        print(f"\n--- Run {run} ---")
        world = PDWorld()
        q_learner = TwoAgentQLearning(alpha=alpha, gamma=gamma)
        
        results = q_learner.run_experiment(
            world, 
            total_steps, 
            policy_sequence_1b,
            seed=100 + run
        )
        
        results['learner'] = q_learner
        results['rewards_history'] = q_learner.rewards_history
        results['cumulative_rewards'] = q_learner.cumulative_rewards
        results['manhattan_distances'] = q_learner.manhattan_distances
        results['steps_per_terminal'] = q_learner.steps_per_terminal
        
        all_results[f'1b_run{run}'] = results
        
        # Visualize
        fig = visualize_results(results, f'Experiment 1b - Run {run}: PRANDOM → PGREEDY')
        plt.savefig(f'outputs/exp1b_run{run}_results.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        # Print Q-table for run 1
        if run == 1:
            print_q_table_summary(q_learner, world, 
                                f'outputs/exp1b_run{run}_qtable.txt')
    
    # ========================================================================
    # EXPERIMENT 1c: PRANDOM then PEXPLOIT
    # ========================================================================
    print("\n" + "="*80)
    print("EXPERIMENT 1c: PRANDOM (500 steps) then PEXPLOIT (7500 steps)")
    print("="*80)
    
    policy_sequence_1c = [
        ('PRANDOM', 500),
        ('PEXPLOIT', 7500)
    ]
    
    for run in [1, 2]:
        print(f"\n--- Run {run} ---")
        world = PDWorld()
        q_learner = TwoAgentQLearning(alpha=alpha, gamma=gamma)
        
        results = q_learner.run_experiment(
            world, 
            total_steps, 
            policy_sequence_1c,
            seed=200 + run
        )
        
        results['learner'] = q_learner
        results['rewards_history'] = q_learner.rewards_history
        results['cumulative_rewards'] = q_learner.cumulative_rewards
        results['manhattan_distances'] = q_learner.manhattan_distances
        results['steps_per_terminal'] = q_learner.steps_per_terminal
        
        all_results[f'1c_run{run}'] = results
        
        # Visualize
        fig = visualize_results(results, f'Experiment 1c - Run {run}: PRANDOM → PEXPLOIT')
        plt.savefig(f'outputs/exp1c_run{run}_results.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        # Print Q-table for both runs (requirement: report final Q-table of 1c)
        print_q_table_summary(q_learner, world, 
                            f'outputs/exp1c_run{run}_qtable.txt')
    
    # ========================================================================
    # COMPARISON SUMMARY
    # ========================================================================
    print("\n" + "="*80)
    print("COMPARISON SUMMARY - EXPERIMENT 1")
    print("="*80)
    
    comparison_data = []
    for exp_key in ['1a', '1b', '1c']:
        for run in [1, 2]:
            key = f'{exp_key}_run{run}'
            if key in all_results:
                results = all_results[key]
                comparison_data.append({
                    'Experiment': exp_key,
                    'Run': run,
                    'Terminal States': results['terminal_states'],
                    'Cumulative Reward': results['cumulative_reward'],
                    'Avg Manhattan Dist': results['avg_manhattan_distance'],
                    'Avg Steps/Terminal': np.mean(results['steps_per_terminal']) if results['steps_per_terminal'] else 0
                })
    
    # Print comparison table
    print(f"\n{'Exp':<6} {'Run':<4} {'Terminals':<10} {'Cum. Reward':<13} {'Avg Dist':<10} {'Avg Steps/Term':<15}")
    print("-" * 80)
    for data in comparison_data:
        print(f"{data['Experiment']:<6} {data['Run']:<4} {data['Terminal States']:<10} "
              f"{data['Cumulative Reward']:<13.1f} {data['Avg Manhattan Dist']:<10.2f} "
              f"{data['Avg Steps/Terminal']:<15.1f}")
    
    # Create comparison plots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Experiment 1 Comparison: All Variations', fontsize=16, fontweight='bold')
    
    experiments = ['1a', '1b', '1c']
    exp_names = ['PRANDOM', 'PRANDOM→PGREEDY', 'PRANDOM→PEXPLOIT']
    colors = ['blue', 'green', 'red']
    
    # Plot cumulative rewards for all experiments (Run 1 only for clarity)
    ax1 = axes[0, 0]
    for exp, name, color in zip(experiments, exp_names, colors):
        key = f'{exp}_run1'
        if key in all_results:
            ax1.plot(all_results[key]['cumulative_rewards'], 
                    label=name, linewidth=1.5, alpha=0.7, color=color)
    ax1.set_xlabel('Steps')
    ax1.set_ylabel('Cumulative Reward')
    ax1.set_title('Cumulative Rewards Comparison (Run 1)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Bar chart: Terminal states reached
    ax2 = axes[0, 1]
    terminals = [[all_results[f'{exp}_run1']['terminal_states'], 
                  all_results[f'{exp}_run2']['terminal_states']] 
                 for exp in experiments]
    x = np.arange(len(experiments))
    width = 0.35
    ax2.bar(x - width/2, [t[0] for t in terminals], width, label='Run 1', alpha=0.8)
    ax2.bar(x + width/2, [t[1] for t in terminals], width, label='Run 2', alpha=0.8)
    ax2.set_xlabel('Experiment')
    ax2.set_ylabel('Terminal States Reached')
    ax2.set_title('Terminal States Comparison')
    ax2.set_xticks(x)
    ax2.set_xticklabels(exp_names, rotation=15, ha='right')
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Bar chart: Average Manhattan distance
    ax3 = axes[1, 0]
    distances = [[all_results[f'{exp}_run1']['avg_manhattan_distance'], 
                  all_results[f'{exp}_run2']['avg_manhattan_distance']] 
                 for exp in experiments]
    ax3.bar(x - width/2, [d[0] for d in distances], width, label='Run 1', alpha=0.8)
    ax3.bar(x + width/2, [d[1] for d in distances], width, label='Run 2', alpha=0.8)
    ax3.set_xlabel('Experiment')
    ax3.set_ylabel('Average Manhattan Distance')
    ax3.set_title('Agent Coordination Comparison (Lower is Better)')
    ax3.set_xticks(x)
    ax3.set_xticklabels(exp_names, rotation=15, ha='right')
    ax3.legend()
    ax3.grid(True, alpha=0.3, axis='y')
    
    # Bar chart: Average cumulative reward
    ax4 = axes[1, 1]
    cum_rewards = [[all_results[f'{exp}_run1']['cumulative_reward'], 
                    all_results[f'{exp}_run2']['cumulative_reward']] 
                   for exp in experiments]
    ax4.bar(x - width/2, [r[0] for r in cum_rewards], width, label='Run 1', alpha=0.8)
    ax4.bar(x + width/2, [r[1] for r in cum_rewards], width, label='Run 2', alpha=0.8)
    ax4.set_xlabel('Experiment')
    ax4.set_ylabel('Total Cumulative Reward')
    ax4.set_title('Total Reward Comparison')
    ax4.set_xticks(x)
    ax4.set_xticklabels(exp_names, rotation=15, ha='right')
    ax4.legend()
    ax4.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('outputs/exp1_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print("\nAll results saved to /outputs")
    print("="*80)


if __name__ == "__main__":
    main()
