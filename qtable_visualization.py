"""
Q-Table and Path Visualization for Task 4
Analyzes learned paths and creates heatmaps
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import LinearSegmentedColormap
import pickle
import sys

# Import from main experiment file
from pd_world_experiment import PDWorld, TwoAgentQLearning

def visualize_q_table_heatmap(q_learner, agent_name, has_block, world):
    """
    Create a heatmap showing the learned policy for a specific agent state
    """
    grid_size = world.grid_size
    
    # Create action preference matrix (which direction has highest Q-value)
    best_action_grid = np.zeros((grid_size, grid_size), dtype=object)
    max_q_grid = np.zeros((grid_size, grid_size))
    
    for row in range(grid_size):
        for col in range(grid_size):
            # Consider a simple state where other agent is far away
            other_pos = (4, 4) if agent_name == 'M' else (0, 0)
            other_has_block = False
            
            if agent_name == 'M':
                state = ((row, col), has_block, other_pos, other_has_block)
            else:
                state = (other_pos, other_has_block, (row, col), has_block)
            
            # Get applicable actions
            if agent_name == 'M':
                world.agent_m = [row, col, has_block]
                world.agent_f = [other_pos[0], other_pos[1], other_has_block]
            else:
                world.agent_f = [row, col, has_block]
                world.agent_m = [other_pos[0], other_pos[1], other_has_block]
            
            actions = world.get_applicable_actions(agent_name)
            
            if not actions:
                best_action_grid[row, col] = 'none'
                continue
            
            # Find action with highest Q-value
            action_qs = [(action, q_learner.get_q_value(state, agent_name, action)) 
                        for action in actions]
            
            # Filter out pickup/dropoff for visualization
            movement_actions = [(a, q) for a, q in action_qs 
                               if a in ['up', 'down', 'left', 'right']]
            
            if movement_actions:
                best_action, max_q = max(movement_actions, key=lambda x: x[1])
                best_action_grid[row, col] = best_action
                max_q_grid[row, col] = max_q
            elif 'pickup' in [a for a, q in action_qs]:
                best_action_grid[row, col] = 'pickup'
                max_q_grid[row, col] = max([q for a, q in action_qs])
            elif 'dropoff' in [a for a, q in action_qs]:
                best_action_grid[row, col] = 'dropoff'
                max_q_grid[row, col] = max([q for a, q in action_qs])
            else:
                best_action_grid[row, col] = 'none'
    
    return best_action_grid, max_q_grid


def plot_policy_arrows(ax, best_action_grid, max_q_grid, title):
    """
    Plot arrows showing the learned policy
    """
    grid_size = best_action_grid.shape[0]
    
    # Create heatmap background
    im = ax.imshow(max_q_grid, cmap='YlOrRd', alpha=0.6, vmin=max_q_grid.min(), vmax=max_q_grid.max())
    
    # Draw grid lines
    for i in range(grid_size + 1):
        ax.axhline(i - 0.5, color='black', linewidth=0.5)
        ax.axvline(i - 0.5, color='black', linewidth=0.5)
    
    # Draw arrows for each cell
    arrow_props = dict(arrowstyle='->', lw=2, color='blue')
    
    for row in range(grid_size):
        for col in range(grid_size):
            action = best_action_grid[row, col]
            
            if action == 'up':
                ax.annotate('', xy=(col, row-0.3), xytext=(col, row+0.3),
                           arrowprops=arrow_props)
            elif action == 'down':
                ax.annotate('', xy=(col, row+0.3), xytext=(col, row-0.3),
                           arrowprops=arrow_props)
            elif action == 'left':
                ax.annotate('', xy=(col-0.3, row), xytext=(col+0.3, row),
                           arrowprops=arrow_props)
            elif action == 'right':
                ax.annotate('', xy=(col+0.3, row), xytext=(col-0.3, row),
                           arrowprops=arrow_props)
            elif action == 'pickup':
                ax.text(col, row, 'P', ha='center', va='center', 
                       fontweight='bold', fontsize=12, color='green')
            elif action == 'dropoff':
                ax.text(col, row, 'D', ha='center', va='center', 
                       fontweight='bold', fontsize=12, color='red')
    
    ax.set_xlim(-0.5, grid_size - 0.5)
    ax.set_ylim(grid_size - 0.5, -0.5)  # Invert y-axis
    ax.set_xticks(range(grid_size))
    ax.set_yticks(range(grid_size))
    ax.set_title(title, fontweight='bold')
    ax.set_xlabel('Column')
    ax.set_ylabel('Row')
    
    return im


def create_comprehensive_visualization(q_learner, world, experiment_name):
    """
    Create comprehensive visualization of learned policies
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    fig.suptitle(f'{experiment_name}: Learned Policies', fontsize=16, fontweight='bold')
    
    # Agent M without block
    best_action_m_no, max_q_m_no = visualize_q_table_heatmap(q_learner, 'M', False, world)
    im1 = plot_policy_arrows(axes[0, 0], best_action_m_no, max_q_m_no, 
                             'Agent M (No Block) - Policy')
    
    # Agent M with block
    best_action_m_yes, max_q_m_yes = visualize_q_table_heatmap(q_learner, 'M', True, world)
    im2 = plot_policy_arrows(axes[0, 1], best_action_m_yes, max_q_m_yes, 
                             'Agent M (With Block) - Policy')
    
    # Agent F without block
    best_action_f_no, max_q_f_no = visualize_q_table_heatmap(q_learner, 'F', False, world)
    im3 = plot_policy_arrows(axes[1, 0], best_action_f_no, max_q_f_no, 
                             'Agent F (No Block) - Policy')
    
    # Agent F with block
    best_action_f_yes, max_q_f_yes = visualize_q_table_heatmap(q_learner, 'F', True, world)
    im4 = plot_policy_arrows(axes[1, 1], best_action_f_yes, max_q_f_yes, 
                             'Agent F (With Block) - Policy')
    
    # Add colorbars
    fig.colorbar(im1, ax=axes[0, 0], label='Q-value')
    fig.colorbar(im2, ax=axes[0, 1], label='Q-value')
    fig.colorbar(im3, ax=axes[1, 0], label='Q-value')
    fig.colorbar(im4, ax=axes[1, 1], label='Q-value')
    
    # Mark pickup and dropoff locations on all subplots
    for ax in axes.flat:
        # Pickup locations (green squares)
        for pickup in world.pickup_locations:
            rect = patches.Rectangle((pickup[1]-0.4, pickup[0]-0.4), 0.8, 0.8,
                                    linewidth=3, edgecolor='green', facecolor='none')
            ax.add_patch(rect)
        
        # Dropoff locations (red squares)
        for dropoff in world.dropoff_locations:
            rect = patches.Rectangle((dropoff[1]-0.4, dropoff[0]-0.4), 0.8, 0.8,
                                    linewidth=3, edgecolor='red', facecolor='none')
            ax.add_patch(rect)
    
    plt.tight_layout()
    return fig


def analyze_path_quality(q_learner, world):
    """
    Analyze the quality of learned paths from pickups to dropoffs
    """
    print("\n" + "="*80)
    print("PATH QUALITY ANALYSIS")
    print("="*80)
    
    results = []
    
    for pickup in world.pickup_locations:
        for dropoff in world.dropoff_locations:
            # Calculate optimal Manhattan distance
            optimal_distance = abs(pickup[0] - dropoff[0]) + abs(pickup[1] - dropoff[1])
            
            # Simulate following the learned policy
            agent_name = 'M'
            current_pos = pickup
            has_block = True
            steps = 0
            max_steps = 50  # Prevent infinite loops
            path = [current_pos]
            
            while current_pos != dropoff and steps < max_steps:
                # Create a simple state
                other_pos = (4, 4)
                other_has_block = False
                
                if agent_name == 'M':
                    state = (current_pos, has_block, other_pos, other_has_block)
                else:
                    state = (other_pos, other_has_block, current_pos, has_block)
                
                # Set world state
                world.agent_m = [current_pos[0], current_pos[1], has_block] if agent_name == 'M' else [other_pos[0], other_pos[1], other_has_block]
                world.agent_f = [other_pos[0], other_pos[1], other_has_block] if agent_name == 'M' else [current_pos[0], current_pos[1], has_block]
                
                # Get applicable actions
                actions = world.get_applicable_actions(agent_name)
                
                if not actions:
                    break
                
                # Choose action with highest Q-value (excluding pickup/dropoff)
                movement_actions = [a for a in actions if a in ['up', 'down', 'left', 'right']]
                
                if not movement_actions:
                    break
                
                action_qs = [(action, q_learner.get_q_value(state, agent_name, action)) 
                            for action in movement_actions]
                best_action = max(action_qs, key=lambda x: x[1])[0]
                
                # Execute action
                if best_action == 'up':
                    current_pos = (current_pos[0] - 1, current_pos[1])
                elif best_action == 'down':
                    current_pos = (current_pos[0] + 1, current_pos[1])
                elif best_action == 'left':
                    current_pos = (current_pos[0], current_pos[1] - 1)
                elif best_action == 'right':
                    current_pos = (current_pos[0], current_pos[1] + 1)
                
                path.append(current_pos)
                steps += 1
            
            if current_pos == dropoff:
                efficiency = optimal_distance / steps if steps > 0 else 1.0
                results.append({
                    'pickup': pickup,
                    'dropoff': dropoff,
                    'optimal': optimal_distance,
                    'learned': steps,
                    'efficiency': efficiency,
                    'path': path
                })
    
    # Print results
    print(f"\n{'From':12} {'To':12} {'Optimal':8} {'Learned':8} {'Efficiency':10} {'Path Quality'}")
    print("-" * 80)
    
    for r in results:
        quality = "Optimal" if r['learned'] == r['optimal'] else \
                 "Good" if r['efficiency'] > 0.8 else \
                 "Fair" if r['efficiency'] > 0.6 else "Poor"
        
        print(f"{str(r['pickup']):12} {str(r['dropoff']):12} {r['optimal']:8} "
              f"{r['learned']:8} {r['efficiency']:10.2%} {quality}")
    
    avg_efficiency = np.mean([r['efficiency'] for r in results])
    print(f"\nAverage Path Efficiency: {avg_efficiency:.2%}")
    print("="*80 + "\n")
    
    return results


def main():
    """Generate visualizations for all experiments"""
    
    # We'll need to reload the Q-learners from the experiment
    # For now, let's re-run simplified versions to get the Q-tables
    
    from pd_world_experiment import PDWorld, TwoAgentQLearning
    
    print("Generating Q-table visualizations...")
    
    # Run quick versions of experiments to get Q-tables
    experiments = {
        'Experiment 1b Run 1': {
            'alpha': 0.3, 'gamma': 0.5,
            'policy_sequence': [('PRANDOM', 500), ('PGREEDY', 7500)],
            'seed': 101
        },
        'Experiment 1c Run 1': {
            'alpha': 0.3, 'gamma': 0.5,
            'policy_sequence': [('PRANDOM', 500), ('PEXPLOIT', 7500)],
            'seed': 201
        }
    }
    
    for exp_name, params in experiments.items():
        print(f"\nProcessing {exp_name}...")
        
        world = PDWorld()
        q_learner = TwoAgentQLearning(alpha=params['alpha'], gamma=params['gamma'])
        
        # Run experiment
        q_learner.run_experiment(world, 8000, params['policy_sequence'], params['seed'])
        
        # Create visualizations
        fig = create_comprehensive_visualization(q_learner, world, exp_name)
        filename = exp_name.lower().replace(' ', '_')
        plt.savefig(f'outputs/{filename}_policy_viz.png', 
                   dpi=150, bbox_inches='tight')
        plt.close()
        
        # Analyze path quality
        path_results = analyze_path_quality(q_learner, world)
        
        print(f"Visualizations saved for {exp_name}")
    
    print("\nAll visualizations complete!")


if __name__ == "__main__":
    main()
