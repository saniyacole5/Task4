# Task 4: Two-Agent Reinforcement Learning - Implementation Guide

## Overview

This implementation provides a complete solution for Task 4 of the AI course, which involves using Q-learning to train two agents to collaboratively solve a pickup-and-delivery problem in a grid world environment.

## Files Structure

### Core Implementation Files

```
├── pd_world_experiment.py        # Main experiment runner
└── qtable_visualization.py       # Q-table and policy visualization tools
```

### Output Files

```
/outputs/
├── Performance Graphs:
│   ├── exp1a_run1_results.png    # PRANDOM policy results (Run 1)
│   ├── exp1a_run2_results.png    # PRANDOM policy results (Run 2)
│   ├── exp1b_run1_results.png    # PGREEDY policy results (Run 1)
│   ├── exp1b_run2_results.png    # PGREEDY policy results (Run 2)
│   ├── exp1c_run1_results.png    # PEXPLOIT policy results (Run 1)
│   ├── exp1c_run2_results.png    # PEXPLOIT policy results (Run 2)
│   └── exp1_comparison.png        # Comparison across all experiments
│
├── Q-Table Analysis:
│   ├── exp1b_run1_qtable.txt     # Q-table summary for PGREEDY
│   ├── exp1c_run1_qtable.txt     # Q-table summary for PEXPLOIT (Run 1)
│   └── exp1c_run2_qtable.txt     # Q-table summary for PEXPLOIT (Run 2)
│
├── Policy Visualizations:
│   └── experiment_1b_run_1_policy_viz.png  # Policy heatmaps
│
└── Reports:
    ├── experiment1_report.md      # Comprehensive analysis
    ├── experiment1_summary.md     # Executive summary
    └── README.md                  # This file
```

## Quick Start

### Running the Complete Experiment

```bash
python pd_world_experiment.py
```

This will:

1. Run all three variations of Experiment 1 (1a, 1b, 1c)
2. Execute each experiment twice with different random seeds
3. Generate performance visualizations
4. Save Q-table summaries
5. Create comparison plots
6. Print detailed results to console

**Expected Runtime:** ~2-3 minutes

### Generating Q-Table Visualizations

```bash
python qtable_visualization.py
```

This will:

1. Re-run experiments 1b and 1c
2. Generate policy heatmaps showing learned behaviors
3. Analyze path quality
4. Save visualization images

**Expected Runtime:** ~2-3 minutes

## Implementation Details

### PDWorld Class

The environment simulation with the following features:

**Grid Configuration:**

- 5×5 grid world
- Pickup locations: (0,1) and (3,1)
- Dropoff locations: (0,4), (2,4), (4,2), (4,4)
- Each dropoff holds 5 blocks

**Agent Mechanics:**

- Two agents: 'M' (Male) and 'F' (Female)
- Agents alternate turns (F acts first)
- Cannot occupy same cell
- Each carries 0 or 1 block

**Actions:**

- Movement: up, down, left, right
- pickup: Pick up block at pickup location
- dropoff: Deliver block at dropoff location

**Rewards:**

- Movement/pickup: -1 (step cost)
- Dropoff: +13 (delivery reward)

### TwoAgentQLearning Class

Q-learning implementation with:

**Learning Parameters:**

- α (alpha): Learning rate (default 0.3)
- γ (gamma): Discount factor (default 0.5)
- Q-table: Dictionary-based storage for state-action values

**Policies:**

1. **PRANDOM**: Random action selection (except for pickup/dropoff when available)
2. **PGREEDY**: Always select highest Q-value action (greedy)
3. **PEXPLOIT**: 80% greedy, 20% random (ε-greedy with ε=0.2)

**State Representation:**

```python
state = (
    (agent_m_row, agent_m_col),  # Agent M position
    agent_m_has_block,            # Boolean: M carrying block?
    (agent_f_row, agent_f_col),  # Agent F position
    agent_f_has_block             # Boolean: F carrying block?
)
```

**Q-Learning Update Rule:**

```
Q(s,a) ← Q(s,a) + α[r + γ·max(Q(s',a')) - Q(s,a)]
```

### Performance Metrics

The system tracks:

- **Cumulative reward**: Total reward accumulated
- **Terminal states reached**: Number of times all dropoffs filled
- **Steps per terminal**: Efficiency measure
- **Manhattan distance**: Agent coordination metric
- **Episode rewards**: Reward per complete episode

## Experiment 1 Configuration

### Experiment 1a: Pure Random Baseline

- Policy: PRANDOM for all 8000 steps
- Purpose: Establish baseline without learning
- Seeds: 43, 44

### Experiment 1b: Greedy After Exploration

- Policy: PRANDOM (500 steps) → PGREEDY (7500 steps)
- Purpose: Test pure exploitation after exploration
- Seeds: 101, 102
- **Best overall performance** ✓

### Experiment 1c: Epsilon-Greedy

- Policy: PRANDOM (500 steps) → PEXPLOIT (7500 steps)
- Purpose: Balance exploration and exploitation
- Seeds: 201, 202
- Report final Q-table (requirement)

## Understanding the Results

### Performance Graphs

Each experiment generates 4 plots:

1. **Cumulative Reward**: Shows total reward accumulation over time

   - Upward trend indicates learning
   - Steeper = faster learning

2. **Reward per Step**: Smoothed average reward

   - Should increase as agents learn efficient paths
   - Spikes occur at dropoff actions (+13 reward)

3. **Manhattan Distance**: Agent coordination metric

   - Lower = agents closer together (more blocking risk)
   - Higher = agents more separated (better coordination)
   - Optimal range: 3-4 cells in 5×5 grid

4. **Steps per Terminal**: Efficiency over episodes
   - Should decrease as agents learn
   - Shows improvement in solving the task

### Q-Table Interpretation

**High Q-values (6-10)**: Indicate good actions

- Highest values near dropoff locations with blocks
- Shows learned paths to goals

**Low Q-values (-5 to 0)**: Indicate less desirable actions

- Common in early states
- Actions that don't lead to goals

**State Space Coverage**: ~2000 states explored

- Small fraction of total possible states
- Indicates efficient learning (not exploring unnecessary states)

## Customization

### Changing Learning Parameters

```python
# In pd_world_experiment.py, modify:
alpha = 0.3  # Learning rate (0.15 - 0.5 typical)
gamma = 0.5  # Discount factor (0.3 - 0.9 typical)
```

### Changing World Configuration

```python
# Create custom world:
world = PDWorld(
    pickup_locations=[(0,1), (3,1)],
    dropoff_locations=[(0,4), (2,4), (4,2), (4,4)]
)
```

### Adding New Policies

```python
# In TwoAgentQLearning class, add method:
def select_action_custom(self, state, agent, applicable_actions):
    # Your policy logic here
    return chosen_action

# Then use in policy_sequence:
policy_sequence = [
    ('PRANDOM', 500),
    ('custom', 7500)  # Add to run_experiment logic
]
```

### Running Longer Experiments

```python
# Increase total steps:
total_steps = 16000  # Double the length

# Or run more initial exploration:
policy_sequence = [
    ('PRANDOM', 2000),    # More exploration
    ('PGREEDY', 14000)    # More exploitation
]
```

## Key Results Summary

| Metric         | 1a (PRANDOM) | 1b (PGREEDY) | 1c (PEXPLOIT) |
| -------------- | ------------ | ------------ | ------------- |
| Avg Reward     | -5886        | **-5872** ✓  | -5963         |
| Avg Terminals  | 7.0          | 7.0          | 6.5           |
| Avg Steps/Term | 1076.2       | **1059.8** ✓ | 1123.2        |
| Coordination   | 3.19         | 3.23         | 3.22          |

**Winner: Experiment 1b (PGREEDY)** - Best reward and efficiency

## Future Experiments

### Experiment 2: Q-Learning vs SARSA

- Compare off-policy (Q-learning) with on-policy (SARSA)
- Expected: SARSA more conservative, Q-learning more aggressive

### Experiment 3: Different Learning Rates

- Test α=0.15, γ=0.45 vs current α=0.3, γ=0.5
- Expected: Lower α → slower but more stable learning

### Experiment 4: Adaptation to Change

- Change pickup locations after 3 terminal states
- Test ability to unlearn old paths and learn new ones
- Expected: Temporary performance drop, then recovery

## References

- Course slides: http://www2.cs.uh.edu/~ceick/ai/2025-World.pptx
- Q-Learning tutorial: http://cs.stanford.edu/people/karpathy/reinforcejs/gridworld_td.html
- Multi-agent RL: https://arxiv.org/abs/1911.10635

---

**Python Version:** 3.12
**Dependencies:** numpy, matplotlib
