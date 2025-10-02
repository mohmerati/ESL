This repository has the implementation of the Exhaustive-serve-longest (ESL) policy, First-come-first-serve, proposed in this [paper](https://www.arxiv.org/abs/2509.25556#:~:text=We%20formulate%20a%20discounted%2Dcost,the%20optimality%20of%20this%20policy.)
# Multiple-servers Multiple-queues Simulation

This repo contains Python code to simulate a queueing system with multiple servers and multiple queues. It defines an environment (`QueueEnv`), several policies for managing the servers (`policy_greedy_longest`, `PolicyFCFSPerTask`, `PolicyCyclicFixedDwell`), and functions to evaluate these policies (`evaluate_policy_with_metrics`) and trace their execution (`rollout_with_trace`, `rollout_with_trace_for_fcfs`). The notebook also includes code to run simulations for different scenarios and visualize the results.

## Environment (`QueueEnv`)

The `QueueEnv` class simulates a system with `M` queues and `R` servers. Key parameters include:

- `M`: Number of queues.
- `R`: Number of servers.
- `p`: Arrival probability for each queue.
- `beta`: Discount factor for future costs.

The `step` method simulates one time step in the environment, taking a joint action for all servers and returning the next state and the cost incurred.

## Policies

We implemented three different policies for the servers:

- `policy_greedy_longest`: A greedy policy where servers prioritize serving the longest queues they are currently at or switch to the longest available queue.
- `PolicyFCFSPerTask`: A policy based on First-Come, First-Served, where servers prioritize serving tasks that have been in their respective queues the longest.
- `PolicyCyclicFixedDwell`: A policy where each server cycles through a block of queues, staying at each queue for a fixed "dwell" time.

## Evaluation and Analysis

- `evaluate_policy_with_metrics`: This function runs multiple episodes of a given policy in the environment and collects metrics such as discounted cost, average queue length, and action fractions (serve, switch, idle).
- `rollout_with_trace` and `rollout_with_trace_for_fcfs`: These functions trace the execution of a policy step by step, providing detailed information about the state, action, arrivals, and queue contents at each time step.

We also include code to:

- Run a grid search over different environment parameters (number of queues, number of servers, arrival probabilities) and policies.
- Collect and store the results in a list of dictionaries.
- Plot the results to compare the performance of different policies under various conditions.
- Save the collected results to a CSV file and the generated plots to Google Drive.

## Usage

1.  **Run the cells sequentially:** You can copy the codes in one notebook, and execute the code cells in the notebook from top to bottom.
2.  **Modify parameters:** Adjust the parameters in the code cells (e.g., `M`, `R`, `alphas`, `T`, `episodes`) to explore different scenarios.
3.  **Analyze results:** Examine the printed output and the generated plots to understand the performance of the different policies.
4.  **Access saved files:** The results are saved to a CSV file and the plots are saved to your Google Drive in the specified directory (`/content/drive/MyDrive/Research/Phase 2/queue_plots_acc`).

This notebook provides a framework for simulating and evaluating different server allocation policies in a multi-server, multi-queue system.
