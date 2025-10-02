from typing import Callable, Optional, Tuple, List, Dict
import numpy as np

Action = Tuple[str, Optional[int]]
JointAction = Tuple[Action, ...]
Policy = Callable[[Tuple[Tuple[int, ...], Tuple[int, ...]], "QueueEnv"], JointAction]
State = Tuple[Tuple[int, ...], Tuple[int, ...]]

class QueueEnv:
    def __init__(self, M: int = 4, R: int = 2, p: Optional[np.ndarray] = None, beta: float = 0.99, seed: int = 42):
        self.M = M
        self.R = R
        self.beta = beta
        self.seed = seed
        self.p = np.array(p, dtype=float)
        assert self.R <= self.M
        self.rng = np.random.default_rng(seed)

    def step(self, state: State, action: JointAction) -> Tuple[State, float]:
        s, x = state
        x_list = list(x)
        cost = float(sum(x_list))
        a_arr = self.rng.binomial(1, self.p).astype(int)
        counts = np.zeros(self.M, dtype=int)
        for r in range(self.R):
            if action[r][0] in ("serve", "idle"):
                target = s[r]
            elif action[r][0] == "switch":
                target = int(action[r][1])
            else:
                raise ValueError("bad action")
            counts[target] += 1
        if np.any(counts > 1):
            raise ValueError("collision constraint violated")
        for r in range(self.R):
            if action[r][0] == "serve" and x_list[s[r]] > 0:
                x_list[s[r]] -= 1
        new_pos = list(s)
        for r in range(self.R):
            if action[r][0] == "switch":
                new_pos[r] = int(action[r][1])
            else:
                new_pos[r] = s[r]
        for i in range(self.M):
            x_list[i] = x_list[i] + int(a_arr[i])
        return (tuple(new_pos), tuple(x_list)), cost

def policy_greedy_longest(state: State, env: QueueEnv) -> JointAction:
    s, x = state
    R, M = env.R, env.M
    x_list = list(x)
    acts: List[Action] = [None] * R  # type: ignore
    reserved_q = set()
    assigned_r = set()
    for r in range(R):
        sr = s[r]
        if x_list[sr] > 0 and sr not in reserved_q:
            acts[r] = ("serve", None)
            reserved_q.add(sr)
            assigned_r.add(r)
    for r in range(R):
        if r in assigned_r:
            continue
        pos_cands = [j for j in range(M) if x_list[j] > 0 and j not in reserved_q]
        if pos_cands:
            j_best = max(pos_cands, key=lambda j: (x_list[j], -j))
            acts[r] = ("switch", j_best)
            reserved_q.add(j_best)
        else:
            acts[r] = ("idle", None)
    return tuple(acts)

def evaluate_policy_with_metrics(env: QueueEnv, policy: Policy, T: int = 20000, episodes: int = 20, start_state: Optional[State] = None) -> Dict[str, object]:
    disc = np.array([env.beta ** k for k in range(T)], dtype=float)

    costs = []
    L_ep = []
    fserve_ep = []
    fswitch_ep = []
    fidle_ep = []

    serve_ct = np.zeros(env.R, dtype=int)
    switch_ct = np.zeros(env.R, dtype=int)
    idle_ct = np.zeros(env.R, dtype=int)
    for _ in range(episodes):
        if hasattr(policy, "reset"):
            try:
                policy.reset()
            except Exception:
                pass
        if start_state is None:
            positions = tuple(range(env.R))
            state = (positions, tuple([0] * env.M))
        else:
            state = start_state
        J = 0.0
        sum_total_q = 0.0
        c_serve = 0
        c_switch = 0
        c_idle = 0
        for t in range(T):
            s, x = state
            sum_total_q += float(sum(x))
            action = policy(state, env)
            for r in range(env.R):
                kind = action[r][0]
                if kind == "serve":
                    c_serve += 1
                elif kind == "switch":
                    c_switch += 1
                else:
                    c_idle += 1
            state, cost = env.step(state, action)
            J += disc[t] * cost
        costs.append(J)

        L_ep.append(sum_total_q / (float(T)*env.M))
        denom = float(env.R * T)
        fserve_ep.append(c_serve / denom)
        fswitch_ep.append(c_switch / denom)
        fidle_ep.append(c_idle / denom)

    costs = np.array(costs, dtype=float)
    mean = float(costs.mean())
    std = float(costs.std(ddof=1)) if episodes > 1 else 0.0
    stderr = float(std / np.sqrt(max(episodes, 1)))
    ci95 = 1.96 * stderr

    L_ep = np.array(L_ep, dtype=float)
    L_mean = float(L_ep.mean())
    L_std = float(L_ep.std(ddof=1)) if episodes > 1 else 0.0
    L_se = float(L_std / np.sqrt(max(episodes, 1)))
    L_ci95 = 1.96 * L_se

    fserve_ep = np.array(fserve_ep, dtype=float)
    fswitch_ep = np.array(fswitch_ep, dtype=float)
    fidle_ep = np.array(fidle_ep, dtype=float)

    def agg(arr: np.ndarray) -> Tuple[float, float, float]:
        m = float(arr.mean())
        s = float(arr.std(ddof=1)) if episodes > 1 else 0.0
        se = float(s / np.sqrt(max(episodes, 1)))
        return m, se, 1.96 * se

    fserve_m, fserve_se, fserve_ci = agg(fserve_ep)
    fswitch_m, fswitch_se, fswitch_ci = agg(fswitch_ep)
    fidle_m, fidle_se, fidle_ci = agg(fidle_ep)

    return {
        "discounted_cost_mean": mean,
        "discounted_cost_stderr": stderr,
        "discounted_cost_ci95": ci95,
        "avg_total_queue_len_mean": L_mean,
        "avg_total_queue_len_stderr": L_se,
        "avg_total_queue_len_ci95": L_ci95,
        "action_fraction_overall_mean": {"serve": fserve_m, "switch": fswitch_m, "idle": fidle_m},
        "action_fraction_overall_stderr": {"serve": fserve_se, "switch": fswitch_se, "idle": fidle_se},
        "action_fraction_overall_ci95": {"serve": fserve_ci, "switch": fswitch_ci, "idle": fidle_ci},
    }

def step_trace(env: QueueEnv, state: State, action: JointAction) -> Dict[str, object]:
    s, x = state
    x0 = list(x)
    cost_pre = float(sum(x0))
    a_arr = env.rng.binomial(1, env.p).astype(int)
    counts = np.zeros(env.M, dtype=int)
    for r in range(env.R):
        ar = action[r]
        if ar[0] in ("serve", "idle"):
            target = s[r]
        elif ar[0] == "switch":
            target = int(ar[1])
        else:
            raise ValueError("bad action")
        counts[target] += 1
    collision_violated = bool(np.any(counts > 1))
    x_after_service = x0.copy()
    for r in range(env.R):
        if action[r][0] == "serve" and x_after_service[s[r]] > 0:
            x_after_service[s[r]] -= 1
    new_pos = list(s)
    for r in range(env.R):
        if action[r][0] == "switch":
            new_pos[r] = int(action[r][1])
        else:
            new_pos[r] = s[r]
    x_after = x_after_service.copy()
    for i in range(env.M):
        x_after[i] = x_after[i] + int(a_arr[i])
    next_state = (tuple(new_pos), tuple(x_after))
    return {
        "s": tuple(s),
        "x": tuple(x0),
        "action": tuple(action),
        "cost_pre": cost_pre,
        "arrivals": tuple(int(v) for v in a_arr.tolist()),
        "counts": tuple(int(c) for c in counts.tolist()),
        "collision_violated": collision_violated,
        "x_after_service": tuple(x_after_service),
        "s_next": tuple(new_pos),
        "x_next": tuple(x_after),
        "next_state": next_state,
    }

def rollout_with_trace(env: QueueEnv, policy: Policy, T: int, start_state: Optional[State] = None, verbose: bool = True) -> List[Dict[str, object]]:
    if start_state is None:
        state = (tuple(range(env.R)), tuple([0] * env.M))
    else:
        state = start_state
    traces: List[Dict[str, object]] = []
    for t in range(T):
        a = policy(state, env)
        tr = step_trace(env, state, a)
        traces.append(tr)
        state = tr["next_state"]
        if verbose:
            print(
                f"t={t} s={tr['s']} x={tr['x']} a={tr['action']} cost_pre={tr['cost_pre']} arr={tr['arrivals']} "
                f"x_srv={tr['x_after_service']} s'={tr['s_next']} x'={tr['x_next']} col={tr['collision_violated']}"
            )
    return traces
# m is number of locastions and R is the number of robots
m = 6
env = QueueEnv(M=m, R=3, p=0.2*np.array([1/2]*m), beta=0.99, seed=42)
policy_greedy_results = evaluate_policy_with_metrics(env, policy_greedy_longest, T=10000, episodes=5)
print("Greedy-longest (R=2):", policy_greedy_results)

# Test cases to make sure the code works fine, just uncomment as needed
# env = QueueEnv(M=3, R=2, p=np.array([0.3, 0.5, 0.2]), beta=0.99, seed=123)
# state0 = ((0, 1), (2, 0, 1))
# rollout_with_trace(env, policy_greedy_longest, T=24, start_state=state0, verbose=True)
# test_state = ((0, 2), (0, 0, 1))
# ga = policy_greedy_longest(test_state, env)
# print("Test action@", test_state, ga)
# assert ga == (("idle", None), ("serve", None))
# state3 = ((0, 1), (0, 5, 0, 0))
# env2 = QueueEnv(M=4, R=2, p=np.array([0.2, 0.2, 0.2, 0.2]), beta=0.99, seed=1)
# ga2 = policy_greedy_longest(state3, env2)
# print("Test action@", state3, ga2)
# assert ga2 == (("idle", None), ("serve", None))
# state4 = ((0, 1), (4, 7, 0, 0))
# ga3 = policy_greedy_longest(state4, env2)
# print("Test action@", state4, ga3)
# state5 = ((0, 3), (4, 7, 0, 0))
# ga4 = policy_greedy_longest(state5, env2)
# print("Test action@", state5, ga4)
# assert ga3 == (("serve", None), ("serve", None))