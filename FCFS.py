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
            ar = action[r]
            if ar[0] in ("serve", "idle"):
                target = s[r]
            elif ar[0] == "switch":
                target = int(ar[1])
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

def assign_to_top_queues(s: Tuple[int, ...], metric: List[Optional[float]], R: int, M: int) -> JointAction:
    nonempty = [j for j in range(M) if metric[j] is not None]
    in_place = {j: sum(1 for r in range(R) if s[r] == j) for j in range(M)}
    ordered = sorted(nonempty, key=lambda j: (-float(metric[j]), -in_place.get(j, 0), j))
    K = min(R, len(ordered))
    targets = ordered[:K]

    acts: List[Action] = [None] * R  # type: ignore
    assigned = set()
    taken_q = set()

    for j in targets:
        for r in range(R):
            if r in assigned:
                continue
            if s[r] == j and j not in taken_q:
                acts[r] = ("serve", None)
                assigned.add(r)
                taken_q.add(j)
                break

    for j in targets:
        if j in taken_q:
            continue
        for r in range(R):
            if r in assigned:
                continue
            acts[r] = ("switch", j)
            assigned.add(r)
            taken_q.add(j)
            break

    for r in range(R):
        if r not in assigned:
            acts[r] = ("idle", None)

    return tuple(acts)

class PolicyFCFSPerTask:
    def __init__(self):
        self.task_ages: Optional[List[List[int]]] = None
        self.prev_state: Optional[State] = None
        self.prev_action: Optional[JointAction] = None
    def reset(self):
        self.task_ages = None
        self.prev_state = None
        self.prev_action = None
    def _init_from_state(self, state: State):
        s, x = state
        self.task_ages = [[0] * x[i] for i in range(len(x))]
    def _update_from_transition(self, prev_state: State, prev_action: JointAction, curr_state: State):
        s_prev, x_prev = prev_state
        s_curr, x_curr = curr_state
        M = len(x_prev)
        for i in range(M):
            self.task_ages[i] = [a + 1 for a in self.task_ages[i]]
        served = [0] * M
        for r, ar in enumerate(prev_action):
            if ar[0] == "serve":
                q = s_prev[r]
                if x_prev[q] > 0 and served[q] == 0:
                    served[q] = 1
        for i in range(M):
            if served[i] == 1 and self.task_ages[i]:
                k = max(range(len(self.task_ages[i])), key=lambda t: self.task_ages[i][t])
                self.task_ages[i].pop(k)
        for i in range(M):
            x_after_service = x_prev[i] - served[i]
            arrivals = x_curr[i] - x_after_service
            if arrivals > 0:
                self.task_ages[i].extend([0] * arrivals)
        for i in range(M):
            diff = len(self.task_ages[i]) - x_curr[i]
            if diff > 0:
                self.task_ages[i].sort()
                for _ in range(diff):
                    if self.task_ages[i]:
                        self.task_ages[i].pop(0)
            elif diff < 0:
                self.task_ages[i].extend([0] * (-diff))
    def __call__(self, state: State, env: QueueEnv) -> JointAction:
        if self.task_ages is None:
            self._init_from_state(state)
        elif self.prev_state is not None and self.prev_action is not None:
            self._update_from_transition(self.prev_state, self.prev_action, state)
        s, x = state
        M, R = env.M, env.R
        head = [None] * M
        for i in range(M):
            if self.task_ages[i]:
                head[i] = float(max(self.task_ages[i]))
            else:
                head[i] = None
        acts = assign_to_top_queues(s, head, R, M)
        self.prev_state = state
        self.prev_action = tuple(acts)
        return self.prev_action

def step_trace_for_fcfs(env: QueueEnv, policy: "PolicyFCFSPerTask", state: State, action: Optional[JointAction] = None) -> Dict[str, object]:
    if policy.task_ages is None:
        policy._init_from_state(state)
    if action is None:
        action = policy(state, env)
        ages_before = [list(a) for a in policy.task_ages] if policy.task_ages is not None else [[] for _ in range(env.M)]
    else:
        if policy.prev_state is not None and policy.prev_action is not None:
            tmp_sync = PolicyFCFSPerTask()
            tmp_sync.task_ages = [list(a) for a in policy.task_ages] if policy.task_ages is not None else [[] for _ in range(env.M)]
            tmp_sync._update_from_transition(policy.prev_state, policy.prev_action, state)
            ages_before = [list(a) for a in tmp_sync.task_ages]
        else:
            ages_before = [list(a) for a in policy.task_ages] if policy.task_ages is not None else [[] for _ in range(env.M)]
    head_before = [max(a) if a else None for a in ages_before]
    tr = step_trace(env, state, action)
    tmp = PolicyFCFSPerTask()
    tmp.task_ages = [list(a) for a in ages_before]
    tmp._update_from_transition(state, action, tr["next_state"])  # type: ignore[arg-type]
    ages_after = [list(a) for a in tmp.task_ages] if tmp.task_ages is not None else [[] for _ in range(env.M)]
    head_after = [max(a) if a else None for a in ages_after]
    out = dict(tr)
    out["ages_before"] = tuple(tuple(row) for row in ages_before)
    out["head_before"] = tuple(head_before)
    out["ages_after"] = tuple(tuple(row) for row in ages_after)
    out["head_after"] = tuple(head_after)
    return out

def rollout_with_trace_for_fcfs(env: QueueEnv, policy: "PolicyFCFSPerTask", T: int, start_state: Optional[State] = None, verbose: bool = True) -> List[Dict[str, object]]:
    if start_state is None:
        state = (tuple(range(env.R)), tuple([0] * env.M))
    else:
        state = start_state
    traces: List[Dict[str, object]] = []
    if policy.task_ages is None:
        policy._init_from_state(state)
    for t in range(T):
        tr = step_trace_for_fcfs(env, policy, state)
        traces.append(tr)
        state = tr["next_state"]
        if verbose:
            print(
                f"t={t} s={tr['s']} x={tr['x']} ages={tr['ages_before']} head={tr['head_before']} a={tr['action']} cost_pre={tr['cost_pre']} arr={tr['arrivals']} "
                f"x_srv={tr['x_after_service']} s'={tr['s_next']} x'={tr['x_next']} col={tr['collision_violated']} ages'={tr['ages_after']} head'={tr['head_after']}"
            )
    return traces


def print_fcfs_snapshot(ages: List[List[int]], s: Tuple[int, ...], label: str):
    M = len(ages)
    R = len(s)
    env_demo = QueueEnv(M=M, R=R, p=np.zeros(M), beta=0.99, seed=0)
    policy = PolicyFCFSPerTask()
    policy.task_ages = [list(a) for a in ages]
    policy.prev_state = None
    policy.prev_action = None
    x = tuple(len(a) for a in ages)
    state = (s, x)
    head = [max(a) if a else None for a in ages]
    act = policy(state, env_demo)
    print(f"[{label}] positions={s} queues={x}")
    print("ages per queue:")
    for i, arr in enumerate(ages):
        print(f"  Q{i}: {arr}")
    print(f"head ages: {head}")
    print(f"chosen action: {act}")

# m is number of locastions and R is the number of robots
# m = 6
# env = QueueEnv(M=m, R=3, p=0.2*np.array([1/2]*m), beta=0.99, seed=42)
# fcfs = PolicyFCFSPerTask()
# fcfs_results = evaluate_policy_with_metrics(env, fcfs, T=10000, episodes=5)
# print("FCFS-per-task (R=2):", fcfs_results)

# Test cases, uncomment as necessary
# rollout_with_trace_for_fcfs(env, fcfs, T=300, start_state=((0,1),(0,0,0,0,0,0)), verbose=True)

# print_fcfs_snapshot(ages=[[6], [5,0], [], [5]], s=(0,3), label="Example A: one oldest at Q1; serve at Q1, switch to next-oldest Q3")
# print_fcfs_snapshot(ages=[[7], [7,1], [], []], s=(2,1), label="Example B: tie on oldest between Q0 and Q1; serve at own (Q1), switch to Q0")
# print_fcfs_snapshot(ages=[[], [], [], []], s=(0,3), label="Example C: all empty; both idle")
# print_fcfs_snapshot(ages=[[], [], [], [4]], s=(3,1), label="Example D: single nonempty at Q3; serve there, other idles")
# print_fcfs_snapshot(ages=[[6], [6], [], []], s=(0,1), label="Example E: two oldest at Q0 and Q1; both serve in place")