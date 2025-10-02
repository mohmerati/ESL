import matplotlib.pyplot as plt

def summarize_metrics_row(metrics: Dict[str, object],
                          M: int, R: int, alpha: float, p_scalar: float,
                          policy_name: str) -> Dict[str, object]:
    af_mean = metrics["action_fraction_overall_mean"]
    af_ci   = metrics["action_fraction_overall_ci95"]
    return {
        "M": M, "R": R, "alpha": alpha, "p": p_scalar, "policy": policy_name,
        "cost_mean": metrics["discounted_cost_mean"],
        "cost_ci95": metrics["discounted_cost_ci95"],
        "ql_mean": metrics["avg_total_queue_len_mean"],
        "ql_ci95": metrics["avg_total_queue_len_ci95"],
        "serve_mean":  af_mean["serve"],  "serve_ci95":  af_ci["serve"],
        "switch_mean": af_mean["switch"], "switch_ci95": af_ci["switch"],
        "idle_mean":   af_mean["idle"],   "idle_ci95":   af_ci["idle"],
    }

def run_grid_and_collect(policies: Dict[str, Policy],
                         M_values=(6,), R_values=(2,3), alphas=(0.2,0.5,0.8),
                         T=10000, episodes=30, beta=0.99, base_seed=42) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for M in M_values:
        for R in R_values:
            ratio = R / M
            for alpha in alphas:
                p_scalar = min(alpha * ratio, 0.85)  # safety cap
                p_vec = np.full(M, p_scalar, dtype=float)
                for policy_name, pol in policies.items():
                    env = QueueEnv(M=M, R=R, p=p_vec, beta=beta, seed=base_seed)
                    metrics = evaluate_policy_with_metrics(env, pol, T=T, episodes=episodes)
                    rows.append(summarize_metrics_row(metrics, M, R, alpha, p_scalar, policy_name))
    return rows

def _filter(rows: List[Dict[str, object]], M: int, R: int) -> List[Dict[str, object]]:
    return [r for r in rows if r["M"] == M and r["R"] == R]

def plot_cost(rows_MR: List[Dict[str, object]], policies_order: List[str], alphas: List[float], title: str):
    fig = plt.figure()
    width = 0.2
    x_base = np.arange(len(alphas))
    for i, pol in enumerate(policies_order):
        means = [next(r["cost_mean"] for r in rows_MR if r["policy"]==pol and r["alpha"]==a) for a in alphas]
        errs  = [next(r["cost_ci95"] for r in rows_MR if r["policy"]==pol and r["alpha"]==a) for a in alphas]
        plt.bar(x_base + i*width, means, width=width, yerr=errs, capsize=4, label=pol)
    plt.xticks(x_base + width*(len(policies_order)-1)/2.0, [f"α={a:.1f}" for a in alphas])
    plt.xlabel("Load scaling α")
    plt.ylabel("Discounted expected holding cost")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    return fig

def plot_ql(rows_MR: List[Dict[str, object]], policies_order: List[str], alphas: List[float], title: str):
    fig = plt.figure()
    width = 0.2
    x_base = np.arange(len(alphas))
    for i, pol in enumerate(policies_order):
        means = [next(r["ql_mean"] for r in rows_MR if r["policy"]==pol and r["alpha"]==a) for a in alphas]
        errs  = [next(r["ql_ci95"] for r in rows_MR if r["policy"]==pol and r["alpha"]==a) for a in alphas]
        plt.bar(x_base + i*width, means, width=width, yerr=errs, capsize=4, label=pol)
    plt.xticks(x_base + width*(len(policies_order)-1)/2.0, [f"α={a:.1f}" for a in alphas])
    plt.xlabel("Load scaling α")
    plt.ylabel("Mean queue length per location")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    return fig

def plot_actions(rows_MR: List[Dict[str, object]], policies_order: List[str], alphas: List[float], title: str):
    # Group by (action, alpha); bars = policies
    actions = ["serve", "switch", "idle"]
    groups = [(act, a) for act in actions for a in alphas]
    fig = plt.figure()
    width = 0.2
    x_base = np.arange(len(groups))
    for i, pol in enumerate(policies_order):
        means = []
        errs  = []
        for act, a in groups:
            mean_key = f"{act}_mean"
            ci_key   = f"{act}_ci95"
            rec = next(r for r in rows_MR if r["policy"]==pol and r["alpha"]==a)
            means.append(rec[mean_key])
            errs.append(rec[ci_key])
        plt.bar(x_base + i*width, means, width=width, yerr=errs, capsize=4, label=pol)
    xt = [f"{act}\nα={a:.1f}" for (act,a) in groups]
    plt.xticks(x_base + width*(len(policies_order)-1)/2.0, xt)
    plt.xlabel("Action × Load")
    plt.ylabel("Fraction of time")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    return fig


policies = {
    "ESL": policy_greedy_longest,
    "FCFS": PolicyFCFSPerTask(),
    "cyclic": PolicyCyclicFixedDwell(),  # consider PolicyCyclicFixedDwell(p_ref=<your p>) for per-scenario consistency
}

M_values = (6,)
R_values = (2, 3)
alphas   = [0.2, 0.5, 0.8]
rows = run_grid_and_collect(policies, M_values=M_values, R_values=R_values, alphas=alphas,
                            T=10000, episodes=100, beta=0.99, base_seed=42)

def save_results_csv(rows: List[Dict[str, object]], path: str = "results.csv"):
    # Pure-python CSV writer to avoid new deps
    if not rows:
        return
    keys = list(rows[0].keys())
    with open(path, "w", encoding="utf-8") as f:
        f.write(",".join(keys) + "\n")
        for r in rows:
            vals = []
            for k in keys:
                v = r[k]
                if isinstance(v, float):
                    vals.append(f"{v:.10g}")
                else:
                    vals.append(str(v))
            f.write(",".join(vals) + "\n")

save_results_csv(rows, "results.csv")
print(f"Collected {len(rows)} rows")

# Make six plots: 3 metrics × 2 ratios
for R in R_values:
    rows_MR = _filter(rows, M=6, R=R)
    title_suffix = f"(N=6, M={R})"
    plot_cost(rows_MR, policies_order=list(policies.keys()), alphas=alphas,
                title=f"Discounted Cost {title_suffix}")
    plot_ql(rows_MR, policies_order=list(policies.keys()), alphas=alphas,
            title=f"Mean Queue Length {title_suffix}")
    plot_actions(rows_MR, policies_order=list(policies.keys()), alphas=alphas,
                    title=f"Action Fractions {title_suffix}")

plt.show()
