###########################################################
            # Evaluation of model 
#########################################################

#  Imports
from new_core import *


def heatmap_final_adoption(
    X0_grid: Iterable[float],
    I0_grid: Iterable[float],
    *,
    a0: float,
    beta_I: float,
    b: float,
    g_I: float,
    n_nodes: int,
    p: float,
    m: int,
    T: int,
    trials_per_cell: int,
    network_type: str,
    strategy_choice_func: str,
    tau: float,
    seed_base: int,
    out_dir: str,
) -> Tuple[List[float], List[float], np.ndarray]:
    """Compute mean final adoption X* for every (X0, I0) cell and save CSV."""
    X0_vals = list(X0_grid)
    I0_vals = list(I0_grid)
    H = np.zeros((len(I0_vals), len(X0_vals)), dtype=float)  # rows: I0, cols: X0
    meta = {}
    for i, I0 in enumerate(I0_vals):
        for j, X0 in enumerate(X0_vals):
            finals = []
            for k in range(trials_per_cell):
                seed = seed_base + i * 10000 + j * 100 + k
                final, _ = run_trial(
                    X0_frac=X0,
                    I0=I0,
                    a0=a0,
                    beta_I=beta_I,
                    b=b,
                    g_I=g_I,
                    T=T,
                    network_type=network_type,
                    n_nodes=n_nodes,
                    p=p,
                    m=m,
                    seed=seed,
                    strategy_choice_func=strategy_choice_func,
                    tau=tau,
                    record_trajectory=False,
                )
                finals.append(final)
            H[i, j] = float(np.mean(finals))
            meta[(I0, X0)] = {"finals": finals, "mean": float(np.mean(finals)), "std": float(np.std(finals))}
    # Save CSV
    df = pd.DataFrame(H, index=[f"I0={v:.3f}" for v in I0_vals], columns=[f"X0={v:.3f}" for v in X0_vals])
    df.to_csv(os.path.join(out_dir, "heatmap_final_adoption.csv"))
    return X0_vals, I0_vals, H

def sensitivity_betaI_with_trials(
    beta_vals: Iterable[float],
    X0_values: Iterable[float],
    *,
    I0: float,
    a0: float,
    b: float,
    g_I: float,
    n_nodes: int,
    p: float,
    m: int,
    T: int,
    trials: int,
    network_type: str,
    strategy_choice_func: str,
    tau: float,
    seed_base: int,
    out_dir: str,
) -> Tuple[List[float], List[float], np.ndarray]:
    """
    Compute final adoption vs beta_I for each initial X0 in X0_values.
    Returns all trial results (shape: [X0, beta_I, trial]) for mean ± std plotting.
    """
    beta_vals = list(beta_vals)
    X0_values = list(X0_values)
    all_trials = np.zeros((len(X0_values), len(beta_vals), trials), dtype=float)

    for i, X0 in enumerate(X0_values):
        for j, beta in enumerate(beta_vals):
            for k in range(trials):
                seed = seed_base + i * 10000 + j * 100 + k
                final, _ = run_trial(
                    X0_frac=X0,
                    I0=I0,
                    a0=a0,
                    beta_I=beta,
                    b=b,
                    g_I=g_I,
                    T=T,
                    network_type=network_type,
                    n_nodes=n_nodes,
                    p=p,
                    m=m,
                    seed=seed,
                    strategy_choice_func=strategy_choice_func,
                    tau=tau,
                    record_trajectory=False,
                )
                all_trials[i, j, k] = final

    # Optional: save mean results to CSV
    means = all_trials.mean(axis=2)
    df = pd.DataFrame(means, index=[f"X0={v:.3f}" for v in X0_values],
                      columns=[f"beta_I={v:.3f}" for v in beta_vals])
    df.to_csv(os.path.join(out_dir, "sensitivity_betaI_mean.csv"))

    return X0_values, beta_vals, all_trials    

def collect_trajectories(
    cases: List[Tuple[float, float]],
    *,
    a0: float,
    beta_I: float,
    b: float,
    g_I: float,
    n_nodes: int,
    p: float,
    m: int,
    T: int,
    network_type: str,
    strategy_choice_func: str,
    tau: float,
    seed_base: int,
    out_dir: str,
) -> Dict[Tuple[float, float], Dict]:
    """Run representative trajectories and save time-series / phase plots."""
    trajs = {}
    for idx, (X0, I0) in enumerate(cases):
        seed = seed_base + idx * 100
        final, (X_traj, I_traj) = run_trial(
            X0_frac=X0,
            I0=I0,
            a0=a0,
            beta_I=beta_I,
            b=b,
            g_I=g_I,
            T=T,
            network_type=network_type,
            n_nodes=n_nodes,
            p=p,
            m=m,
            seed=seed,
            strategy_choice_func=strategy_choice_func,
            tau=tau,
            record_trajectory=True,
        )
        trajs[(X0, I0)] = {"final": final, "X": X_traj, "I": I_traj}
        # Save plots
        t = np.arange(len(X_traj))
        plt.figure(figsize=(6, 4))
        plt.plot(t, X_traj, label="X(t)")
        plt.plot(t, I_traj, label="I(t)")
        plt.xlabel("time step")
        plt.ylabel("value")
        plt.title(f"Time series X/I  X0={X0:.2f}, I0={I0:.2f}")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"timeseries_X0_{int(X0*100)}_I0_{int(I0*100)}.png"))
        plt.close()

        plt.figure(figsize=(5, 5))
        plt.plot(I_traj, X_traj, marker=".", linewidth=1)
        plt.xlabel("I(t)")
        plt.ylabel("X(t)")
        plt.title(f"Phase plot X vs I  X0={X0:.2f}, I0={I0:.2f}")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"phase_X0_{int(X0*100)}_I0_{int(I0*100)}.png"))
        plt.close()
    return trajs


# Heatmap analysis 
def run_heatmap_analysis(
    out_dir,
    X0_grid,
    I0_grid,
    a0,
    beta_I,
    b,
    g_I,
    n_nodes,
    p,
    m,
    T,
    trials_per_cell,
    network_type,
    strategy_choice_func,
    tau,
    seed_base,
):
    print("Starting heatmap sweep (this may take some time)...")

    X0_vals, I0_vals, H = heatmap_final_adoption(
        X0_grid=X0_grid,
        I0_grid=I0_grid,
        a0=a0,
        beta_I=beta_I,
        b=b,
        g_I=g_I,
        n_nodes=n_nodes,
        p=p,
        m=m,
        T=T,
        trials_per_cell=trials_per_cell,
        network_type=network_type,
        strategy_choice_func=strategy_choice_func,
        tau=tau,
        seed_base=seed_base,
        out_dir=out_dir,
    )

    # Save heatmap figure
    plt.figure(figsize=(7, 5))
    im = plt.imshow(
        H,
        origin="lower",
        aspect="auto",
        extent=[X0_vals[0], X0_vals[-1], I0_vals[0], I0_vals[-1]],
    )
    plt.colorbar(im, label=r'Mean final adoption $X^*$')
    plt.xlabel(r'Initial adoption $X_0$')
    plt.ylabel(r'Initial infrastructure $I_0$')
    plt.title(fr'Heatmap: Mean final adoption $X^*$ vs ($X_0$, $I_0$) ($\beta_I$={beta_I:.2f})')
    plt.tight_layout()
    fname = os.path.join(out_dir, "heatmap_final_adoption.png")
    plt.savefig(fname)
    plt.close()

    print("Heatmap saved to:", fname)

    return X0_vals, I0_vals, H


# Sensitivity analysis 
def run_sensitivity_analysis(
    out_dir,
    beta_vals,
    X0_sens,
    I0,
    a0,
    b,
    g_I,
    n_nodes,
    p,
    m,
    T,
    trials,
    network_type,
    strategy_choice_func,
    tau,
    seed_base,
):
    print("Starting sensitivity sweep over beta_I ...")

    X0_list, betas, sens_res = sensitivity_betaI_with_trials(
        beta_vals=beta_vals,
        X0_values=X0_sens,
        I0=I0,
        a0=a0,
        b=b,
        g_I=g_I,
        n_nodes=n_nodes,
        p=p,
        m=m,
        T=T,
        trials=trials,
        network_type=network_type,
        strategy_choice_func=strategy_choice_func,
        tau=tau,
        seed_base=seed_base,
        out_dir=out_dir,
    )

    # Compute mean across trials
    mean_vals = np.mean(sens_res, axis=2)

    # Plot only the mean
    plt.figure(figsize=(7, 5))
    for i, x0 in enumerate(X0_list):
        plt.plot(betas, mean_vals[i, :], label=f"X0={x0:.2f}", marker="o", linewidth=1)

    plt.ylim(-0.05, 1.05)
    plt.xlabel(r'$\beta_I$')
    plt.ylabel(r'Mean Final Adoption $X^*$')
    plt.title(r'Sensitivity: Final Adoption vs $\beta_I$')
    plt.legend(title=r"Initial Adoption $X_0$")
    plt.tight_layout()

    fname = os.path.join(out_dir, "sensitivity_betaI_mean_only.png")
    plt.savefig(fname)
    plt.close()
    print("Sensitivity plot saved to:", fname)

    return X0_list, betas, mean_vals


# Phase trajectory plots 
def run_phase_plots(
    out_dir,
    phase_cases,
    a0,
    beta_I,
    b,
    g_I,
    n_nodes,
    p,
    m,
    T,
    network_type,
    strategy_choice_func,
    tau,
    seed_base,
):
    print("Collecting representative trajectories...")

    trajs = collect_trajectories(
        phase_cases,
        a0=a0,
        beta_I=beta_I,
        b=b,
        g_I=g_I,
        n_nodes=n_nodes,
        p=p,
        m=m,
        T=T,
        network_type=network_type,
        strategy_choice_func=strategy_choice_func,
        tau=tau,
        seed_base=seed_base,
        out_dir=out_dir,
    )

    print("Phase plots saved to:", out_dir)
    return trajs




###########################################
#      Network Structure Analysis 
#############################################

# Speed of adoption
    # Quantified in a number
def calculate_adoption_rate(X_traj: np.ndarray) -> float:
    initial_adoption = X_traj[0]
    final_adoption = X_traj[-1]
    total_steps = len(X_traj)
    return (final_adoption - initial_adoption) / total_steps
    
    # Time to treashold ~> high adoption
        # May not capture overall dynamics 
def time_to_reach_threshold(X_traj: np.ndarray, threshold: float) -> int:
    for t, X in enumerate(X_traj):
        if X >= threshold:
            return t
    return len(X_traj)  # Return total steps if threshold not reached


# Probabaility of reaching high-adoption equilibrium 
    # Finan adoption fractio, but no account for fluctuation 
def calculate_final_adoption(X_traj: np.ndarray) -> float:
    return X_traj[-1]
    
    # Stability analysis 
def is_high_adoption_equilibrium(X_traj: np.ndarray, threshold: float) -> bool:
    return np.mean(X_traj[-10:]) >= threshold  # Check last 10 steps for stability

    # Probalistic measure to reach high adoption
def monte_carlo_high_adoption_probability(trials: int, model_params: dict, threshold: float) -> float:
    successes = 0
    for _ in range(trials):
        final_adoption, _ = run_trial(**model_params)
        if final_adoption >= threshold:
            successes += 1
    return successes / trials

# Cluster formation 
    # Clustering coeficient analysis ~> tendency 
def calculate_clustering_coefficient(G: nx.Graph) -> float:
    return nx.average_clustering(G)

    # Spatial Adoption patterns  
def plot_spatial_adoption(G: nx.Graph, node_agent_map: Dict[int, EVAgent], out_dir: str):
    fig, ax = plt.subplots(figsize=(6,6))
    pos = nx.spring_layout(G, seed=42)
    color_map = ["green" if agent.strategy=="C" else "red" for agent in node_agent_map.values()]
    nx.draw(G, pos, node_color=color_map, with_labels=True, ax=ax)
    plt.savefig(os.path.join(out_dir, "spatial_adoption.png"))
    plt.close(fig)  

    