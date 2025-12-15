# Policy implemnentation 


#  Imports
from new_core import *
from new_plotting import *


# --- 1. Helper Function for Targeted Seeding ---
def apply_seeding(model, X0_frac, method="random"):
    """
    Applies specific seeding strategies to the provided model instance.
    Adapts the logic to work with model.node_agent_map.
    """
    nodes = list(model.G.nodes())
    total_nodes = len(nodes)
    k_ev = int(round(X0_frac * total_nodes))
    
    # Reset all agents to Defect first
    for agent in model.node_agent_map.values():
        agent.strategy = "D"
        agent.next_strategy = "D"

    target_nodes = []

    if method == "random":
        target_nodes = random.sample(nodes, k_ev)
        
    elif method == "degree":
        # Sort nodes by degree (high to low)
        degrees = dict(model.G.degree())
        # Sort keys by value descending
        sorted_nodes = sorted(degrees, key=degrees.get, reverse=True)
        target_nodes = sorted_nodes[:k_ev]

    # Apply "C" strategy to chosen targets
    for node_id in target_nodes:
        model.node_agent_map[node_id].strategy = "C"
        model.node_agent_map[node_id].next_strategy = "C"

# --- 2. Simulation Runner ---
def run_batch(n_trials, steps, method, params):
    """
    Runs a batch of simulations and returns trajectories and final states.
    """
    X_trajectories = []
    I_trajectories = []
    final_X = []
    final_I = []

    for i in range(n_trials):
        # Initialize model with 0 initial EVs so we can manually seed them
        model = EVStagHuntModel(
            initial_ev=0, 
            a0=params['a0'], beta_I=params['beta_I'], b=params['b'], 
            g_I=params['g_I'], I0=params['I0'], 
            seed=None, # Random seed for stochasticity
            network_type=params['network_type'], 
            n_nodes=params['n_nodes'], p=params['p'], m=params['m'],
            strategy_choice_func=params['func']
        )
        
        # Apply the specific seeding method (Random vs Degree)
        apply_seeding(model, params['X0_frac'], method=method)
        
        # storage for this trial
        trial_X = []
        trial_I = []
        
        # Run time steps
        for t in range(steps):
            trial_X.append(model.get_adoption_fraction())
            trial_I.append(model.infrastructure)
            model.step()
            
        # Capture final state
        trial_X.append(model.get_adoption_fraction())
        trial_I.append(model.infrastructure)
        
        X_trajectories.append(trial_X)
        I_trajectories.append(trial_I)
        final_X.append(trial_X[-1])
        final_I.append(trial_I[-1])

    return np.array(X_trajectories), np.array(I_trajectories), np.array(final_X), np.array(final_I)


def execute_simulations_and_visualize(TRIALS, STEPS, PARAMS):
    # --- 4. Execute Simulations ---
    print("Running Baseline (Random Seeding)...")
    base_X_traj, base_I_traj, base_final_X, base_final_I = run_batch(TRIALS, STEPS, "random", PARAMS)

    print("Running Intervention (Degree Seeding)...")
    seed_X_traj, seed_I_traj, seed_final_X, seed_final_I = run_batch(TRIALS, STEPS, "degree", PARAMS)

    # Calculate Means
    base_mean_X = np.mean(base_X_traj, axis=0)
    base_mean_I = np.mean(base_I_traj, axis=0)
    seed_mean_X = np.mean(seed_X_traj, axis=0)
    seed_mean_I = np.mean(seed_I_traj, axis=0)

    # --- 5. Visualization ---
    fig, axs = plt.subplots(2, 2, figsize=(14, 10))
    time_axis = range(STEPS + 1)

    # Style settings
    color_X = '#1f77b4' # Blue
    color_I = '#ff7f0e' # Orange
    alpha_hist = 0.6
    bins_count = 20

    # --- Top Left: Baseline Dynamics ---
    axs[0, 0].plot(time_axis, base_mean_X, label=r'Mean Adoption ($X^*$)', color=color_X, linewidth=2)
    axs[0, 0].plot(time_axis, base_mean_I, label=r'Mean Infrastructure ($I^*$)', color=color_I, linewidth=2, linestyle='--')
    axs[0, 0].set_title(r'Baseline: Random Seeding ($X_0=0.20$ $I_0=0.00$)', fontsize=12, fontweight='bold')
    axs[0, 0].set_xlabel('Time Step')
    axs[0, 0].set_ylabel('Fraction')
    axs[0, 0].set_ylim(0, 1.05)
    axs[0, 0].grid(True, alpha=0.3)
    axs[0, 0].legend()

    # --- Bottom Left: Baseline Histogram ---
    axs[1, 0].hist(
        [base_final_X, base_final_I], # Group data into a list for combined plotting
        bins=bins_count, 
        range=(0,1), 
        alpha=alpha_hist, 
        color=[color_X, color_I], # Set colors for the two datasets
        label=['Final Adoption ($X$)', 'Final Infrastructure ($I$)']
    )
    axs[1, 0].set_title(r'Baseline: Distribution of Final Outcomes ($X_0=0.20$ $I_0=0.00$)', fontsize=12, fontweight='bold')
    axs[1, 0].set_xlabel('Value (0-1)')
    axs[1, 0].set_ylabel('Frequency')
    axs[1, 0].set_ylim (0, 105)
    axs[1, 0].legend()

    # --- Top Right: Targeted Seeding Dynamics ---
    axs[0, 1].plot(time_axis, seed_mean_X, label=r'Mean Adoption ($X^*$)', color=color_X, linewidth=2)
    axs[0, 1].plot(time_axis, seed_mean_I, label=r'Mean Infrastructure ($I^*$)', color=color_I, linewidth=2, linestyle='--')
    axs[0, 1].set_title(r'Intervention: Degree Centrality Seeding ($X_0=0.20$ $I_0=0.00$)', fontsize=12, fontweight='bold')
    axs[0, 1].set_xlabel('Time Step')
    axs[0, 1].set_ylabel('Fraction')
    axs[0, 1].set_ylim(0, 1.05)
    axs[0, 1].grid(True, alpha=0.3)
    axs[0, 1].legend()

    # --- Bottom Right: Targeted Seeding Histogram ---
    axs[1, 1].hist(
        [seed_final_X, seed_final_I], # Group data into a list for combined plotting
        bins=bins_count, 
        range=(0,1), 
        alpha=alpha_hist, 
        color=[color_X, color_I], # Set colors for the two datasets
        label=['Final Adoption ($X$)', 'Final Infrastructure ($I$)']
    )
    axs[1, 1].set_title(r'Intervention: Distribution of Final Outcomes ($X_0=0.20$ $I_0=0.00$)', fontsize=12, fontweight='bold')
    axs[1, 1].set_xlabel('Value (0-1)')
    axs[1, 1].set_ylabel('Frequency')
    axs[1, 1].set_ylim (0, 105)
    axs[1, 1].legend()

    plt.tight_layout()
    plt.show()



#######################
# --- Helper function for color lightening ---
def lighten_color(hex_color, factor=0.5):
    """
    Lightens a given color by blending it with white.
    Factor of 0.0 is original color, 1.0 is white.
    """
    # Convert hex to RGB
    rgb_tuple = tuple(int(hex_color.lstrip('#')[i:i+2], 16) for i in (0, 2, 4))
    
    # Calculate new RGB values blended with white (255)
    new_rgb = [int(r + (255 - r) * factor) for r in rgb_tuple]
    
    # Convert new RGB back to hex
    return f'#{new_rgb[0]:02x}{new_rgb[1]:02x}{new_rgb[2]:02x}'


def get_key_prefix(seeding_method):
    return 'Rand' if seeding_method == 'random' else 'Deg'


def run_all_scenarios(
    RESULTS,
    TRIALS,
    STEPS,
    PARAMS,
    SHOCK_TIMES,
    SHOCK_MAG,
    run_batch,
    get_key_prefix
):
    print("Running Baseline 1: Random Seeding, No Shocks...")
    X_traj, I_traj, final_X, final_I = run_batch(TRIALS, STEPS, "random", PARAMS)
    RESULTS['Rand_NoShock'] = (X_traj, I_traj, final_X, final_I)

    print("Running Baseline 2: Targeted Seeding, No Shocks...")
    X_traj, I_traj, final_X, final_I = run_batch(TRIALS, STEPS, "degree", PARAMS)
    RESULTS['Deg_NoShock'] = (X_traj, I_traj, final_X, final_I)

    for t_shock in SHOCK_TIMES:
        key = f'{get_key_prefix("random")}_Shock_{t_shock}'
        print(f"Running Scenario: Random Seeding, Shock at t={t_shock}...")
        X_traj, I_traj, final_X, final_I = run_batch(
            TRIALS,
            STEPS,
            "random",
            PARAMS,
            shock_steps={t_shock},
            shock_magnitude=SHOCK_MAG
        )
        RESULTS[key] = (X_traj, I_traj, final_X, final_I)

    for t_shock in SHOCK_TIMES:
        key = f'{get_key_prefix("degree")}_Shock_{t_shock}'
        print(f"Running Scenario: Targeted Seeding, Shock at t={t_shock}...")
        X_traj, I_traj, final_X, final_I = run_batch(
            TRIALS,
            STEPS,
            "degree",
            PARAMS,
            shock_steps={t_shock},
            shock_magnitude=SHOCK_MAG
        )
        RESULTS[key] = (X_traj, I_traj, final_X, final_I)



##############################################
#        Infastructure Shocks 
###############################################3


# New run batch withouth final states     
def run_batch_single_shock(n_trials, steps, method, params, shock_time, shock_amount):
    """
    Runs simulations with a single global infrastructure shock at a given time.
    shock_amount: additive change to I (e.g., +0.2 or -0.1)
    shock_time: timestep where shock occurs
    """
    X_trajectories = []
    I_trajectories = []

    for _ in range(n_trials):
        model = EVStagHuntModel(
            initial_ev=0,
            a0=params['a0'], beta_I=params['beta_I'], b=params['b'],
            g_I=params['g_I'], I0=params['I0'], seed=None,
            network_type=params['network_type'], n_nodes=params['n_nodes'],
            p=params['p'], m=params['m'],
            strategy_choice_func=params['func']
        )

        apply_seeding(model, params['X0_frac'], method)

        trial_X = []
        trial_I = []

        for t in range(steps):
            # Apply global infrastructure shock
            if t == shock_time:
                model.infrastructure = np.clip(model.infrastructure + shock_amount, 0, 1)

            trial_X.append(model.get_adoption_fraction())
            trial_I.append(model.infrastructure)
            model.step()

        # Add final state
        trial_X.append(model.get_adoption_fraction())
        trial_I.append(model.infrastructure)

        X_trajectories.append(trial_X)
        I_trajectories.append(trial_I)

    return np.array(X_trajectories), np.array(I_trajectories)


# General infastructure shocks
def run_multiple_single_shocks_overlay(n_trials, steps, method, params, shock_times, shock_amount):
    all_X_traj, all_I_traj, labels = [], [], []
    for t_shock in shock_times:
        X_traj, I_traj = run_batch_single_shock(
            n_trials, steps, method, params,
            shock_time=t_shock, shock_amount=shock_amount
        )
        all_X_traj.append(X_traj)
        all_I_traj.append(I_traj)
        labels.append(f"Shock at t={t_shock}")

    # --- Mean Trajectories Plot ---
    time_axis = range(steps + 1)
    plt.figure(figsize=(12,6))

    colors_X = ["blue", "red", "darkgreen"]   # X colors for each shock
    colors_I = ["lightblue", "darkorange", "lightgreen"]  # I colors for each shock

    for i in range(len(shock_times)):
        mean_X = np.mean(all_X_traj[i], axis=0)
        mean_I = np.mean(all_I_traj[i], axis=0)
        plt.plot(time_axis, mean_X, color=colors_X[i], label=f'X(t) - {labels[i]}', linewidth=2)
        plt.plot(time_axis, mean_I, color=colors_I[i], linestyle='--', label=f'I(t) - {labels[i]}', linewidth=2)

    plt.title("Mean Adoption and Infrastructure for Multiple Shock Times")
    plt.xlabel("Time Step")
    plt.ylabel("Fraction")
    plt.ylim(0,1.05)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.show()

    # --- Side-by-Side Histogram with Space Between X and I ---
    plt.figure(figsize=(14,6))
    bins_count = 20
    bin_edges = np.linspace(0,1,bins_count+1)
    n_shocks = len(shock_times)
    total_width = (bin_edges[1]-bin_edges[0])
    width = total_width / 3  # smaller width to add space between X/I sets

    for i in range(n_shocks):
        final_X = all_X_traj[i][:,-1]
        final_I = all_I_traj[i][:,-1]
        offset_X = i*total_width*2   # space between shock sets
        offset_I = offset_X + width  # small gap between X and I bars

        plt.bar(bin_edges[:-1]+offset_X, np.histogram(final_X, bins=bin_edges)[0],
                width=width, color=colors_X[i], alpha=0.75, label=f'Final X - {labels[i]}')
        plt.bar(bin_edges[:-1]+offset_I, np.histogram(final_I, bins=bin_edges)[0],
                width=width, color=colors_I[i], alpha=0.75, label=f'Final I - {labels[i]}')

    plt.title("Distribution of Final Adoption and Infrastructure (Side-by-Side with Space)")
    plt.xlabel("Value (0–1)")
    plt.ylabel("Frequency")
    plt.legend()
    plt.show()

############################


def plot_results(RESULTS, STEPS, SHOCK_TIMES):
    # --- 5. Visualization (4 Rows, 2 Columns) ---
    RESULTS = {}

    # Plotting setup
    fig, axs = plt.subplots(4, 2, figsize=(12, 16), sharex='col')
    time_axis = range(STEPS + 1)
    BASE_I_COLOR = '#ff7f0e' 
    alpha_hist = 0.6
    bins_count = 20
    y_lim_hist_freq = 105 
    LIGHTENING_FACTOR = 0.5 

    # Color palette for the three adoption trajectories (keys updated)
    COLOR_X_SHOCKS = {
        25: '#1f77b4',  # Blue (Early shock)
        75: '#2ca02c', # Green (Mid shock)
        125: '#d62728', # Red (Late shock)
    }

    # New plotting approach: Iterate over (seeding_method, has_shock) pairs
    PLOT_ORDER = [
        # Row 0: Random Seeding, No Shocks (Dynamics & Histograms)
        {'row': 0, 'seeding_method': 'random', 'has_shock': False, 'key_prefix': 'Rand', 'base_color_key': 25},
        # Row 1: Targeted Seeding, No Shocks (Dynamics & Histograms)
        {'row': 1, 'seeding_method': 'degree', 'has_shock': False, 'key_prefix': 'Deg', 'base_color_key': 25},
        # Row 2: Random Seeding, With Shocks (Dynamics & Histograms)
        {'row': 2, 'seeding_method': 'random', 'has_shock': True, 'key_prefix': 'Rand'},
        # Row 3: Targeted Seeding, With Shocks (Dynamics & Histograms)
        {'row': 3, 'seeding_method': 'degree', 'has_shock': True, 'key_prefix': 'Deg'},
    ]

    # --- Legend Size Consistency ---
    LEGEND_FONT_NO_SHOCK = 8
    LEGEND_FONT_SHOCK = 7

    for item in PLOT_ORDER:
        row = item['row']
        seeding_method = item['seeding_method']
        has_shock = item['has_shock']
        prefix = item['key_prefix']
        
        # --- Data Retrieval ---
        if not has_shock:
            key = f"{prefix}_NoShock"
            X_data, I_data, final_X, final_I = RESULTS[key]
            base_color = COLOR_X_SHOCKS[item['base_color_key']]
        else:
            no_shock_key = f"{prefix}_NoShock"
            _, I_no_shock, _, _ = RESULTS[no_shock_key]

        # --- Column 0: Dynamics Trajectories (X/I over time) ---
        ax_dyn = axs[row, 0]
        
        # Title Setup
        if not has_shock:
            ax_dyn.set_title(f'{seeding_method.capitalize()} Seeding: No I Shocks', fontsize=11, fontweight='bold')
        else:
            ax_dyn.set_title(f'{seeding_method.capitalize()} Seeding: I Shocks', fontsize=11, fontweight='bold')

        ax_dyn.set_ylabel('Fraction (X, I)')
        ax_dyn.set_ylim(0, 1.05)
        ax_dyn.grid(True, alpha=0.3)
        
        # Dynamics Plotting Logic
        if not has_shock:
            # Plot Base (No Shock)
            mean_X = np.mean(X_data, axis=0)
            mean_I = np.mean(I_data, axis=0)
            
            ax_dyn.plot(time_axis, mean_X, label=r'Mean Adoption ($\bar{X}$)', color=base_color, linewidth=2)
            ax_dyn.plot(time_axis, mean_I, label=r'Mean Infrastructure ($\bar{I}$)', color=BASE_I_COLOR, linewidth=2, linestyle='--')
            
            ax_dyn.legend(loc='upper left', fontsize=LEGEND_FONT_NO_SHOCK) 
            
        else:  # Shock scenarios
            for t_shock in SHOCK_TIMES:
                shock_key = f'{prefix}_Shock_{t_shock}'
                X_shock, I_shock, _, _ = RESULTS[shock_key]
                
                shock_color = COLOR_X_SHOCKS[t_shock]

                mean_X_shock = np.mean(X_shock, axis=0)
                ax_dyn.plot(
                    time_axis,
                    mean_X_shock,
                    label=f'Adoption $t={t_shock}$',
                    color=shock_color,
                    linewidth=2
                )

                mean_I_shock = np.mean(I_shock, axis=0)
                ax_dyn.plot(
                    time_axis,
                    mean_I_shock,
                    label=f'I $t={t_shock}$',
                    color=shock_color,
                    linewidth=2,
                    linestyle='--'
                )
                
                ax_dyn.axvline(x=t_shock, color=shock_color, linestyle=':', alpha=0.7)

            ax_dyn.legend(loc='upper left', fontsize=LEGEND_FONT_SHOCK, ncol=2)

        # --- Column 1: Final Outcomes Distribution (Histograms) ---
        ax_hist = axs[row, 1]
        
        # Title Setup
        if not has_shock:
            ax_hist.set_title(f'{seeding_method.capitalize()} Seeding: No I Shocks (Final Outcomes)', fontsize=11, fontweight='bold')
        else:
            ax_hist.set_title(f'{seeding_method.capitalize()} Seeding: I Shocks (Final Outcomes)', fontsize=11, fontweight='bold')

        ax_hist.set_xlabel('Value (0-1)')
        ax_hist.set_ylabel('Frequency')
        ax_hist.set_ylim(0, y_lim_hist_freq) 
        ax_hist.grid(True, alpha=0.3)
        
        # Histogram Plotting Logic
        if not has_shock:
            light_base_I_color = lighten_color(BASE_I_COLOR, factor=LIGHTENING_FACTOR)
            
            ax_hist.hist(
                [final_X, final_I],
                bins=bins_count, range=(0, 1), alpha=alpha_hist,
                color=[base_color, light_base_I_color],
                label=[r'Final Adoption ($X_T$)', r'Final Infrastructure ($I_T$)']
            )
            ax_hist.legend(loc='best', fontsize=LEGEND_FONT_NO_SHOCK)
        else:
            hist_data = [RESULTS[f'{prefix}_Shock_{t}'][2] for t in SHOCK_TIMES]
            hist_colors_X = [COLOR_X_SHOCKS[t] for t in SHOCK_TIMES]
            hist_labels = [f'Final Adoption $t={t}$' for t in SHOCK_TIMES]

            I_shock_data = [RESULTS[f'{prefix}_Shock_{t}'][3] for t in SHOCK_TIMES]
            hist_colors_I = [lighten_color(COLOR_X_SHOCKS[t], factor=LIGHTENING_FACTOR) for t in SHOCK_TIMES]
            I_shock_labels = [f'Final Infrastructure $t={t}$' for t in SHOCK_TIMES]

            combined_hist_data = hist_data + I_shock_data
            combined_hist_colors = hist_colors_X + hist_colors_I
            combined_hist_labels = hist_labels + I_shock_labels

            ax_hist.hist(
                combined_hist_data,
                bins=bins_count, range=(0, 1), alpha=alpha_hist,
                color=combined_hist_colors,
                label=combined_hist_labels
            )
            ax_hist.legend(loc='best', fontsize=LEGEND_FONT_SHOCK, ncol=2)

    # Set common X labels for the bottom row of each plot type
    axs[3, 0].set_xlabel('Time Step')
    axs[3, 1].set_xlabel('Value (0-1)')

    # Adjust layout to prevent overlap and display the figure
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    plt.show()





    

##########################################
#    Additional policy implementation   
########################################

def apply_targeted_shock(model, shock_frac, method="degree"):
    """
    Applies a one-time targeted adoption shock to currently non-adopting agents.
    """
    nodes = list(model.G.nodes())
    total_nodes = len(nodes)
    k_shock = int(round(shock_frac * total_nodes))

    # Identify current defectors only
    defectors = [
        n for n in nodes
        if model.node_agent_map[n].strategy == "D"
    ]

    if len(defectors) == 0:
        return

    if method == "random":
        targets = random.sample(defectors, min(k_shock, len(defectors)))

    elif method == "degree":
        degrees = dict(model.G.degree(defectors))
        targets = sorted(degrees, key=degrees.get, reverse=True)[:k_shock]

    for n in targets:
        model.node_agent_map[n].strategy = "C"
        model.node_agent_map[n].next_strategy = "C"

        
def run_batch_single_shock(
    n_trials, steps, method, params, shock_time, shock_amount
):
    """
    Runs simulations with NO targeted seeding at t=0.
    A single discrete shock is applied at shock_time.
    """
    X_trajectories = []
    I_trajectories = []

    for _ in range(n_trials):
        model = EVStagHuntModel(
            initial_ev=0,
            a0=params['a0'], beta_I=params['beta_I'], b=params['b'],
            g_I=params['g_I'], I0=params['I0'],
            seed=None,
            network_type=params['network_type'],
            n_nodes=params['n_nodes'], p=params['p'], m=params['m'],
            strategy_choice_func=params['func']
        )

        # ---- RANDOM BASELINE ONLY ----
        apply_seeding(model, params['X0_frac'], method="random")

        trial_X, trial_I = [], []

        for t in range(steps):
            trial_X.append(model.get_adoption_fraction())
            trial_I.append(model.infrastructure)

            # ---- DISCRETE TARGETED SHOCK ----
            if t == shock_time:
                apply_targeted_shock(
                    model,
                    shock_frac=shock_amount,
                    method=method
                )

            model.step()

        trial_X.append(model.get_adoption_fraction())
        trial_I.append(model.infrastructure)

        X_trajectories.append(trial_X)
        I_trajectories.append(trial_I)

    return np.array(X_trajectories), np.array(I_trajectories)

def run_multiple_single_shocks_overlay_adjusted(
    n_trials, steps, method, params, shock_times, shock_amount
):
    all_X_traj, all_I_traj, labels = [], [], []

    for t_shock in shock_times:
        X_traj, I_traj = run_batch_single_shock(
            n_trials, steps, method, params,
            shock_time=t_shock,
            shock_amount=shock_amount
        )
        all_X_traj.append(X_traj)
        all_I_traj.append(I_traj)
        labels.append(f"Shock at t={t_shock}")

    # ---------- Mean Trajectories ----------
    time_axis = range(steps + 1)
    plt.figure(figsize=(12, 6))

    colors_X = ["blue", "red", "darkgreen"]
    colors_I = ["lightblue", "darkorange", "lightgreen"]

    for i in range(len(shock_times)):
        plt.plot(
            time_axis,
            np.mean(all_X_traj[i], axis=0),
            color=colors_X[i],
            linewidth=2,
            label=f"X(t) – {labels[i]}"
        )
        plt.plot(
            time_axis,
            np.mean(all_I_traj[i], axis=0),
            color=colors_I[i],
            linestyle="--",
            linewidth=2,
            label=f"I(t) – {labels[i]}"
        )

    plt.title("Mean Adoption and Infrastructure\n(Discrete Targeted Shocks Only)")
    plt.xlabel("Time Step")
    plt.ylabel("Fraction")
    plt.ylim(0, 1.05)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.show()

    # ---------- Side-by-Side Histograms ----------
    plt.figure(figsize=(14, 6))
    bins = np.linspace(0, 1, 21)
    width = 0.015

    for i in range(len(shock_times)):
        final_X = all_X_traj[i][:, -1]
        final_I = all_I_traj[i][:, -1]

        x_pos = bins[:-1] + i * (width * 3)

        plt.bar(
            x_pos,
            np.histogram(final_X, bins=bins)[0],
            width=width,
            color=colors_X[i],
            alpha=0.75,
            label=f"Final X – {labels[i]}"
        )
        plt.bar(
            x_pos + width, 
            np.histogram(final_I, bins=bins)[0],
            width=width,
            color=colors_I[i],
            alpha=0.75,
            label=f"Final I – {labels[i]}"
        )

    plt.title("Final Adoption and Infrastructure Distributions\n(Side-by-Side with Space)")
    plt.xlabel("Value (0–1)")
    plt.ylabel("Frequency")
    plt.legend()
    plt.show()

    return np.array(all_X_traj), np.array(all_I_traj)
