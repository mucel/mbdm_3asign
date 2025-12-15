# Stag Hunt model implementation and evaluation 

#  Imports
from __future__ import annotations
from mesa import Agent, Model
from mesa.time import SimultaneousActivation
from mesa.space import NetworkGrid
from mesa.datacollection import DataCollector
import networkx as nx
import numpy as np
import pandas as pd
from typing import Tuple, List, Iterable, Dict, Optional
import matplotlib.pyplot as plt
import random
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed 
import math
import os
from dataclasses import dataclass
from collections import defaultdict



# Strategy update functions
def choose_strategy_imitate(agent, neighbors):
    """Choose strategy of the highest-payoff neighbour (including self)."""
    candidates = neighbors + [agent]
    best = max(candidates, key=lambda a: a.payoff)
    return best.strategy


def choose_strategy_logit(agent, neighbors, a_I, b, tau):
    """Choose strategy using logit / softmax choice.

    Parameters
    - agent: the agent choosing a strategy
    - neighbors: list of neighbour agents
    - a_I: effective coordination payoff given current infrastructure
    - b: defection payoff
    - tau: temperature parameter for softmax
    """
    # compute expected payoffs for C and D
    pi_C = 0.0
    pi_D = 0.0
    for other in neighbors:
        s_j = other.strategy
        if s_j == "C":
            pi_C += a_I
            pi_D += b
        else:
            pi_C += 0.0
            pi_D += b

    # softmax choice
    denom = np.exp(pi_C / tau) + np.exp(pi_D / tau)
    P_C = np.exp(pi_C / tau) / denom if denom > 0 else 0.5
    return "C" if random.random() < P_C else "D"


# Agent and Model classes
class EVAgent:
    def __init__(self, node_id: int, model: "EVStagHuntModel", init_strategy: str = "D"):
        self.node_id = node_id
        self.model = model
        self.strategy = init_strategy  # "C" or "D"
        self.payoff = 0.0
        self.next_strategy = init_strategy

    def step(self) -> None:
        """Compute payoff from interactions with neighbours."""
        I = self.model.infrastructure
        a0 = self.model.a0
        beta_I = self.model.beta_I
        b = self.model.b
        a_I = a0 + beta_I * I

        neighbors = list(self.model.G.neighbors(self.node_id))
        if not neighbors:
            self.payoff = 0.0
            return

        payoff = 0.0
        for nbr in neighbors:
            other = self.model.node_agent_map[nbr]
            s_i = self.strategy
            s_j = other.strategy
            if s_i == "C" and s_j == "C":
                payoff += a_I
            elif s_i == "C" and s_j == "D":
                payoff += 0.0
            elif s_i == "D" and s_j == "C":
                payoff += b
            else:  # D vs D
                payoff += b
        self.payoff = payoff

    def advance(self) -> None:
        """Compute next_strategy according to model's strategy_choice_func but DO NOT commit."""
        func = self.model.strategy_choice_func
        neighbors = [self.model.node_agent_map[n] for n in self.model.G.neighbors(self.node_id)]

        if func == "imitate":
            candidates = neighbors + [self]
            best = max(candidates, key=lambda a: a.payoff)
            self.next_strategy = best.strategy
        elif func == "logit":
            a_I = self.model.a0 + self.model.beta_I * self.model.infrastructure
            pi_C = 0.0
            pi_D = 0.0
            for other in neighbors:
                if other.strategy == "C":
                    pi_C += a_I
                    pi_D += self.model.b
                else:
                    pi_C += 0.0
                    pi_D += self.model.b
            tau = getattr(self.model, "tau", 1.0)
            denom = math.exp(pi_C / tau) + math.exp(pi_D / tau)
            P_C = math.exp(pi_C / tau) / denom if denom > 0 else 0.5
            self.next_strategy = "C" if random.random() < P_C else "D"
        else:
            raise ValueError(f"Unknown strategy choice function: {func}")

    def commit(self) -> None:
        """Apply next_strategy synchronously."""
        self.strategy = self.next_strategy


class EVStagHuntModel:
    def __init__(
        self,
        initial_ev: int,
        a0: float,
        beta_I: float,
        b: float,
        g_I: float,
        I0: float,
        seed: int | None,
        network_type: str,
        n_nodes: int,
        p: float,
        m: int,
        strategy_choice_func: str = "imitate",
        tau: float = 1.0,
    ):
        self.random = random.Random(seed)
        self.seed = seed
        self.a0 = a0
        self.beta_I = beta_I
        self.b = b
        self.g_I = g_I
        self.infrastructure = float(I0)
        self.strategy_choice_func = strategy_choice_func
        self.tau = tau

        # Build network
        if network_type == "ER":
            G = nx.erdos_renyi_graph(n_nodes, p, seed=seed)
        elif network_type == "WS":
            G = nx.watts_strogatz_graph(n_nodes, max(2, 2*m), p, seed=seed)
        elif network_type == "BA":
            G = nx.barabasi_albert_graph(n_nodes, max(1, m), seed=seed)

        # if graph disconnected, keep largest connected component (avoids isolated nodes effects)
        if not nx.is_connected(G) and len(G) > 0:
            comp = max(nx.connected_components(G), key=len)
            G = G.subgraph(comp).copy()
        self.G = G

        # create agents: one per node
        nodes = list(self.G.nodes())
        total_nodes = len(nodes)
        k_ev = max(0, min(initial_ev, total_nodes))
        ev_nodes = set(self.random.sample(nodes, k_ev))
        self.node_agent_map: Dict[int, EVAgent] = {}
        for node in nodes:
            init_strategy = "C" if node in ev_nodes else "D"
            agent = EVAgent(node, self, init_strategy)
            self.node_agent_map[node] = agent

    def get_adoption_fraction(self) -> float:
        agents = list(self.node_agent_map.values())
        if not agents:
            return 0.0
        return sum(1 for a in agents if a.strategy == "C") / len(agents)

    def step(self) -> None:
        # compute payoffs
        for agent in list(self.node_agent_map.values()):
            agent.step()
        # choose next strategies (based on payoffs)
        for agent in list(self.node_agent_map.values()):
            agent.advance()
        # commit synchronously
        for agent in list(self.node_agent_map.values()):
            agent.commit()
        # update infrastructure: I <- clip(I + g_I*(X - I), 0, 1)
        X = self.get_adoption_fraction()
        I = self.infrastructure
        dI = self.g_I * (X - I)
        self.infrastructure = float(min(1.0, max(0.0, I + dI)))


def run_trial(
        X0_frac: float, 
        I0: float,
        *,
        a0: float,
        beta_I: float,
        b: float,
        g_I: float,
        T: int,
        network_type: str,
        n_nodes: int,
        p: float,
        m: int,
        seed: Optional[int],
        strategy_choice_func: str,
        tau: float = 1.0,
        record_trajectory: bool = False,
        ) -> Tuple[float, Tuple[np.ndarray, np.ndarray]]:
        initial_ev = int(round(X0_frac * n_nodes))
        model = EVStagHuntModel(
            initial_ev=initial_ev,
            a0=a0,
            beta_I=beta_I,
            b=b,
            g_I=g_I,
            I0=I0,
            seed=seed,
            network_type=network_type,
            n_nodes=n_nodes,
            p=p,
            m=m,
            strategy_choice_func=strategy_choice_func,
            tau=tau,
            )
        X_traj = []
        I_traj = []
        for t in range(T):
            if record_trajectory:
                X_traj.append(model.get_adoption_fraction())
                I_traj.append(model.infrastructure)
            model.step()
            if record_trajectory:
                X_traj.append(model.get_adoption_fraction())
                I_traj.append(model.infrastructure)
        
        final = model.get_adoption_fraction()
                
        return final, (np.array(X_traj), np.array(I_traj))
