import random as rd
from multiprocessing import Pool

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np


def generate_random_uniform_values(n=1000, generator=rd.Random(0), min_val=0,
                                   max_val=1):
    """Create a random uniform distribution of opinions in the interval
    [min_val, max_val]."""

    opinion = []
    for i in range(n):
        opinion.append(generator.uniform(min_val, max_val))
    return opinion


def check_seeding(seeding):
    """Check if all nodes are included in the seeding."""

    nodes = np.zeros(len(seeding), dtype=bool)
    for s in seeding:
        nodes[s] = True
    all_nodes = True
    for n in nodes:
        if not n:
            all_nodes = False
            break
    return all_nodes


def dw_model(initial_op=generate_random_uniform_values(n=1000),
             graph=nx.complete_graph(1000), seeding=None,
             simulation_steps=100000, convergence=0.1,
             threshold_bc=np.full(1000, 0.25), seed=0):
    """Simulate the Deffuant-Weisbuch model.

    Parameters
    ----------
    initial_op : list[float]
        List with the initial opinion of each agent.
    graph : nx.Graph
        Graph representing the social network between agents.
    seeding : list[int]
        Distribution of agents in the graph nodes.
    simulation_steps : int
        Number of simulation steps.
    convergence : float
        Convergence speed.
    threshold_bc : list[float]
        List with the confidence threshold of each agent.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    tuple
        A tuple with the following elements:

        list[list[int, float]]
            2D points with intermediate opinions vs timestep.
        list[float]
            List with the final opinion of each agent.

    """
    if not seeding:
        seeding = range(len(initial_op))

    # Check that the number of nodes matches the number of opinions
    if not len(initial_op) == graph.number_of_nodes():
        print("ERROR: graph size don't match #agents!")
        exit

    # Check that the number of opinions matches the size of the seeding
    if not len(initial_op) == len(seeding):
        print("ERROR: seeding don't match #agents!")
        exit

    # Check that all nodes/agents are included in the seeding
    if not check_seeding(seeding):
        print("ERROR: seeding is not complete!")
        exit

    # Check that the number of agents matches the number of thresholds
    if not len(initial_op) == len(threshold_bc):
        print("ERROR: theshold_bc don't match #agents!")
        exit

    # Initialize the seed
    rd.seed(seed)

    # Copy the initial opinions to an auxiliary list
    opinions = np.copy(initial_op)

    # List to save the intermediate opinions
    data_plot = [[], []]

    # List of edges
    edges = list(graph.edges())

    # Iterate until a maximum number of iterations...
    for i in range(simulation_steps):

        # ... choose a random edge
        random_edge = rd.choice(edges)
        # ... and its agents
        ag1 = seeding[random_edge[0]]
        ag2 = seeding[random_edge[1]]
        # ... get their opinions
        op1 = opinions[ag1]
        op2 = opinions[ag2]

        # BOUNDED CONFIDENCE
        if abs(op1 - op2) < threshold_bc[ag1]:
            opinions[ag1] += convergence * (op2 - op1)
        if abs(op1 - op2) < threshold_bc[ag2]:
            opinions[ag2] += convergence * (op1 - op2)

        # Add the intermediate opinion to the data_plot list
        data_plot[0].append(i)
        data_plot[1].append(opinions[ag1])
        data_plot[0].append(i)
        data_plot[1].append(opinions[ag2])

    return data_plot, opinions


def hk_model(initial_op=generate_random_uniform_values(n=1000),
             graph=nx.complete_graph(1000), seeding=None,
             simulation_steps=100000, threshold_bc=np.full(100, 0.25), seed=0):
    """Simulate the Hegselmann-Krause model.

    Parameters
    ----------
    initial_op : list[float]
        List with the initial opinion of each agent.
    graph : nx.Graph
        Graph representing the social network between agents.
    seeding : list[int]
        Distribution of agents in the graph nodes.
    simulation_steps : int
        Number of simulation steps.
    threshold_bc : list[float]
        List with the confidence threshold of each agent.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    tuple
        A tuple with the following elements:

        list[list[int, float]]
            2D points with intermediate opinions vs timestep.
        list[float]
            List with the final opinion of each agent.
    """
    if not seeding:
        seeding = range(len(initial_op))

    # Check that the number of nodes matches the number of opinions
    if not len(initial_op) == graph.number_of_nodes():
        print("ERROR: graph size don't match #agents!")
        exit

    # Check that the number of opinions matches the size of the seeding
    if not len(initial_op) == len(seeding):
        print("ERROR: seeding don't match #agents!")
        exit

    # Check that all nodes/agents are included in the seeding
    if not check_seeding(seeding):
        print("ERROR: seeding is not complete!")
        exit

    # Check that the number of agents matches the number of thresholds
    if not len(initial_op) == len(threshold_bc):
        print("ERROR: theshold_bc don't match #agents!")
        exit

    # Initialize the seed
    rd.seed(seed)

    # Copy the initial opinions to an auxiliary list
    opinions = np.copy(initial_op)

    # List to save the intermediate opinions
    data_plot = [[], []]

    # Number of agents
    num_agents = len(opinions)

    # List of neighbors of each agent
    neighbors = []
    for node in range(num_agents):
        neighbors_of_node = []
        for neigh in graph.neighbors(node):
            neighbors_of_node.append(neigh)
        neighbors.append(neighbors_of_node)

    # Iterate until a maximum number of iterations...
    for i in range(simulation_steps):

        # ... choose a random agent
        node = rd.randint(0, num_agents - 1)
        ag = seeding[node]
        # ... get its opinion
        opinion = opinions[ag]

        # BOUNDED CONFIDENCE
        neighbors_with_confidence = 0
        sum_opinion_neighbors = 0

        # Iterate over the neighbors of the agent
        for neigh in neighbors[node]:
            ag_neighbor = seeding[neigh]
            op_neighbor = opinions[ag_neighbor]

            # Check if the neighbor's opinion is within the confidence
            # threshold
            if abs(op_neighbor - opinion) < threshold_bc[ag]:
                neighbors_with_confidence += 1
                sum_opinion_neighbors += op_neighbor

        if neighbors_with_confidence > 0:
            # Consider the opinion of the agent itself
            sum_opinion_neighbors += opinion
            neighbors_with_confidence += 1
            # Update the opinion of the agent
            opinions[ag] = sum_opinion_neighbors / neighbors_with_confidence
            # Add the intermediate opinion to the data_plot list
            data_plot[0].append(i)
            data_plot[1].append(opinions[ag])

    # Return the intermediate opinions and the final opinions
    return data_plot, opinions


def run_hk_model_mc(mc, initial_op=generate_random_uniform_values(n=1000),
                    graph=nx.complete_graph(1000), seeding=None,
                    simulation_steps=100000, threshold_bc=np.full(100, 0.25)):
    final_opinions = []
    data_plot = []

    args = [(initial_op, graph, seeding, simulation_steps, threshold_bc, i) for
            i in range(mc)]

    with Pool() as pool:
        for result in pool.starmap(hk_model, args):
            data_plot.append(result[0])
            final_opinions.append(result[1])

    return data_plot, final_opinions


def plot_opinions(initial_op, intermediate_op, final_op, title, filename="",
                  simulation_steps=100000, alpha=0.1):
    """Plot the opinion dynamics of the agents."""
    fig, ax = plt.subplots(1, 2, figsize=(6, 3),
                           gridspec_kw={'width_ratios': [4, 1]})
    fig.tight_layout()

    # SCATTER PLOT: left plot
    # Intermediate opinions (blue)
    ax[0].scatter(intermediate_op[0], intermediate_op[1], s=1, color="blue",
                  alpha=alpha, label="interm op.")
    # Initial opinions (green)
    ax[0].scatter(np.full(len(initial_op), 0), initial_op, s=5, color="green",
                  alpha=alpha, label="initial op.")
    # Final opinions (red)
    ax[0].scatter(np.full(len(final_op), simulation_steps), final_op, s=5,
                  color="red", alpha=alpha, label="final op.")

    ax[0].set_xlim([-2 * simulation_steps // 100,
                    simulation_steps + 2 * simulation_steps // 100])
    ax[0].set_ylim([0, 1])
    ax[0].set_xlabel('time step')
    ax[0].set_ylabel('opinion')
    ax[0].ticklabel_format(axis="x", style="sci", scilimits=(0, 0))

    handles, labels = ax[0].get_legend_handles_labels()
    order = [1, 0, 2]
    leg = ax[0].legend([handles[idx] for idx in order],
                       [labels[idx] for idx in order], loc='center left',
                       bbox_to_anchor=(1.4, 0.85), fancybox=True)
    for lh in leg.legend_handles:
        lh.set_alpha(1)

    # HISTOGRAMS: right plot
    ax[1].hist(initial_op, bins=50, range=(0, 1), orientation="horizontal",
               color="green")
    ax[1].hist(final_op, bins=50, range=(0, 1), orientation="horizontal",
               color="red")

    ax[1].set_xscale("log")
    ax[1].set_ylim([0, 1])
    ax[1].set_xlim([len(final_op) // 100, len(final_op)])
    ax[1].set_ylabel('final opinion')
    ax[1].set_xlabel('%ag')
    ax[1].xaxis.tick_top()
    ax[1].yaxis.tick_right()
    ax[1].yaxis.set_label_position("right")
    ax[1].set_xticks(
        [len(final_op) // 100, len(final_op) // 10, len(final_op)])
    ax[1].set_xticklabels(["1%", "10%", "100%"])

    plt.subplots_adjust(wspace=0.03, hspace=0)
    ax[0].set_title(title)

    if not filename == "":
        plt.savefig(filename, bbox_inches="tight")
    else:
        plt.show()
