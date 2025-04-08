
# graph_utils.py
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

def compute_transition_matrix(predicted_states, num_states=3):
    matrix = np.zeros((num_states, num_states))
    for i in range(len(predicted_states) - 1):
        matrix[predicted_states[i], predicted_states[i + 1]] += 1
    row_sums = matrix.sum(axis=1, keepdims=True)
    transition_probs = np.divide(matrix, row_sums, where=row_sums != 0)
    return transition_probs

def build_dpgm_graph(transition_matrix, state_labels=None):
    num_states = transition_matrix.shape[0]
    G = nx.DiGraph()
    if state_labels is None:
        state_labels = [f"State {i}" for i in range(num_states)]
    for i in range(num_states):
        G.add_node(i, label=state_labels[i])
    for i in range(num_states):
        for j in range(num_states):
            prob = transition_matrix[i, j]
            if prob > 0:
                G.add_edge(i, j, weight=prob)
    return G

def plot_dpgm_graph(G, state_labels=None):
    pos = nx.circular_layout(G)
    edge_labels = nx.get_edge_attributes(G, 'weight')
    edge_labels = {k: f"{v:.2f}" for k, v in edge_labels.items()}
    node_labels = {n: G.nodes[n]['label'] for n in G.nodes}
    plt.figure(figsize=(6, 6))
    nx.draw(G, pos, with_labels=True, labels=node_labels, node_color='skyblue', node_size=1500,
            edge_color='gray', arrows=True, width=2)
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=10)
    plt.title("Directed Probabilistic Graph of Brain State Transitions")
    plt.axis('off')
    plt.tight_layout()
    plt.show()

def generate_dpgm(predicted_states, state_labels=None):
    transition_matrix = compute_transition_matrix(predicted_states, num_states=len(set(predicted_states)))
    G = build_dpgm_graph(transition_matrix, state_labels)
    plot_dpgm_graph(G, state_labels)
    return G, transition_matrix
