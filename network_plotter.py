# import functionfile_speedygreedy as ff
import networkx as netx
import random
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy as dc

number_of_nodes = 6
graph_parameter = 0.2
directed_graph = True
number_of_actuators = 2
number_of_sensors = 3
node_c = ['C0'] * number_of_nodes + ['C1'] * number_of_actuators + ['C2'] * number_of_sensors

fig = plt.figure(figsize=(6, 3))
ax_net = fig.subplots(1, 2, sharex=True, sharey=True)
for a in ax_net:
    a.set_aspect('equal')
    a.axis('off')

# G = netx.generators.gn_graph(number_of_nodes)
G = netx.generators.random_k_out_graph(number_of_nodes, k=3, alpha=1, self_loops=False)
# print(netx.to_numpy_array(G))
node_pos = netx.circular_layout(G)
node_list = netx.nodes(G)

arch_list = ['B'+str(i) for i in range(number_of_actuators)] + ['C'+str(i) for i in range(number_of_sensors)]
# netx.draw_networkx(G, ax=ax_net[0], pos=netx.circular_layout(G), with_labels=False)
# plt.show()


def rand_remove(G_ref):
    sample_remove_edges = random.sample(list(G_ref.edges()), number_of_nodes)
    print(sample_remove_edges)
    M1 = netx.to_numpy_array(G_ref)
    G_ref.remove_edges_from(sample_remove_edges)
    M2 = netx.to_numpy_array(G_ref)
    # print(M1-M2)


G1 = dc(G)
rand_remove(G1)
G2 = dc(G)
rand_remove(G2)


def add_architecture(G_ref, arch_count, arch_type):
    G_mod = dc(G_ref)
    arch_location = random.sample(range(number_of_nodes), arch_count)
    for i in range(arch_count):
        if arch_type == 'B':
            G_mod.add_edge('B'+str(i), arch_location[i])
        elif arch_type == 'C':
            G_mod.add_edge(arch_location[i], 'C' + str(i))
        else:
            raise Exception('Check arch_type')
    return G_mod


G1 = add_architecture(G1, number_of_actuators, 'B')
G1 = add_architecture(G1, number_of_sensors, 'C')

G2 = add_architecture(G2, number_of_actuators, 'B')
G2 = add_architecture(G2, number_of_sensors, 'C')

plot_properties = {'node_color': node_c, 'with_labels': False, 'node_size': 100, 'arrowstyle': '->'}

# netx.draw_networkx(G1, ax=ax_net[0], pos=netx.circular_layout(G1), **plot_properties)
# netx.draw_networkx(G2, ax=ax_net[1], pos=netx.circular_layout(G2), **plot_properties)

netx.draw_networkx(G1, ax=ax_net[0], pos=netx.shell_layout(G1, nlist=[node_list, arch_list]), **plot_properties)
netx.draw_networkx(G2, ax=ax_net[1], pos=netx.shell_layout(G2, nlist=[node_list, arch_list]), **plot_properties)

# netx.draw_networkx(G1, ax=ax_net[0], pos=netx.spiral_layout(G1), node_color=node_c, with_labels=False)
# netx.draw_networkx(G2, ax=ax_net[1], pos=netx.spiral_layout(G2), node_color=node_c, with_labels=False)

# netx.draw_networkx(G1, ax=ax_net[0], pos=netx.rescale_layout_dict(netx.spring_layout(G1, pos=netx.shell_layout(G1, nlist=[node_list, arch_list]), fixed=node_list)), **plot_properties)
# netx.draw_networkx(G2, ax=ax_net[1], pos=netx.rescale_layout_dict(netx.spring_layout(G2, pos=netx.shell_layout(G2, nlist=[node_list, arch_list]), fixed=node_list)), **plot_properties)

plt.show()
fig.savefig('Images/sample_graph.pdf', format='pdf')


# def generate_connected_graph():
#     # G = netx.generators.gnr_graph(number_of_nodes, graph_parameter)
#     # A = netx.to_numpy_array(G)
#     # print(A)
#     # G = netx.from_numpy_array(A, create_using=netx.DiGraph)
#
#     # G = netx.Graph()
#     # connected_graph_check = False
#     # while not connected_graph_check:
#     #     G = netx.generators.erdos_renyi_graph(number_of_nodes, graph_parameter)
#     #     # G = netx.generators.barabasi_albert_graph(number_of_nodes, graph_parameter)
#     #     connected_graph_check = netx.is_connected(G)
#
#     G = netx.generators.erdos_renyi_graph(number_of_nodes, graph_parameter, directed=directed_graph)
#
#     return G
#
#
# def generate_arch(G_in, size, archtype):
#     # arch_mat = np.zeros((number_of_nodes, size))
#     arch_loc = np.random.choice(number_of_nodes, size, replace=False)
#     for i in range(size):
#         if archtype == 'B':
#             G_in.add_edge(archtype + str(i), arch_loc[i])
#         else:
#             G_in.add_edge(arch_loc[i], archtype + str(i))
#
#
# # G = netx.cycle_graph(number_of_nodes)
# # pos_list = netx.circular_layout(G)
# # node_list = list(G.nodes())
#
#
# def plot_net(ax):
#     G_net = generate_connected_graph()
#     generate_arch(G_net, number_of_actuators, 'B')
#     generate_arch(G_net, number_of_sensors, 'C')
#
#     all_list = list(netx.nodes(G_net))
#     print(netx.to_numpy_array(G_net))
#
#     # net_total = np.block([[A, B, np.zeros_like(C).T],
#     #                      [np.zeros_like(B).T, np.zeros((number_of_actuators, number_of_actuators)), np.zeros((number_of_actuators, number_of_sensors))],
#     #                      [C, np.zeros((number_of_sensors, number_of_actuators)), np.zeros((number_of_sensors, number_of_sensors))]])
#     # G_net = netx.from_numpy_array(net_total, create_using=netx.DiGraph)
#     # if len(pos_list) == 0:
#     #     pos_spring = netx.spring_layout(G_net)
#     #     for k in pos_spring:
#     #         if k in node_list:
#     #             pos_list[k] = pos_spring[k]
#
#     # netx.draw_networkx(G_net, ax=ax, pos=netx.spring_layout(G_net, pos=pos_list), node_color=node_color, with_labels=False, node_size=150)
#
#     netx.draw_networkx(G_net, ax=ax, pos=netx.shell_layout(G_net, nlist=[all_list[:number_of_nodes], all_list[number_of_nodes:]]), node_color=node_color, with_labels=False, node_size=150)
#
#     ax.set_aspect('equal')
#     ax.axis('off')
#
#
# fig = plt.figure(figsize=(6, 3))
# ax_net = fig.subplots(1, 2, sharex=True, sharey=True)
# plot_net(ax_net[0])
# plot_net(ax_net[1])
# plt.show()
# fig.savefig('Images/sample_graph.pdf', format='pdf')
