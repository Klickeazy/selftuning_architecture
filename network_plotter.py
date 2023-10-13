# import functionfile_speedygreedy as ff
import networkx as netx
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

number_of_nodes = 10
graph_parameter = 0.2
directed_graph = True
actuator_size = 2
sensor_size = 3
node_color = ['C0']*number_of_nodes + ['C1']*actuator_size + ['C2']*sensor_size


def generate_connected_graph():
    # G = netx.generators.gnr_graph(number_of_nodes, graph_parameter)
    # A = netx.to_numpy_array(G)
    # print(A)
    # G = netx.from_numpy_array(A, create_using=netx.DiGraph)

    # G = netx.Graph()
    # connected_graph_check = False
    # while not connected_graph_check:
    #     G = netx.generators.erdos_renyi_graph(number_of_nodes, graph_parameter)
    #     # G = netx.generators.barabasi_albert_graph(number_of_nodes, graph_parameter)
    #     connected_graph_check = netx.is_connected(G)

    G = netx.generators.erdos_renyi_graph(number_of_nodes, graph_parameter, directed=directed_graph)

    return G


def generate_arch(G_in, size, archtype):
    # arch_mat = np.zeros((number_of_nodes, size))
    arch_loc = np.random.choice(number_of_nodes, size, replace=False)
    for i in range(size):
        if archtype == 'B':
            G_in.add_edge(archtype + str(i), arch_loc[i])
        else:
            G_in.add_edge(arch_loc[i], archtype + str(i))


# G = netx.cycle_graph(number_of_nodes)
# pos_list = netx.circular_layout(G)
# node_list = list(G.nodes())


def plot_net(ax):
    G_net = generate_connected_graph()
    generate_arch(G_net, actuator_size, 'B')
    generate_arch(G_net, sensor_size, 'C')

    all_list = list(netx.nodes(G_net))
    print(netx.to_numpy_array(G_net))

    # net_total = np.block([[A, B, np.zeros_like(C).T],
    #                      [np.zeros_like(B).T, np.zeros((actuator_size, actuator_size)), np.zeros((actuator_size, sensor_size))],
    #                      [C, np.zeros((sensor_size, actuator_size)), np.zeros((sensor_size, sensor_size))]])
    # G_net = netx.from_numpy_array(net_total, create_using=netx.DiGraph)
    # if len(pos_list) == 0:
    #     pos_spring = netx.spring_layout(G_net)
    #     for k in pos_spring:
    #         if k in node_list:
    #             pos_list[k] = pos_spring[k]

    # netx.draw_networkx(G_net, ax=ax, pos=netx.spring_layout(G_net, pos=pos_list), node_color=node_color, with_labels=False, node_size=150)

    netx.draw_networkx(G_net, ax=ax, pos=netx.shell_layout(G_net, nlist=[all_list[:number_of_nodes], all_list[number_of_nodes:]]), node_color=node_color, with_labels=False, node_size=150)

    ax.set_aspect('equal')
    ax.axis('off')


fig = plt.figure(figsize=(6, 3))
ax_net = fig.subplots(1, 2, sharex=True, sharey=True)
plot_net(ax_net[0])
plot_net(ax_net[1])
plt.show()
fig.savefig('Images/sample_graph.pdf', format='pdf')
