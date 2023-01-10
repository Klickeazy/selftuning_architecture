import numpy as np
import time
import matplotlib.pyplot as plt
import pickle


class Item:
    """
    Class to hold value and id of an item, display it and copy it
    """
    def __init__(self, id_in, value_in):
        self.id = str(id_in)
        self.value = value_in

    def print_item(self):
        print("ID: " + self.id + " - Value: " + str(self.value))
        return None

    def copy_item(self, item_in):
        self.id = item_in.id
        self.value = item_in.value


def selection_value(selection):
    """
    :param selection:   list of Item objects with id and value
    :return:     sum of item values
    """
    value = 0
    for i in selection:
        value += i.value
    return value


def constraint_check_selection(selection, limit):
    """
    :param selection:   list of Item objects with id and value
    :param limit:   number of items to select
    :return:    check if below limits for further selection - return false if not
    """
    if len(selection) < limit:
        return True
    else:
        return False


def constraint_check_rejection(rejection, limit):
    if len(rejection) >= limit:
        return True
    else:
        return False


def greedy_selection(choices, limit):
    selection_start = time.time()
    selection = []
    while constraint_check_selection(selection, limit):
        max_item = Item(None, 0)
        for c in choices:
            if selection_value([max_item] + selection) < selection_value([c] + selection):
                max_item = c
        if max_item.id is None:
            print('No valid item values')
            exit()
        selection.append(max_item)
        choices.remove(max_item)
    selection_end = time.time()
    return {"set": selection, "value": selection_value(selection), "time": np.round(selection_end-selection_start, 5)}


def greedy_rejection(choices, limit):
    rejection_start = time.time()
    rejection = choices[:]
    while constraint_check_rejection(rejection, limit):
        min_item = Item(None, np.inf)
        for r in rejection:
            if selection_value([i for i in rejection if i != min_item]) > selection_value([i for i in rejection if i != r]):
                min_item = r
        if min_item.id is None:
            print('No valid item values')
            exit()
        rejection.remove(min_item)
    rejection_end = time.time()
    return {"set": rejection, "value": selection_value(rejection), "time": np.round(rejection_end-rejection_start, 5)}


def greedy_comparison(choices, limit):
    choices = items[:]
    selection = greedy_selection(choices, limit)
    choices = items[:]
    rejection = greedy_rejection(choices, limit)

    return {"selection": selection, "rejection": rejection, "limit": limit}


def print_comparison(comp_values):
    print('Greedy Selection: Pick ', comp_values["limit"], ' from ', len(items))
    print('Value: ', comp_values["selection"]["value"])
    print('Time: ', comp_values["selection"]["time"], 'ms')

    print('Greedy Rejection: Pick ', comp_values["limit"], ' from ', len(items))
    print('Rejection value: ', comp_values["rejection"]["value"])
    print('Selection time: ', comp_values["rejection"]["time"], 'ms')


def test_limit_range(items, step_size=1):
    print("Start limit range test")
    selection_data = {"limit": [], "value": [], "time": []}
    rejection_data = {"limit": [], "value": [], "time": []}
    count = 0
    for i in range(1, len(items)+1, step_size):
        if not count % 10:
            print("\r", i, end="")
        vals = greedy_comparison(items, i)
        selection_data["limit"].append(vals["limit"])
        selection_data["value"].append(vals["selection"]["value"])
        selection_data["time"].append(vals["selection"]["time"])

        rejection_data["limit"].append(vals["limit"])
        rejection_data["value"].append(vals["rejection"]["value"])
        rejection_data["time"].append(vals["rejection"]["time"])

        count += 1

    print("End limit range test")

    return {"selection": selection_data, "rejection": rejection_data}


def test_limit_range_visualize(values):
    fig = plt.figure(layout='tight')
    gs = fig.add_gridspec(1, 1)

    ax1 = fig.add_subplot(gs[0, 0], projection='3d')
    ax1.scatter(values["selection"]["limit"], values["selection"]["value"], values["selection"]["time"], color='C0', label='Selection')
    ax1.scatter(values["rejection"]["limit"], values["rejection"]["value"], values["rejection"]["time"], color='C1', label='Rejection')

    # ax2 = fig.add_subplot(gs[0, 1], projection='3d')
    # ax2.scatter(values["selection"]["limit"], values["rejection"]["value"], values["rejection"]["time"])

    ax1.set_xlabel('Limit')
    # ax2.set_xlabel('Limit')
    ax1.set_ylabel('Value')
    # ax2.set_ylabel('Value')
    ax1.set_zlabel('Time')
    # ax2.set_zlabel('Time')
    ax1.set_title('Selection vs Rejection')
    # ax2.set_title('Rejection')
    ax1.legend()

    plt.show()


def test_limit_range_pickledump(values, fname):
    f = open(fname, "wb")
    pickle.dump(values, f)
    f.close()


def test_limit_range_pickleload(fname):
    f = open(fname, "rb")
    val = pickle.load(fname)
    f.close()
    return  val


if __name__ == "__main__":

    n_items = 100
    n_step = 10
    items = []
    for n in range(0, n_items):
        items.append(Item(n, np.random.randint(1, 100)))

    values = test_limit_range(items, n_step)
    test_limit_range_visualize(values)
    fname = "DataDumps/values_n"+str(n_items)+"_step"+str(n_step)+".bin"
    test_limit_range_pickledump(values, fname)

