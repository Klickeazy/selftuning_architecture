import numpy as np
import time
import matplotlib.pyplot as plt
import pickle
from pathlib import Path


class Item:
    """
    Class to hold value and id of an item, display it and copy it
    class terms: ID (str), Value (int)
    """
    def __init__(self, id_in, value_in):
        self.id = str(id_in)
        self.value = value_in

    def print_item(self):
        """
        Print Item contents
        :return:
        """
        print("ID: " + self.id + " - Value: " + str(self.value))
        return None

    def copy_item(self, item_in):
        self.id = item_in.id
        self.value = item_in.value


def selection_value(selection):
    """
    Calculate cumulative value of all Items in selection
    :param selection:   list of Item objects with id and value
    :return:     sum of item values
    """
    value = 0
    for i in selection:
        value += i.value
    return value


def constraint_check_selection(selection, limit):
    """
    Check constraints for greedy selection - maximize selection up to limit
    :param selection:   list of Item objects with id and value
    :param limit:   final number of items after selection
    :return:    check if below limits for further selection - return false if not
    """
    if len(selection) < limit:
        return True
    else:
        return False


def constraint_check_rejection(rejection, limit):
    """
    Check constraints for greedy rejection - maximize selection up to limit
    :param rejection:   list of Item objects with id and value
    :param limit: final number of items after rejection
    :return:    check if below limits for further selection - return false if not
    """
    if len(rejection) >= limit:
        return True
    else:
        return False


def greedy_selection(choices, limit, selection=None):
    selection_start = time.time()
    if selection is None:
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


def greedy_rejection(choices, limit, rejection=None):
    rejection_start = time.time()
    if rejection is None:
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


def test_limit_range(items, step_size=1, start=1):
    print("Start limit range test")
    selection_data = {"limit": [], "value": [], "time": []}
    rejection_data = {"limit": [], "value": [], "time": []}
    # count = 0
    for i in range(start, len(items)+1, step_size):
        # if not ((i-start)//step_size) % 10:
        #     print("\r", i, end="")
        print("\r", i, end="")
        vals = greedy_comparison(items, i)
        selection_data["limit"].append(vals["limit"])
        selection_data["value"].append(vals["selection"]["value"])
        selection_data["time"].append(vals["selection"]["time"])

        rejection_data["limit"].append(vals["limit"])
        rejection_data["value"].append(vals["rejection"]["value"])
        rejection_data["time"].append(vals["rejection"]["time"])

        # count += 1

    print("\r End limit range test")

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
    if Path(fname).is_file():
        print('Overwriting old test data')
    with open(fname, "wb") as f:
        pickle.dump(values, f)
        f.close()


def test_limit_range_pickleload(fname):
    try:
        with open(fname, "rb") as f:
            val = pickle.load(f)
            f.close()
            return val
    except FileNotFoundError:
        print('Test not run yet')


if __name__ == "__main__":

    print('Code run start')

    n_items = 1000
    n_step = 50

    run_code = 1

    if run_code:
        items = []
        for n in range(0, n_items):
            items.append(Item(n, np.random.randint(1, 100)))

        values = test_limit_range(items, n_step)
        fname = "DataDumps/values_n"+str(n_items)+"_step"+str(n_step)+".bin"
        test_limit_range_pickledump(values, fname)
        test_limit_range_visualize(values)

    else:
        fname = "DataDumps/values_n" + str(n_items) + "_step" + str(n_step) + ".bin"
        values = test_limit_range_pickleload(fname)
        test_limit_range_visualize(values)

    print('Code run complete')