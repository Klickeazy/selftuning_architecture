import time
import numpy as np
from copy import deepcopy as dc


class Item:
    """
    Item class containing item id string and int cost value
    """
    rand_id_count = [1]  # Random Item ID string updater

    def __init__(self, id_in=None, value_in=None, low=-500, high=500):
        if id_in is None:
            id_in = "R"+str(self.rand_id_count[0])
            self.rand_id_count[0] += 1
        self.item_id = str(id_in)
        if value_in is None:
            value_in = 0
            while not value_in:
                value_in = np.random.randint(low, high)
        self.item_value = value_in

    def display_item(self):
        print('ID: ', self.item_id, ' - Value: ', self.item_value)

    def compare_item(self, item_check):
        if not isinstance(item_check, Item):
            raise Exception('Incorrect data type')
        return item_check.item_id == self.item_id and item_check.item_value == self.item_value


class ItemList:
    """
    Set of items with array of id strings, array of int values and int net value
    Functions to calculate net value, change item (add/remove), find item
    """
    rand_id_count = [0]  # Random List ID string updater

    def __init__(self, item_set=None, name=None, n_random_items=None, low=-500, high=500):
        self.list_value = None
        if item_set is None:
            self.items = []
            if n_random_items is not None:
                for _ in range(0, n_random_items):
                    self.items.append(Item(low=low, high=high))
        else:
            self.items = item_set[:]
        if name is None:
            self.list_id = "L" + str(self.rand_id_count[0])
            self.rand_id_count[0] += 1
        else:
            self.list_id = dc(name)
        self.list_value_update()
        self.duplicate_item_check()

    def list_value_update(self):
        self.list_value = None
        if len(self.items) > 0:
            self.list_value = sum([i.item_value for i in self.items])

    def display_list(self):
        print(self.list_id, ':: N: ', len(self.items), ' - Net value: ', self.list_value)
        print('IDs(Value): ', [i.item_id+'('+str(i.item_value)+')' for i in self.items])

    def find_by_item(self, check_item):
        if not isinstance(check_item, Item):
            raise Exception('Incorrect data type')
        check = 0
        idx = None
        for i in range(0, len(self.items)):
            if self.items[i].compare_item(check_item):
                check = 1
                idx = i
                break
        return {'check': check, 'idx': idx}

    def find_by_item_idx(self, idx):
        if idx not in range(-len(self.items), len(self.items)):
            raise Exception('Exceeds number of items in list(', len(self.items), ')')
        else:
            return self.items[idx]

    def find_by_item_id(self, item_id):
        check = 0
        idx = []
        for i in range(0, len(self.items)):
            if self.items[i].item_id == item_id:
                idx = i
                check = 1
                break
        return {'check': check, 'idx': idx}

    def find_by_item_value(self, item_value):
        check = 0
        idxs = []
        for i in range(0, len(self.items)):
            if self.items[i].item_value == item_value:
                idxs.append(i)
                check += 1
        return {'check': check, 'idxs': idxs}

    def duplicate_item_check(self):
        for i in range(0, len(self.items)):
            for j in range(i+1, len(self.items)):
                if self.items[i].compare_item(self.items[j]):
                    print('Duplicate items', i, ' and ', j)

    def add_item(self, new_item=None, value_update=True, idx=None, duplicate_print_check=False):
        if new_item is None:
            new_item = Item()
        elif not isinstance(new_item, Item):
            raise Exception('Incorrect data type')
        if not self.find_by_item(new_item)['check']:
            if idx is None:
                self.items.append(new_item)
            elif idx not in range(-len(self.items), len(self.items)):
                raise Exception('Check index')
            else:
                self.items = self.items[:idx] + [new_item] + self.items[idx:]
        if value_update:
            self.list_value_update()
        if duplicate_print_check:
            print('Duplicate item not added')
            new_item.display_item()

    def add_item_list(self, new_items, duplicate_print_check=False):
        if not isinstance(new_items, ItemList):
            raise Exception('Incorrect data type')
        for i in new_items.items:
            self.add_item(i, value_update=False, duplicate_print_check=duplicate_print_check)
        self.list_value_update()

    def remove_item_by_idx(self, idx):
        if idx not in range(-len(self.items), len(self.items)):
            raise Exception('Check index')
        else:
            self.items = self.items[:idx] + self.items[idx+1:]

    def remove_item(self, old_item, value_update=True, duplicate_print_check=False):
        if not isinstance(old_item, Item):
            raise Exception('Incorrect data type')
        search = self.find_by_item(old_item)
        if search['check']:
            self.remove_item_by_idx(search['idx'])
            if value_update:
                self.list_value_update()
        elif duplicate_print_check:
            print('Item not found')

    def remove_item_list(self, old_items, duplicate_print_check=False):
        if not isinstance(old_items, ItemList):
            raise Exception('Incorrect data type')
        for i in old_items.items:
            self.remove_item(i, value_update=False, duplicate_print_check=duplicate_print_check)
        self.list_value_update()

    def compare_lists(self, list2):
        if not isinstance(list2, ItemList):
            raise Exception('Incorrect data type')
        compare = {'unique': ItemList(name='Unique'), 'intersect': ItemList(name='Intersect'), 'absent':ItemList(name='absent')}
        for i in self.items:
            if list2.find_by_item(i)['check']:
                compare['intersect'].add_item(i)
            else:
                compare['unique'].add_item(i)
        for i in list2.items:
            if not self.find_by_item(i)['check']:
                compare['absent'].add_item(i)
        return compare


def max_limit(work_set, limit):
    return len(work_set.items) < limit


def min_limit(work_set, limit):
    return len(work_set.items) >= limit


def knapsack_value(work_set):
    return work_set.list_value


def item_index_from_policy(values, policy):
    if policy == "max":
        return values.index(max(values))
    elif policy == "min":
        return values.index(min(values))
    else:
        raise Exception('Check policy')


def greedy_selection(total_set, work_set, limit, number_of_changes=None, fixed_set=None, failure_set=None, max_greedy_limit=max_limit, min_greedy_limit=min_limit, cost_metric=knapsack_value, policy="max", t_start=time.time(), no_select=False, status_check=False):
    work_iteration, available_set = initialize_greedy(total_set, work_set, fixed_set, failure_set)
    choice_history = []
    work_history = []
    value_history = []
    count_of_changes = 0
    while max_greedy_limit(work_iteration, limit[1]):
        work_history.append(dc(work_iteration))
        choice_iteration = available_set.compare_lists(work_iteration)['unique']
        if fixed_set is not None:
            choice_iteration = fixed_set.compare_lists(choice_iteration)['absent']
        choice_history.append(choice_iteration)
        if len(choice_iteration.items) == 0:
            if status_check:
                print('No selections possible')
            break
        iteration_cases = []
        values = []
        if no_select and min_greedy_limit(work_iteration, limit[0]):
            iteration_cases.append(dc(work_iteration))
            values.append(cost_metric(iteration_cases[-1]))
        for i in range(0, len(choice_iteration.items)):
            iteration_cases.append(dc(work_iteration))
            iteration_cases[-1].add_item(choice_iteration.items[i])
            values.append(cost_metric(iteration_cases[-1]))
        value_history.append(values)
        target_idx = item_index_from_policy(values, policy)
        work_iteration = dc(iteration_cases[target_idx])
        if len(work_iteration.compare_lists(work_history[-1])['unique'].items) == 0:
            if status_check:
                print('No valuable selections')
            break
        count_of_changes += 1
        if number_of_changes is not None and count_of_changes == number_of_changes:
            if status_check:
                print('Maximum number of changes done')
            break
    work_history.append(work_iteration)
    work_iteration.list_id = "Greedy Selection"
    return {'work_set': work_iteration, 'work_history': work_history, 'choice_history': choice_history, 'value_history': value_history, 'time': time.time()-t_start}


def greedy_rejection(total_set, work_set, limit, number_of_changes=None, fixed_set=None, failure_set=None, max_greedy_limit=max_limit, min_greedy_limit=min_limit, cost_metric=knapsack_value, policy="max", t_start=time.time(), no_reject=False, status_check=False):
    work_iteration, available_set = initialize_greedy(total_set, work_set, fixed_set, failure_set)
    choice_history = []
    work_history = []
    value_history = []
    count_of_changes = 0
    while min_greedy_limit(work_iteration, limit[0]+1):
        work_history.append(dc(work_iteration))
        choice_iteration = dc(work_iteration)
        if fixed_set is not None:
            choice_iteration = available_set.compare_lists(choice_iteration)['absent']
        choice_history.append(choice_iteration)
        if len(choice_iteration.items) == 0:
            if status_check:
                print('No rejections possible')
            break
        iteration_cases = []
        values = []
        if no_reject and max_greedy_limit(work_iteration, limit[1]+1):
            iteration_cases.append(dc(work_iteration))
            values.append(cost_metric(iteration_cases[-1]))
        for i in range(0, len(choice_iteration.items)):
            iteration_cases.append(dc(work_iteration))
            iteration_cases[-1].remove_item(choice_iteration.items[i])
            values.append(cost_metric(iteration_cases[-1]))
        value_history.append(values)
        target_idx = item_index_from_policy(values, policy)
        work_iteration = dc(iteration_cases[target_idx])
        if len(work_iteration.compare_lists(work_history[-1])['absent'].items) == 0:
            if status_check:
                print('No valuable rejections')
            break
        count_of_changes += 1
        if number_of_changes is not None and count_of_changes == number_of_changes:
            if status_check:
                print('Maximum number of changes done')
            break
    work_history.append(work_iteration)
    work_iteration.list_id = "Greedy Rejection"
    return {'work_set': work_iteration, 'work_history': work_history, 'choice_history': choice_history, 'value_history': value_history, 'time': time.time() - t_start}


def initialize_greedy(total_set, work_set, fixed_set, failure_set):
    if not isinstance(total_set, ItemList) or not isinstance(work_set, ItemList) or not (isinstance(fixed_set, ItemList) or fixed_set is None) or not (isinstance(failure_set, ItemList) or failure_set is None):
        raise Exception('Incorrect data type')

    work_iteration = dc(work_set)
    available_set = dc(total_set)
    if failure_set is not None:
        work_iteration = work_iteration.compare_lists(failure_set)['unique']
        available_set = available_set.compare_lists(failure_set)['unique']
    return work_iteration, available_set


def greedy_simultaneous(total_set, work_set, limit, iterations=1, changes_per_iteration=1, fixed_set=None, max_greedy_limit=max_limit, min_greedy_limit=min_limit, cost_metric=knapsack_value, policy="max", t_start=time.time(), status_check=False):
    if not isinstance(total_set, ItemList) or not isinstance(work_set, ItemList) or not (isinstance(fixed_set, ItemList) or fixed_set is None):
        raise Exception('Incorrect data type')

    work_iteration = dc(work_set)
    work_history = [work_iteration]
    value_history = []
    for _ in range(0, iterations):
        # Keep same
        iteration_cases = [work_iteration]
        values = [cost_metric(iteration_cases[-1])]
        all_values = [cost_metric(iteration_cases[-1])]

        # Select one
        select = greedy_selection(total_set, work_iteration, limit, number_of_changes=changes_per_iteration, fixed_set=fixed_set, max_greedy_limit=max_greedy_limit, min_greedy_limit=min_greedy_limit, cost_metric=cost_metric, policy=policy, no_select=True, status_check=status_check)
        iteration_cases.append(select['work_set'])
        values.append(cost_metric(iteration_cases[-1]))
        all_values += select['value_history']

        # Reject one
        reject = greedy_rejection(total_set, work_iteration, limit, number_of_changes=changes_per_iteration, fixed_set=fixed_set, max_greedy_limit=max_greedy_limit, min_greedy_limit=min_greedy_limit, cost_metric=cost_metric, policy=policy, no_reject=True, status_check=status_check)
        iteration_cases.append(reject['work_set'])
        values.append(cost_metric(iteration_cases[-1]))
        all_values += reject['value_history']

        # Swap: add then drop
        swap_select1 = greedy_selection(total_set, work_iteration, limit+np.array([0, changes_per_iteration]), number_of_changes=changes_per_iteration, fixed_set=fixed_set, max_greedy_limit=max_greedy_limit, min_greedy_limit=min_greedy_limit, cost_metric=cost_metric, policy=policy, no_select=True, status_check=False)
        swap_reject1 = greedy_rejection(total_set, swap_select1['work_set'], limit, number_of_changes=changes_per_iteration, fixed_set=fixed_set, max_greedy_limit=max_greedy_limit, min_greedy_limit=min_greedy_limit, cost_metric=cost_metric, policy=policy, no_reject=True, status_check=status_check)
        iteration_cases.append(swap_reject1['work_set'])
        values.append(cost_metric(iteration_cases[-1]))
        all_values += swap_reject1['value_history']

        # # Swap: drop then add
        # swap_reject2 = greedy_rejection(total_set, work_iteration, limit-np.array([1, 0]), fixed_set=fixed_set, max_greedy_limit=max_greedy_limit, min_greedy_limit=min_greedy_limit, cost_metric=cost_metric, policy=policy, no_reject=True, status_check=False)
        # swap_select2 = greedy_selection(total_set, swap_reject2['work_set'], limit, fixed_set=fixed_set, max_greedy_limit=max_greedy_limit, min_greedy_limit=min_greedy_limit, cost_metric=cost_metric, policy=policy, no_select=True, status_check=status_check)
        # iteration_cases.append(swap_select2['work_set'])
        # values.append(cost_metric(iteration_cases[-1]))

        # print('Simultaneous iteration values', values)

        target_idx = item_index_from_policy(values, policy)

        work_iteration = iteration_cases[target_idx]
        work_history.append(work_iteration)
        value_history.append(all_values)
        if len(work_history[-1].compare_lists(work_history[-2])['unique'].items) == 0:
            print('No changes to work_set')
            break

    return {'work_set': work_iteration, 'work_history': work_history, 'value_history': value_history, 'time': time.time()-t_start}
