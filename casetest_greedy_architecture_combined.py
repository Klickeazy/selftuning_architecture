import numpy as np
import greedy_architecture_combined as gac

if __name__ == "__main__":
    S = gac.System(architecture={'rand': True})
    S.enhanced_system()
    print(S.trajectory['enhanced'])

    greedy_selection = gac.greedy_architecture_selection(S)
    S_greedy = greedy_selection['work_iteration']

