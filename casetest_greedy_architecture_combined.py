import numpy as np
import greedy_architecture_combined as gac

if __name__ == "__main__":
    S = gac.System(architecture={'rand': True})
    print(S.vector['x'])
    print(S.vector['x_estimate'])
    S.enhanced_system()
    print(S.vector['enhanced'])
