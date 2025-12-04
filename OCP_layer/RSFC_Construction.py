import numpy as np
import pickle

def generate_rsfc_construction(xm, ym, name, global_param):
    """
    Constructs Relative Safe Flight Corridors (RSFC) for a pair of vehicles.
    """
    nfe = global_param.Optimal['nfe']
    box_edge_length = np.zeros((nfe, 4))

    for i in range(nfe):
        xc = xm[i]
        yc = ym[i]
        lb = get_aabb_length(xc, yc)
        
        # Store bounds: [x_min, x_max, y_min, y_max] derived from [left, right, down, up]
        box_edge_length[i, :] = [xc - lb[1], xc + lb[3], yc - lb[2], yc + lb[0]]

    global_param.Cur_group['RSFC'][name] = box_edge_length

def get_aabb_length(xc, yc):
    """
    Determines the maximum expansion length in 4 directions (Up, Left, Down, Right)
    until a collision with the relative obstacle (origin) or max size is reached.
    """
    # lb indices: 0: Up, 1: Left, 2: Down, 3: Right
    lb = np.zeros(4)
    is_completed = np.zeros(4)
    
    max_expansion = 2.0
    step_size = 0.5

    while np.sum(is_completed) < 4:
        for i in range(4):
            if is_completed[i]:
                continue
            
            test_lb = lb.copy()
            if test_lb[i] + step_size > max_expansion:
                is_completed[i] = 1
                continue
            
            test_lb[i] = test_lb[i] + step_size
            
            if is_current_expansion_valid(xc, yc, test_lb):
                lb = test_lb.copy()
            else:
                is_completed[i] = 1
                
    return lb

def is_current_expansion_valid(xc, yc, test_lb):
    """
    Checks if the proposed AABB intersects with the collision circle at the origin.
    Input test_lb order: [Up, Left, Down, Right]
    """
    xmin = xc - test_lb[1]
    xmax = xc + test_lb[3]
    ymin = yc - test_lb[2]
    ymax = yc + test_lb[0]

    collision_radius = 4.0

    # Calculate the squared distance from the origin (0,0) to the AABB
    min_dist_x = 0 if xmin <= 0 <= xmax else min(abs(xmin), abs(xmax))
    min_dist_y = 0 if ymin <= 0 <= ymax else min(abs(ymin), abs(ymax))

    dist_sq = min_dist_x**2 + min_dist_y**2

    # Return 1 if valid (no collision), 0 if collision
    if dist_sq <= collision_radius**2:
        return 0
    else:
        return 1