import time
import pickle
import pandas as pd
import numpy as np
from scipy.spatial import KDTree

def find_nearest_points_batch(all_points, reference_points, num_nearest=50):
    points = np.array(all_points)
    references = np.array(reference_points)

    tree = KDTree(points)
    distances, indices = tree.query(references, k=num_nearest)
    nearest_points = points[indices]

    return nearest_points

def save_global_parameters(global_param):
    file_path = 'global_param.pkl'
    with open(file_path, 'wb') as f:
        pickle.dump(global_param, f)

def load_global_parameters():
    file_path = 'global_param.pkl'
    with open(file_path, 'rb') as f:
        global_param = pickle.load(f)
    return global_param

def get_aabb_length(number, xm, ym, all_points):
    lb = np.zeros((number, 4))
    is_completed = np.zeros((number, 4))
    total_tasks = number * 4

    while np.sum(is_completed) < total_tasks:
        for i in range(4):
            if np.sum(is_completed[:, i]) == number:
                continue
            
            test_lb = lb.copy()
            condition = (test_lb[:, i] + 0.2) > 2.0
            is_completed[condition, i] = 1

            delta = np.ones(number) * 0.2
            delta[is_completed[:, i] == 1] = 0
            test_lb[:, i] += delta
            
            flag = is_current_expansion_valid(xm, ym, test_lb, lb, i, all_points)
            is_completed[np.logical_not(flag), i] = 1
            lb[:, i] = np.where(flag, test_lb[:, i], lb[:, i])
            
    return lb

def is_current_expansion_valid(xc, yc, test_lb, lb, i, all_points):
    ax = xc - lb[:, 1]
    ay = yc + lb[:, 0]
    bx = xc + lb[:, 3]
    by = yc + lb[:, 0]
    cx = xc + lb[:, 3]
    cy = yc - lb[:, 2]
    dx = xc - lb[:, 1]
    dy = yc - lb[:, 2]

    if i == 0:
        xmin, xmax = ax, bx
        ymin, ymax = cy, yc + test_lb[:, 0]
    elif i == 1:
        xmin, xmax = xc - test_lb[:, 1], bx
        ymin, ymax = dy, ay
    elif i == 2:
        xmin, xmax = dx, cx
        ymin, ymax = yc - test_lb[:, 2], by
    elif i == 3:
        xmin, xmax = ax, xc + test_lb[:, 3]
        ymin, ymax = dy, by

    rectangles = np.column_stack((xmin - 1, ymin - 1, xmax + 1, ymax + 1))
    is_valid = check_points_outside_rectangles(all_points, rectangles)
    
    return is_valid

def check_points_outside_rectangles(all_points, rectangles):
    x_min = rectangles[:, 0, np.newaxis]
    y_min = rectangles[:, 1, np.newaxis]
    x_max = rectangles[:, 2, np.newaxis]
    y_max = rectangles[:, 3, np.newaxis]

    outside_x = (all_points[:, :, 0] < x_min) | (all_points[:, :, 0] > x_max)
    outside_y = (all_points[:, :, 1] < y_min) | (all_points[:, :, 1] > y_max)
    outside = outside_x | outside_y

    return np.all(outside, axis=1)

def sfc_main(ig_traj, global_param):
    keys = list(ig_traj.keys())
    all_points = []
    xm = []
    ym = []
    x_flat = []
    y_flat = []
    
    num_steps = 200
    total_vehicles = len(keys)
    near_points = np.zeros((total_vehicles * num_steps, 50, 2))

    for i in range(total_vehicles):
        key_i = keys[i]
        traj = ig_traj[key_i]
        start_point = global_param.Cur_group['stage2']['source_id'][i]
        end_point = global_param.Cur_group['stage2']['goal_id'][i]
        
        name = f'lane_{start_point}_{end_point}'
        sfc_map_points = global_param.Cur_group['SFC_detail_map'][name]
        
        all_points.append(sfc_map_points)
        xm.append(traj['x'])
        ym.append(traj['y'])
        x_flat.extend(traj['x'])
        y_flat.extend(traj['y'])

    number_of_points = num_steps * total_vehicles
    
    for i in range(total_vehicles):
        reference_points = np.column_stack((xm[i], ym[i]))
        batch_near_points = find_nearest_points_batch(all_points[i], reference_points)
        near_points[i * num_steps : (i + 1) * num_steps, :, :] = batch_near_points

    x_flat_arr = np.array(x_flat)
    y_flat_arr = np.array(y_flat)
    
    lb = get_aabb_length(number_of_points, x_flat_arr, y_flat_arr, near_points)
    
    corridors = np.column_stack((
        x_flat_arr - lb[:, 1], 
        x_flat_arr + lb[:, 3], 
        y_flat_arr - lb[:, 2], 
        y_flat_arr + lb[:, 0]
    ))

    for i in range(total_vehicles):
        name = f'NO_{i}'
        box = corridors[i * num_steps : (i + 1) * num_steps, :]
        global_param.Cur_group['SFC'][name] = box
        
        df = pd.DataFrame(box, columns=['x_min', 'x_max', 'y_min', 'y_max'])
        df.to_csv(f'SFC{i}.csv', index=False)

    save_global_parameters(global_param)
    return 0