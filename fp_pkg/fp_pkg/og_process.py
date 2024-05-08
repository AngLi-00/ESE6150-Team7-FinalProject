import math
import numpy as np
from numba import njit

@njit(cache=True)
def process_occ_grid(occupancy_grid, resolution, ranges, angles, m_ranges, m_angles, dilate):
    # Process Regular Coords
    x_detect = ranges * np.cos(angles)
    y_detect = ranges * np.sin(angles)

    x_coords = np.round(x_detect / resolution) + occupancy_grid.shape[1] // 2
    y_coords = np.round(y_detect / resolution) + occupancy_grid.shape[0] // 2

    x_coords = x_coords.astype(np.int64)
    y_coords = y_coords.astype(np.int64)

    filter = (x_coords >= 0) & (x_coords < occupancy_grid.shape[1]) & \
        (y_coords >= 0) & (y_coords < occupancy_grid.shape[0])
    
    xc_filt = x_coords[filter]
    yc_filt = y_coords[filter]

    # Process Max Coords
    xm_detect = m_ranges * np.cos(m_angles)
    ym_detect = m_ranges * np.sin(m_angles)

    xm_coords = np.round(xm_detect / resolution) + occupancy_grid.shape[1] // 2
    ym_coords = np.round(ym_detect / resolution) + occupancy_grid.shape[0] // 2

    xm_coords = xm_coords.astype(np.int64)
    ym_coords = ym_coords.astype(np.int64)

    # Fill In OG
    if dilate:
        occupancy_grid.fill(0)
    else:
        occupancy_grid.fill(-1)

        origin = (occupancy_grid.shape[1] // 2, occupancy_grid.shape[0] // 2)

        xf = np.append(x_coords, xm_coords)
        yf = np.append(y_coords, ym_coords)

        for x1, y1 in zip(xf, yf):
            x0, y0 = origin
            points = []
            dx = abs(x1 - x0)
            dy = -abs(y1 - y0)
            sx = 1 if x0 < x1 else -1
            sy = 1 if y0 < y1 else -1
            err = dx + dy  # Error value e_xy
            while True:
                points.append([x0, y0])
                if x0 == x1 and y0 == y1:
                    break
                e2 = 2 * err
                if e2 >= dy: 
                    err += dy
                    x0 += sx
                if e2 <= dx: 
                    err += dx
                    y0 += sy
            
            points = np.array(points)
            points_x = points[:, 0]
            points_y = points[:, 1]

            # Filter coords outside OG
            pfilter = (points_x >= 0) & (points_x < occupancy_grid.shape[1]) & \
                (points_y >= 0) & (points_y < occupancy_grid.shape[0])
            
            for xp, yp in zip(points_x[pfilter], points_y[pfilter]):
                occupancy_grid[yp, xp] = 0

    for xf, yf in zip(xc_filt, yc_filt):
        occupancy_grid[yf, xf] = 1

    return occupancy_grid

@njit(cache=True)
def process_og_cam(occupancy_grid, resolution, x_points, y_points, dilate):

    # Process Regular Coords
    x_detect = x_points
    y_detect = y_points

    x_coords = np.round(x_detect / resolution) + occupancy_grid.shape[1] // 2
    y_coords = np.round(y_detect / resolution) + occupancy_grid.shape[0] // 2

    x_coords = x_coords.astype(np.int64)
    y_coords = y_coords.astype(np.int64)

    filter = (x_coords >= 0) & (x_coords < occupancy_grid.shape[1]) & \
        (y_coords >= 0) & (y_coords < occupancy_grid.shape[0])
    
    xc_filt = x_coords[filter]
    yc_filt = y_coords[filter]

    # Process Max Coords
    angles = np.arange(-np.pi / 4, np.pi / 4, 0.5)
    r_max = 5.0

    xm_detect = r_max * np.cos(angles)
    ym_detect = r_max * np.sin(angles)

    xm_coords = np.round(xm_detect / resolution) + occupancy_grid.shape[1] // 2
    ym_coords = np.round(ym_detect / resolution) + occupancy_grid.shape[0] // 2

    xm_coords = xm_coords.astype(np.int64)
    ym_coords = ym_coords.astype(np.int64)

    # Fill In OG
    if dilate:
        occupancy_grid.fill(0)

    else:
        occupancy_grid.fill(-1)

        det_set = set()

        for xf, yf in zip(xc_filt, yc_filt):
            det_set.add((xf, yf))

        origin = (occupancy_grid.shape[1] // 2, occupancy_grid.shape[0] // 2)

        x_bress = np.append(x_coords, xm_coords)
        y_bress = np.append(y_coords, ym_coords)

        for x1, y1 in zip(x_bress, y_bress):
            x0, y0 = origin
            points = []
            dx = abs(x1 - x0)
            dy = -abs(y1 - y0)
            sx = 1 if x0 < x1 else -1
            sy = 1 if y0 < y1 else -1
            err = dx + dy  # Error value e_xy
            while True:
                points.append([x0, y0])
                if x0 == x1 and y0 == y1:
                    break
                if (x0, y0) in det_set:
                    break
                e2 = 2 * err
                if e2 >= dy: 
                    err += dy
                    x0 += sx
                if e2 <= dx: 
                    err += dx
                    y0 += sy
            
            points = np.array(points)
            points_x = points[:, 0]
            points_y = points[:, 1]

            # Filter coords outside OG
            pfilter = (points_x >= 0) & (points_x < occupancy_grid.shape[1]) & \
                (points_y >= 0) & (points_y < occupancy_grid.shape[0])
            
            for xp, yp in zip(points_x[pfilter], points_y[pfilter]):
                occupancy_grid[yp, xp] = 0

    for xf, yf in zip(xc_filt, yc_filt):
        occupancy_grid[yf, xf] = 1

    return occupancy_grid