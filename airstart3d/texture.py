
import os

import numpy as np
import matplotlib.pyplot as plt

from scipy.ndimage import sobel, laplace, gaussian_filter


'''
The texture classes render texture based on location and time.

We need a unique coding for time and location
tile_x: int
tile_y: int
seconds: unix?

Prerender for times?
1-Minutes:
1440 per day -> 42mb per tile

5-Minute:
288 per day -> 8mb per tile

8x9 tiles = 72 tiles
-> 3gb for 1 Minute
-> 576mb for 5 Minute -> Render on demand!





'''

class ElevationTexture:

    def __init__(self, tile_x, tile_y, elevation_data, folder):
        assert tile_x == int(tile_x), 'tile_x is not int'
        assert tile_y == int(tile_y), 'tile_y is not int'
        self.tile_x = tile_x
        self.tile_y = tile_y
        self.elevation_data = elevation_data
        self.width = elevation_data.shape[0]
        self.folder = folder
        self.str_direction = ""

    def filename(self):
        return f"airstart3d/textures/{self.folder}/tile_{self.tile_x}_{self.tile_y}{self.str_direction}.png"
    
    def render(self, sun_direction=None):
        if sun_direction is not None:
            self.str_direction = f"_{sun_direction.x:.4f}_{sun_direction.y:.4f}_{sun_direction.z:.4f}" # hackedihackhack
        tile = self.filename()
        if os.path.exists(tile): # do not cache at the moment!
            print("Use cached Texture!")
            return tile
        _, ax = self.prepare_plot()
        if sun_direction is not None:
            self.render_directional_tile(ax, sun_direction)
        else:
            self.render_tile(ax)
        self.close_plot(ax, tile)
        return tile
    
    def prepare_plot(self):
        return plt.subplots()

    def close_plot(self, ax, path):
        ax.set_aspect('equal', adjustable='box')
        ax.grid(False)
        ax.axis('off')
        plt.savefig(path, dpi=150, bbox_inches='tight', pad_inches=0, transparent=False)
        plt.close()



class ContourTexture (ElevationTexture):

    def __init__(self, tile_x, tile_y, elevation_data):
        super().__init__(tile_x, tile_y, elevation_data, "contours")

    def render_tile(self, ax):        
        # create contour texture:
        X = np.arange(self.width)
        Y = -np.arange(self.width)
        X, Y = np.meshgrid(X, Y)

        # iso lines:
        ax.contour(X, Y, self.elevation_data, levels=np.arange(0, 4000, 50), vmin=500, vmax=3000)

class SlopeTexture (ElevationTexture):

    def __init__(self, tile_x, tile_y, elevation_data):
        super().__init__(tile_x, tile_y, elevation_data, "slope")
    
    def render_tile(self, ax):
        Z = self.elevation_data
        dzdx = sobel(Z, axis=1)
        dzdy = sobel(Z, axis=0)

        # compute slope
        slope = np.sqrt(dzdx**2 + dzdy**2)
        ax.imshow(slope, cmap='bwr', origin='upper', vmin=0, vmax=200)

class CurvatureTexture (ElevationTexture):

    def __init__(self, tile_x, tile_y, elevation_data):
        super().__init__(tile_x, tile_y, elevation_data, "curvature")
    
    def render_tile(self, ax):
        gZ = gaussian_filter(self.elevation_data, 1.0, mode="nearest")
        # compute curvature
        curvature = -laplace(gZ)
        # fix border:
        curvature = curvature[1:-1, 1:-1]
        # render
        ax.imshow(curvature, cmap='bwr', origin='upper')

class ThermalTexture (ElevationTexture):
    
    def __init__(self, tile_x, tile_y, elevation_data, skip_d8=False):
        super().__init__(tile_x, tile_y, elevation_data, "thermal")
        self.skip_d8 = skip_d8

    def render_directional_tile(self, ax, sun_direction):
        thermal = compute_thermal_differential(self.elevation_data, sun_direction, self.skip_d8)
        ax.imshow(thermal)


### Thermal computation


def compute_thermal_differential(elevation_data, sun_direction, skip_d8):

    DO_PLOT = False
    if DO_PLOT:    
        plt.close()
        fig, ax = plt.subplots()

    # Compute Area of each tile:
    Z = elevation_data
    real_area = np.zeros(elevation_data.shape)        
    for row in range(real_area.shape[0]-1):
        for col in range(real_area.shape[1]-1):
            x = np.array([25, Z[row+1, col]-Z[row, col], 0])
            z = np.array([0, Z[row, col+1] - Z[row, col], 25])
            area = np.linalg.norm(np.cross(x, z))
            real_area[row, col] = area/625    
    real_area = np.sqrt(np.clip(real_area, 0, 2)) # 2x => 45Grad    
    #real_area.fill(1.0) # do not use real_area
    
    if DO_PLOT:
        ax.imshow(real_area)
        plt.show()
        #plt.close()
        fig, ax = plt.subplots()

    # Compute normals
    Z = elevation_data
    dzdx = sobel(Z, axis=1)
    dzdy = sobel(Z, axis=0)
    
    sun_direction = np.array([sun_direction.x, sun_direction.y, sun_direction.z])
    sun_intensity = np.zeros(elevation_data.shape)        
    # get vector of sun
    for row in range(sun_intensity.shape[0]):
        for col in range(sun_intensity.shape[1]):
            x = np.array([1, dzdx[row, col], 0])
            y = np.array([0, dzdy[row, col], 1])
            n = -np.cross(x, y)
            n = n / np.linalg.norm(n)
            sun_intensity[row, col] = 1.0 * np.dot(n, sun_direction) * real_area[row, col]    
    sun_intensity = np.clip(sun_intensity, 0, 2) # should not be higher..
    
    # inversion:
    #sun_intensity[elevation_data < 1000] = 0.

    if DO_PLOT:
        ax.imshow(sun_intensity)
        plt.show()
        fig, ax = plt.subplots()

    if skip_d8:
        return sun_intensity

    # D8 Flow Algorithm:
    Z = gaussian_filter(-elevation_data, 2.0) # negative elevation        
    # Grid size        
    rows, cols = Z.shape

    d8_dirs = np.array([[6,  7,   8],
                        [5,  0,   1],
                        [4,  3,   2]])

    # Offsets for neighbors (dx, dy)
    dx = [-1,  0,  1, -1, 1, -1, 0, 1]
    dy = [-1, -1, -1,  0, 0,  1, 1, 1]

    # Initialize flow direction grid
    flow_dir = np.zeros_like(Z, dtype=int)

    # Compute flow direction for each cell (excluding edges)
    for y in range(1, rows - 1):
        for x in range(1, cols - 1):
            min_slope = 0
            best_direction = 0

            # Get elevation of the current cell
            elev = Z[y, x]

            # Check all 8 neighbors
            for i in range(8):
                nx, ny = x + dx[i], y + dy[i]  # Neighbor coordinates
                neighbor_elev = Z[ny, nx]

                # Compute slope (difference in elevation)
                slope = elev - neighbor_elev  # No need to divide by distance for D8

                # Find the steepest downward slope
                if slope > min_slope:
                    min_slope = slope
                    best_direction = d8_dirs[i // 3, i % 3]  # Get corresponding D8 value

            # Assign flow direction
            flow_dir[y, x] = best_direction
        
        ## Compute accumulated flow:
    #flow_rate = 0.1 # this is introduced at each cell    

    directions = {
        0: (0, 0),
        6: (-1, -1),
        7: (0, -1),
        8: (1, -1),
        5: (-1, 0),
        1: (1, 0),
        4: (-1, 1),
        3: (0, 1),
        2: (1, 1)
    }

    dt = 1.0
    total_time = 10
    k = 0.1
    flow_accumulated = np.zeros(flow_dir.shape)
    next_accumulated = np.zeros(flow_dir.shape)
    #flow_accumulated = sun_intensity.copy() # initialize
    #cell_accumulated = np.zeros(flow_dir.shape)
    #release_threshold = 1200 * 30 # so many timesteps

    for t in range(int(total_time/ dt)):
        next_accumulated.fill(0)
        for row in range(flow_accumulated.shape[0]):
            for col in range(flow_accumulated.shape[1]):                
                
                next_accumulated[row, col] += sun_intensity[row, col]

                direction = flow_dir[row, col]
                dc, dr = directions[direction]
                nr, nc = row+dr, col+dc

                if 1 <= nr < flow_accumulated.shape[0]-1 and 1 <= nc < flow_accumulated.shape[1]-1: # ignore border
                    next_accumulated[nr, nc] += k*flow_accumulated[row, col] # send current flow
                
                # if at 00: release thermal
                #if dc == 0 and dr == 0:
                #    next_accumulated[nr, nc] = 0.

                #cell_accumulated[row, col] += sun_intensity[row, col]
                #if cell_accumulated[row, col] > release_threshold:
                #    # release own cell state
                #    if 1 <= nr < flow_accumulated.shape[0]-1 and 1 <= nc < flow_accumulated.shape[1]-1: # ignore border
                #        next_accumulated[nr, nc] += cell_accumulated[row, col]
                #    
                #    # reset state
                #    cell_accumulated[row, col] = 0.
                
        flow_accumulated = next_accumulated
                                                   
        #        if 1 <= nr < flow_accumulated.shape[0]-1 and 1 <= nc < flow_accumulated.shape[1]-1: # ignore border
        #            #next_accumulated[nr, nc] += (sun_intensity[row, col] + k*flow_accumulated[row, col])*dt
        #            # spread only:
        #            next_accumulated[nr, nc] += (k*flow_accumulated[row, col])
        #flow_accumulated = gaussian_filter(flow_accumulated, 0.1) + 0.1 * np.clip(next_accumulated.copy(), 0, 1200)

        if DO_PLOT:
            # redraw
            ax.imshow(flow_accumulated, origin="upper")
            fig.canvas.draw()
            plt.pause(0.05)

    if DO_PLOT:
        # final draw:
        ax.imshow(flow_accumulated, origin="upper")

    return flow_accumulated