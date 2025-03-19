

from datetime import datetime
import rasterio
import numpy as np
import os
from scipy.ndimage import sobel, laplace, gaussian_filter

from vpython import *

import matplotlib.pyplot as plt

from tif_viewer import swissraster_to_rgb

from airstart3d.sun import Sun

'''
This script reads elevation data from the EU-DEM project
'''


''' i will borrow that: '''
class plot3D:
    def __init__(self, f, L, xmin, xmax, ymin, ymax, zmin, zmax, texture=None):
        # The x axis is labeled y, the z axis is labeled x, and the y axis is labeled z.
        # This is done to mimic fairly standard practive for plotting
        #     the z value of a function of x and y.
        self.f = f
        self.texture = texture
        if not xmin: self.xmin = 0
        else: self.xmin = xmin
        if not xmax: self.xmax = 1
        else: self.xmax = xmax
        if not ymin: self.ymin = 0
        else: self.ymin = ymin
        if not ymax: self.ymax = 1
        else: self.ymax = ymax
        if not zmin: self.zmin = 0
        else: self.zmin = zmin
        if not zmax: self.zmax = 1
        else: self.zmax = zmax
        
        self.L = L
        #R = L/100
        #d = L-2
        #xaxis = cylinder(pos=vec(0,0,0), axis=vec(0,0,d), radius=R, color=color.yellow)
        #yaxis = cylinder(pos=vec(0,0,0), axis=vec(d,0,0), radius=R, color=color.yellow)
        #zaxis = cylinder(pos=vec(0,0,0), axis=vec(0,d,0), radius=R, color=color.yellow)
        #k = 1.02
        #h = 0.05*L
        #text(pos=xaxis.pos+k*xaxis.axis, text='x', height=h, align='center', billboard=True, emissive=True)
        #text(pos=yaxis.pos+k*yaxis.axis, text='y', height=h, align='center', billboard=True, emissive=True)
        #text(pos=zaxis.pos+k*zaxis.axis, text='z', height=h, align='center', billboard=True, emissive=True)
    
        self.vertices = []
        for x in range(L):
            for y in range(L):
                val = self.evaluate(x,y)
                x_rel = (x/L)*(self.xmax-self.xmin)+self.xmin
                y_rel = (y/L)*(self.ymax-self.ymin)+self.ymin
                self.vertices.append(self.make_vertex(x_rel, y_rel, x, y, val ))
        
        self.make_quads()
        self.make_normals()
        
    def evaluate(self, x, y):
        return self.f(x, y) # absolute evaluation
        # 
        #d = self.L-2
        #return (d/(self.zmax-self.zmin)) * (self.f(self.xmin+x*(self.xmax-self.xmin)/d, self.ymin+y*(self.ymax-self.ymin)/d)-self.zmin)
    
    def make_quads(self):
        # Create the quad objects, based on the vertex objects already created.
        for x in range(self.L-2):
            for y in range(self.L-2):
                v0 = self.get_vertex(x,y)
                v1 = self.get_vertex(x+1,y)
                v2 = self.get_vertex(x+1, y+1)
                v3 = self.get_vertex(x, y+1)
                quad(vs=[v0, v1, v2, v3], texture=self.texture)
                #quad(vs=[v2, v1, v0, v3], texture=self.texture)
        
    def make_normals(self):
        # Set the normal for each vertex to be perpendicular to the lower left corner of the quad.
        # The vectors a and b point to the right and up around a vertex in the xy plance.
        for i in range(self.L*self.L):
            x = int(i/self.L)
            y = i % self.L
            if x == self.L-1 or y == self.L-1: continue
            v = self.vertices[i]
            a = self.vertices[i+self.L].pos - v.pos
            b = self.vertices[i+1].pos - v.pos
            v.normal = cross(a,b)
    
    def replot(self):
        for i in range(self.L*self.L):
            x = int(i/self.L)
            y = i % self.L
            v = self.vertices[i]
            v.pos.y = self.evaluate(x,y)
        self.make_normals()
                
    def make_vertex(self,x,y,tex_x, tex_y, value):        
        #texpos = vector(tex_x/self.L, tex_y/self.L, 0)
        texpos = vector(tex_y/self.L, 1.-tex_x/self.L, 0)
        return vertex(pos=vec(y,value,x), texpos=texpos, shininess=0.0, normal=vec(0,1,0))
        
    def get_vertex(self,x,y):
        return self.vertices[x*self.L+y]
        
    def get_pos(self,x,y):
        return self.get_vertex(x,y).pos

''' end of it '''


def get_tile_coordinates(lat, lon, tile_size=5):
    # Determine the bottom-left corner of the 5x5 degree tile
    lat_min = (lat // tile_size) * tile_size
    lon_min = (lon // tile_size) * tile_size

    # Calculate the top-right corner of the tile
    lat_max = lat_min + tile_size
    lon_max = lon_min + tile_size

    return lat_min, lon_min, lat_max, lon_max

def extract_elevation_from_tile(tile_path, lat, lon, grid_size_m=25):
    with rasterio.open(tile_path) as src:
        # Convert lat/lon to pixel coordinates within the tile
        row, col = src.index(lon, lat)
        
        # Calculate the pixel buffer for the 25x25m grid around the point
        # We are working with EU-DEM (25m resolution), so the grid size is 25m
        window = rasterio.windows.Window(col, row, grid_size_m, grid_size_m)
        
        # Read the data for the 25x25m window
        elevation_data = src.read(1, window=window)
        
        return elevation_data

def read_elevation_data_4258(lon, lat, width):
    tile_path = "/home/benjamin/Downloads/EU_DEM_mosaic_5deg/eudem_dem_4258_europe.tif"    

    # offline fix
    #import os.path
    #if not os.path.isfile(tile_path):
    #    return np.load("airstart3d/elevation_data.npy")

    with rasterio.open(tile_path) as src:
        row, col = src.index(lon, lat)
        grid = int(width//25)
        half_grid = grid // 2
        #window = rasterio.windows.Window(col-half_grid, row-half_grid, grid, grid)
        window = rasterio.windows.Window(col, row, grid, grid)
        return src.read(1, window=window)   

def read_swissraster_utm32(x, y, width):
    #tile_path = "/home/benjamin/Downloads/SwissRaster25/swiss-map-raster25_2021_1229_utm32T.tif" # requires colormap
    tile_path = "data/airstart3d/SwissRaster25/Interlaken/merged_raster_utm32T.tif" # full region
    with rasterio.open(tile_path) as src:
        row, col = src.index(x, y)
        window = rasterio.windows.Window(col, row, width, width)
        data = src.read(1, window=window)
        rgb_img = swissraster_to_rgb(data, src.colormap(1))
        fig, ax = plt.subplots()
        ax.imshow(rgb_img)
        ax.set_aspect('equal', adjustable='box')
        ax.grid(False)
        ax.axis('off')
        plt.savefig(f'airstart3d/textures/swissraster/tile_{x}_{y}.png', dpi=150, bbox_inches='tight', pad_inches=0, transparent=False)

def read_elevation_data_32632(x, y, width):
    tile_path = "data/airstart3d/elevation/eudem_dem_32632_switzerland.tif"
    with rasterio.open(tile_path) as src:
        row, col = src.index(x, y)
        grid = int(width//25)
        half_grid = grid // 2        
        window = rasterio.windows.Window(col, row, grid, grid)
        data = src.read(1, window=window)
    
    # create contour texture:
    X = np.arange(grid)
    Y = -np.arange(grid)
    X, Y = np.meshgrid(X, Y)
    fig, ax = plt.subplots()
    # iso lines:
    ax.contour(X, Y, data, levels=np.arange(0, 4000, 50), vmin=500, vmax=3000)
    #img = ax.imshow(elevation_data, cmap="gray", origin="upper")

    ax.set_aspect('equal', adjustable='box')
    ax.grid(False)
    ax.axis('off')
    plt.savefig(f'airstart3d/textures/contours/tile_{x}_{y}.png', dpi=150, bbox_inches='tight', pad_inches=0, transparent=False)

    Z = data
    gZ = gaussian_filter(data, 1.0, mode="nearest")
    dzdx = sobel(Z, axis=1)
    dzdy = sobel(Z, axis=0)

    # compute slope
    slope = np.sqrt(dzdx**2 + dzdy**2)

    # compute curvature
    curvature = laplace(gZ)
    # fix border:
    curvature = curvature[1:-1, 1:-1]
    #curvature[:, 0] = 0
    #curvature[0, :] = 0
    #curvature[:, -1] = 0        
    #curvature[-1, :] = 0

    # compute South aspect:
    aspect = np.arctan2(dzdy, dzdx) * (180/np.pi)
    aspect[aspect < 0] += 360 # inside [0, 360]
    
    # max is 180:
    # aspect[aspect > 180] -= 180 # inside[0, 180]
    
    # show the curvature:
    #plt.imshow(curvature, cmap='bwr_r', origin='upper', vmin=-10, vmax=10)

    # Aspect:
    # plt.imshow(aspect, cmap="gray", origin='upper')

    plt.close()
    fig, ax = plt.subplots()
    ax.imshow(slope, cmap='bwr', origin='upper', vmin=0, vmax=200)
    ax.set_aspect('equal', adjustable='box')
    ax.grid(False)
    ax.axis('off')
    plt.savefig(f'airstart3d/textures/slope/tile_{x}_{y}.png', dpi=150, bbox_inches='tight', pad_inches=0, transparent=False)

    plt.close()
    fig, ax = plt.subplots()
    ax.imshow(curvature, cmap='bwr_r', origin='upper', vmin=-10, vmax=10)
    ax.set_aspect('equal', adjustable='box')
    ax.grid(False)
    ax.axis('off')
    plt.savefig(f'airstart3d/textures/curvature/tile_{x}_{y}.png', dpi=150, bbox_inches='tight', pad_inches=0, transparent=False)

    thermal = compute_thermal_differential(x, y, data)
    plt.close()
    fig, ax = plt.subplots()
    ax.imshow(thermal, vmin=0, vmax=1.15, origin='upper')
    ax.set_aspect('equal', adjustable='box')
    ax.grid(False)
    ax.axis('off')
    plt.savefig(f'airstart3d/textures/thermal/tile_{x}_{y}.png', dpi=150, bbox_inches='tight', pad_inches=0, transparent=False)
    

    return data
 
def compute_thermal_differential(x, y, elevation_data):

    DO_PLOT = True and __name__ == '__main__'
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
    
    sun = Sun(datetime(2025, 3, 8, 12, 30), '46.5', '7.9')
    sun_direction = sun.get_sun_direction(sun.observer)
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
    
    #return sun_intensity # stop here!


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
        print('t:', t)
        #print('max cell: ', np.max(cell_accumulated))
        print('max flow: ', np.max(flow_accumulated))
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
        
        # release thermals:
        #next_accumulated[flow_dir == 0] = 0.
        flow_accumulated = np.clip(next_accumulated, 0, 100)
                                                   
                
                
                
                

        #        if 1 <= nr < flow_accumulated.shape[0]-1 and 1 <= nc < flow_accumulated.shape[1]-1: # ignore border
        #            #next_accumulated[nr, nc] += (sun_intensity[row, col] + k*flow_accumulated[row, col])*dt
        #            # spread only:
        #            next_accumulated[nr, nc] += (k*flow_accumulated[row, col])
        #flow_accumulated = gaussian_filter(flow_accumulated, 0.1) + 0.1 * np.clip(next_accumulated.copy(), 0, 1200)

        if DO_PLOT:
            # redraw
            ax.imshow(flow_accumulated, origin="upper")
            #ax.imshow(cell_accumulated, origin="upper")
            fig.canvas.draw()
            plt.pause(0.05)


    if DO_PLOT:
        # final draw:
        ax.imshow(flow_accumulated, origin="upper")
        #ax.imshow(cell_accumulated, origin="upper")

        ax.set_aspect('equal', adjustable='box')
        ax.grid(False)
        ax.axis('off')
        plt.savefig('airstart3d/textures/test/test.png', dpi=150, bbox_inches='tight', pad_inches=0, transparent=False)                
        plt.show()
    return flow_accumulated


if __name__ == '__main__':
        


    # Setup Scene Lights
    for i in range( 0, len(scene.lights) ):
        LL = scene.lights[i]
        LL.color *= 0.2
        print( LL.pos, LL.direction, LL.color )        
    scene.ambient = color.white * 0.5

    # Coordinates for Grindelwald, Switzerland
    #lat = 46.6242
    #lon = 8.0390

    # Swiss Cup Flex:
    #lat=46.66903082499045
    #lon=7.957116401763962
    #width=3356.926548743599

    # in shorter
    lat=46.679594465640186
    lon=7.9541015625    
    width=3356.9786285001073
    #width *= 4

    # x, y at top_left?
    #x=420011.130248355
    #y=5170090.784371755    
    x = 420011-2000
    y = 5170090
    width = 5000

    # 1. Determine the 5x5 degree tile for Grindelwald
    #lat_min, lon_min, lat_max, lon_max = get_tile_coordinates(lat, lon)

    #print(f"Tile coordinates: ({lat_min}, {lon_min}) to ({lat_max}, {lon_max})")

    # 2. Assuming you already have the tile downloaded, specify the tile file path
    #tile_file_path = f"path_to_your_tiles/DEM_{lat_min}_{lon_min}_to_{lat_max}_{lon_max}.tif"

    # 3. Extract elevation data for the 25x25m grid around Grindelwald
    #elevation_grid = extract_elevation_from_tile(tile_file_path, lat, lon)
    #print(f"Elevation grid (25x25m) for Grindelwald:\n{elevation_grid}")

    # old system
    #elevation_data = read_elevation_data(lon, lat, width)
    # proper projection
    elevation_data = read_elevation_data_32632(x, y, width)


    USE_MATPLOTLIB = False
    USE_VPYTHON = True
    if USE_MATPLOTLIB:
        # plot contour
        grid = elevation_data.shape[0]
        X = np.arange(grid)
        Y = -np.arange(grid)
        X, Y = np.meshgrid(X, Y)

        plt.close()
        

        # iso lines:
        #ax.contour(X, Y, elevation_data, levels=50)
        #ax.imshow(elevation_data, cmap="gray", origin="upper")


        compute_thermal_differential(x, y, elevation_data)


       
    
    if USE_VPYTHON:
        # use vpython

        L = elevation_data.shape[0]
        #L = L // 4
        if scene is None:
            scene = canvas(title="3D Elevation Profile", width=800, height=400)
            scene.center = vec(0.05*L,0.2*L,0)
            scene.range = 1.3*L

        t = 0
        dt = 0.02
        def f(x, y):
            # Return the value of the function of x and y:
            #return 0.7+0.2*sin(10*x)*cos(10*y)*sin(5*t)
            #grid = int(width//25)
            #i = np.clip(int(x*grid), 0, grid-1)
            #j = np.clip(int(y*grid), 0, grid-1)
            #return elevation_data[x*4,y*4]-2000
            return elevation_data[x,y]-2000

        np.save("airstart3d/elevation_data.npy", elevation_data)

        #660.643, -1500, -495.034>
        #texture = f'airstart3d/textures/contours/tile_{x}_{y}.png'
        #texture = f'airstart3d/textures/slope/tile_{x}_{y}.png'
        #texture = f'airstart3d/textures/curvature/tile_{x}_{y}.png'
        texture = f'airstart3d/textures/thermal/tile_{x}_{y}.png'
        #texture = 'airstart3d/textures/test/test.png'
        w2 = width//2
        p = plot3D(f, L, -w2, w2, -w2, w2, 0, 1000, texture=texture) # function, xmin, xmax, ymin, ymax (defaults 0, 1, 0, 1, 0, 1)



    '''

            # Create the VPython scene
            scene = canvas(title="2D Elevation Profile", width=800, height=400)

            # Generate synthetic elevation data (replace with real data)
            x_vals = np.linspace(0, width, grid)
            y_vals = np.linspace(0, width, grid)
            x,y = np.meshgrid(x_vals, y_vals)

            # Create the curve for elevation
            elevation_curve = curve(color=color.white)

            # Add points to the curve
            for i in range(grid):
                for j in range(grid):
                    elevation_curve.append(vector(x_vals[i], y_vals[j], elevation_data[i, j]))        

            # Add axis lines
            #curve(vector(-10, 0, 0), vector(10, 0, 0), color=color.red)  # X-axis
            #curve(vector(-10, -5, 0), vector(-10, 5, 0), color=color.blue)  # Elevation axis


            # Generate synthetic elevation data (for example purposes)
            #size = 200  # Grid size
            size = grid
            x_vals = np.linspace(0, width, size)
            y_vals = np.linspace(0, width, size)
            X, Y = np.meshgrid(x_vals, y_vals)
            #Z = np.sin(X) * np.cos(Y) * 5  # Sample elevation function
            Z = elevation_data

            # VPython scene setup
            scene = canvas(title="Elevation Plot", width=800, height=600)

            # Create points in 3D space
            for i in range(size):
                for j in range(size):
                    x, y, z = X[i, j], Z[i, j], -Y[i, j]
                    sphere(pos=vector(x, y, z), radius=25, color=vector(0.7, 0.7, 0.7))  # Gray color


    '''



    
