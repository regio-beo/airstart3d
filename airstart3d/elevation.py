

from datetime import datetime
import rasterio
import numpy as np
import os


from vpython import *

import matplotlib.pyplot as plt

from tif_viewer import swissraster_to_rgb

from airstart3d.sun import Sun
from airstart3d.texture import *

'''
This script reads elevation data from the EU-DEM project
'''

def plot_triggers(tile_x, tile_y, elevation_data, pos, width): 
    # deprecated   
    triggers = np.load(f"airstart3d/textures/curvature/trigger_{tile_x}_{tile_y}.npy")
    for row in range(triggers.shape[0]):
        for col in range(triggers.shape[1]):
            length = triggers[row, col]
            if length > 1.1:
                print("create sphere")
                trigger_pos = vector(col*25+pos.x, elevation_data[row+1, col+1], row*25+pos.z)
                axis = vector(0, 1, -0.2)
                cylinder(pos=trigger_pos, axis=axis, length=length*650, radius=length*50, color=color.yellow)

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


    #curvature[:, 0] = 0
    #curvature[0, :] = 0
    #curvature[:, -1] = 0        
    #curvature[-1, :] = 0

    # compute South aspect:
    #aspect = np.arctan2(dzdy, dzdx) * (180/np.pi)
    #aspect[aspect < 0] += 360 # inside [0, 360]
    
    # max is 180:
    # aspect[aspect > 180] -= 180 # inside[0, 180]
    
    # show the curvature:
    #plt.imshow(curvature, cmap='bwr_r', origin='upper', vmin=-10, vmax=10)

    # Aspect:
    # plt.imshow(aspect, cmap="gray", origin='upper')

    #plt.close()
    #fig, ax = plt.subplots()
    
    #ax.set_aspect('equal', adjustable='box')
    #ax.grid(False)
    #ax.axis('off')
    #plt.savefig(f'airstart3d/textures/slope/tile_{x}_{y}.png', dpi=150, bbox_inches='tight', pad_inches=0, transparent=False)


    # Thermal Trigger

    #plt.close()
    #fig, ax = plt.subplots()
    #thermal = compute_thermal_differential(x, y, data)
    #curvature[curvature > -5] = 0.
    #curvature[curvature < -5] = 1.0    
    #curvature = curvature * thermal[1:-1, 1:-1]
    #np.save(f"airstart3d/textures/curvature/trigger_{x}_{y}.npy", curvature)
    
    #ax.set_aspect('equal', adjustable='box')
    #ax.grid(False)
    #ax.axis('off')
    #plt.savefig(f'airstart3d/textures/curvature/tile_{x}_{y}.png', dpi=150, bbox_inches='tight', pad_inches=0, transparent=False)

    #thermal = compute_thermal_differential(x, y, data)
    #plt.close()
    #fig, ax = plt.subplots()
    #ax.imshow(thermal, vmin=0, vmax=1.15, origin='upper')
    #ax.set_aspect('equal', adjustable='box')
    #ax.grid(False)
    #ax.axis('off')
    #plt.savefig(f'airstart3d/textures/thermal/tile_{x}_{y}.png', dpi=150, bbox_inches='tight', pad_inches=0, transparent=False)
    
    return data
 



