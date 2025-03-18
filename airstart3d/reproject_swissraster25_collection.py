import os
import rasterio
import numpy as np
from rasterio.warp import calculate_default_transform, reproject, Resampling
from rasterio.merge import merge
from rasterio.io import MemoryFile

MAP_DIR = "/home/benjamin/Downloads/SwissRaster25/Interlaken"

# Output file path for the merged raster
output_tif = f"{MAP_DIR}/merged_raster_utm32T.tif"

if os.path.exists(output_tif):
    print("Error: output tif already exists!")
    exit(1)

# List of input raster files to stitch together
input_files = os.listdir(MAP_DIR)
input_files = list(filter(lambda file: file.lower().endswith('.tif'), input_files))
print('input files to reproject:', input_files)

# Define Destination CRS (EPSG:32632)
dst_crs = "EPSG:32632"  # UTM Zone 32N (Switzerland)

# Initialize an empty list to hold all reprojected rasters
reprojected_rasters = []

# Loop over each input file and reproject to EPSG:32632
for input_tif in input_files:
    with rasterio.open(f'{MAP_DIR}/{input_tif}') as src:
        print(f"Processing {input_tif}")

        # Extract colormap (only valid if there's a single band)
        colormap = src.colormap(1) if src.count == 1 else None
        
        # Compute transformation for reprojecting to EPSG:32632
        transform, width, height = calculate_default_transform(
            src.crs, dst_crs, src.width, src.height, *src.bounds, resolution=1
        )
        
        # Update metadata for the destination raster
        kwargs = src.meta.copy()
        kwargs.update({
            "crs": dst_crs,
            "transform": transform,
            "width": width,
            "height": height,
            "nodata": src.nodata,
            "dtype": "uint8",  # Ensure the dtype remains uint8 (if it's indexed)
            "compress": "lzw"   # Optional: Compression to reduce file size
        })
        
        # Create a temporary in-memory array to hold the reprojected data
        reprojected_data = np.empty((src.count, height, width), dtype=src.dtypes[0])
        
        # Create an in-memory file to hold the reprojected data
        memfile = MemoryFile()  # Create the memory file outside the 'with' block
        with memfile.open(**kwargs) as dst:
            # Reproject each band
            for i in range(1, src.count + 1):
                reproject(
                    source=rasterio.band(src, i),
                    destination=rasterio.band(dst, i),
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=transform,
                    dst_crs=dst_crs,
                    resampling=Resampling.nearest
                )
            
            # Append the opened MemoryFile dataset (not closed yet) to the list
            reprojected_rasters.append(memfile.open())

# Now, merge all reprojected rasters into one large raster
merged_data, merged_transform = merge(reprojected_rasters)

# Update metadata for the merged output raster
kwargs.update({
    "transform": merged_transform,
    "width": merged_data.shape[2],
    "height": merged_data.shape[1],
})

# Write the merged raster to the output file
with rasterio.open(output_tif, "w", **kwargs) as dst:
    for i in range(merged_data.shape[0]):
        dst.write(merged_data[i], i + 1)
    # If colormap exists, copy it
    if colormap:
        dst.write_colormap(1, colormap)

print(f"✅ Stitching and reprojecting complete! Merged raster saved as {output_tif}")
