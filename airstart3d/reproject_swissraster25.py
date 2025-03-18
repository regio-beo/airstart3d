import rasterio
from rasterio.warp import calculate_default_transform, reproject, Resampling

# Input and output file paths
input_tif = "/home/benjamin/Downloads/SwissRaster25/swiss-map-raster25_2021_1229_komb_1.25_2056_airstart.tif"
output_tif = "/home/benjamin/Downloads/SwissRaster25/swiss-map-raster25_2021_1229_utm32T.tif"

# Define Destination CRS (UTM 32T)
dst_crs = "EPSG:32632"  # UTM Zone 32N (Switzerland)

# Open the source raster
with rasterio.open(input_tif) as src:
    print(f"Source CRS: {src.crs}")  
    print(f"Data Type: {src.dtypes}")  
    print(f"Band Count: {src.count}")  
    print(f"Source NoData Value: {src.nodata}")  

    # Extract colormap (only valid if there's a single band)
    colormap = src.colormap(1) if src.count == 1 else None

    # Compute transformation for UTM 32T (keep resolution consistent)
    transform, width, height = calculate_default_transform(
        src.crs, dst_crs, src.width, src.height, *src.bounds, resolution=1
    )

    # Update metadata while keeping dtype as uint8
    kwargs = src.meta.copy()
    kwargs.update({
        "crs": dst_crs,
        "transform": transform,
        "width": width,
        "height": height,
        "nodata": src.nodata,  
        "dtype": "uint8",  # Ensure colors remain uint8
        "compress": "lzw"   # Optional: Compression to reduce file size
    })

    # Create the output raster
    with rasterio.open(output_tif, "w", **kwargs) as dst:
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
        
        # If colormap exists, copy it
        if colormap:
            dst.write_colormap(1, colormap)

print("✅ Reprojection complete with colormap preserved! Saved as", output_tif)
