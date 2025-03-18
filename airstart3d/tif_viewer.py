import rasterio
import matplotlib.pyplot as plt
import numpy as np

src_tif = "/home/benjamin/Downloads/EU_DEM_mosaic_5deg/eudem_dem_4258_europe.tif"  # Input raster in EPSG:4326
utm32_tif = "/home/benjamin/Downloads/EU_DEM_mosaic_5deg/eudem_dem_32632_switzerland.tif"  # Output raster in EPSG:32632

swissraster = "/home/benjamin/Downloads/SwissRaster25/swiss-map-raster25_2021_1229_komb_1.25_2056_airstart.tif"
swissraster_utm32 = "/home/benjamin/Downloads/SwissRaster25/swiss-map-raster25_2021_1229_utm32T.tif"

swissraster_merged_interlaken = "/home/benjamin/Downloads/SwissRaster25/Interlaken/merged_raster_utm32T.tif"

def swissraster_to_rgb(img, cmap):
    # get values and colors:
    # Read the first band
    # Get the colormap
    # Convert colormap to a NumPy array (normalized to [0,1] if needed)
    palette = np.array([cmap[i] for i in range(256)])  # Shape (256, 4) -> (R, G, B, A)

    # Extract RGB and Alpha channels separately
    r = np.array([cmap[i][0] for i in range(256)])  # Red channel
    g = np.array([cmap[i][1] for i in range(256)])  # Green channel
    b = np.array([cmap[i][2] for i in range(256)])  # Blue channel
    a = np.array([cmap[i][3] for i in range(256)])  # Alpha channel (transparency)

    # Convert indexed image to RGB(A) using the colormap
    rgb_img = np.stack([
        r[img],  # Apply Red channel
        g[img],  # Apply Green channel
        b[img]   # Apply Blue channel
    ], axis=-1)
    return rgb_img


if __name__ == "__main__":

    # Open the multi-page TIFF file
    with rasterio.open(swissraster_merged_interlaken) as dataset:
        print(f"Number of pages (bands): {dataset.count}")

        print(f"Overviews (subdatasets): {dataset.overviews(1)}")  # List of available overview levels
        print(f"Dataset shape: {dataset.shape}")  # Height, Width

        for level, overview in enumerate(dataset.overviews(1)):
            print(f"Reading overview at level {level} (downsample factor: {overview})")
            overview_data = dataset.read(1, out_shape=(
                1,
                dataset.height // overview,
                dataset.width // overview
            ))
            print(f"Overview {level} shape: {overview_data.shape}")
        
        rgb_img = swissraster_to_rgb(dataset.read(1), dataset.colormap(1))

        # Plot the raster with the colormap
        plt.figure(figsize=(10, 10))
        plt.imshow(rgb_img)
        plt.colorbar(label="Color Index")
        plt.axis("off")
        plt.title("SwissMap Raster 25")
        plt.show()


    #with rasterio.open(swissraster) as src:
    #    #plt.imshow(src.read(1), cmap="terrain")
    #    #plt.colorbar(label="Elevation (m)")
    #    plt.imshow(src.read(1))
    #    plt.title("EU-DEM (Switzerland, UTM 32T, 25m)")
    #    plt.show()