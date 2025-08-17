import geopandas as gpd
from shapely.geometry import Polygon
import subprocess

def df_to_gdf(df):
    # Create polygons from xmin, ymin, xmax, ymax
    polygons = [
        Polygon([
            (row.xmin, row.ymin),
            (row.xmax, row.ymin),
            (row.xmax, row.ymax),
            (row.xmin, row.ymax),
            (row.xmin, row.ymin)
        ])
        for row in df.itertuples(index=False)
    ]
    
    # Use the first row's EPSG for CRS (assuming all rows same EPSG)
    crs = f"EPSG:{df['epsg'].iloc[0]}"
    
    gdf = gpd.GeoDataFrame(df, geometry=polygons, crs=crs)
    return gdf

# Function to identify overlapping GFW extents for a given Sentinel-2 tile
def identify_master_extent(s2_tile, gfw_gdf):
    # Ensure both are in the same coordinate reference system (CRS)
    s2_tile = s2_tile.to_crs(gfw_gdf.crs)

    # Perform spatial join to identify overlapping GFW extents
    overlapping_gfw_extents = gpd.sjoin(gfw_gdf, s2_tile, how="inner", predicate="intersects")
    return overlapping_gfw_extents

def download_file(url, output_path):
    try:
        # Run the wget command with the specified URL and output path
        result = subprocess.run(['wget', '-O', output_path, url], check=True)
        print(f"File downloaded successfully to {output_path}")
    except subprocess.CalledProcessError as e:
        print(f"Error occurred: {e}")
