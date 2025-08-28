import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "geoai"))

import math
import geopandas as gpd
import numpy as np
import pandas as pd
import pystac_client
import stackstac
import torch
import yaml
from box import Box
from matplotlib import pyplot as plt
from rasterio.enums import Resampling
from shapely import Point
from sklearn import decomposition, svm
from torchvision.transforms import v2
import datetime

from claymodel.module import ClayMAEModule
from geoai.clay import Clay


def get_data():
    """Common data loading function for both approaches"""
    # Point over Monchique Portugal
    lat, lon = 37.30939, -8.57207

    # Dates of a large forest fire
    start = "2018-07-01"
    end = "2018-09-01"

    STAC_API = "https://earth-search.aws.element84.com/v1"
    COLLECTION = "sentinel-2-l2a"

    # Search the catalogue
    catalog = pystac_client.Client.open(STAC_API)
    search = catalog.search(
        collections=[COLLECTION],
        datetime=f"{start}/{end}",
        bbox=(lon - 1e-5, lat - 1e-5, lon + 1e-5, lat + 1e-5),
        max_items=100,
        query={"eo:cloud_cover": {"lt": 80}},
    )

    all_items = search.get_all_items()

    # Reduce to one per date (there might be some duplicates
    # based on the location)
    items = []
    dates = []
    for item in all_items:
        if item.datetime.date() not in dates:
            items.append(item)
            dates.append(item.datetime.date())

    print(f"Found {len(items)} items")

    # Extract coordinate system from first item
    epsg_str = items[0].properties["proj:code"]
    epsg = int(epsg_str.split(":")[-1])  # Convert 'EPSG:32629' to 32629

    # Convert point of interest into the image projection
    # (assumes all images are in the same projection)
    poidf = gpd.GeoDataFrame(
        pd.DataFrame(),
        crs="EPSG:4326",
        geometry=[Point(lon, lat)],
    ).to_crs(epsg_str)

    coords = poidf.iloc[0].geometry.coords[0]

    # Create bounds in projection
    size = 256
    gsd = 10
    bounds = (
        coords[0] - (size * gsd) // 2,
        coords[1] - (size * gsd) // 2,
        coords[0] + (size * gsd) // 2,
        coords[1] + (size * gsd) // 2,
    )

    # Retrieve the pixel values, for the bounding box in
    # the target projection. In this example we use only
    # the RGB and NIR bands.
    stack = stackstac.stack(
        items,
        bounds=bounds,
        snap_bounds=False,
        epsg=epsg,
        resolution=gsd,
        dtype="float64",
        rescale=False,
        fill_value=np.nan,
        assets=["blue", "green", "red", "nir"],
        resampling=Resampling.nearest,
    )

    stack = stack.compute()

    return stack, lat, lon, bounds, epsg_str, gsd


def generate_embeddings_core_clay_individual(stack, lat, lon):
    """Generate embeddings using core Clay library but processing one by one like wrapper"""
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    ckpt = "~/.cache/clay/clay-v1.5.ckpt"
    torch.set_default_device(device)

    model = ClayMAEModule.load_from_checkpoint(
        ckpt,
        model_size="large",
        metadata_path="geoai/config/clay_metadata.yaml",
        dolls=[16, 32, 64, 128, 256, 768, 1024],
        doll_weights=[1, 1, 1, 1, 1, 1, 1],
        mask_ratio=0.0,
        shuffle=False,
    )
    model.eval()
    model = model.to(device)

    # Extract mean, std, and wavelengths from metadata
    platform = "sentinel-2-l2a"
    metadata = Box(yaml.safe_load(open("geoai/config/clay_metadata.yaml")))
    mean = []
    std = []
    waves = []
    # Use the band names to get the correct values in the correct order.
    for band in stack.band:
        mean.append(metadata[platform].bands.mean[str(band.values)])
        std.append(metadata[platform].bands.std[str(band.values)])
        waves.append(metadata[platform].bands.wavelength[str(band.values)])

    # Prepare the normalization transform function using the mean and std values.
    transform = v2.Compose(
        [
            v2.Normalize(mean=mean, std=std),
        ]
    )

    # Prep datetimes embedding using a normalization function from the model code.
    def normalize_timestamp(date):
        week = date.isocalendar().week * 2 * np.pi / 52
        hour = date.hour * 2 * np.pi / 24
        return (math.sin(week), math.cos(week)), (math.sin(hour), math.cos(hour))

    def normalize_latlon(lat, lon):
        lat = lat * np.pi / 180
        lon = lon * np.pi / 180
        return (math.sin(lat), math.cos(lat)), (math.sin(lon), math.cos(lon))

    # Process each image individually (like the wrapper does)
    datetimes = stack.time.values.astype("datetime64[s]").tolist()
    embeddings_list = []

    for i, datetime_obj in enumerate(datetimes):
        # Extract single image
        pixels_single = torch.from_numpy(stack[i].data.astype(np.float32))
        pixels_single = transform(pixels_single).unsqueeze(
            0
        )  # Add batch dimension like wrapper

        # Process individual time
        times = normalize_timestamp(datetime_obj)
        week_norm = times[0]
        hour_norm = times[1]

        # Process individual location
        latlons = normalize_latlon(lat, lon)
        lat_norm = latlons[0]
        lon_norm = latlons[1]

        # Prepare datacube for single image
        datacube = {
            "platform": platform,
            "time": torch.tensor(
                week_norm + hour_norm,  # Same as wrapper: week_norm + hour_norm
                dtype=torch.float32,
                device=device,
            ).unsqueeze(
                0
            ),  # Add batch dimension like wrapper
            "latlon": torch.tensor(
                lat_norm + lon_norm,
                dtype=torch.float32,
                device=device,  # Same as wrapper
            ).unsqueeze(
                0
            ),  # Add batch dimension like wrapper
            "pixels": pixels_single.to(device),
            "gsd": torch.tensor(stack.gsd.values, device=device),  # Single GSD value
            "waves": torch.tensor(waves, device=device),
        }

        with torch.no_grad():
            unmsk_patch, unmsk_idx, msk_idx, msk_matrix = model.model.encoder(datacube)
            # Extract only class token like wrapper
            embedding = unmsk_patch[:, 0, :].cpu().numpy()
            embeddings_list.append(embedding)

    # Stack all embeddings
    embeddings = np.vstack(embeddings_list)

    return embeddings


def compare_embeddings(embeddings1, embeddings2, label1="Method 1", label2="Method 2"):
    """Compare two sets of embeddings"""
    print(f"{label1} embeddings shape: {embeddings1.shape}")
    print(f"{label2} embeddings shape: {embeddings2.shape}")

    # Check if shapes match
    if embeddings1.shape != embeddings2.shape:
        print("ERROR: Embedding shapes do not match!")
        return False

    # Calculate various similarity metrics
    # 1. Check if embeddings are identical (within tolerance)
    tolerance_levels = [1e-10, 1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4]
    for tol in tolerance_levels:
        are_identical = np.allclose(embeddings1, embeddings2, atol=tol)
        print(f"Embeddings identical (tolerance={tol}): {are_identical}")
        if are_identical:
            break

    # 2. Maximum absolute difference
    max_diff = np.max(np.abs(embeddings1 - embeddings2))
    print(f"Maximum absolute difference: {max_diff}")

    # 3. Mean Squared Error
    mse = np.mean((embeddings1 - embeddings2) ** 2)
    print(f"Mean Squared Error: {mse}")

    # 4. Mean Absolute Error
    mae = np.mean(np.abs(embeddings1 - embeddings2))
    print(f"Mean Absolute Error: {mae}")

    # 5. Check individual embeddings for exact matches
    exact_matches = 0
    for i in range(embeddings1.shape[0]):
        if np.array_equal(embeddings1[i], embeddings2[i]):
            exact_matches += 1
        else:
            diff = np.max(np.abs(embeddings1[i] - embeddings2[i]))
            print(f"  Image {i}: max diff = {diff}")

    print(f"Exact matches: {exact_matches}/{embeddings1.shape[0]}")

    return {
        "max_diff": max_diff,
        "mse": mse,
        "mae": mae,
        "exact_matches": exact_matches,
        "total_images": embeddings1.shape[0],
    }


def main():
    print("=== Detailed Clay Embeddings Comparison ===")
    print("Loading data...")

    # Get common data
    stack, lat, lon, bounds, epsg_str, gsd = get_data()

    print("\nGenerating embeddings using core Clay (individual processing)...")
    embeddings_core_individual = generate_embeddings_core_clay_individual(
        stack, lat, lon
    )

    print("Generating embeddings using GeoAI wrapper...")
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # Create custom metadata for just the 4 bands we're using
    custom_metadata = {
        "band_order": ["blue", "green", "red", "nir"],
        "rgb_indices": [2, 1, 0],
        "gsd": 10,
        "bands": {
            "mean": {"blue": 1105.0, "green": 1355.0, "red": 1552.0, "nir": 2743.0},
            "std": {"blue": 1809.0, "green": 1757.0, "red": 1888.0, "nir": 1742.0},
            "wavelength": {"blue": 0.493, "green": 0.56, "red": 0.665, "nir": 0.842},
        },
    }

    clay_model = Clay(custom_metadata=custom_metadata, device=str(device))

    # Convert WGS84 bounds for Clay model
    # First convert stack bounds back to WGS84
    proj_bounds = (bounds[0], bounds[1], bounds[2], bounds[3])
    bounds_gdf = gpd.GeoDataFrame(
        geometry=[Point(bounds[0], bounds[1]), Point(bounds[2], bounds[3])],
        crs=epsg_str,
    ).to_crs("EPSG:4326")

    wgs84_bounds = (
        bounds_gdf.iloc[0].geometry.x,  # min_lon
        bounds_gdf.iloc[0].geometry.y,  # min_lat
        bounds_gdf.iloc[1].geometry.x,  # max_lon
        bounds_gdf.iloc[1].geometry.y,  # max_lat
    )

    # Process each image through Clay model
    embeddings_list = []
    datetimes = stack.time.values.astype("datetime64[s]").tolist()

    for i, datetime_obj in enumerate(datetimes):
        # Extract image for this time step [H, W, C]
        image = stack[i].values.transpose(
            1, 2, 0
        )  # Convert from [C, H, W] to [H, W, C]

        # Convert numpy datetime64 to Python datetime
        if hasattr(datetime_obj, "astype"):
            timestamp = datetime_obj.astype("datetime64[s]").astype("int")
            date = datetime.datetime.fromtimestamp(timestamp)
        else:
            date = datetime_obj

        # Generate embedding using geoai wrapper
        embedding = clay_model.generate(
            image=image,
            bounds=wgs84_bounds,
            date=date,
            gsd=gsd,
            only_cls_token=True,  # Get only the class token (global embedding)
        )

        embeddings_list.append(embedding.cpu().numpy())

    # Stack all embeddings
    embeddings_wrapper = np.vstack(embeddings_list)

    print("\n=== Comparison Results ===")
    results = compare_embeddings(
        embeddings_core_individual,
        embeddings_wrapper,
        "Core Clay (individual)",
        "GeoAI wrapper",
    )

    return results


if __name__ == "__main__":
    results = main()
