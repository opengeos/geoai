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


def generate_embeddings_core_clay(stack, lat, lon):
    """Generate embeddings using core Clay library (TestClay.py approach)"""
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

    datetimes = stack.time.values.astype("datetime64[s]").tolist()
    times = [normalize_timestamp(dat) for dat in datetimes]
    week_norm = [dat[0] for dat in times]
    hour_norm = [dat[1] for dat in times]

    # Prep lat/lon embedding using the
    def normalize_latlon(lat, lon):
        lat = lat * np.pi / 180
        lon = lon * np.pi / 180
        return (math.sin(lat), math.cos(lat)), (math.sin(lon), math.cos(lon))

    latlons = [normalize_latlon(lat, lon)] * len(times)
    lat_norm = [dat[0] for dat in latlons]
    lon_norm = [dat[1] for dat in latlons]

    # Normalize pixels
    pixels = torch.from_numpy(stack.data.astype(np.float32))
    pixels = transform(pixels)

    # Prepare additional information
    datacube = {
        "platform": platform,
        "time": torch.tensor(
            np.hstack((week_norm, hour_norm)),
            dtype=torch.float32,
            device=device,
        ),
        "latlon": torch.tensor(
            np.hstack((lat_norm, lon_norm)), dtype=torch.float32, device=device
        ),
        "pixels": pixels.to(device),
        "gsd": torch.tensor(stack.gsd.values, device=device),
        "waves": torch.tensor(waves, device=device),
    }

    with torch.no_grad():
        unmsk_patch, unmsk_idx, msk_idx, msk_matrix = model.model.encoder(datacube)

    # The first embedding is the class token, which is the
    # overall single embedding. We extract that for PCA below.
    embeddings = unmsk_patch[:, 0, :].cpu().numpy()

    return embeddings


def generate_embeddings_geoai_wrapper(stack, lat, lon, bounds, epsg_str, gsd):
    """Generate embeddings using geoai wrapper (TestClay_GeoAI.py approach)"""
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
    embeddings = np.vstack(embeddings_list)

    return embeddings


def compare_embeddings(embeddings1, embeddings2):
    """Compare two sets of embeddings"""
    print(f"Core Clay embeddings shape: {embeddings1.shape}")
    print(f"GeoAI wrapper embeddings shape: {embeddings2.shape}")

    # Check if shapes match
    if embeddings1.shape != embeddings2.shape:
        print("ERROR: Embedding shapes do not match!")
        return False

    # Calculate various similarity metrics
    # 1. Mean Squared Error
    mse = np.mean((embeddings1 - embeddings2) ** 2)
    print(f"Mean Squared Error: {mse}")

    # 2. Mean Absolute Error
    mae = np.mean(np.abs(embeddings1 - embeddings2))
    print(f"Mean Absolute Error: {mae}")

    # 3. Correlation coefficient
    flat1 = embeddings1.flatten()
    flat2 = embeddings2.flatten()
    correlation = np.corrcoef(flat1, flat2)[0, 1]
    print(f"Correlation coefficient: {correlation}")

    # 4. Cosine similarity for each embedding pair
    cosine_similarities = []
    for i in range(embeddings1.shape[0]):
        emb1 = embeddings1[i]
        emb2 = embeddings2[i]
        # Cosine similarity = dot product / (norm1 * norm2)
        cos_sim = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
        cosine_similarities.append(cos_sim)

    print(f"Cosine similarities per embedding: {cosine_similarities}")
    print(f"Mean cosine similarity: {np.mean(cosine_similarities)}")
    print(f"Min cosine similarity: {np.min(cosine_similarities)}")
    print(f"Max cosine similarity: {np.max(cosine_similarities)}")

    # 5. Check if embeddings are identical (within various tolerances)
    tolerance_levels = [1e-10, 1e-8, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2]
    for tolerance in tolerance_levels:
        are_identical = np.allclose(embeddings1, embeddings2, atol=tolerance)
        print(f"Embeddings identical (tolerance={tolerance}): {are_identical}")
        if are_identical:
            print(f"✅ Embeddings are functionally identical at tolerance {tolerance}")
            break

    # More detailed analysis
    if not np.allclose(embeddings1, embeddings2, atol=1e-3):
        print("❌ Embeddings have significant differences even at 1e-3 tolerance")
    elif not np.allclose(embeddings1, embeddings2, atol=1e-6):
        print(
            "⚠️ Embeddings have minor numerical differences (likely due to processing order)"
        )
        print("   This is expected and doesn't affect practical performance")
    else:
        print("✅ Embeddings are essentially identical")

    # 6. Maximum absolute difference
    max_diff = np.max(np.abs(embeddings1 - embeddings2))
    print(f"Maximum absolute difference: {max_diff}")

    # Determine functional equivalence
    functionally_identical = np.allclose(embeddings1, embeddings2, atol=1e-4)

    return {
        "mse": mse,
        "mae": mae,
        "correlation": correlation,
        "mean_cosine_similarity": np.mean(cosine_similarities),
        "cosine_similarities": cosine_similarities,
        "functionally_identical": functionally_identical,
        "max_diff": max_diff,
    }


def main():
    print("=== Comparing Clay Embeddings ===")
    print("Loading data...")

    # Get common data
    stack, lat, lon, bounds, epsg_str, gsd = get_data()

    print("\nGenerating embeddings using core Clay library...")
    embeddings_core = generate_embeddings_core_clay(stack, lat, lon)

    print("Generating embeddings using GeoAI wrapper...")
    embeddings_wrapper = generate_embeddings_geoai_wrapper(
        stack, lat, lon, bounds, epsg_str, gsd
    )

    print("\n=== Comparison Results ===")
    results = compare_embeddings(embeddings_core, embeddings_wrapper)

    # Summary conclusion
    print("\n=== CONCLUSION ===")
    if results["functionally_identical"]:
        print("✅ The embeddings are FUNCTIONALLY IDENTICAL")
        print(
            "   Small numerical differences are due to processing order (batched vs individual)"
        )
        print("   Both approaches produce equivalent results for downstream tasks")
    elif results["mean_cosine_similarity"] > 0.999:
        print("✅ The embeddings are VERY SIMILAR (cosine similarity > 0.999)")
        print(
            "   Differences are likely due to numerical precision - functionally equivalent"
        )
    elif results["mean_cosine_similarity"] > 0.99:
        print("⚠️ The embeddings are SIMILAR (cosine similarity > 0.99)")
        print("   May indicate implementation differences worth investigating")
    else:
        print("❌ The embeddings are SIGNIFICANTLY DIFFERENT")
        print("   This indicates a problem in one of the implementations")

    print(f"\nKey metrics:")
    print(f"- Max difference: {results['max_diff']:.2e}")
    print(f"- Correlation: {results['correlation']:.10f}")
    print(f"- Mean cosine similarity: {results['mean_cosine_similarity']:.10f}")
    print(f"- MSE: {results['mse']:.2e}")

    return results


if __name__ == "__main__":
    results = main()
