import sys
import os

# sys.path.append(os.path.join(os.path.dirname(__file__), 'geoai'))

import geopandas as gpd
import numpy as np
import pandas as pd
import pystac_client
import stackstac
import torch
from matplotlib import pyplot as plt
from rasterio.enums import Resampling
from shapely import Point
from sklearn import decomposition, svm
import datetime

from geoai.clay import Clay


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

print(stack)

stack = stack.compute()

# Initialize Clay model using geoai wrapper with custom metadata for 4 bands
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
    geometry=[Point(bounds[0], bounds[1]), Point(bounds[2], bounds[3])], crs=epsg_str
).to_crs("EPSG:4326")

wgs84_bounds = (
    bounds_gdf.iloc[0].geometry.x,  # min_lon
    bounds_gdf.iloc[0].geometry.y,  # min_lat
    bounds_gdf.iloc[1].geometry.x,  # max_lon
    bounds_gdf.iloc[1].geometry.y,  # max_lat
)

# Process all images as a batch through Clay model
datetimes = stack.time.values.astype("datetime64[s]").tolist()

# Convert numpy datetime64 to Python datetime objects
dates_list = []
for datetime_obj in datetimes:
    if hasattr(datetime_obj, "astype"):
        timestamp = datetime_obj.astype("datetime64[s]").astype("int")
        date = datetime.datetime.fromtimestamp(timestamp)
    else:
        date = datetime_obj
    dates_list.append(date)

# Prepare batched data
# Convert stack data from [B, C, H, W] to [B, H, W, C] for Clay model
batch_images = torch.from_numpy(stack.values).permute(0, 2, 3, 1)  # [B, H, W, C]

# Create list of bounds (same bounds for all images in this case)
bounds_list = [wgs84_bounds] * len(dates_list)

# Generate embeddings using batch processing
embeddings_tensor = clay_model.generate(
    image=batch_images,
    bounds=bounds_list,
    date=dates_list,
    gsd=gsd,
    only_cls_token=True,  # Get only the class token (global embedding)
)

# Convert to numpy for downstream processing
embeddings = embeddings_tensor.cpu().numpy()

print(f"Generated embeddings shape: {embeddings.shape}")

# Label the images we downloaded
# 0 = Cloud
# 1 = Forest
# 2 = Fire
labels = np.array([0, 1, 0, 1, 1, 1, 1, 2, 2, 2, 2, 2])

# Split into fit and test manually, ensuring we have all 3 classes in both sets
fit = [0, 1, 3, 4, 7, 8, 9]
test = [2, 5, 6, 10, 11]

# Train a Support Vector Machine model
clf = svm.SVC()
clf.fit(embeddings[fit] + 100, labels[fit])

# Predict classes on test set
prediction = clf.predict(embeddings[test] + 100)

# Perfect match for SVM
match = np.sum(labels[test] == prediction)
print(f"Matched {match} out of {len(test)} correctly")
