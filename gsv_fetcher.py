"""
gsv_fetcher.py - Google Street View Static API logic
Handles grid generation and image fetching for a given bounding box.
"""

import math
import hashlib
import io
from typing import Optional

import requests
from PIL import Image

# ── Constants ─────────────────────────────────────────────────────────────────
GSV_BASE_URL = "https://maps.googleapis.com/maps/api/streetview"
GSV_METADATA_URL = "https://maps.googleapis.com/maps/api/streetview/metadata"

# A 20x20-pixel "grey" image is returned by GSV when no imagery exists.
# We detect this by checking image entropy; very low entropy = grey placeholder.
GREY_IMAGE_ENTROPY_THRESHOLD = 0.5

# Earth's mean radius in metres (used for lat/lng <-> metre conversion)
EARTH_RADIUS_M = 6_371_000


def metres_to_lat_degrees(metres: float) -> float:
    """Convert a distance in metres to degrees of latitude."""
    return metres / EARTH_RADIUS_M * (180.0 / math.pi)


def metres_to_lng_degrees(metres: float, lat: float) -> float:
    """Convert a distance in metres to degrees of longitude at a given latitude."""
    return metres / (EARTH_RADIUS_M * math.cos(math.radians(lat))) * (180.0 / math.pi)


def generate_grid_points(
    bounds: dict, spacing_m: int = 150
) -> list[tuple[float, float]]:
    """
    Generate a regular grid of (lat, lng) sampling points within a bounding box.

    Args:
        bounds: dict with keys 'north', 'south', 'east', 'west' (float degrees)
        spacing_m: distance between adjacent grid points in metres

    Returns:
        List of (lat, lng) tuples covering the bounding box.
    """
    north = bounds["north"]
    south = bounds["south"]
    east = bounds["east"]
    west = bounds["west"]

    # Convert spacing from metres to degrees at the centre latitude
    centre_lat = (north + south) / 2.0
    lat_step = metres_to_lat_degrees(spacing_m)
    lng_step = metres_to_lng_degrees(spacing_m, centre_lat)

    points = []
    lat = south
    while lat <= north:
        lng = west
        while lng <= east:
            points.append((round(lat, 7), round(lng, 7)))
            lng += lng_step
        lat += lat_step

    return points


def make_filename(lat: float, lng: float, heading: Optional[int] = None) -> str:
    """Create a deterministic filename from lat/lng coordinates and heading."""
    heading_token = "auto" if heading is None else str(int(heading))
    key = f"{lat:.6f}_{lng:.6f}_{heading_token}"
    short_hash = hashlib.md5(key.encode()).hexdigest()[:8]
    return f"gsv_{lat:.5f}_{lng:.5f}_{heading_token}_{short_hash}.jpg"


def fetch_gsv_image(
    lat: float,
    lng: float,
    api_key: str,
    size: str = "640x640",
    heading: Optional[int] = None,
    pitch: int = 0,
    fov: int = 90,
    source: str = "outdoor",
) -> tuple[Optional[bytes], str, str]:
    """
    Fetch a single Street View image for the given coordinates.

    Args:
        lat, lng: Coordinates of the desired location.
        api_key: Google Street View Static API key.
        size: Image dimensions string (e.g. '640x640').
        heading: Camera heading in degrees (0-360). None lets GSV choose.
        pitch: Camera pitch (-90 to 90). 0 = horizontal.
        fov: Field of view (10-120 degrees).
        source: Street View imagery source restriction. Defaults to outdoor.

    Returns:
        (image_bytes, filename, status) tuple. image_bytes is None on failure.
    """
    filename = make_filename(lat, lng, heading=heading)

    params: dict = {
        "size": size,
        "location": f"{lat},{lng}",
        "pitch": pitch,
        "fov": fov,
        "key": api_key,
        "source": source,
        "return_error_code": "true",   # Return HTTP 404 instead of grey placeholder
    }
    if heading is not None:
        params["heading"] = heading

    try:
        response = requests.get(GSV_BASE_URL, params=params, timeout=15)

        if response.status_code == 200:
            return response.content, filename, "ok"
        elif response.status_code == 404:
            # GSV explicitly says no imagery available here
            return None, filename, "no_street_view"
        else:
            # Other HTTP errors (auth issues, quota, etc.)
            return None, filename, f"http_{response.status_code}"

    except requests.RequestException:
        return None, filename, "request_exception"


def check_image_valid(image_bytes: bytes) -> bool:
    """
    Return True if the image looks like real Street View imagery.

    GSV sometimes returns a low-information grey image when imagery is scarce.
    We detect this by computing image entropy: real photos have significantly
    higher entropy than near-uniform placeholder images.

    Args:
        image_bytes: Raw JPEG bytes from the API.

    Returns:
        True if the image appears to contain real imagery.
    """
    try:
        img = Image.open(io.BytesIO(image_bytes)).convert("L")  # greyscale
        # Compute histogram-based entropy
        histogram = img.histogram()
        total_pixels = sum(histogram)
        if total_pixels == 0:
            return False
        entropy = -sum(
            (count / total_pixels) * math.log2(count / total_pixels)
            for count in histogram
            if count > 0
        )
        return entropy > GREY_IMAGE_ENTROPY_THRESHOLD
    except Exception:
        return False


def fetch_gsv_metadata(lat: float, lng: float, api_key: str) -> dict:
    """
    Query the GSV Metadata API to check whether imagery exists at a location.
    This does NOT consume billable quota for the image itself.

    Returns a dict with 'status' key: 'OK', 'ZERO_RESULTS', or 'REQUEST_DENIED'.
    """
    params = {
        "location": f"{lat},{lng}",
        "key": api_key,
    }
    try:
        resp = requests.get(GSV_METADATA_URL, params=params, timeout=10)
        return resp.json()
    except Exception:
        return {"status": "ERROR"}
