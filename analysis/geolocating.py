"""Script to assign Japanese firms to their respective prefectures."""

import json

import geopandas as gpd
from shapely.geometry import Point

from graphsim.networks.marklines import load_marklines
from graphsim.zhao.improved_node import ImprovedZhaoNode
from graphsim.zhao.search import ImprovedSearch
from graphsim import paths

if __name__ == "__main__":
    DATA_DIR = paths.DATA_DIR
    RESULTS_DIR = paths.GEO_DIR
    GEOJSON_URL = (
        "https://raw.githubusercontent.com/dataofjapan/land/master/japan.geojson"
    )

    # Load Marklines network and data
    net, data, id_firm_map = load_marklines(
        DATA_DIR,
        node_cls=ImprovedZhaoNode,
        search_cls=ImprovedSearch,
    )

    # Filter Japanese firms
    japanese_firms = [firm for firm in id_firm_map.values() if firm.nation_id == 1]

    # Load geospatial data for the prefectures
    prefectures_geo = gpd.read_file(GEOJSON_URL)

    # Rename the column for the English names of the prefectures for easier access.
    prefectures_geo = prefectures_geo.rename(
        columns={"nam": "prefecture_name", "nam_ja": "prefecture_name_ja"},
    )

    # Create a dictionary to store the results.
    results = {name: [] for name in prefectures_geo["prefecture_name"]}
    unassigned_objects = []

    for firm in japanese_firms:
        # Create a geometric point from the firm's coordinates.
        # Important: Shapely and GeoPandas expect (longitude, latitude).
        point = Point(firm.longitude, firm.latitude)

        is_assigned = False
        # Check if the point is within each prefecture.
        for index, prefecture in prefectures_geo.iterrows():
            # The 'contains' method checks if the prefecture's geometry contains the point.
            if prefecture.geometry.contains(point):
                prefecture_name = prefecture["prefecture_name"]
                results[prefecture_name].append(firm.id)
                is_assigned = True
                break

        if not is_assigned:
            unassigned_objects.append(firm.id)
            print(
                f"-> Firm '{firm.id}' at ({firm.longitude}, {firm.latitude}) could not be assigned to any prefecture."
            )

    # Print the number of firms per prefecture
    sorted_results = sorted(results.items(), key=lambda x: len(x[1]), reverse=True)
    for prefecture_name, objects_in_prefecture in sorted_results:
        print(
            f"Prefecture: {prefecture_name}, Number of Firms: {len(objects_in_prefecture)}"
        )

    # Output the unassigned objects
    if unassigned_objects:
        print("\nUnassigned Objects:")
        for obj in unassigned_objects:
            print(f"  - {obj}")

    # Save the results to a file
    with open(RESULTS_DIR / "japanese_firms_geolocation.json", "w") as f:
        json.dump(results, f)
