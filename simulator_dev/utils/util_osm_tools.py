import pyrosm

# --- scripts ---
def util_osm_get_buildings(filepath):
    osm = pyrosm.OSM(filepath)
    map_buildings = osm.get_buildings()
    return map_buildings
