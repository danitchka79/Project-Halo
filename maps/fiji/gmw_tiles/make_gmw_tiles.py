"""Render Global Mangrove Watch extent data for Fiji into XYZ PNG map tiles.

Each source is a set of 1x1 degree uint8 grids at 1e-4 deg (~10 m) per pixel
(GMW v4.0.19 2020 GeoTIFFs as they come; the v4.1.12 annual vectors are first
rasterised onto that same grid). Tiles are sampled at MAX_Z by nearest
neighbour, then each lower zoom averages 2x2 children so small patches fade
rather than vanish.
"""
import math, os, sqlite3, struct, sys, glob
import numpy as np
from PIL import Image
import rasterio
from rasterio.features import rasterize
from rasterio.transform import from_origin

MAX_Z, MIN_Z = 14, 6
RES = 1e-4  # degrees per source pixel


def tile_rows_lat(z, y):
    n = 2 ** z * 256
    yp = y * 256 + np.arange(256) + 0.5
    return np.degrees(np.arctan(np.sinh(np.pi * (1 - 2 * yp / n))))


def tile_cols_lon(z, x):
    n = 2 ** z * 256
    return (x * 256 + np.arange(256) + 0.5) / n * 360 - 180


def lonlat_to_tile(lon, lat, z):
    n = 2 ** z
    x = int((lon + 180) / 360 * n)
    y = int((1 - math.asinh(math.tan(math.radians(lat))) / math.pi) / 2 * n)
    return min(max(x, 0), n - 1), min(max(y, 0), n - 1)


def tiles_from_degree_grids(grids):
    """grids: iterable of (left, top, uint8 array 10000x10000). Returns {(x,y): bool[256,256]} at MAX_Z."""
    out = {}
    for left, top, arr in grids:
        if not arr.any():
            continue
        x0, y0 = lonlat_to_tile(left, top, MAX_Z)
        x1, y1 = lonlat_to_tile(left + 1 - 1e-9, top - 1 + 1e-9, MAX_Z)
        for x in range(x0, x1 + 1):
            lons = tile_cols_lon(MAX_Z, x)
            ci = np.floor((lons - left) / RES).astype(int)
            cok = (ci >= 0) & (ci < arr.shape[1])
            if not cok.any():
                continue
            for y in range(y0, y1 + 1):
                lats = tile_rows_lat(MAX_Z, y)
                ri = np.floor((top - lats) / RES).astype(int)
                rok = (ri >= 0) & (ri < arr.shape[0])
                if not rok.any():
                    continue
                sub = np.zeros((256, 256), bool)
                sub[np.ix_(rok, cok)] = arr[np.ix_(ri[rok], ci[cok])] > 0
                if sub.any():
                    if (x, y) in out:
                        out[(x, y)] |= sub
                    else:
                        out[(x, y)] = sub
    return out


def write_pyramid(base, outdir, rgb, alpha_max):
    level = {k: v.astype(np.float32) for k, v in base.items()}
    count = 0
    for z in range(MAX_Z, MIN_Z - 1, -1):
        for (x, y), cov in level.items():
            a = np.clip(cov * (1 if z == MAX_Z else 2.5), 0, 1) * alpha_max
            img = np.zeros((256, 256, 4), np.uint8)
            img[..., :3] = rgb
            img[..., 3] = (a * 255).round().astype(np.uint8)
            d = os.path.join(outdir, str(z), str(x))
            os.makedirs(d, exist_ok=True)
            Image.fromarray(img, 'RGBA').save(os.path.join(d, f'{y}.png'), optimize=True)
            count += 1
        if z == MIN_Z:
            break
        parents = {}
        for (x, y), cov in level.items():
            half = cov.reshape(128, 2, 128, 2).mean(axis=(1, 3))
            p = parents.setdefault((x // 2, y // 2), np.zeros((256, 256), np.float32))
            oy, ox = (y % 2) * 128, (x % 2) * 128
            p[oy:oy + 128, ox:ox + 128] = half
        level = parents
    print(f'{outdir}: {count} tiles')


def geotiff_grids(pattern):
    for f in sorted(glob.glob(pattern)):
        with rasterio.open(f) as r:
            yield r.bounds.left, r.bounds.top, r.read(1)


# ---- GeoPackage polygons (no GDAL needed) ----
def _wkb_geom(b, o=0):
    bo = '<' if b[o] == 1 else '>'
    t = struct.unpack_from(bo + 'I', b, o + 1)[0]
    o += 5
    dims = 2
    if t > 0x20000000:  # EWKB flags
        if t & 0x80000000: dims += 1
        if t & 0x40000000: dims += 1
        if t & 0x20000000: o += 4
        t &= 0xFFFF
    base = t % 1000
    dims += {1: 1, 2: 1, 3: 2}.get(t // 1000, 0)

    def ring(o):
        n = struct.unpack_from(bo + 'I', b, o)[0]; o += 4
        pts = np.frombuffer(b, dtype=bo + 'f8', count=n * dims, offset=o).reshape(n, dims)[:, :2]
        return pts, o + n * dims * 8

    if base == 3:
        nr = struct.unpack_from(bo + 'I', b, o)[0]; o += 4
        rings = []
        for _ in range(nr):
            r, o = ring(o); rings.append(r)
        return [rings], o
    if base == 6:
        npoly = struct.unpack_from(bo + 'I', b, o)[0]; o += 4
        polys = []
        for _ in range(npoly):
            p, o = _wkb_geom(b, o); polys += p
        return polys, o
    raise ValueError(f'unsupported WKB type {t}')


def gpkg_polygons(path, bbox):
    con = sqlite3.connect(path)
    table, gcol = con.execute("select table_name, column_name from gpkg_geometry_columns").fetchone()
    minx, miny, maxx, maxy = bbox
    q = (f'select t."{gcol}" from "{table}" t join "rtree_{table}_{gcol}" r on t.fid = r.id '
         f'where r.maxx >= ? and r.minx <= ? and r.maxy >= ? and r.miny <= ?')
    for (blob,) in con.execute(q, (minx, maxx, miny, maxy)):
        flags = blob[3]
        env = {0: 0, 1: 32, 2: 48, 3: 48, 4: 64}[(flags >> 1) & 7]
        polys, _ = _wkb_geom(blob, 8 + env)
        yield from polys


def vector_grids(path, degree_tiles):
    """Rasterise the polygons onto each 1x1 degree grid (same grid as the GeoTIFFs)."""
    polys = []
    for b in [(176, -21, 180, -15), (-180, -21, -178, -15)]:
        polys += list(gpkg_polygons(path, b))
    print(f'{path}: {len(polys)} polygons')
    for left, top in degree_tiles:
        tf = from_origin(left, top, RES, RES)
        shapes = []
        for rings in polys:
            x0, y0 = rings[0].min(axis=0); x1, y1 = rings[0].max(axis=0)
            if x1 < left or x0 > left + 1 or y1 < top - 1 or y0 > top:
                continue
            shapes.append(({'type': 'Polygon', 'coordinates': [r.tolist() for r in rings]}, 1))
        if not shapes:
            continue
        yield left, top, rasterize(shapes, out_shape=(10000, 10000), transform=tf, dtype='uint8')


if __name__ == '__main__':
    which, out = sys.argv[1], sys.argv[2]
    if which == 'hires':
        base = tiles_from_degree_grids(geotiff_grids('v4019/*.tif'))
        write_pyramid(base, out, (0x00, 0x5f, 0x73), 0.85)   # GMW dark teal
    else:
        degree_tiles = [(lon, lat) for lat in range(-15, -21, -1)
                        for lon in list(range(176, 180)) + [-180, -179]]
        base = tiles_from_degree_grids(vector_grids(which, degree_tiles))
        write_pyramid(base, out, (0x06, 0xc4, 0xbd), 0.6)   # GMW habitat-extent teal
