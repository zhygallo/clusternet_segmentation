"""Script for generating rasterized masks from shapefiles"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import rasterio
from rasterio import features
import geopandas as gpd
import click
import os


def get_rasterization(input_shp, meta_data={}, ref_tif='', output_tif=''):
    """
    Get rasterized image from shapefile.
    Args:
        input_shp:  string.     path to the shapefile (.shp).
        meta_data:  dictionary. meta_data passed to rasterio.open() function
                                in case reference image is not specified.
        ref_tif:    string.     path to the reference .tif image.
                                When specified meta_data is ignored.
        output_tif: string.     path to the output rasterized image.

    Returns:
        numpy array. rasterized image.
    """

    # Checking the correctness of the arguments
    assert os.path.isfile(input_shp), 'no file: input_shp'
    assert type(meta_data) is dict, 'meta_data has to be dictionary'
    if ref_tif != '':
        assert os.path.isfile(ref_tif), 'no file: ref_tif'

    shape_df = gpd.read_file(input_shp)

    if ref_tif != '':
        rst = rasterio.open(ref_tif)
        meta_data = rst.meta.copy()
        meta_data['count'] = 1

    with rasterio.open(output_tif, 'w', **meta_data) as out:
        shapes = ((geom, 1) for geom in shape_df.geometry)
        # create a generator of geom, value pairs to use in rasterizing
        result = features.rasterize(shapes=shapes, out_shape=out.shape, fill=0, transform=out.transform)
        if output_tif:
            out.write_band(1, result)

    return result


@click.command()
@click.argument('input_shp', type=click.STRING)
@click.option('--ref_tif', type=click.STRING, default='', help='path to the reference .tif image')
@click.option('--output_tif', type=click.STRING, default='', help='path to the output rasterized image.')
def main(input_shp, ref_tif, output_tif):
    meta_data = {
        'driver': 'GTiff',
        'dtype': 'uint8',
        'nodata': None,
        'width': 13500,
        'height': 15300,
        'count': 1,
        'crs': {'init': 'epsg:26911'},
        'transform': (0.5, 0.0, 481649.0, 0.0, -0.5, 3623101.0)}

    files = os.listdir(input_shp)
    files = [file for file in files if file.endswith('.shp')]
    for file in files:
        input_shp_path = os.path.join(input_shp, file)
        ref_tif_path = os.path.join(ref_tif, file.split('.')[0]+'.tif')
        output_tif_paht = os.path.join(output_tif, file.split('.')[0]+'.tif')
        get_rasterization(input_shp=input_shp_path, meta_data=meta_data, ref_tif=ref_tif_path, output_tif=output_tif_paht)


if __name__ == '__main__':
    main()
