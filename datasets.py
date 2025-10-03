from torchgeo.datasets import RasterDataset

# Aplicación de clases torchgeo
class PNOAImages(RasterDataset): # This can be a single large image raster
    filename_glob = "*.tif"

class SidewalkLabels(RasterDataset): # But the labels must be split into train, val and test directories
    filename_glob = "*.tif"
    is_image = False