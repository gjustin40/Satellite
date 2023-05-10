# Copyright (c) OpenMMLab. All rights reserved.
from mmseg.datasets.builder import DATASETS
from mmseg.datasets.custom import CustomDataset

@DATASETS.register_module(force=True)
class SpaceNet6SARDataset(CustomDataset):
    """SpaceNet6 SAR Dataset.

    In segmentation map annotation for SpaceNet6 SAR Dataset, 0 is the ignore index.
    SpaceNet6 SAR data has 4 channels (HH, HV, VH, VV)
    Also dtype is float32
    ``reduce_zero_label`` should be set to True. The ``img_suffix`` and
    ``seg_map_suffix`` are both fixed to '.png'.
    """
    CLASSES = ('background', 'building')

    PALETTE = [[0, 0, 0], [255, 255, 255]]

    def __init__(self, **kwargs):
        super(SpaceNet6SARDataset, self).__init__(
            img_suffix='.tif',
            seg_map_suffix='.png',
            reduce_zero_label=False,
            **kwargs)
