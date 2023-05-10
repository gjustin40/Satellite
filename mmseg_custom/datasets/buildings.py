# Copyright (c) OpenMMLab. All rights reserved.
from mmseg.datasets.builder import DATASETS
from mmseg.datasets.custom import CustomDataset

@DATASETS.register_module(force=True)
class BuildingsDataset(CustomDataset):
    """Buidlings dataset.

    In segmentation map annotation for Potsdam dataset, 0 is the ignore index.
    ``reduce_zero_label`` should be set to True. The ``img_suffix`` and
    ``seg_map_suffix`` are both fixed to '.png'.
    """
    CLASSES = ('background', 'building', 'etc')

    PALETTE = [[0, 0, 0], [255, 255, 255], [128,128,128]]

    def __init__(self, **kwargs):
        super(BuildingsDataset, self).__init__(
            img_suffix='.png',
            seg_map_suffix='.png',
            reduce_zero_label=False,
            **kwargs)
