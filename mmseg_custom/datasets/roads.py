# Copyright (c) OpenMMLab. All rights reserved.
from mmseg.datasets.builder import DATASETS
from mmseg.datasets.custom import CustomDataset

@DATASETS.register_module(force=True)
class RoadDataset(CustomDataset):
    """Road Dataset.

    In segmentation map annotation for Road dataset, 0 is the ignore index.
    ``reduce_zero_label`` should be set to True. The ``img_suffix`` and
    ``seg_map_suffix`` are both fixed to '.png'.
    """
    CLASSES = ('background', 'road')

    PALETTE = [[0, 0, 0], [255, 255, 255]]

    def __init__(self, **kwargs):
        super(RoadDataset, self).__init__(
            img_suffix='.png',
            seg_map_suffix='.png',
            reduce_zero_label=False,
            **kwargs)
