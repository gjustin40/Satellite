# Copyright (c) OpenMMLab. All rights reserved.
from mmseg.datasets.builder import DATASETS
from mmseg.datasets.custom import CustomDataset

@DATASETS.register_module(force=True)
class LandcoveraiDataset(CustomDataset):
    """LandCoverAI dataset.
    The ``img_suffix`` and ``seg_map_suffix`` are both fixed to '.png'.
    """

    CLASSES = ('Background', 'Building', 'Woodland', 'Water', 'Road')

    PALETTE = [[255, 255, 255], [255, 0, 0], [0, 255, 0], [0, 0, 255], [255,255,0]]

    def __init__(self, **kwargs):
        super(LandcoveraiDataset, self).__init__(
            img_suffix='.png',
            seg_map_suffix='.png',
            reduce_zero_label=False,
            **kwargs)