input_size = 512
num_things_classes = 0
num_stuff_classes = 2
num_classes = 2
norm_cfg = dict(type='SyncBN', requires_grad=True)
pretrained = 'pretrained/beit_large_patch16_224_pt22k_ft22k_4channels.pth'
model = dict(
    type='EncoderDecoder',
    pretrained='pretrained/beit_large_patch16_224_pt22k_ft22k_4channels.pth',
    backbone=dict(
        type='BEiTAdapter',
        img_size=512,
        in_chans=4,
        patch_size=16,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4,
        qkv_bias=True,
        use_abs_pos_emb=False,
        use_rel_pos_bias=True,
        init_values=1e-06,
        drop_path_rate=0.3,
        conv_inplane=64,
        n_points=4,
        deform_num_heads=16,
        cffn_ratio=0.25,
        deform_ratio=0.5,
        with_cp=True,
        interaction_indexes=[[0, 5], [6, 11], [12, 17], [18, 23]]),
    decode_head=dict(
        type='UPerHead',
        in_channels=[1024, 1024, 1024, 1024],
        in_index=[0, 1, 2, 3],
        pool_scales=(1, 2, 3, 6),
        channels=1024,
        dropout_ratio=0.1,
        num_classes=2,
        norm_cfg=dict(type='SyncBN', requires_grad=True),
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
    auxiliary_head=dict(
        type='FCNHead',
        in_channels=1024,
        in_index=2,
        channels=256,
        num_convs=1,
        concat_input=False,
        dropout_ratio=0.1,
        num_classes=2,
        norm_cfg=dict(type='SyncBN', requires_grad=True),
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.4)),
    train_cfg=dict(),
    test_cfg=dict(mode='slide', crop_size=(640, 640), stride=(426, 426)))
crop_size = (512, 512)
img_scale = (512, 512)
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53, 103.53],
    std=[58.395, 57.12, 57.375, 57.375],
    to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile', color_type='unchanged'),
    dict(type='LoadAnnotations', reduce_zero_label=False),
    dict(type='Resize', img_scale=(512, 512)),
    dict(type='RandomCrop', crop_size=(512, 512), cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    dict(
        type='Normalize',
        mean=[123.675, 116.28, 103.53, 103.53],
        std=[58.395, 57.12, 57.375, 57.375],
        to_rgb=True),
    dict(type='Pad', size=(512, 512), pad_val=0, seg_pad_val=255),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg'])
]
test_pipeline = [
    dict(type='LoadImageFromFile', color_type='unchanged'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(512, 512),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='ResizeToMultiple', size_divisor=32),
            dict(type='RandomFlip'),
            dict(
                type='Normalize',
                mean=[123.675, 116.28, 103.53, 103.53],
                std=[58.395, 57.12, 57.375, 57.375],
                to_rgb=True),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img'])
        ])
]
dataset_type = 'BuildingsDataset'
data_root = '/home/yh.sakong/data/sn6_building/preprocessed/optical'
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=1,
    train=dict(
        type='RepeatDataset',
        times=40000,
        dataset=dict(
            type='BuildingsDataset',
            data_root=data_root,
            img_dir='train/images_png',
            ann_dir='train/labels_png',
            pipeline=[
                dict(type='LoadImageFromFile', color_type='unchanged'),
                dict(type='LoadAnnotations', reduce_zero_label=False),
                dict(type='Resize', img_scale=(512, 512)),
                dict(
                    type='RandomCrop',
                    crop_size=(512, 512),
                    cat_max_ratio=0.75),
                dict(type='RandomFlip', prob=0.5),
                dict(
                    type='Normalize',
                    mean=[123.675, 116.28, 103.53, 103.53],
                    std=[58.395, 57.12, 57.375, 57.375],
                    to_rgb=True),
                dict(type='Pad', size=(512, 512), pad_val=0, seg_pad_val=255),
                dict(type='DefaultFormatBundle'),
                dict(type='Collect', keys=['img', 'gt_semantic_seg'])
            ])),
    val=dict(
        type='BuildingsDataset',
        data_root=data_root,
        img_dir='val/images_png',
        ann_dir='val/labels_png',
        pipeline=[
            dict(type='LoadImageFromFile', color_type='unchanged'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=(512, 512),
                flip=False,
                transforms=[
                    dict(type='Resize', keep_ratio=True),
                    dict(type='ResizeToMultiple', size_divisor=32),
                    dict(type='RandomFlip'),
                    dict(
                        type='Normalize',
                        mean=[123.675, 116.28, 103.53, 103.53],
                        std=[58.395, 57.12, 57.375, 57.375],
                        to_rgb=True),
                    dict(type='ImageToTensor', keys=['img']),
                    dict(type='Collect', keys=['img'])
                ])
        ]),
    test=dict(
        type='BuildingsDataset',
        data_root=data_root,
        img_dir='val/images_png',
        ann_dir='val/labels_png',
        pipeline=[
            dict(type='LoadImageFromFile', color_type='unchanged'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=(512, 512),
                flip=False,
                transforms=[
                    dict(type='Resize', keep_ratio=True),
                    dict(type='ResizeToMultiple', size_divisor=32),
                    dict(type='RandomFlip'),
                    dict(
                        type='Normalize',
                        mean=[123.675, 116.28, 103.53, 103.53],
                        std=[58.395, 57.12, 57.375, 57.375],
                        to_rgb=True),
                    dict(type='ImageToTensor', keys=['img']),
                    dict(type='Collect', keys=['img'])
                ])
        ]))
log_config = dict(
    interval=200, hooks=[dict(type='TextLoggerHook', by_epoch=False)])
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
auto_resume = False
workflow = [('train', 1)]
cudnn_benchmark = True
optimizer = dict(
    type='AdamW',
    lr=2e-05,
    betas=(0.9, 0.999),
    weight_decay=0.05,
    constructor='LayerDecayOptimizerConstructor',
    paramwise_cfg=dict(num_layers=24, layer_decay_rate=0.9))
optimizer_config = dict()
lr_config = dict(
    policy='poly',
    warmup='linear',
    warmup_iters=1500,
    warmup_ratio=1e-06,
    power=1.0,
    min_lr=0.0,
    by_epoch=False)
runner = dict(type='IterBasedRunner', max_iters=80000)
checkpoint_config = dict(by_epoch=False, interval=2000, max_keep_ckpts=1)
evaluation = dict(
    interval=2000, metric=['mIoU', 'mDice', 'mFscore'], save_best='mDice')
work_dir = 'work_dirs/sn6/upernet_sn6_optical_4channels'
gpu_ids = range(0, 2)
