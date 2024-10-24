# model settings
norm_cfg = dict(type='SyncBN', requires_grad=True)
data_preprocessor = dict(
    type='DualInputSegDataPreProcessor',
    mean=[123.675, 116.28, 103.53] * 2,
    std=[58.395, 57.12, 57.375] * 2,
    bgr_to_rgb=True,
    size_divisor=32,
    pad_val=0,
    seg_pad_val=255,
    test_cfg=dict(size_divisor=32))
model = dict(
    type='APDEncoderDecoder',
    pretrained=None,
    backbone=dict(
        type='SiaResAPD_18',
        depth=18,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        dilations=(1, 1, 1, 1),
        strides=(1, 2, 2, 2),
        norm_cfg=norm_cfg,
        norm_eval=False,
        style='pytorch',
        contract_dilation=True),
    decode_head=dict(
        type='Seg_head',
        in_channels=[256, 256, 256, 64, 2, 2],
        in_index=[0, 1, 2, 3, 4, 5],
        channels=256,
        dropout_ratio=0.1,
        num_classes=2,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=[dict(
            type='mmseg.CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
            dict(type='mmseg.CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
            dict(type='ContrastiveLoss', use_sigmoid=False, loss_weight=1.0)]),
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))