vintext_textdet_data_root = 'data/vintext'

vintext_textdet_train = dict(
    type='OCRDataset',
    data_root=vintext_textdet_data_root,
    ann_file='instances_training.json',
    data_prefix=dict(img_path='imgs/training/'),
    filter_cfg=dict(filter_empty_gt=True, min_size=32),
    pipeline=None)

vintext_textdet_test = dict(
    type='OCRDataset',
    data_root=vintext_textdet_data_root,
    ann_file='instances_test.json',
    data_prefix=dict(img_path='imgs/test/'),
    test_mode=True,
    pipeline=None)
