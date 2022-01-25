"""
@Fire
https://github.com/fire717
"""
data_dir = '/workspace/data/mpii/'
project_dir = '/workspace/projects/movenet/'

cfg = {
    ##### Global Setting
    "cuda": True,
    "num_workers": 0,
    "random_seed": 42,
    "cfg_verbose": True,

    "save_dir": project_dir + "output/",
    "num_classes": 13,
    "width_mult": 1.0,
    "img_size": 192,

    ##### Train Setting
    'pre-separated_data': True,
    'training_data_split': 80,
    'img_path': data_dir + "tos_synthetic_export/",
    # 'train_label_path': data_dir + 'train.json',
    # 'val_label_path': data_dir + 'val.json',
    'train_label_path': data_dir + 'mpii_hvd_test.json',
    'val_label_path': data_dir + 'mpii_hvd_test.json',

    'balance_data': False,

    'log_interval': 10,
    'save_best_only': True,

    'pin_memory': True,
    'newest_ckpt': project_dir + 'output/newest.json',

    ##### Horovod Hyperparameters
    'use_adasum': False,
    'gradient_predivide_factor': 1.0,  # gradient predivide factor to be applied in optimizer

    ##### Train Hyperparameters
    'learning_rate': 0.001,  # 1.25e-4
    # 'batch_size': 32,
    'batch_size': 16,
    'batches_per_allreduce': 1,  # number of batches processed locally before executing allreduce across workers; it multiplies total batch size
    # 'epochs': 150,
    'epochs': 5,
    'optimizer': 'Adam',  # Adam  SGD
    'scheduler': 'MultiStepLR-70,100-0.1',  # default  SGDR-5-2  CVPR   step-4-0.8 MultiStepLR
    # multistepLR-<<>milestones>-<<decay multiplier>>
    'weight_decay': 5.e-4,  # 0.0001,

    'class_weight': None,  # [1., 1., 1., 1., 1., 1., 1., ]
    'clip_gradient': 5,  # 1,

    ##### Test
    'test_img_path': data_dir + "cropped/imgs",
    'predict_output_path': project_dir + "predict/",

    # "../data/eval/imgs",
    # "../data/eval/imgs",
    # "../data/all/imgs"
    # "../data/true/mypc/crop_upper1"
    # ../data/coco/small_dataset/imgs
    # "../data/testimg"
    'exam_label_path': data_dir + '/all/data_all_new.json',

    'eval_img_path': data_dir + 'tos_synthetic_export/',
    'eval_label_path': data_dir + 'annotations/val.json',
}
