from sacred import Experiment

ex = Experiment("DINO")

@ex.config
def config():
    exp_name = "config"
    loss_names = {'dino': 1}
    seed = 0

    # Dataset setting
    imagenet_dir = "/datasets01/imagenet_full_size/061417/"
    image_size = 224
    # Model
    patch_size = 16
    arch = "vit_small"

    # Optimizer Setting
    optim_type = "adamw"
    learning_rate = 0.0005
    weight_decay = 0.04
    weight_decay_end = 0.4
    decay_power = 1
    max_epoch = 100
    max_steps = None
    warmup_steps = None
    warmup_epoch = 10
    end_lr = 1e-6

    # PL Trainer Setting
    resume_from = None
    fast_dev_run = False
    val_check_interval = 1.0
    test_only = False

    log_dir = "result"
    per_gpu_batchsize = 64  # you should define this manually with per_gpu_batch_size=#
    num_gpus = 1
    num_nodes = 1
    load_path = ""
    num_workers = 8
    precision = 16

    # DINO setting
    nmb_centroids = 65536
    norm_last_layer = True
    use_subword_in_last_layer = False
    use_bn_in_head = False
    dino_img_key = 'imagenet_image'
    dino_label_key = 'imagenet_label'

    # teacher temp schedule
    warmup_teacher_temp = 0.04
    teacher_temp = 0.04
    warmup_teacher_temp_epoch = 0
    
    freeze_last_layer = 1
    global_crops_scale = [0.4, 1.0]
    local_crops_scale = [0.05, 0.4]
    local_crops_number = 8
    momentum_teacher = 0.996
