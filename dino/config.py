from sacred import Experiment

ex = Experiment("DINO")

@ex.config
def config():
    exp_name = "config"
    seed = 0

    # Dataset setting
    get_recall_metric = False
    data_root = ""
    dataset_ratio = 1 # Between 0 and 1 the ratio of paired image-text
    imagenet_dir = "/datasets01/imagenet_full_size/061417/"
    datasets = ["imagenet"]
    train_transform = "multicrop"
    val_transform = "multicrop"
    image_size = 224
    max_image_len = -1
    draw_false_image = 1
    image_only = False

    # Text Setting
    text_dataset = None
    vqav2_label_size = 3129
    max_text_len = 40
    tokenizer = "bert-base-uncased"
    vocab_size = 30522
    whole_word_masking = True
    mlm_prob = 0.15
    draw_false_text = 0

    # Model
    patch_size = 16
    arch = "vit_small"
    vit = "dino_vit_small_patch16_224"
    hidden_size = 384
    num_layers = 12
    num_heads = 6
    mlp_ratio=4
    drop_rate=0.1

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
    from_scratch = False # If True then not using pretrained weight, but can still go through load_path
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

    # KNN eval
    nb_knn = 20
    knn_temp = 0.07

@ex.named_config
def task_dino_mlm():
    text_dataset = 'cc'
