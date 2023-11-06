import os

from torch.utils.tensorboard import SummaryWriter


def get_run_directory(base_dir):
    i = 1
    while os.path.exists(f"{base_dir}_{i}"):
        i += 1
    return f"{base_dir}_{i}"


def get_writer(alg_name):
    run_directory = get_run_directory(f"walker2d_tensorboard/{alg_name}")
    writer = SummaryWriter(run_directory, max_queue=1000000, flush_secs=30)
    return writer
