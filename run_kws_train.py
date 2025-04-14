import os
import argparse
import logging
import time
import yaml
import torch

from src.kws_trainer import KWSTrainer
from src.utils.utils import dict_to_object, print_dicts
from src.utils.logger import get_logger


def get_args():
    parser = argparse.ArgumentParser(description='training your network')
    parser.add_argument('--config', type=str, default='configs/tcn.yml', help='config file')
    parser.add_argument('--gpus',
                        default='-1',
                        help='gpu lists, seperated with `,`, -1 for cpu')
    parser.add_argument('--model_dir', default="models", help='save model dir')
    parser.add_argument('--checkpoint', default=None, help='checkpoint model dir')
    parser.add_argument('--pretrained', default=None, help='pretrained model dir')
    parser.add_argument('--log_filename', default='train.log', help='output log file name')
    parser.add_argument('--tensorboard_dir',
                        default='tensorboard',
                        help='tensorboard log dir')
    parser.add_argument('--ddp.dist_backend',
                        dest='dist_backend',
                        default='nccl',
                        choices=['nccl', 'gloo'],
                        help='distributed backend')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    # 获取参数
    args = get_args()

    # 读取配置文件
    with open(args.config, 'r', encoding='utf-8') as f:
        yaml_config = yaml.load(f, Loader=yaml.FullLoader)
        configs = dict_to_object(yaml_config)

    log_file = f'{args.model_dir}/{configs.use_model}/{args.log_filename}' if args.log_filename else ''
    logger = get_logger(__name__, log_file)

    local_rank = 0
    if args.gpus != '-1':
        assert (torch.cuda.is_available())  # 检查是否支持CUDA

        nranks = torch.cuda.device_count()  # 获取有多少张显卡训练
        local_rank = int(os.environ['LOCAL_RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        gpu_id = args.gpus.split(',')[local_rank]
        logger.info(f"Training on GPU: total {nranks}, world_size {world_size}, gpu {gpu_id} local_rank {local_rank}")

        torch.cuda.set_device(int(gpu_id))
        if world_size > 1:
            torch.distributed.init_process_group(backend=args.dist_backend)
        is_distributed = True if len(args.gpus.split(',')) > 1 else False
    else:
        logger.info("Training on CPU.")
        gpu_id = args.gpus
        is_distributed = False

    # 只在 rank0 上打印log
    if local_rank == 0:
        # Print arguments into logs
        logger.info("---------- Print arguments ----------")
        print_dicts(args)
        # Print config parameters into logs
        logger.info(f"----------{args.config}----------")
        print_dicts(configs, ' ')

    # 获取训练器
    trainer = KWSTrainer(configs, gpu_id, log_file, is_distributed)

    # 训练
    trainer.train(args.model_dir, args.checkpoint, args.pretrained)
