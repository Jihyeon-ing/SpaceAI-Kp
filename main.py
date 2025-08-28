from gpu_config import configure_gpu
configure_gpu('0')

import os
import argparse
from exp import Exp
import warnings
warnings.filterwarnings('ignore')

def create_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--res_dir', default='./results', type=str)
    parser.add_argument('--cp_path', default='checkpoints', type=str)
    parser.add_argument('--ex_name', default='ver_250826')
    parser.add_argument('--model_name', default='tsmixer', type=str)

    parser.add_argument('--use_gpu', default=True, type=bool)
    parser.add_argument('--gpu', default='0', type=str)
    parser.add_argument('--seed', default=777, type=int)
    parser.add_argument('--num_workers', default=40, type=int)

    #data parameters
    parser.add_argument('--batch_size', default=1, type=int, help='batch size')
    parser.add_argument('--val_batch_size', default=1, type=int, help='batch size of validation/test set')
    parser.add_argument('--input_len', default=24, type=int, help='input length')
    parser.add_argument('--tar_len', default=12, type=int, help='target length')
    parser.add_argument('--n_features', default=6, type=int, help='number of input features')

    # training parameters
    parser.add_argument('--alpha', default=0.8, type=int)
    parser.add_argument('--epochs', default=200, type=int)
    parser.add_argument('--lr', default=0.001, type=float)
    parser.add_argument('--flag', default='train', type=str)

    # model parameters
    parser.add_argument('--hidden_dim', default=64, type=int, help='hidden channel size')
    parser.add_argument('--n_blocks', default=2, type=int, help='number of tsmixer block')
    parser.add_argument('--dropout', default=0., type=int, help='number of tsmixer block')

    # test parameters
    parser.add_argument('--test_epoch', default=176, type=int)
    parser.add_argument('--save_result', default=False, type=bool)
    return parser


if __name__ == '__main__':
    args = create_parser().parse_args()
    config = args.__dict__


    if args.flag == 'train':
        Exp(args).train()

    else:
        Exp(args).test()
