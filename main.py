import torch
import argparse
from UGATIT import UGATIT
from utils import *
from mel_utils import print_and_summarize

"""parsing and configuration"""

def parse_args():
    desc = "Pytorch implementation of U-GAT-IT"
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('--phase', type=str, default='train', help='[train / test]')
    parser.add_argument('--light', type=str2bool, default=False, help='[U-GAT-IT full version / U-GAT-IT light version]')
    parser.add_argument('--dataset', type=str, default='YOUR_DATASET_NAME', help='dataset_name')

    parser.add_argument('--iteration', type=int, default=1000000, help='The number of training iterations')
    parser.add_argument('--batch_size', type=int, default=1, help='The size of batch size')
    parser.add_argument('--print_freq', type=int, default=1000, help='The number of image print freq')
    parser.add_argument('--save_freq', type=int, default=100000, help='The number of model save freq')
    parser.add_argument('--decay_flag', type=str2bool, default=True, help='The decay_flag')

    parser.add_argument('--lr', type=float, default=0.0001, help='The learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.0001, help='The weight decay')
    parser.add_argument('--adv_weight', type=int, default=1, help='Weight for GAN')
    parser.add_argument('--cycle_weight', type=int, default=10, help='Weight for Cycle')
    parser.add_argument('--identity_weight', type=int, default=10, help='Weight for Identity')
    parser.add_argument('--cam_weight', type=int, default=1000, help='Weight for CAM')

    parser.add_argument('--ch', type=int, default=64, help='base channel number per layer')
    parser.add_argument('--n_res', type=int, default=4, help='The number of resblock')
    parser.add_argument('--n_dis', type=int, default=6, help='The number of discriminator layer')

    parser.add_argument('--img_size', type=int, default=256, help='The size of image')
    parser.add_argument('--img_ch', type=int, default=3, help='The size of image channel')

    parser.add_argument('--result_dir', type=str, default='results', help='Directory name to save the results')
    parser.add_argument('--device', type=str, default='cuda', choices=['cpu', 'cuda'], help='Set gpu mode; [cpu, cuda]')
    parser.add_argument('--benchmark_flag', type=str2bool, default=False)
    parser.add_argument('--resume', type=str2bool, default=False)

    parser.add_argument('--mel_spectrogram', type=str2bool, default=False, help='Input mel spectrograms instead of images')
    parser.add_argument('--gen_lr', type=float, required=False, help='Learning rate for generator')
    parser.add_argument('--dis_lr', type=float, required=False, help='Learning rate for discriminator')

    parser.add_argument('--use_noise', type=str2bool, default=False, help='Use noise data in DATASET/noise')
    parser.add_argument('--gen_noise_A', type=float, default=0., help='Proportion of noise to add to A samples before generation')
    parser.add_argument('--gen_noise_B', type=float, default=0., help='Proportion of noise to add to B samples before generation')
    parser.add_argument('--dis_noise_A', type=float, default=0., help='Proportion of noise to add to A2B samples after generation, before discriminator')
    parser.add_argument('--dis_noise_B', type=float, default=0., help='Proportion of noise to add to B2A samples after generation, before discriminator')
    parser.add_argument('--identity_noise_A', type=float, default=0., help='Proportion of noise to add to identity loss A2A')
    parser.add_argument('--identity_noise_B', type=float, default=0., help='Proportion of noise to add to identity loss B2B')
    parser.add_argument('--cycle_noise_A', type=float, default=0., help='Proportion of noise to add to cycle loss A2B2A')
    parser.add_argument('--cycle_noise_B', type=float, default=0., help='Proportion of noise to add to cycle loss B2A2B')
    parser.add_argument('--test_stride', type=int, default=None, help='Distance to move adjacent windows when generating contiguous audio')
    parser.add_argument('--test_interpolate', type=str2bool, default=False, help='When generating continuous audio, whether to linearly interpolate between windows or trim and concatenate')
    parser.add_argument('--deterministic', type=str2bool, default=False, help='Fix seed for random ops')

    parser.add_argument('--print_input', type=str2bool, default=False, help='Print stats about inputs')
    parser.add_argument('--print_wandg', type=str2bool, default=False, help='Print stats about weights and gradients')
    parser.add_argument('--print_gen_layer', type=int, default=-1, help='Print parameters at specified index for generators')
    parser.add_argument('--print_dis_layer', type=int, default=-1, help='Print parameters at specified index for discriminators')
    parser.add_argument('--save_when_interrupted', type=str2bool, default=False, help='Save model when early termination occurs')

    parser.add_argument('--DEBUG_use_old', type=str2bool, default=False, help='')

    return check_args(parser.parse_args())

"""checking arguments"""
def check_args(args):
    # --result_dir
    check_folder(os.path.join(args.result_dir, args.dataset, 'model'))
    check_folder(os.path.join(args.result_dir, args.dataset, 'img'))
    check_folder(os.path.join(args.result_dir, args.dataset, 'test'))

    # --epoch
    try:
        assert args.epoch >= 1
    except:
        print('number of epochs must be larger than or equal to one')

    # --batch_size
    try:
        assert args.batch_size >= 1
    except:
        print('batch size must be larger than or equal to one')
    return args

"""main"""
def main():
    # parse arguments
    args = parse_args()
    if args is None:
      exit()
    print(args)

    # open session
    if args.DEBUG_use_old:
        from UGATIT_old import UGATIT as UGATIT_old
        gan_old = UGATIT_old(args)
        gan_old.build_model()
    gan = UGATIT(args)

    # build graph
    gan.build_model()

    if args.deterministic:
        torch.manual_seed(0)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    try:  # this try/catch is for early stopping program via keyboard interrupt
        if args.phase == 'train' :
            outputs = [x.detach().cpu() for x in gan.train()]
            del gan
            if args.DEBUG_use_old:
                if args.deterministic:
                    torch.manual_seed(0)
                    torch.backends.cudnn.deterministic = True
                    torch.backends.cudnn.benchmark = False
                old_outputs = gan_old.train()
                print(type(outputs[0]), type(old_outputs[0]))
                for i, j in zip(outputs, old_outputs):
                    # for (_, i), (_, j) in zip(list(a.items()), list(b.items())):
                        print_and_summarize(i, j)
                        print_and_summarize(i - j)
                        print(torch.min(i - j), torch.max(i - j))

            print(" [*] Training finished!")

        if args.phase == 'test' :
            gan.test()
            print(" [*] Test finished!")

    except (KeyboardInterrupt, SystemExit):
        print('Interrupted')
        if args.save_when_interrupted:
            print('Trying to save checkpoint...')
            gan.save(os.path.join(args.result_dir, 'model'), 'interrupted')

if __name__ == '__main__':
    main()
