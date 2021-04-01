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

    parser.add_argument('--gen_lr', type=float, required=False, help='Learning rate for generator')
    parser.add_argument('--dis_lr', type=float, required=False, help='Learning rate for discriminator')

    parser.add_argument('--num_workers', type=int, default=1, help='Number of worker processes for dataloader')
    parser.add_argument('--mel_spectrogram', type=str2bool, default=False, help='Input mel spectrograms instead of images')
    parser.add_argument('--mel_channels', type=int, default=80, help='The (vertical) height of mel spectrogram (img_size is time/horizontal width)')
    parser.add_argument('--use_noise', type=str2bool, default=False, help='Use noise data in DATASET/noise')
    parser.add_argument('--gen_noise_A', type=float, default=0., help='Proportion of noise to add to A samples before generation')
    parser.add_argument('--gen_noise_B', type=float, default=0., help='Proportion of noise to add to B samples before generation')
    parser.add_argument('--dis_noise_A', type=float, default=0., help='Proportion of noise to add to A2B samples after generation, before discriminator')
    parser.add_argument('--dis_noise_B', type=float, default=0., help='Proportion of noise to add to B2A samples after generation, before discriminator')
    parser.add_argument('--dis_noise_A2B', type=float, default=None, help='This proportion is only for training discriminator on fake samples')
    parser.add_argument('--dis_noise_B2A', type=float, default=None, help='This proportion is only for training discriminator on fake samples')
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
    parser.add_argument('--adjust_noise_volume', type=str2bool, default=False, help='Use noise normalizer to adjust noise/source distribution so that noise (usually) is not louder than source')
    parser.add_argument('--scale_source_volume_A', type=float, default=0., help='Random augmentation for generator only, of A samples by scaling volume')
    parser.add_argument('--scale_source_volume_B', type=float, default=0., help='Random augmentation for generator only, of B samples by scaling volume')
    parser.add_argument('--scale_target_volume_A', type=float, default=0., help='Random scaling of volume (gen and dis) of A when loaded')
    parser.add_argument('--scale_target_volume_B', type=float, default=0., help='Random scaling of volume (gen and dis) of B when loaded')
    parser.add_argument('--scale_source_volume_noise', type=float, default=0., help='Random scaling of noise to match sample volume prior to adding with sample when gen_noise_A/gen_noise_B is set')
    parser.add_argument('--scale_target_volume_noise', type=float, default=0., help='Random scaling (gen and dis) of noise samples when loaded')
    parser.add_argument('--noise_margin', type=float, default=0.03, help='Min mean diff between noise and B samples to trigger scaling')
    parser.add_argument('--noise_weight_A', type=float, default=0, help='Weight for discriminator A differentiation between noise and target')
    parser.add_argument('--noise_weight_B', type=float, default=0, help='Weight for discriminator B differentiation between noise and target')
    parser.add_argument('--NOISE_n_samples', type=int, default=100, help='This is an experiment to check effect of noise distribution on loss, set `phase` to NOISE_test')

    parser.add_argument('--augment_volume', type=str2bool, default=False, help='Augment data at source with sinusoidal volume changes, salt and pepper noise, and stretching')
    parser.add_argument('--w_sin_freq_A', type=float, default=0., help='Rate of sinusoidal volume changes in time domain, for A samples')
    parser.add_argument('--h_sin_freq_A', type=float, default=0., help='Rate of sinusoidal volume changes in freq domain, for A samples')
    parser.add_argument('--w_sin_amp_A', type=float, default=0., help='Amplitude of sinusoidal volume changes in time domain, for A samples')
    parser.add_argument('--h_sin_amp_A', type=float, default=0., help='Amplitude of sinusoidal volume changes in freq domain, for A samples')
    parser.add_argument('--sp_amp_A', type=float, default=0., help='Amplitude of salt and pepper noise, for A samples')
    parser.add_argument('--w_stretch_max_A', type=int, default=0, help='Max num pixels to stretch by in time domain, for A samples')
    parser.add_argument('--h_translate_max_A', type=int, default=0, help='Max num pixels to translate by in freq domain, for A samples')

    parser.add_argument('--w_sin_freq_B', type=float, default=0., help='Rate of sinusoidal volume changes in time domain, for B samples')
    parser.add_argument('--h_sin_freq_B', type=float, default=0., help='Rate of sinusoidal volume changes in freq domain, for B samples')
    parser.add_argument('--w_sin_amp_B', type=float, default=0., help='Amplitude of sinusoidal volume changes in time domain, for B samples')
    parser.add_argument('--h_sin_amp_B', type=float, default=0., help='Amplitude of sinusoidal volume changes in freq domain, for B samples')
    parser.add_argument('--sp_amp_B', type=float, default=0., help='Amplitude of salt and pepper noise, for B samples')
    parser.add_argument('--w_stretch_max_B', type=int, default=0, help='Max num pixels to stretch by in time domain, for B samples')
    parser.add_argument('--h_translate_max_B', type=int, default=0, help='Max num pixels to translate by in freq domain, for B samples')

    parser.add_argument('--w_sin_freq_noise', type=float, default=0., help='Rate of sinusoidal volume changes in time domain, for noise samples')
    parser.add_argument('--h_sin_freq_noise', type=float, default=0., help='Rate of sinusoidal volume changes in freq domain, for noise samples')
    parser.add_argument('--w_sin_amp_noise', type=float, default=0., help='Amplitude of sinusoidal volume changes in time domain, for noise samples')
    parser.add_argument('--h_sin_amp_noise', type=float, default=0., help='Amplitude of sinusoidal volume changes in freq domain, for noise samples')
    parser.add_argument('--sp_amp_noise', type=float, default=0., help='Amplitude of salt and pepper noise, for noise samples')
    parser.add_argument('--w_stretch_max_noise', type=int, default=0, help='Max num pixels to stretch by in time domain, for noise samples')
    parser.add_argument('--h_translate_max_noise', type=int, default=0, help='Max num pixels to translate by in freq domain, for noise samples')

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
    gan = UGATIT(args)

    # build graph
    gan.build_model()

    if args.deterministic:
        torch.manual_seed(0)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    try:  # this try/catch is for early stopping program via keyboard interrupt
        if args.phase == 'train' :
            gan.train()
            print(" [*] Training finished!")

        if args.phase == 'test' :
            gan.test()
            print(" [*] Test finished!")

        if args.phase == 'NOISE_test' :
            cols = gan.noise_test(args.NOISE_n_samples)
            print(cols)
            print(" [*] NOISE_test finished!")

        if args.phase == 'dis' :
            gan.run_dis_only()
            print(" [*] Dis finished!")

        if args.phase == 'inject_gen' :
            gan.inject_gen()
            print(" [*] Gen finished!")

        if args.phase == 'seeded_gen' :
            gan.seeded_gen()
            print(" [*] Gen finished!")

        if args.phase == 'staggered_gen' :
            gan.staggered_gen()
            print(" [*] Gen finished!")

    except (KeyboardInterrupt, SystemExit):
        print('Interrupted')
        if args.save_when_interrupted:
            print('Trying to save checkpoint...')
            gan.save(os.path.join(args.result_dir, 'model'), 'interrupted')

if __name__ == '__main__':
    main()
