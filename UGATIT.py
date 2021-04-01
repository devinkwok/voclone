import time, itertools
from dataset import ImageFolder, MelFolder, SequentialMelLoader
from torchvision import transforms
from torch.utils.data import DataLoader
from networks import *
from utils import *
from glob import glob
import torch.nn.functional as F

import pandas as pd
from mel_utils import *
from iterseq import *


class UGATIT(object) :
    """Defines model training.
    """
    def __init__(self, args):
        self.light = args.light

        if self.light :
            self.model_name = 'UGATIT_light'
        else :
            self.model_name = 'UGATIT'

        self.result_dir = args.result_dir
        self.dataset = args.dataset

        self.iteration = args.iteration
        self.decay_flag = args.decay_flag

        self.batch_size = args.batch_size
        self.print_freq = args.print_freq
        self.save_freq = args.save_freq

        self.weight_decay = args.weight_decay
        self.ch = args.ch

        """ Weight """
        self.adv_weight = args.adv_weight
        self.cycle_weight = args.cycle_weight
        self.identity_weight = args.identity_weight
        self.cam_weight = args.cam_weight

        """ Generator """
        self.n_res = args.n_res

        """ Discriminator """
        self.n_dis = args.n_dis

        self.img_size = args.img_size
        self.img_ch = args.img_ch

        self.device = args.device
        self.benchmark_flag = args.benchmark_flag
        self.resume = args.resume

        if torch.backends.cudnn.enabled and self.benchmark_flag:
            print('set benchmark !')
            torch.backends.cudnn.benchmark = True

        # new params
        self.num_workers = args.num_workers
        self.mel_spectrogram = args.mel_spectrogram
        self.mel_channels = args.mel_channels
        assert self.mel_channels <= self.img_size  # can't have more channels than the image input size
        self.print_input = args.print_input
        self.print_wandg = args.print_wandg
        if args.gen_lr is None:
            self.gen_lr = args.lr
        else:
            self.gen_lr = args.gen_lr
        if args.dis_lr is None:
            self.dis_lr = args.lr
        else:
            self.dis_lr = args.dis_lr

        self.use_noise = args.use_noise
        self.gen_noise_A = args.gen_noise_A
        self.gen_noise_B = args.gen_noise_B
        self.dis_noise_A = args.dis_noise_A
        self.dis_noise_B = args.dis_noise_B
        if args.dis_noise_A2B is None:
            self.dis_noise_A2B = self.dis_noise_A
        else:
            self.dis_noise_A2B = args.dis_noise_A2B
        if args.dis_noise_B2A is None:
            self.dis_noise_B2A = self.dis_noise_B
        else:
            self.dis_noise_B2A = args.dis_noise_B2A

        self.identity_noise_A = args.identity_noise_A
        self.identity_noise_B = args.identity_noise_B
        self.cycle_noise_A = args.cycle_noise_A
        self.cycle_noise_B = args.cycle_noise_B
        self.test_interpolate = args.test_interpolate
        self.test_stride = args.test_stride
        if self.test_stride is None:
            self.test_stride = self.img_size

        self.print_gen_layer = args.print_gen_layer
        self.print_dis_layer = args.print_dis_layer
        self.adjust_noise_volume = args.adjust_noise_volume
        self.scale_target_volume_A = args.scale_target_volume_A
        self.scale_target_volume_B = args.scale_target_volume_B
        self.scale_source_volume_A = args.scale_source_volume_A
        self.scale_source_volume_B = args.scale_source_volume_B
        self.scale_target_volume_noise = args.scale_target_volume_noise
        self.scale_source_volume_noise = args.scale_source_volume_noise
        self.noise_margin = args.noise_margin
        self.noise_weight_A = args.noise_weight_A
        self.noise_weight_B = args.noise_weight_B

        self.augment_volume = args.augment_volume
        self.w_sin_freq_A = args.w_sin_freq_A
        self.h_sin_freq_A = args.h_sin_freq_A
        self.w_sin_amp_A = args.w_sin_amp_A
        self.h_sin_amp_A = args.h_sin_amp_A
        self.sp_amp_A = args.sp_amp_A
        self.w_stretch_max_A = args.w_stretch_max_A
        self.h_translate_max_A = args.h_translate_max_A

        self.w_sin_freq_B = args.w_sin_freq_B
        self.h_sin_freq_B = args.h_sin_freq_B
        self.w_sin_amp_B = args.w_sin_amp_B
        self.h_sin_amp_B = args.h_sin_amp_B
        self.sp_amp_B = args.sp_amp_B
        self.w_stretch_max_B = args.w_stretch_max_B
        self.h_translate_max_B = args.h_translate_max_B

        self.w_sin_freq_noise = args.w_sin_freq_noise
        self.h_sin_freq_noise = args.h_sin_freq_noise
        self.w_sin_amp_noise = args.w_sin_amp_noise
        self.h_sin_amp_noise = args.h_sin_amp_noise
        self.sp_amp_noise = args.sp_amp_noise
        self.w_stretch_max_noise = args.w_stretch_max_noise
        self.h_translate_max_noise = args.h_translate_max_noise

        print()
        print("##### Information #####")
        print("# mel_spectrogram : ", self.mel_spectrogram)
        print("# light : ", self.light)
        print("# dataset : ", self.dataset)
        print("# batch_size : ", self.batch_size)
        print("# iteration per epoch : ", self.iteration)
        print()
        print("##### Generator #####")
        print("# residual blocks : ", self.n_res)
        print()
        print("##### Discriminator #####")
        print("# discriminator layer : ", self.n_dis)
        print()
        print("##### Weight #####")
        print("# adv_weight : ", self.adv_weight)
        print("# cycle_weight : ", self.cycle_weight)
        print("# identity_weight : ", self.identity_weight)
        print("# cam_weight : ", self.cam_weight)

    ##################################################################################
    # Model
    ##################################################################################

    @property
    def base_transform(self):
        """Default transformation to apply to all mel spectrograms in order
        to make them image-like. Applies value normalization and padding.

        Returns:
            torchvision.transforms.Compose: Object containing transforms
        """
        return transforms.Compose([
            transforms.Lambda(lambda x: mel_transform(x).unsqueeze(0)),
            transforms.Lambda(lambda x: pad_mel_channels(x, self.img_size)),
            # transforms.Lambda(lambda x: print_and_summarize(x, prefix='    in dist', do_print=self.print_input)),
        ])

    def build_model(self):
        """Initialize data loaders, data transforms, generators, discriminators,
        loss functions, and optimizers.
        """
        if self.mel_spectrogram:
            self.gen_transform_A = nn.Identity()
            if self.scale_source_volume_A > 0.:
                self.gen_transform_A = transforms.Compose([
                        transforms.Lambda(lambda x: scale_volume(x,  1. - self.scale_source_volume_A, 1. + self.scale_source_volume_A)),
                        ])
            self.gen_transform_B = nn.Identity()
            if self.scale_source_volume_B > 0.:
                self.gen_transform_A = transforms.Compose([
                        transforms.Lambda(lambda x: scale_volume(x,  1. - self.scale_source_volume_B, 1. + self.scale_source_volume_B)),
                        ])
            transform_A = transforms.Compose([
                transforms.Lambda(lambda x: mel_transform(x).unsqueeze(0)),
                transforms.Lambda(lambda x: augment_volume(x, self.w_sin_freq_A, self.h_sin_freq_A, self.w_sin_amp_A, self.h_sin_amp_A, self.sp_amp_A)),
                transforms.Lambda(lambda x: stretch_time_and_translate_ch(x, self.img_size, self.mel_channels, self.w_stretch_max_A, self.h_translate_max_A, pad_val=-1)),
                transforms.Lambda(lambda x: pad_mel_channels(x, self.img_size)),
                # transforms.Lambda(lambda x: print_and_summarize(x, prefix='    in dist', do_print=self.print_input)),
            ])
            transform_B = transforms.Compose([
                transforms.Lambda(lambda x: mel_transform(x).unsqueeze(0)),
                transforms.Lambda(lambda x: augment_volume(x, self.w_sin_freq_B, self.h_sin_freq_B, self.w_sin_amp_B, self.h_sin_amp_B, self.sp_amp_B)),
                transforms.Lambda(lambda x: stretch_time_and_translate_ch(x, self.img_size, self.mel_channels, self.w_stretch_max_B, self.h_translate_max_B, pad_val=-1)),
                transforms.Lambda(lambda x: pad_mel_channels(x, self.img_size)),
                # transforms.Lambda(lambda x: print_and_summarize(x, prefix='    in dist', do_print=self.print_input)),
            ])
            transform_noise = transforms.Compose([
                transforms.Lambda(lambda x: mel_transform(x).unsqueeze(0)),
                transforms.Lambda(lambda x: augment_volume(x, self.w_sin_freq_noise, self.h_sin_freq_noise, self.w_sin_amp_noise, self.h_sin_amp_noise, self.sp_amp_noise)),
                transforms.Lambda(lambda x: stretch_time_and_translate_ch(x, self.img_size, self.mel_channels, self.w_stretch_max_noise, self.h_translate_max_noise, pad_val=-1)),
                transforms.Lambda(lambda x: pad_mel_channels(x, self.img_size)),
                # transforms.Lambda(lambda x: print_and_summarize(x, prefix='    in dist', do_print=self.print_input)),
            ])
            if self.scale_target_volume_A > 0.:
                transform_A = transforms.Compose([transform_A,
                    transforms.Lambda(lambda x: scale_volume(x,  1. - self.scale_target_volume_A, 1. + self.scale_target_volume_A)),
                    ])
            if self.scale_target_volume_B > 0.:
                transform_B = transforms.Compose([transform_B,
                    transforms.Lambda(lambda x: scale_volume(x,  1. - self.scale_target_volume_B, 1. + self.scale_target_volume_B)),
                    ])
            if self.scale_target_volume_noise > 0.:
                transform_noise = transforms.Compose([transform_noise,
                    transforms.Lambda(lambda x: scale_volume(x,  1. - self.scale_target_volume_noise, 1. + self.scale_target_volume_noise)),
                    ])
            self.trainA = MelDir(os.path.join('dataset', self.dataset, 'trainA'))
            self.trainB = MelDir(os.path.join('dataset', self.dataset, 'trainB'))
            self.testA = MelFolder(os.path.join('dataset', self.dataset, 'testA'), self.base_transform)
            self.testB = MelFolder(os.path.join('dataset', self.dataset, 'testB'), self.base_transform)
            self.trainA_loader = StridedSequence(self.trainA, self.img_size + self.w_stretch_max_A, transforms=transform_A).get_data_loader(self.batch_size, self.num_workers)
            self.trainB_loader = StridedSequence(self.trainB, self.img_size + self.w_stretch_max_B, transforms=transform_B).get_data_loader(self.batch_size, self.num_workers)
            self.testA_loader = SequentialMelLoader(self.testA, self.img_size, stride=self.test_stride)
            self.testB_loader = SequentialMelLoader(self.testB, self.img_size, stride=self.test_stride)
            self.trainA_visits = {}
            self.trainB_visits = {}
            if self.use_noise:
                self.noise_data = MelDir(os.path.join('dataset', self.dataset, 'noise'))
                self.noise_loader = StridedSequence(self.noise_data, self.img_size + self.w_stretch_max_noise, transforms=transform_noise).get_data_loader(self.batch_size, self.num_workers)
                self.noise_visits = {}
            print('Datasets: A {} ({}), B {} ({}), noise {} ({})'.format(
                                self.trainA_loader.dataset.n_seq, self.trainA_loader.dataset.stride,
                                self.trainB_loader.dataset.n_seq, self.trainB_loader.dataset.stride,
                                self.noise_loader.dataset.n_seq, self.noise_loader.dataset.stride))
        else:
            train_transform = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.Resize((self.img_size + 30, self.img_size+30)),
                transforms.RandomCrop(self.img_size),
                transforms.ToTensor(),
                transforms.Lambda(lambda x: print_and_summarize(x, prefix='before', do_print=self.print_input)),
                transforms.Normalize(mean=(0.5), std=(0.5)),
                transforms.Lambda(lambda x: print_and_summarize(x, prefix='    after', do_print=self.print_input)),
            ])
            test_transform = transforms.Compose([
                transforms.Resize((self.img_size, self.img_size)),
                transforms.ToTensor(),
                transforms.Lambda(lambda x: print_and_summarize(x, prefix='before', do_print=self.print_input)),
                transforms.Normalize(mean=(0.5), std=(0.5)),
                transforms.Lambda(lambda x: print_and_summarize(x, prefix='    after', do_print=self.print_input)),
            ])
            self.trainA = ImageFolder(os.path.join('dataset', self.dataset, 'trainA'), train_transform)
            self.trainB = ImageFolder(os.path.join('dataset', self.dataset, 'trainB'), train_transform)
            self.testA = ImageFolder(os.path.join('dataset', self.dataset, 'testA'), test_transform)
            self.testB = ImageFolder(os.path.join('dataset', self.dataset, 'testB'), test_transform)
            self.trainA_loader = DataLoader(self.trainA, batch_size=self.batch_size, shuffle=True)
            self.trainB_loader = DataLoader(self.trainB, batch_size=self.batch_size, shuffle=True)
            self.testA_loader = DataLoader(self.testA, batch_size=1, shuffle=False)
            self.testB_loader = DataLoader(self.testB, batch_size=1, shuffle=False)

        """ Define Generator, Discriminator """
        self.genA2B = ResnetGenerator(input_nc=self.img_ch, output_nc=self.img_ch, ngf=self.ch, n_blocks=self.n_res, img_size=self.img_size, light=self.light).to(self.device)
        self.genB2A = ResnetGenerator(input_nc=self.img_ch, output_nc=self.img_ch, ngf=self.ch, n_blocks=self.n_res, img_size=self.img_size, light=self.light).to(self.device)
        self.disGA = Discriminator(input_nc=self.img_ch, ndf=self.ch, n_layers=self.n_dis+1).to(self.device)
        self.disGB = Discriminator(input_nc=self.img_ch, ndf=self.ch, n_layers=self.n_dis+1).to(self.device)
        self.disLA = Discriminator(input_nc=self.img_ch, ndf=self.ch, n_layers=self.n_dis-1).to(self.device)
        self.disLB = Discriminator(input_nc=self.img_ch, ndf=self.ch, n_layers=self.n_dis-1).to(self.device)

        """ Define Loss """
        self.L1_loss = nn.L1Loss().to(self.device)
        self.MSE_loss = nn.MSELoss().to(self.device)
        self.BCE_loss = nn.BCEWithLogitsLoss().to(self.device)

        """ Trainer """
        self.G_optim = torch.optim.Adam(itertools.chain(self.genA2B.parameters(), self.genB2A.parameters()), lr=self.gen_lr, betas=(0.5, 0.999), weight_decay=self.weight_decay)
        self.D_optim = torch.optim.Adam(itertools.chain(self.disGA.parameters(), self.disGB.parameters(), self.disLA.parameters(), self.disLB.parameters()), lr=self.dis_lr, betas=(0.5, 0.999), weight_decay=self.weight_decay)

        """ Define Rho clipper to constraint the value of rho in AdaILN and ILN"""
        self.Rho_clipper = RhoClipper(0, 1)
        self.start_iter = 1

        if self.resume:
            self.load_latest_checkpoint()


    def get_samples(self, is_test=False):
        """Draws mel samples.

        Args:
            is_test (bool, optional): If True, draw from test dataloaders,
                otherwise train dataloders. Defaults to False.

        Returns:
            tuple of torch.Tensor: two batches of mel spectrograms
                from source A and source B dataloaders respectively
        """
        if is_test:
            try:
                real_A, (key_A, coord_A) = next(self.testA_iter)
            except:
                self.testA_iter = iter(self.testA_loader)
                real_A, (key_A, coord_A) = next(self.testA_iter)

            try:
                real_B, (key_B, coord_B) = next(self.testB_iter)
            except:
                self.testB_iter = iter(self.testB_loader)
                real_B, (key_B, coord_B) = next(self.testB_iter)
        else:
            try:
                real_A, (key_A, coord_A) = next(self.trainA_iter)
            except:
                self.trainA_iter = iter(self.trainA_loader)
                real_A, (key_A, coord_A) = next(self.trainA_iter)

            try:
                real_B, (key_B, coord_B) = next(self.trainB_iter)
            except:
                self.trainB_iter = iter(self.trainB_loader)
                real_B, (key_B, coord_B) = next(self.trainB_iter)
            if key_A not in self.trainA_visits:
                self.trainA_visits[key_A] = 1
            else:
                self.trainA_visits[key_A] += 1
            if key_B not in self.trainB_visits:
                self.trainB_visits[key_B] = 1
            else:
                self.trainB_visits[key_B] += 1
        self.noise_A, self.noise_B = None, None
        if self.use_noise:  # add noise to data samples if required
            try:
                self.noise_A, (key_NA, coord_NA) = next(self.noise_iter)
            except:
                self.noise_iter = iter(self.noise_loader)
                self.noise_A, (key_NA, coord_NA) = next(self.noise_iter)
            try:
                self.noise_B, (key_NB, coord_NB) = next(self.noise_iter)
            except:
                self.noise_iter = iter(self.noise_loader)
                self.noise_B, (key_NB, coord_NB) = next(self.noise_iter)
            if not is_test:
                real_A = self.noisy(real_A, self.gen_noise_A, True)
                real_B = self.noisy(real_B, self.gen_noise_B, False)

            self.noise_A = self.noise_A.to(self.device)
            self.noise_B = self.noise_B.to(self.device)
            if key_NA not in self.noise_visits:
                self.noise_visits[key_NA] = 1
            else:
                self.noise_visits[key_NA] += 1
            if key_NB not in self.noise_visits:
                self.noise_visits[key_NB] = 1
            else:
                self.noise_visits[key_NB] += 1

        # static_A = detect_static(unpad_mel_channels(inv_mel_transform(real_A), self.mel_channels))
        # static_B = detect_static(unpad_mel_channels(inv_mel_transform(real_B), self.mel_channels))
        # if static_A is not None:
        #     print("                      !!!!", key_A, coord_A, static_A)
        # if static_B is not None:
        #     print("                      !!!!", key_B, coord_B, key_NB, coord_NB, static_B)
        return real_A.to(self.device), real_B.to(self.device)


    def noisy(self, mel, noise_prop, is_A=True):  # wrapper including printing
        """Mixes noise into a mel spectrogram.
        Noise is either drawn from noise_A or noise_B dataloader.

        Args:
            mel (torch.Tensor): source spectrogram
            noise_prop (float): random proportion (0 to 1) of samples to add noise to
            is_A (bool, optional): whether to use noise from source A (True) or source B. Defaults to True.

        Returns:
            torch.Tensor: spectrogram with mel and noise combined
        """
        if noise_prop <= 0.:
            return mel
        if torch.rand([1]).item() < noise_prop:
            if self.adjust_noise_volume:
                mel, noise = noise_normalizer(mel, self.noise_A, self.noise_B, self.noise_margin, self.scale_source_volume_noise)
            else:
                if is_A:
                    noise = self.noise_A
                else:
                    noise = self.noise_B
            combined = add_mels(mel, noise)
            return combined
        return mel
    

    def get_dis(self, sample, noise_prop, is_A=True):
        """Discriminator output.

        Args:
            sample (torch.Tensor): mel spectrogram
            noise_prop (float): random proportion of samples to add noise to before discriminator,
                this is useful for normalizing when only one class has noise
            is_A (bool, optional): whether to use A2B (True) or B2A generator. Defaults to True.

        Returns:
            tuple of torch.Tensor: G discriminator logit, G class activation mapping (CAM) logit,
                L discriminator logit, L class activation mapping (CAM) logit...
                G and L refer to greater and less? One is higher dimensional than the other
        """
        tensor = self.noisy(sample, noise_prop, is_A)
        if is_A:
            G_logit, G_cam_logit, _ = self.disGA(tensor)
            L_logit, L_cam_logit, _ = self.disLA(tensor)
        else:
            G_logit, G_cam_logit, _ = self.disGB(tensor)
            L_logit, L_cam_logit, _ = self.disLB(tensor)
        return G_logit, G_cam_logit, L_logit, L_cam_logit


    def get_gen(self, real_X, is_A=True):
        """Generator output.

        Args:
            real_X (torch.Tensor): source mel spectrogram
            is_A (bool, optional): whether to use A2B (True) or B2A generator. Defaults to True.

        Returns:
            tuple of torch.Tensor: fake sample, class activation mapping (CAM) logit, heatmap,
                cycle gen sample (i.e. A2B2A), cycle gen heatmap,
                identity gen sample (i.e. A2A), identity CAM logit, identity heatmap
        """
        if is_A:
            genX2Y = self.genA2B
            genY2X = self.genB2A
            cycle_noise = self.cycle_noise_A
            id_noise = self.identity_noise_A
        else:
            genX2Y = self.genB2A
            genY2X = self.genA2B
            cycle_noise = self.cycle_noise_B
            id_noise = self.identity_noise_B
        fake_X2Y, fake_X2Y_cam_logit, fake_X2Y_heatmap = genX2Y(real_X)
        # add background noise to cycle
        fake_X2Y2X, _, fake_X2Y2X_heatmap = genY2X(self.noisy(fake_X2Y, cycle_noise, is_A))
        # add background noise to identity
        fake_X2X, fake_X2X_cam_logit, fake_X2X_heatmap = genY2X(self.noisy(real_X, id_noise, is_A))

        fake_X2Y = strip_padded(fake_X2Y, self.mel_channels)

        return fake_X2Y, fake_X2Y_cam_logit, fake_X2Y_heatmap, \
                fake_X2Y2X, fake_X2Y2X_heatmap, \
                fake_X2X, fake_X2X_cam_logit, fake_X2X_heatmap


    def collate_examples(self, n_samples=None, is_A=True, is_test=False):
        """Generates samples and combines resulting spectrograms to report model progress.

        Args:
            n_samples (int, optional): Number of samples to generate.
                If None, use all samples in data loaders. Defaults to None.
            is_A (bool, optional): Generate A to B if True, otherwise B to A. Defaults to True.
            is_test (bool, optional): If True, use test dataloader, otherwise use train dataloader.
                Defaults to False.

        Returns:
            torch.Tensor: generated mel spectrograms concatenated together
        """
        if n_samples is None:
            if is_test:
                if is_A:
                    n_samples = len(self.testA_loader)
                else:
                    n_samples = len(self.testB_loader)
            else:
                if is_A:
                    n_samples = len(self.trainA_loader)
                else:
                    n_samples = len(self.trainB_loader)
        X2Y = None
        for _ in range(n_samples):
            real_A, real_B = self.get_samples(is_test=is_test)
            if is_A:
                real_X = self.gen_transform_A(real_A)
            else:
                real_X = self.gen_transform_B(real_B)
            fake_X2Y, _, fake_X2Y_heatmap, fake_X2Y2X, fake_X2Y2X_heatmap, fake_X2X, _, fake_X2X_heatmap, \
                = self.get_gen(real_X, is_A=is_A)

            if self.mel_spectrogram:
                col = np.concatenate((real_X.detach().cpu(),
                                    fake_X2X.detach().cpu(),
                                    fake_X2Y.detach().cpu(),
                                    fake_X2Y2X.detach().cpu()), 0)
                if self.use_noise:
                    if is_A:
                        col = np.concatenate((self.noise_A.detach().cpu(), col), 0)
                    else:
                        col = np.concatenate((self.noise_B.detach().cpu(), col), 0)
            else:
                col = np.concatenate((RGB2BGR(tensor2numpy(denorm(real_X[0]))),
                                    cam(tensor2numpy(fake_X2X_heatmap[0]), self.img_size),
                                    RGB2BGR(tensor2numpy(denorm(fake_X2X[0]))),
                                    cam(tensor2numpy(fake_X2Y_heatmap[0]), self.img_size),
                                    RGB2BGR(tensor2numpy(denorm(fake_X2Y[0]))),
                                    cam(tensor2numpy(fake_X2Y2X_heatmap[0]), self.img_size),
                                    RGB2BGR(tensor2numpy(denorm(fake_X2Y2X[0])))), 0)
            if X2Y is None:
                X2Y = col
            else:
                X2Y = np.concatenate((X2Y, col), 1)
        return X2Y


    def train(self):
        """Training loop.
        """
        self.genA2B.train(), self.genB2A.train(), self.disGA.train(), self.disGB.train(), self.disLA.train(), self.disLB.train()

        # training loop
        print('training start !')
        start_time = time.time()
        for step in range(self.start_iter, self.iteration + 1):
            if self.decay_flag and step > (self.iteration // 2):
                self.G_optim.param_groups[0]['lr'] -= (self.gen_lr / (self.iteration // 2))
                self.D_optim.param_groups[0]['lr'] -= (self.dis_lr / (self.iteration // 2))

            real_A, real_B = self.get_samples()
            
            # Update D with noise

            # Update D
            self.D_optim.zero_grad()

            fake_A2B, _, _ = self.genA2B(self.gen_transform_A(real_A))
            fake_B2A, _, _ = self.genB2A(self.gen_transform_B(real_B))
            fake_A2B = strip_padded(fake_A2B, self.mel_channels)
            fake_B2A = strip_padded(fake_B2A, self.mel_channels)

            # if dis_noise_A > 0., this means add noise to A, or that A has no noise and B has noise
            # then A2B will have no noise while B2A will have noise, so compare A2B+noise with B, and B2A with A+noise
            # vice versa if B has no noise and A has noise
            # if both A and B have no noise, then setting both dis_noise_A and dis_noise_B > 0.
            # means compare A2B+noise with B+noise, and B2A+noise with A+noise
            real_GA_logit, real_GA_cam_logit, real_LA_logit, real_LA_cam_logit = self.get_dis(real_A, self.dis_noise_A, is_A=True)
            real_GB_logit, real_GB_cam_logit, real_LB_logit, real_LB_cam_logit = self.get_dis(real_B, self.dis_noise_B, is_A=False)
            fake_GA_logit, fake_GA_cam_logit, fake_LA_logit, fake_LA_cam_logit = self.get_dis(fake_B2A, self.dis_noise_B2A, is_A=True)
            fake_GB_logit, fake_GB_cam_logit, fake_LB_logit, fake_LB_cam_logit = self.get_dis(fake_A2B, self.dis_noise_A2B, is_A=False)

            # print('logits')
            # print(torch.std_mean(real_GA_logit), torch.std_mean(real_GA_cam_logit), torch.std_mean(real_LA_logit), torch.std_mean(real_LA_cam_logit))
            # print(torch.std_mean(real_GB_logit), torch.std_mean(real_GB_cam_logit), torch.std_mean(real_LB_logit), torch.std_mean(real_LB_cam_logit))
            # print(torch.std_mean(fake_GA_logit), torch.std_mean(fake_GA_cam_logit), torch.std_mean(fake_LA_logit), torch.std_mean(fake_LA_cam_logit))
            # print(torch.std_mean(fake_GB_logit), torch.std_mean(fake_GB_cam_logit), torch.std_mean(fake_LB_logit), torch.std_mean(fake_LB_cam_logit))
            D_ad_loss_GA = self.MSE_loss(real_GA_logit, torch.ones_like(real_GA_logit).to(self.device)) \
                + self.MSE_loss(fake_GA_logit, torch.zeros_like(fake_GA_logit).to(self.device))
            D_ad_cam_loss_GA = self.MSE_loss(real_GA_cam_logit, torch.ones_like(real_GA_cam_logit).to(self.device)) \
                + self.MSE_loss(fake_GA_cam_logit, torch.zeros_like(fake_GA_cam_logit).to(self.device))
            D_ad_loss_LA = self.MSE_loss(real_LA_logit, torch.ones_like(real_LA_logit).to(self.device)) \
                + self.MSE_loss(fake_LA_logit, torch.zeros_like(fake_LA_logit).to(self.device))
            D_ad_cam_loss_LA = self.MSE_loss(real_LA_cam_logit, torch.ones_like(real_LA_cam_logit).to(self.device)) \
                + self.MSE_loss(fake_LA_cam_logit, torch.zeros_like(fake_LA_cam_logit).to(self.device))
            D_ad_loss_GB = self.MSE_loss(real_GB_logit, torch.ones_like(real_GB_logit).to(self.device)) \
                + self.MSE_loss(fake_GB_logit, torch.zeros_like(fake_GB_logit).to(self.device))
            D_ad_cam_loss_GB = self.MSE_loss(real_GB_cam_logit, torch.ones_like(real_GB_cam_logit).to(self.device)) \
                + self.MSE_loss(fake_GB_cam_logit, torch.zeros_like(fake_GB_cam_logit).to(self.device))
            D_ad_loss_LB = self.MSE_loss(real_LB_logit, torch.ones_like(real_LB_logit).to(self.device)) \
                + self.MSE_loss(fake_LB_logit, torch.zeros_like(fake_LB_logit).to(self.device))
            D_ad_cam_loss_LB = self.MSE_loss(real_LB_cam_logit, torch.ones_like(real_LB_cam_logit).to(self.device)) \
                + self.MSE_loss(fake_LB_cam_logit, torch.zeros_like(fake_LB_cam_logit).to(self.device))

            D_loss_A = self.adv_weight * (D_ad_loss_GA + D_ad_cam_loss_GA + D_ad_loss_LA + D_ad_cam_loss_LA)
            D_loss_B = self.adv_weight * (D_ad_loss_GB + D_ad_cam_loss_GB + D_ad_loss_LB + D_ad_cam_loss_LB)
            Discriminator_loss = D_loss_A + D_loss_B

            D_loss_noise = torch.zeros_like(Discriminator_loss)
            if self.noise_weight_A > 0:
                noise_GA_logit, noise_GA_cam_logit, noise_LA_logit, noise_LA_cam_logit = self.get_dis(self.noise_A, 0., is_A=True)
                D_noise_loss_A = self.MSE_loss(noise_GA_logit, torch.zeros_like(noise_GA_logit).to(self.device)) \
                    + self.MSE_loss(noise_GA_cam_logit, torch.zeros_like(noise_GA_cam_logit).to(self.device)) \
                    + self.MSE_loss(noise_LA_logit, torch.zeros_like(noise_LA_logit).to(self.device)) \
                    + self.MSE_loss(noise_LA_cam_logit, torch.zeros_like(noise_LA_cam_logit).to(self.device))
                D_loss_noise += self.noise_weight_A * D_noise_loss_A
            if self.noise_weight_B > 0:
                noise_GB_logit, noise_GB_cam_logit, noise_LB_logit, noise_LB_cam_logit = self.get_dis(self.noise_B, 0., is_A=False)
                D_noise_loss_B = self.MSE_loss(noise_GB_logit, torch.zeros_like(noise_GB_logit).to(self.device)) \
                    + self.MSE_loss(noise_GB_cam_logit, torch.zeros_like(noise_GB_cam_logit).to(self.device)) \
                    + self.MSE_loss(noise_LB_logit, torch.zeros_like(noise_LB_logit).to(self.device)) \
                    + self.MSE_loss(noise_LB_cam_logit, torch.zeros_like(noise_LB_cam_logit).to(self.device))
                D_loss_noise += self.noise_weight_B * D_noise_loss_B
            Discriminator_loss += D_loss_noise

            if self.print_wandg and torch.isnan(Discriminator_loss):
                print_weights_and_grads({
                    'genA2B': self.genA2B, 'genB2A': self.genB2A, 'disGA': self.disGA, 'disLA': self.disLA, 'disGB': self.disGB, 'disLB': self.disLB,
                    })
                break
            Discriminator_loss.backward()
            self.D_optim.step()

            # Update G
            self.G_optim.zero_grad()

            fake_A2B, fake_A2B_cam_logit, _, fake_A2B2A, _, fake_A2A, fake_A2A_cam_logit, _, = self.get_gen(self.gen_transform_A(real_A), is_A=True)
            fake_B2A, fake_B2A_cam_logit, _, fake_B2A2B, _, fake_B2B, fake_B2B_cam_logit, _ = self.get_gen(self.gen_transform_B(real_B), is_A=False)
            # add background noise to fake if source has none
            fake_GA_logit, fake_GA_cam_logit, fake_LA_logit, fake_LA_cam_logit = self.get_dis(fake_B2A, self.dis_noise_B2A, is_A=True)
            fake_GB_logit, fake_GB_cam_logit, fake_LB_logit, fake_LB_cam_logit = self.get_dis(fake_A2B, self.dis_noise_A2B, is_A=False)

            G_ad_loss_GA = self.MSE_loss(fake_GA_logit, torch.ones_like(fake_GA_logit).to(self.device))
            G_ad_cam_loss_GA = self.MSE_loss(fake_GA_cam_logit, torch.ones_like(fake_GA_cam_logit).to(self.device))
            G_ad_loss_LA = self.MSE_loss(fake_LA_logit, torch.ones_like(fake_LA_logit).to(self.device))
            G_ad_cam_loss_LA = self.MSE_loss(fake_LA_cam_logit, torch.ones_like(fake_LA_cam_logit).to(self.device))
            G_ad_loss_GB = self.MSE_loss(fake_GB_logit, torch.ones_like(fake_GB_logit).to(self.device))
            G_ad_cam_loss_GB = self.MSE_loss(fake_GB_cam_logit, torch.ones_like(fake_GB_cam_logit).to(self.device))
            G_ad_loss_LB = self.MSE_loss(fake_LB_logit, torch.ones_like(fake_LB_logit).to(self.device))
            G_ad_cam_loss_LB = self.MSE_loss(fake_LB_cam_logit, torch.ones_like(fake_LB_cam_logit).to(self.device))

            G_recon_loss_A = self.L1_loss(fake_A2B2A, real_A)
            G_recon_loss_B = self.L1_loss(fake_B2A2B, real_B)

            G_identity_loss_A = self.L1_loss(fake_A2A, real_A)
            G_identity_loss_B = self.L1_loss(fake_B2B, real_B)

            G_cam_loss_A = self.BCE_loss(fake_B2A_cam_logit, torch.ones_like(fake_B2A_cam_logit).to(self.device)) + self.BCE_loss(fake_A2A_cam_logit, torch.zeros_like(fake_A2A_cam_logit).to(self.device))
            G_cam_loss_B = self.BCE_loss(fake_A2B_cam_logit, torch.ones_like(fake_A2B_cam_logit).to(self.device)) + self.BCE_loss(fake_B2B_cam_logit, torch.zeros_like(fake_B2B_cam_logit).to(self.device))

            G_loss_A =  self.adv_weight * (G_ad_loss_GA + G_ad_cam_loss_GA + G_ad_loss_LA + G_ad_cam_loss_LA) + self.cycle_weight * G_recon_loss_A + self.identity_weight * G_identity_loss_A + self.cam_weight * G_cam_loss_A
            G_loss_B = self.adv_weight * (G_ad_loss_GB + G_ad_cam_loss_GB + G_ad_loss_LB + G_ad_cam_loss_LB) + self.cycle_weight * G_recon_loss_B + self.identity_weight * G_identity_loss_B + self.cam_weight * G_cam_loss_B

            Generator_loss = G_loss_A + G_loss_B
            if self.print_wandg and torch.isnan(Generator_loss):
                print_weights_and_grads({
                    'genA2B': self.genA2B, 'genB2A': self.genB2A, 'disGA': self.disGA, 'disLA': self.disLA, 'disGB': self.disGB, 'disLB': self.disLB,
                    })
                break
            Generator_loss.backward()
            self.G_optim.step()

            # clip parameter of AdaILN and ILN, applied after optimizer step
            self.genA2B.apply(self.Rho_clipper)
            self.genB2A.apply(self.Rho_clipper)

            # print debug info
            print("[%5d/%5d] time: %4.2f d_loss_A: %.4f, d_loss_B: %.4f, d_loss_noise: %.4f, g_loss: %.4f" % (step, self.iteration, time.time() - start_time, D_loss_A, D_loss_B, D_loss_noise, Generator_loss))
            if self.print_wandg:
                print(summarize(*wandg_stats(self.genA2B, self.genB2A, self.disGA, self.disLA, self.disGB, self.disLB)))
            if self.print_gen_layer >= 0:
                a2b = [p for p in self.genA2B.parameters()][self.print_gen_layer]
                b2a = [p for p in self.genB2A.parameters()][self.print_gen_layer]
                print(a2b.shape, summarize(a2b, b2a, include_grad=True), b2a.shape)
            if self.print_dis_layer >= 0:
                print(summarize([p for p in self.disGA.parameters()][self.print_gen_layer],
                                [p for p in self.disLA.parameters()][self.print_gen_layer],
                                [p for p in self.disGB.parameters()][self.print_gen_layer],
                                [p for p in self.disLB.parameters()][self.print_gen_layer],
                                include_grad=True))

            # save progress
            if step % self.print_freq == 0:
                self.genA2B.eval(), self.genB2A.eval(), self.disGA.eval(), self.disGB.eval(), self.disLA.eval(), self.disLB.eval()

                A2B = np.concatenate([
                        self.collate_examples(n_samples=5, is_A=True, is_test=False),
                        self.collate_examples(n_samples=5, is_A=True, is_test=True)], 1)
                B2A = np.concatenate([
                    self.collate_examples(n_samples=5, is_A=False, is_test=False),
                    self.collate_examples(n_samples=5, is_A=False, is_test=True)], 1)

                if self.mel_spectrogram:
                    A2B = A2B.transpose(2, 1, 0, 3).reshape(self.img_size, -1)
                    B2A = B2A.transpose(2, 1, 0, 3).reshape(self.img_size, -1)
                    self.save_mel(A2B, 'img', 'A2B_%07d.mel' % step)
                    self.save_mel(B2A, 'img', 'B2A_%07d.mel' % step)
                    # save this to check loading stats
                    def save_dict(d, name):
                        with open(os.path.join(self.result_dir, self.dataset, 'img', 'visited' + name + '_%07d.txt' % step), 'w') as f:
                            for key, val in d.items():
                                if val > 0:
                                    print(key, val, file=f)
                    save_dict(self.trainA_visits, 'trainA')
                    save_dict(self.trainB_visits, 'trainB')
                    save_dict(self.noise_visits, 'noise')
                else:
                    cv2.imwrite(os.path.join(self.result_dir, self.dataset, 'img', 'A2B_%07d.png' % step), A2B * 255.0)
                    cv2.imwrite(os.path.join(self.result_dir, self.dataset, 'img', 'B2A_%07d.png' % step), B2A * 255.0)

                self.genA2B.train(), self.genB2A.train(), self.disGA.train(), self.disGB.train(), self.disLA.train(), self.disLB.train()

            if step % self.save_freq == 0:
                self.save(os.path.join(self.result_dir, self.dataset, 'model'), step)
            if step % 1000 == 0:
                self.save(self.result_dir, 'latest')

    def save_mel(self, mel, subdir, name):
        """Saves a mel spectrogram tensor in a format that can be converted to audio.

        Args:
            mel (torch.Tensor): mel spectrogram after training normalizations (needs to be un-transformed)
            subdir (str): directory under result_dir to save to
            name (str): file name
        """
        torch.save(
            unpad_mel_channels(torch.tensor(inv_mel_transform(mel)), self.mel_channels).numpy().squeeze(),
            os.path.join(self.result_dir, self.dataset, subdir, name))

    def save(self, dir, step):
        """Saves model checkpoint.

        Args:
            dir (str): checkpoint directory
            step (int): iteration number
        """
        params = {}
        params['genA2B'] = self.genA2B.state_dict()
        params['genB2A'] = self.genB2A.state_dict()
        params['disGA'] = self.disGA.state_dict()
        params['disGB'] = self.disGB.state_dict()
        params['disLA'] = self.disLA.state_dict()
        params['disLB'] = self.disLB.state_dict()
        if type(step) is int:
            torch.save(params, os.path.join(dir, self.dataset + '_params_%07d.pt' % step))
        else:
            torch.save(params, os.path.join(dir, self.dataset + '_params_%s.pt' % step))

    def load_latest_checkpoint(self):
        """Loads model from latest checkpoint of result_dir.
        """
        model_list = glob(os.path.join(self.result_dir, self.dataset, 'model', '*.pt'))
        if not len(model_list) == 0:
            model_list.sort()
            self.start_iter = int(model_list[-1].split('_')[-1].split('.')[0])
            self.load(os.path.join(self.result_dir, self.dataset, 'model'), self.start_iter)
            print(" [*] Load SUCCESS")
            if self.decay_flag and self.start_iter > (self.iteration // 2):
                self.G_optim.param_groups[0]['lr'] -= (self.gen_lr / (self.iteration // 2)) * (self.start_iter - self.iteration // 2)
                self.D_optim.param_groups[0]['lr'] -= (self.dis_lr / (self.iteration // 2)) * (self.start_iter - self.iteration // 2)

    def load(self, dir, step):
        """Loads model checkpoint.

        Args:
            dir (str): checkpoint directory
            step (int): iteration number
        """
        params = torch.load(os.path.join(dir, self.dataset + '_params_%07d.pt' % step))
        self.genA2B.load_state_dict(params['genA2B'])
        self.genB2A.load_state_dict(params['genB2A'])
        self.disGA.load_state_dict(params['disGA'])
        self.disGB.load_state_dict(params['disGB'])
        self.disLA.load_state_dict(params['disLA'])
        self.disLB.load_state_dict(params['disLB'])

    def test(self):
        """Generates samples without training.
        """
        self.genA2B.eval(), self.genB2A.eval()
        if self.mel_spectrogram:
            real_A_out, fake_A2A_out, fake_A2B_out, fake_A2B2A_out = None, None, None, None
        for n, (real_A, _) in enumerate(self.testA_loader):
            print('Generating A2B sample ', n)
            real_A = real_A.to(self.device)
            fake_A2B, _, fake_A2B_heatmap = self.genA2B(real_A)
            fake_A2B2A, _, fake_A2B2A_heatmap = self.genB2A(fake_A2B)
            fake_A2A, _, fake_A2A_heatmap = self.genB2A(real_A)

            if self.mel_spectrogram:
                real_A_out = join(real_A_out, real_A.detach().cpu(), self.img_size - self.test_stride, interpolate=self.test_interpolate)
                fake_A2A_out = join(fake_A2A_out, fake_A2A.detach().cpu(), self.img_size - self.test_stride, interpolate=self.test_interpolate)
                fake_A2B_out = join(fake_A2B_out, fake_A2B.detach().cpu(), self.img_size - self.test_stride, interpolate=self.test_interpolate)
                fake_A2B2A_out = join(fake_A2B2A_out, fake_A2B2A.detach().cpu(), self.img_size - self.test_stride, interpolate=self.test_interpolate)
            else:
                A2B = np.concatenate((RGB2BGR(tensor2numpy(denorm(real_A[0]))),
                                  cam(tensor2numpy(fake_A2A_heatmap[0]), self.img_size),
                                  RGB2BGR(tensor2numpy(denorm(fake_A2A[0]))),
                                  cam(tensor2numpy(fake_A2B_heatmap[0]), self.img_size),
                                  RGB2BGR(tensor2numpy(denorm(fake_A2B[0]))),
                                  cam(tensor2numpy(fake_A2B2A_heatmap[0]), self.img_size),
                                  RGB2BGR(tensor2numpy(denorm(fake_A2B2A[0])))), 0)
                cv2.imwrite(os.path.join(self.result_dir, self.dataset, 'test', 'A2B_%d.png' % (n + 1)), A2B * 255.0)
        if self.mel_spectrogram:
            self.save_mel(real_A_out, 'test', 'real_A_%d.mel' % n)
            self.save_mel(fake_A2A_out, 'test', 'fake_A2A_%d.mel' % n)
            self.save_mel(fake_A2B_out, 'test', 'fake_A2B_%d.mel' % n)
            self.save_mel(fake_A2B2A_out, 'test', 'fake_A2B2A_%d.mel' % n)

        if self.mel_spectrogram:
            real_B_out, fake_B2B_out, fake_B2A_out, fake_B2A2B_out = None, None, None, None
        for n, (real_B, _) in enumerate(self.testB_loader):
            print('Generating B2A sample ', n)
            real_B = real_B.to(self.device)
            fake_B2A, _, fake_B2A_heatmap = self.genB2A(real_B)
            fake_B2A2B, _, fake_B2A2B_heatmap = self.genA2B(fake_B2A)
            fake_B2B, _, fake_B2B_heatmap = self.genA2B(real_B)


            if self.mel_spectrogram:
                real_B_out = join(real_B_out, real_B.detach().cpu(), self.img_size - self.test_stride, interpolate=self.test_interpolate)
                fake_B2B_out = join(fake_B2B_out, fake_B2B.detach().cpu(), self.img_size - self.test_stride, interpolate=self.test_interpolate)
                fake_B2A_out = join(fake_B2A_out, fake_B2A.detach().cpu(), self.img_size - self.test_stride, interpolate=self.test_interpolate)
                fake_B2A2B_out = join(fake_B2A2B_out, fake_B2A2B.detach().cpu(), self.img_size - self.test_stride, interpolate=self.test_interpolate)
            else:
                B2A = np.concatenate((RGB2BGR(tensor2numpy(denorm(real_B[0]))),
                                  cam(tensor2numpy(fake_B2B_heatmap[0]), self.img_size),
                                  RGB2BGR(tensor2numpy(denorm(fake_B2B[0]))),
                                  cam(tensor2numpy(fake_B2A_heatmap[0]), self.img_size),
                                  RGB2BGR(tensor2numpy(denorm(fake_B2A[0]))),
                                  cam(tensor2numpy(fake_B2A2B_heatmap[0]), self.img_size),
                                  RGB2BGR(tensor2numpy(denorm(fake_B2A2B[0])))), 0)
                cv2.imwrite(os.path.join(self.result_dir, self.dataset, 'test', 'B2A_%d.png' % (n + 1)), B2A * 255.0)
        if self.mel_spectrogram:
            self.save_mel(real_B_out, 'test', 'real_B_%d.mel' % n)
            self.save_mel(fake_B2B_out, 'test', 'fake_B2B_%d.mel' % n)
            self.save_mel(fake_B2A_out, 'test', 'fake_B2A_%d.mel' % n)
            self.save_mel(fake_B2A2B_out, 'test', 'fake_B2A2B_%d.mel' % n)

    def noise_test(self, n_samples):
        """Experimental, gather statistical properties of sample + noise spectrograms (mean and variance)
        to see if discriminators are using differences in volume as a "shortcut".
        """
        def stats(mel):
            std, mean = torch.std_mean(mel)
            logits = self.get_dis(mel, self.dis_noise_B, is_A=True)
            prob = 0.
            for logit in logits:
                prob += torch.mean(logit).item()
            prob = prob / 4.
            return prob, mean.item(), std.item()

        with torch.no_grad():
            cols = []
            self.use_noise = False
            for _ in range(n_samples):
                try:
                    real_B, (key_B, coord_B) = next(self.trainB_iter)
                except:
                    self.trainB_iter = iter(self.trainB_loader)
                    real_B, (key_B, coord_B) = next(self.trainB_iter)

                real_B = real_B.to(self.device)
                fake_B2A, _, _ = self.genB2A(real_B)

                rows = []
                for _ in range(n_samples):
                    try:
                        self.noise_B, (key_NB, coord_NB) = next(self.noise_iter)
                    except:
                        self.noise_iter = iter(self.noise_loader)
                        self.noise_B, (key_NB, coord_NB) = next(self.noise_iter)
                    self.noise_B = self.noise_B.to(self.device)
                    real_B_noisy = self.noisy(real_B, self.gen_noise_B, False)

                    noisy_fake_B2A = self.noisy(fake_B2A, self.gen_noise_B, False)
                    fake_B2A_noisy, _, _ = self.genB2A(real_B_noisy)
                    # mels = np.concatenate((real_B.detach().cpu(),
                    #                     fake_B2A.detach().cpu(),
                    #                     noisy_fake_B2A.detach().cpu(),
                    #                     fake_B2A_noisy.detach().cpu()), 3)
                    # self.save_mel(mels, '', 'mel_{}_{}_{}_{}'.format(key_B[0], coord_B.item(), key_NB[0], coord_NB.item()))

                    rows.append([
                        key_NB[0], coord_NB.item(),
                        *torch.std_mean(self.noise_B),
                        *stats(real_B),
                        *stats(fake_B2A),
                        *stats(noisy_fake_B2A),
                        *stats(fake_B2A_noisy),
                        ])
                cols.append([
                    key_B[0], coord_B.item(),
                    rows,
                    ])
            torch.save(cols, os.path.join(self.result_dir, self.dataset, 'NOISE_test_result_table.pickle'))
            return cols

    def run_dis_only(self):
        """Use discriminator to classify new samples as A/B or noise.
        May help in finding more training samples.
        """
        rows = []
        base_dir = os.path.join('dataset', 'classify')
        for subdir in os.listdir(base_dir):
            data = MelFolder(os.path.join(base_dir, subdir), self.base_transform)
            dataloader = SequentialMelLoader(data, self.img_size, stride=self.test_stride)
            strip_before_name = len(os.path.join(base_dir, subdir))
            strip_after_name = len('.mel')
            for n, (sample, location) in enumerate(dataloader):
                logit_tensors = self.get_dis(sample.to(self.device), 0., is_A=True)
                name, position = location
                name = name[strip_before_name: - strip_after_name]
                if n % self.print_freq == 0:
                    print(n, location, summarize(*logit_tensors, include_shape=True))
                row = [subdir, name, position]
                for logit in logit_tensors:
                    row += [torch.min(logit).item(), torch.mean(logit).item(), torch.max(logit).item()]
                rows.append(row)
            df = pd.DataFrame(rows, columns=['subdir', 'name', 'position',
                        'GA_logit_min', 'GA_logit_mean', 'GA_logit_max',
                        'GA_cam_logit_min', 'GA_cam_logit_mean', 'GA_cam_logit_max',
                        'LA_logit_min', 'LA_logit_mean', 'LA_logit_max',
                        'LA_cam_logit_min', 'LA_cam_logit_mean', 'LA_cam_logit_max'])
        df.to_csv(os.path.join(self.result_dir, 'ugatit-dis-outputs.csv'))

    # following are 3 methods for generating contiguous sequences of audio
    def inject_gen(self):
        """Injects a piece of audio from previously generated sample
        to subsequent sample, to encourage consistent sounding output.
        """
        fname = 'dataset/cohen160-with-interview/testB/motp-verses-padded.mel'
        seed_pos = 4750
        stride = self.img_size // 4
        mel = torch.load(fname)
        source = mel_transform(mel).unsqueeze(0).unsqueeze(0)
        print(source.shape)
        output = torch.zeros_like(source)
        cur_img = source[:,:,:, seed_pos:seed_pos + self.img_size]
        cur_pos = seed_pos
        while cur_pos < source.shape[-1] - self.img_size * 2:
            print(cur_pos, cur_img.shape)
            with torch.no_grad():
                gen_img, _, _ = self.genB2A(cur_img.to(self.device))
                gen_img = gen_img.cpu()[:,:,:, stride:3*stride]
            output[:,:,:, cur_pos+stride:cur_pos+3*stride] += gen_img  # these overlap
            cur_img[:,:,:, :2*stride] = gen_img
            cur_img[:,:,:, 2*stride:] = source[:,:,:, cur_pos+3*stride:cur_pos+5*stride]
            cur_pos += stride
        
        final_out = inv_mel_transform(output / 2).squeeze()  # averaged over overlapping segments
        torch.save(final_out, 'results/cohen160-with-interview/inject-gen-motp.mel')

    def staggered_gen(self):
        """Generates overlapping samples, averages via linear ramp in overlaps.
        """
        fname = 'dataset/cohen160-with-interview/testB/motp-chorus.mel'
        n_overlaps = 4
        stride = self.img_size // n_overlaps
        mel = torch.load(fname)
        source = mel_transform(mel).unsqueeze(0).unsqueeze(0)
        print(source.shape)
        output = torch.zeros_like(source)
        half_size = self.img_size // 2
        ramp = torch.arange(0, half_size, dtype=torch.float).expand(1,1,self.mel_channels, -1) / half_size
        linear_ramp = torch.zeros(1, 1, self.mel_channels, self.img_size)
        linear_ramp[:,:,:, :half_size] = ramp
        linear_ramp[:,:,:, half_size:] = ramp.flip(3)
        print(linear_ramp)
        for i in range(0, source.shape[-1] - 160, stride):
            with torch.no_grad():
                gen_img, _, _ = self.genB2A(source[:,:,:,i:i+self.img_size].to(self.device))
                gen_img = gen_img.cpu()
            output[:,:,:,i:i+self.img_size] += torch.exp(inv_mel_transform(gen_img)) * linear_ramp

        # averaged over overlapping segments
        final_out = torch.log(output / n_overlaps).squeeze()
        torch.save(final_out, 'results/cohen160-with-interview/staggered-gen-motp.mel')

    def seeded_gen(self):
        """Injects a separate piece of constant audio
        to every sample, to encourage consistent sounding output.
        """
        fname = 'dataset/cohen160-with-interview/testB/motp-chorus.mel'
        # seed_pos = 5100  # 0:59
        seed_pos = 5100 - 3380 -160  # 0:18
        n_overlaps = 2
        seed_size = 40
        gen_size = self.img_size - seed_size
        stride = gen_size // n_overlaps
        mel = torch.load(fname)
        source = mel_transform(mel).unsqueeze(0).unsqueeze(0)
        print(source.shape)
        output = torch.zeros_like(source)
        seed_img = source[:,:,:, seed_pos:seed_pos+self.img_size]  #include this at the end of each clip
        torch.save(inv_mel_transform(seed_img).squeeze(), 'results/cohen160-with-interview/seed_img.mel')

        half_size = gen_size // 2
        ramp = torch.arange(0, half_size, dtype=torch.float).expand(1,1,self.mel_channels, -1) / half_size
        linear_ramp = torch.zeros(1, 1, self.mel_channels, gen_size)
        linear_ramp[:,:,:, :half_size] = ramp
        linear_ramp[:,:,:, half_size:] = ramp.flip(3)
        print(linear_ramp)
        for i in range(0, source.shape[-1] - self.img_size, stride):
            with torch.no_grad():
                img_in = torch.zeros_like(seed_img)
                img_in[:,:,:, :gen_size] = source[:,:,:, i:i + gen_size]
                img_in[:,:,:, gen_size:] = seed_img[:,:,:, self.img_size // 2 - seed_size // 2:self.img_size // 2 + seed_size // 2]
                gen_img, _, _ = self.genB2A(img_in.to(self.device))
                gen_img = gen_img.cpu()[:,:,:, :gen_size]  # discard the seed
            output[:,:,:,i:i+gen_size] += torch.exp(inv_mel_transform(gen_img)) * linear_ramp

        # averaged over overlapping segments
        final_out = torch.log(output / n_overlaps).squeeze()
        torch.save(final_out, 'results/cohen160-with-interview/seeded-gen-motp.mel')
