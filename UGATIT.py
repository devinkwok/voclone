import time, itertools
from dataset import ImageFolder, MelFolder, SequentialMelLoader
from torchvision import transforms
from torch.utils.data import DataLoader
from networks import *
from utils import *
from glob import glob


def crossfade(mel_1, mel_2, overlap, interpolate=False):
    if mel_1 is None:
        return mel_2
    if overlap < 1:
        return torch.cat((mel_1, mel_2), axis=3)
    elif interpolate:
        weights = torch.arange(overlap, dtype=torch.float) / overlap
        overlap_region = mel_1[:, :, :, -overlap:] * weights + mel_2[:, :, :, :overlap] * torch.flip(weights, [0])
    else:
        trim_1 = overlap // 2
        trim_2 = overlap - trim_1
        overlap_region = torch.cat((mel_1[:, :, :, -overlap: - trim_1], mel_2[:, :, :, trim_2:overlap]), axis=3)
    return torch.cat((mel_1[:, :, :, :-overlap], overlap_region, mel_2[:, :, :, overlap:]), axis=3)


def mel_transform(mel_spectrogram):
    mel = (mel_spectrogram + 11.) / 13. * 2. - 1.
    return mel.unsqueeze(0)


def inv_mel_transform(mel_spectrogram):
    mel = (mel_spectrogram + 1.) / 2. * 13. - 11.
    return mel.squeeze()


def print_and_pass_input(tensor, prefix='', do_print=True):
    if do_print:
        print(prefix, tensor.shape, summarize(tensor))
    return tensor


def random_crop(tensor, target_width, std_epsilon=0.2, mean_epsilon=-0.9):
    # if random crop results in uniform values, try again
    width = tensor.shape[2]
    std, mean = -1, mean_epsilon - 1
    while std < std_epsilon or mean < mean_epsilon:
        start = torch.randint(width - target_width, [1]).item()
        cropped = tensor[:,:, start:start + target_width]
        std, mean = torch.std_mean(cropped)
    return cropped


def print_weights_and_grads(module_dict):
    for name, module in module_dict.items():
        print(name, module)
        for param in module.parameters():
            print('   ', param.shape, summarize(param, include_grad=True), sep='\t\t')


def summarize(*tensors, include_grad=False):
    if include_grad:
        strings = ['<{:0.2f} [{:0.2f}/{:0.2f}] {:0.2f}> ({:0.2f} {{{:0.2f}/{:0.2f}}} {:0.2f})'.format(
                *get_stats(t), *get_stats(t.grad)) for t in tensors]
    else:
        strings = ['<{:0.2f} [{:0.2f}/{:0.2f}] {:0.2f}>'.format(
                *get_stats(t)) for t in tensors]
    return ' || '.join(strings)


def get_stats(tensor):
    std, mean = torch.std_mean(tensor)
    return torch.min(tensor).item(), mean.item(), std.item(), torch.max(tensor).item()


def wandg_stats(*modules):
    stats = [[], [], [], [], [], [], [], []]
    for m in modules:
        for p in m.parameters():
            vals = (*get_stats(p), *get_stats(p.grad))
            [arr.append(val) for val, arr in zip(vals, stats)]
    return [torch.tensor(x) for x in stats]


# adds log-mel spectrograms together at given proportion (transforms first by exp)
def add_noise(source, noise, noise_prop):
    with torch.no_grad():
        return mel_transform(torch.log(torch.exp(inv_mel_transform(source)) \
                        + torch.exp(inv_mel_transform(noise))))
        # source_prop = 1. - noise_prop
        # out = torch.log(torch.exp(inv_mel_transform(source)) * source_prop \
        #                 + torch.exp(inv_mel_transform(noise)) * noise_prop)
        # torch.save(inv_mel_transform(torch.cat([source.detach().cpu(), noise.detach().cpu(), out.detach().cpu()],
        #             axis=2)), os.path.join('results', 'debug', 'noise-debug.mel'))
        # return mel_transform(out)


class UGATIT(object) :
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
        self.mel_spectrogram = args.mel_spectrogram
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


    def build_model(self):
        """ DataLoader """
        if self.mel_spectrogram:
            train_transform = transforms.Compose([
                transforms.Lambda(lambda x: print_and_pass_input(x, prefix='before', do_print=self.print_input)),
                transforms.Lambda(mel_transform),
                transforms.Lambda(lambda x: random_crop(x, self.img_size)),
                transforms.Lambda(lambda x: print_and_pass_input(x, prefix='    after', do_print=self.print_input)),
            ])
            test_transform = transforms.Compose([
                transforms.Lambda(lambda x: print_and_pass_input(x, prefix='before', do_print=self.print_input)),
                transforms.Lambda(mel_transform),
                transforms.Lambda(lambda x: print_and_pass_input(x, prefix='    after', do_print=self.print_input)),
            ])
            self.trainA = MelFolder(os.path.join('dataset', self.dataset, 'trainA'), train_transform)
            self.trainB = MelFolder(os.path.join('dataset', self.dataset, 'trainB'), train_transform)
            self.testA = MelFolder(os.path.join('dataset', self.dataset, 'testA'), test_transform)
            self.testB = MelFolder(os.path.join('dataset', self.dataset, 'testB'), test_transform)
            self.testA_loader = SequentialMelLoader(self.testA, self.img_size, stride=self.test_stride)
            self.testB_loader = SequentialMelLoader(self.testB, self.img_size, stride=self.test_stride)
            if self.use_noise:
                self.noise_data = MelFolder(os.path.join('dataset', self.dataset, 'noise'), train_transform)
                self.noise_loader = DataLoader(self.noise_data, batch_size=self.batch_size, shuffle=True)
        else:
            train_transform = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.Resize((self.img_size + 30, self.img_size+30)),
                transforms.RandomCrop(self.img_size),
                transforms.ToTensor(),
                transforms.Lambda(lambda x: print_and_pass_input(x, prefix='before', do_print=self.print_input)),
                transforms.Normalize(mean=(0.5), std=(0.5)),
                transforms.Lambda(lambda x: print_and_pass_input(x, prefix='    after', do_print=self.print_input)),
            ])
            test_transform = transforms.Compose([
                transforms.Resize((self.img_size, self.img_size)),
                transforms.ToTensor(),
                transforms.Lambda(lambda x: print_and_pass_input(x, prefix='before', do_print=self.print_input)),
                transforms.Normalize(mean=(0.5), std=(0.5)),
                transforms.Lambda(lambda x: print_and_pass_input(x, prefix='    after', do_print=self.print_input)),
            ])
            self.trainA = ImageFolder(os.path.join('dataset', self.dataset, 'trainA'), train_transform)
            self.trainB = ImageFolder(os.path.join('dataset', self.dataset, 'trainB'), train_transform)
            self.testA = ImageFolder(os.path.join('dataset', self.dataset, 'testA'), test_transform)
            self.testB = ImageFolder(os.path.join('dataset', self.dataset, 'testB'), test_transform)
            self.testA_loader = DataLoader(self.testA, batch_size=1, shuffle=False)
            self.testB_loader = DataLoader(self.testB, batch_size=1, shuffle=False)

        self.trainA_loader = DataLoader(self.trainA, batch_size=self.batch_size, shuffle=True)
        self.trainB_loader = DataLoader(self.trainB, batch_size=self.batch_size, shuffle=True)

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


    def train(self):
        self.genA2B.train(), self.genB2A.train(), self.disGA.train(), self.disGB.train(), self.disLA.train(), self.disLB.train()

        start_iter = 1
        if self.resume:
            model_list = glob(os.path.join(self.result_dir, self.dataset, 'model', '*.pt'))
            if not len(model_list) == 0:
                model_list.sort()
                start_iter = int(model_list[-1].split('_')[-1].split('.')[0])
                self.load(os.path.join(self.result_dir, self.dataset, 'model'), start_iter)
                print(" [*] Load SUCCESS")
                if self.decay_flag and start_iter > (self.iteration // 2):
                    self.G_optim.param_groups[0]['lr'] -= (self.lr / (self.iteration // 2)) * (start_iter - self.iteration // 2)
                    self.D_optim.param_groups[0]['lr'] -= (self.lr / (self.iteration // 2)) * (start_iter - self.iteration // 2)

        # training loop
        print('training start !')
        start_time = time.time()
        for step in range(start_iter, self.iteration + 1):
            if self.decay_flag and step > (self.iteration // 2):
                self.G_optim.param_groups[0]['lr'] -= (self.lr / (self.iteration // 2))
                self.D_optim.param_groups[0]['lr'] -= (self.lr / (self.iteration // 2))

            try:
                real_A, _ = next(trainA_iter)
            except:
                trainA_iter = iter(self.trainA_loader)
                real_A, _ = next(trainA_iter)

            try:
                real_B, _ = next(trainB_iter)
            except:
                trainB_iter = iter(self.trainB_loader)
                real_B, _ = next(trainB_iter)

            if self.use_noise:  # add noise to data samples if required
                try:
                    noise, _ = next(noise_iter)
                except:
                    noise_iter = iter(self.noise_loader)
                    noise, _ = next(noise_iter)
                if self.gen_noise_A > 0.:
                    print('gen_real_A', summarize(real_A))
                    real_A = add_noise(real_A, noise, self.gen_noise_A)
                if self.gen_noise_B > 0.:
                    real_B = add_noise(real_B, noise, self.gen_noise_B)
                    print('gen_real_B', summarize(real_B))
                noise = noise.to(self.device)
            real_A, real_B = real_A.to(self.device), real_B.to(self.device)


            # Update D
            self.D_optim.zero_grad()

            fake_A2B, _, _ = self.genA2B(real_A)
            fake_B2A, _, _ = self.genB2A(real_B)

            # if dis_noise_A > 0., this means add noise to A, or that A has no noise and B has noise
            # then A2B will have no noise while B2A will have noise, so compare A2B+noise with B, and B2A with A+noise
            # vice versa if B has no noise and A has noise
            # if both A and B have no noise, then setting both dis_noise_A and dis_noise_B > 0.
            # means compare A2B+noise with B+noise, and B2A+noise with A+noise
            if (not self.gen_noise_A > 0.) and self.dis_noise_A > 0.:
                real_A_for_dis = add_noise(real_A, noise, self.dis_noise_A)
                fake_A2B_for_dis = add_noise(fake_A2B, noise, self.dis_noise_A)
                print('dis_real_A', summarize(real_A_for_dis), 'dis_fake_A2B', summarize(fake_A2B_for_dis))
            else:
                real_A_for_dis = real_A
                fake_A2B_for_dis = fake_A2B
            if (not self.gen_noise_B > 0.) and self.dis_noise_B > 0.:
                real_B_for_dis = add_noise(real_B, noise, self.dis_noise_B)
                fake_B2A_for_dis = add_noise(fake_B2A, noise, self.dis_noise_A)
                print('dis_real_B', summarize(real_B_for_dis), 'dis_fake_B2A', summarize(fake_B2A_for_dis))
            else:
                real_B_for_dis = real_B
                fake_B2A_for_dis = fake_B2A

            real_GA_logit, real_GA_cam_logit, _ = self.disGA(real_A_for_dis)
            real_LA_logit, real_LA_cam_logit, _ = self.disLA(real_A_for_dis)
            real_GB_logit, real_GB_cam_logit, _ = self.disGB(real_B_for_dis)
            real_LB_logit, real_LB_cam_logit, _ = self.disLB(real_B_for_dis)

            fake_GA_logit, fake_GA_cam_logit, _ = self.disGA(fake_B2A_for_dis)
            fake_LA_logit, fake_LA_cam_logit, _ = self.disLA(fake_B2A_for_dis)
            fake_GB_logit, fake_GB_cam_logit, _ = self.disGB(fake_A2B_for_dis)
            fake_LB_logit, fake_LB_cam_logit, _ = self.disLB(fake_A2B_for_dis)

            D_ad_loss_GA = self.MSE_loss(real_GA_logit, torch.ones_like(real_GA_logit).to(self.device)) + self.MSE_loss(fake_GA_logit, torch.zeros_like(fake_GA_logit).to(self.device))
            D_ad_cam_loss_GA = self.MSE_loss(real_GA_cam_logit, torch.ones_like(real_GA_cam_logit).to(self.device)) + self.MSE_loss(fake_GA_cam_logit, torch.zeros_like(fake_GA_cam_logit).to(self.device))
            D_ad_loss_LA = self.MSE_loss(real_LA_logit, torch.ones_like(real_LA_logit).to(self.device)) + self.MSE_loss(fake_LA_logit, torch.zeros_like(fake_LA_logit).to(self.device))
            D_ad_cam_loss_LA = self.MSE_loss(real_LA_cam_logit, torch.ones_like(real_LA_cam_logit).to(self.device)) + self.MSE_loss(fake_LA_cam_logit, torch.zeros_like(fake_LA_cam_logit).to(self.device))
            D_ad_loss_GB = self.MSE_loss(real_GB_logit, torch.ones_like(real_GB_logit).to(self.device)) + self.MSE_loss(fake_GB_logit, torch.zeros_like(fake_GB_logit).to(self.device))
            D_ad_cam_loss_GB = self.MSE_loss(real_GB_cam_logit, torch.ones_like(real_GB_cam_logit).to(self.device)) + self.MSE_loss(fake_GB_cam_logit, torch.zeros_like(fake_GB_cam_logit).to(self.device))
            D_ad_loss_LB = self.MSE_loss(real_LB_logit, torch.ones_like(real_LB_logit).to(self.device)) + self.MSE_loss(fake_LB_logit, torch.zeros_like(fake_LB_logit).to(self.device))
            D_ad_cam_loss_LB = self.MSE_loss(real_LB_cam_logit, torch.ones_like(real_LB_cam_logit).to(self.device)) + self.MSE_loss(fake_LB_cam_logit, torch.zeros_like(fake_LB_cam_logit).to(self.device))

            D_loss_A = self.adv_weight * (D_ad_loss_GA + D_ad_cam_loss_GA + D_ad_loss_LA + D_ad_cam_loss_LA)
            D_loss_B = self.adv_weight * (D_ad_loss_GB + D_ad_cam_loss_GB + D_ad_loss_LB + D_ad_cam_loss_LB)

            Discriminator_loss = D_loss_A + D_loss_B
            if self.print_wandg and torch.isnan(Discriminator_loss):
                print_weights_and_grads({
                    'genA2B': self.genA2B, 'genB2A': self.genB2A, 'disGA': self.disGA, 'disLA': self.disLA, 'disGB': self.disGB, 'disLB': self.disLB,
                    })
                break
            Discriminator_loss.backward()
            self.D_optim.step()

            # Update G
            self.G_optim.zero_grad()

            fake_A2B, fake_A2B_cam_logit, _ = self.genA2B(real_A)
            fake_B2A, fake_B2A_cam_logit, _ = self.genB2A(real_B)

            # add background noise to cycle
            if self.cycle_noise_A > 0.:
                fake_A2B_for_cycle = add_noise(fake_A2B, noise, self.cycle_noise_A)
                print('fake_A2B_for_cycle', summarize(fake_A2B_for_cycle))
            else:
                fake_A2B_for_cycle = fake_A2B
            if self.cycle_noise_B > 0.:
                fake_B2A_for_cycle = add_noise(fake_B2A, noise, self.cycle_noise_B)
                print('fake_B2A_for_cycle', summarize(fake_B2A_for_cycle))
            else:
                fake_B2A_for_cycle = fake_B2A
            fake_A2B2A, _, _ = self.genB2A(fake_A2B_for_cycle)
            fake_B2A2B, _, _ = self.genA2B(fake_B2A_for_cycle)

            # add background noise to identity
            if self.identity_noise_A > 0.:
                real_A_for_id = add_noise(real_A, noise, self.identity_noise_A)
                print('real_A_for_id', summarize(real_A_for_id))
            else:
                real_A_for_id = real_A
            if self.identity_noise_B > 0.:
                real_B_for_id = add_noise(real_B, noise, self.identity_noise_B)
                print('real_B_for_id', summarize(real_B_for_id))
            else:
                real_B_for_id = real_B
            fake_A2A, fake_A2A_cam_logit, _ = self.genB2A(real_A_for_id)
            fake_B2B, fake_B2B_cam_logit, _ = self.genA2B(real_B_for_id)

            # add background noise to fake if source has none
            if (not self.gen_noise_A > 0.) and self.dis_noise_A > 0.:
                fake_A2B_for_dis = add_noise(fake_A2B, noise, self.dis_noise_A)
                print('fake_A2B_for_dis', summarize(fake_A2B_for_dis))
            else:
                fake_A2B_for_dis = fake_A2B
            if (not self.gen_noise_B > 0.) and self.dis_noise_B > 0.:
                fake_B2A_for_dis = add_noise(fake_B2A, noise, self.dis_noise_A)
                print('fake_B2A_for_dis', summarize(fake_B2A_for_dis))
            else:
                fake_B2A_for_dis = fake_B2A
            fake_GA_logit, fake_GA_cam_logit, _ = self.disGA(fake_B2A_for_dis)
            fake_LA_logit, fake_LA_cam_logit, _ = self.disLA(fake_B2A_for_dis)
            fake_GB_logit, fake_GB_cam_logit, _ = self.disGB(fake_A2B_for_dis)
            fake_LB_logit, fake_LB_cam_logit, _ = self.disLB(fake_A2B_for_dis)

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

            print("[%5d/%5d] time: %4.4f d_loss: %.8f, g_loss: %.8f" % (step, self.iteration, time.time() - start_time, Discriminator_loss, Generator_loss))
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

            if step % self.print_freq == 0:
                train_sample_num = 5
                test_sample_num = 5
                if self.mel_spectrogram:
                    A2B = torch.zeros((1, self.img_size, 0))
                    B2A = torch.zeros((1, self.img_size, 0))
                else:
                    A2B = np.zeros((self.img_size * 7, 0, self.img_ch))
                    B2A = np.zeros((self.img_size * 7, 0, self.img_ch))

                self.genA2B.eval(), self.genB2A.eval(), self.disGA.eval(), self.disGB.eval(), self.disLA.eval(), self.disLB.eval()
                for _ in range(train_sample_num):
                    try:
                        real_A, _ = next(trainA_iter)
                    except:
                        trainA_iter = iter(self.trainA_loader)
                        real_A, _ = next(trainA_iter)

                    try:
                        real_B, _ = next(trainB_iter)
                    except:
                        trainB_iter = iter(self.trainB_loader)
                        real_B, _ = next(trainB_iter)
                    real_A, real_B = real_A.to(self.device), real_B.to(self.device)

                    fake_A2B, _, fake_A2B_heatmap = self.genA2B(real_A)
                    fake_B2A, _, fake_B2A_heatmap = self.genB2A(real_B)

                    fake_A2B2A, _, fake_A2B2A_heatmap = self.genB2A(fake_A2B)
                    fake_B2A2B, _, fake_B2A2B_heatmap = self.genA2B(fake_B2A)

                    fake_A2A, _, fake_A2A_heatmap = self.genB2A(real_A)
                    fake_B2B, _, fake_B2B_heatmap = self.genA2B(real_B)

                    if self.mel_spectrogram:
                        if self.use_noise:
                            A2B = torch.cat((A2B, noise.detach().cpu()), axis=2)
                            B2A = torch.cat((B2A, noise.detach().cpu()), axis=2)
                        A2B = torch.cat((A2B, real_A[0].detach().cpu(),
                                            fake_A2A[0].detach().cpu(),
                                            fake_A2B[0].detach().cpu(),
                                            fake_A2B2A[0].detach().cpu()), axis=2)
                        B2A = torch.cat((B2A, real_B[0].detach().cpu(),
                                            fake_B2B[0].detach().cpu(),
                                            fake_B2A[0].detach().cpu(),
                                            fake_B2A2B[0].detach().cpu()), axis=2)
                    else:
                        A2B = np.concatenate((A2B, np.concatenate((RGB2BGR(tensor2numpy(denorm(real_A[0]))),
                                                            cam(tensor2numpy(fake_A2A_heatmap[0]), self.img_size),
                                                            RGB2BGR(tensor2numpy(denorm(fake_A2A[0]))),
                                                            cam(tensor2numpy(fake_A2B_heatmap[0]), self.img_size),
                                                            RGB2BGR(tensor2numpy(denorm(fake_A2B[0]))),
                                                            cam(tensor2numpy(fake_A2B2A_heatmap[0]), self.img_size),
                                                            RGB2BGR(tensor2numpy(denorm(fake_A2B2A[0])))), 0)), 1)

                        B2A = np.concatenate((B2A, np.concatenate((RGB2BGR(tensor2numpy(denorm(real_B[0]))),
                                                            cam(tensor2numpy(fake_B2B_heatmap[0]), self.img_size),
                                                            RGB2BGR(tensor2numpy(denorm(fake_B2B[0]))),
                                                            cam(tensor2numpy(fake_B2A_heatmap[0]), self.img_size),
                                                            RGB2BGR(tensor2numpy(denorm(fake_B2A[0]))),
                                                            cam(tensor2numpy(fake_B2A2B_heatmap[0]), self.img_size),
                                                            RGB2BGR(tensor2numpy(denorm(fake_B2A2B[0])))), 0)), 1)

                for _ in range(test_sample_num):
                    try:
                        real_A, _ = next(testA_iter)
                    except:
                        testA_iter = iter(self.testA_loader)
                        real_A, _ = next(testA_iter)

                    try:
                        real_B, _ = next(testB_iter)
                    except:
                        testB_iter = iter(self.testB_loader)
                        real_B, _ = next(testB_iter)
                    real_A, real_B = real_A.to(self.device), real_B.to(self.device)

                    fake_A2B, _, fake_A2B_heatmap = self.genA2B(real_A)
                    fake_B2A, _, fake_B2A_heatmap = self.genB2A(real_B)

                    fake_A2B2A, _, fake_A2B2A_heatmap = self.genB2A(fake_A2B)
                    fake_B2A2B, _, fake_B2A2B_heatmap = self.genA2B(fake_B2A)

                    fake_A2A, _, fake_A2A_heatmap = self.genB2A(real_A)
                    fake_B2B, _, fake_B2B_heatmap = self.genA2B(real_B)


                    if self.mel_spectrogram:
                        A2B = torch.cat((A2B, real_A[0].detach().cpu(),
                                            fake_A2A[0].detach().cpu(),
                                            fake_A2B[0].detach().cpu(),
                                            fake_A2B2A[0].detach().cpu()), axis=2)
                        B2A = torch.cat((B2A, real_B[0].detach().cpu(),
                                            fake_B2B[0].detach().cpu(),
                                            fake_B2A[0].detach().cpu(),
                                            fake_B2A2B[0].detach().cpu()), axis=2)
                        torch.save(inv_mel_transform(A2B), os.path.join(self.result_dir, self.dataset, 'img', 'A2B_%07d.mel' % step))
                        torch.save(inv_mel_transform(B2A), os.path.join(self.result_dir, self.dataset, 'img', 'B2A_%07d.mel' % step))
                    else:
                        A2B = np.concatenate((A2B, np.concatenate((RGB2BGR(tensor2numpy(denorm(real_A[0]))),
                                                            cam(tensor2numpy(fake_A2A_heatmap[0]), self.img_size),
                                                            RGB2BGR(tensor2numpy(denorm(fake_A2A[0]))),
                                                            cam(tensor2numpy(fake_A2B_heatmap[0]), self.img_size),
                                                            RGB2BGR(tensor2numpy(denorm(fake_A2B[0]))),
                                                            cam(tensor2numpy(fake_A2B2A_heatmap[0]), self.img_size),
                                                            RGB2BGR(tensor2numpy(denorm(fake_A2B2A[0])))), 0)), 1)

                        B2A = np.concatenate((B2A, np.concatenate((RGB2BGR(tensor2numpy(denorm(real_B[0]))),
                                                            cam(tensor2numpy(fake_B2B_heatmap[0]), self.img_size),
                                                            RGB2BGR(tensor2numpy(denorm(fake_B2B[0]))),
                                                            cam(tensor2numpy(fake_B2A_heatmap[0]), self.img_size),
                                                            RGB2BGR(tensor2numpy(denorm(fake_B2A[0]))),
                                                            cam(tensor2numpy(fake_B2A2B_heatmap[0]), self.img_size),
                                                            RGB2BGR(tensor2numpy(denorm(fake_B2A2B[0])))), 0)), 1)

                        cv2.imwrite(os.path.join(self.result_dir, self.dataset, 'img', 'A2B_%07d.png' % step), A2B * 255.0)
                        cv2.imwrite(os.path.join(self.result_dir, self.dataset, 'img', 'B2A_%07d.png' % step), B2A * 255.0)
            self.genA2B.train(), self.genB2A.train(), self.disGA.train(), self.disGB.train(), self.disLA.train(), self.disLB.train()

            if step % self.save_freq == 0:
                self.save(os.path.join(self.result_dir, self.dataset, 'model'), step)
            if step % 1000 == 0:
                self.save(self.result_dir, 'latest')

    def save(self, dir, step):
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

    def load(self, dir, step):
        params = torch.load(os.path.join(dir, self.dataset + '_params_%07d.pt' % step))
        self.genA2B.load_state_dict(params['genA2B'])
        self.genB2A.load_state_dict(params['genB2A'])
        self.disGA.load_state_dict(params['disGA'])
        self.disGB.load_state_dict(params['disGB'])
        self.disLA.load_state_dict(params['disLA'])
        self.disLB.load_state_dict(params['disLB'])

    def test(self):
        model_list = glob(os.path.join(self.result_dir, self.dataset, 'model', '*.pt'))
        if not len(model_list) == 0:
            model_list.sort()
            iter = int(model_list[-1].split('_')[-1].split('.')[0])
            self.load(os.path.join(self.result_dir, self.dataset, 'model'), iter)
            print(" [*] Load SUCCESS")
        else:
            print(" [*] Load FAILURE")
            return

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
                real_A_out = crossfade(real_A_out, real_A.detach().cpu(), self.img_size - self.test_stride, interpolate=self.test_interpolate)
                fake_A2A_out = crossfade(fake_A2A_out, fake_A2A.detach().cpu(), self.img_size - self.test_stride, interpolate=self.test_interpolate)
                fake_A2B_out = crossfade(fake_A2B_out, fake_A2B.detach().cpu(), self.img_size - self.test_stride, interpolate=self.test_interpolate)
                fake_A2B2A_out = crossfade(fake_A2B2A_out, fake_A2B2A.detach().cpu(), self.img_size - self.test_stride, interpolate=self.test_interpolate)
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
            torch.save(inv_mel_transform(real_A_out), os.path.join(self.result_dir, self.dataset, 'test', 'real_A_%d.mel' % n))
            torch.save(inv_mel_transform(fake_A2A_out), os.path.join(self.result_dir, self.dataset, 'test', 'fake_A2A_%d.mel' % n))
            torch.save(inv_mel_transform(fake_A2B_out), os.path.join(self.result_dir, self.dataset, 'test', 'fake_A2B_%d.mel' % n))
            torch.save(inv_mel_transform(fake_A2B2A_out), os.path.join(self.result_dir, self.dataset, 'test', 'fake_A2B2A_%d.mel' % n))

        if self.mel_spectrogram:
            real_B_out, fake_B2B_out, fake_B2A_out, fake_B2A2B_out = None, None, None, None
        for n, (real_B, _) in enumerate(self.testB_loader):
            print('Generating B2A sample ', n)
            real_B = real_B.to(self.device)
            fake_B2A, _, fake_B2A_heatmap = self.genB2A(real_B)
            fake_B2A2B, _, fake_B2A2B_heatmap = self.genA2B(fake_B2A)
            fake_B2B, _, fake_B2B_heatmap = self.genA2B(real_B)


            if self.mel_spectrogram:
                real_B_out = crossfade(real_B_out, real_B.detach().cpu(), self.img_size - self.test_stride, interpolate=self.test_interpolate)
                fake_B2B_out = crossfade(fake_B2B_out, fake_B2B.detach().cpu(), self.img_size - self.test_stride, interpolate=self.test_interpolate)
                fake_B2A_out = crossfade(fake_B2A_out, fake_B2A.detach().cpu(), self.img_size - self.test_stride, interpolate=self.test_interpolate)
                fake_B2A2B_out = crossfade(fake_B2A2B_out, fake_B2A2B.detach().cpu(), self.img_size - self.test_stride, interpolate=self.test_interpolate)
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
            torch.save(inv_mel_transform(real_B_out), os.path.join(self.result_dir, self.dataset, 'test', 'real_B_%d.mel' % n))
            torch.save(inv_mel_transform(fake_B2B_out), os.path.join(self.result_dir, self.dataset, 'test', 'fake_B2B_%d.mel' % n))
            torch.save(inv_mel_transform(fake_B2A_out), os.path.join(self.result_dir, self.dataset, 'test', 'fake_B2A_%d.mel' % n))
            torch.save(inv_mel_transform(fake_B2A2B_out), os.path.join(self.result_dir, self.dataset, 'test', 'fake_B2A2B_%d.mel' % n))
