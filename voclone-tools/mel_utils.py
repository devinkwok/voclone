import math
import torch
import torchvision.transforms.functional as tF
import torch.nn.functional as F

def augment_volume(mel, w_sin_freq, h_sin_freq, w_sin_amp, h_sin_amp, sp_amp, max_change=1.15, min_change=0.5, epsilon=0.1):
    if w_sin_freq <= 0 and w_sin_amp <= 0 and h_sin_freq <= 0 and h_sin_amp <= 0 and sp_amp <= 0:
        return mel

    with torch.no_grad():
        width = mel.shape[-1]
        height = mel.shape[-2]

        factor = torch.ones_like(mel)
        time_freq, time_offset, ch_freq, ch_offset, time_weight, ch_weight = torch.rand([6]).numpy()
        if w_sin_freq > 0 and w_sin_amp > 0:
            time_freq = time_freq * 2 * math.pi * w_sin_freq
            time_offset = time_offset * 2 * math.pi
            time_weight = time_weight * w_sin_amp
            time_sine = torch.sin(torch.arange(width, dtype=torch.float
                                ) / width * time_freq + time_offset)
            time_sine = time_sine.unsqueeze(0).expand(height, -1) * time_weight
            factor = factor + time_sine
        
        if h_sin_freq > 0 and h_sin_amp > 0:
            ch_freq = ch_freq * 2 * math.pi * h_sin_freq
            ch_offset = time_offset * 2 * math.pi
            ch_weight = ch_weight * h_sin_amp
            ch_sine = torch.sin(torch.arange(height, dtype=torch.float
                                    ) / height * ch_freq + ch_offset)
            ch_sine = ch_sine.unsqueeze(1).expand(-1, width) * ch_weight
            factor = factor + ch_sine

        if sp_amp > 0:
            sp_noise = torch.randn_like(mel) * sp_amp
            factor = factor + sp_noise

        floor = torch.min(mel)  # keep floor fixed
        ceil = torch.max(mel)
        ceil = torch.max(mel)
        max_factor = (1. - epsilon - floor) / (ceil - floor)
        torch.clamp(factor, min_change, min(max_factor, max_change))
        return (mel - floor) * factor + floor

def stretch_time_and_translate_ch(mel, target_w, target_h, w_stretch_max, h_translate_max, pad_val=0):
    if w_stretch_max <= 0 and h_translate_max <= 0:
        return mel
    with torch.no_grad():
        width = mel.shape[-1]
        height = mel.shape[-2]
        mel = F.pad(mel, pad=(0, 0, h_translate_max, h_translate_max), mode='constant', value=pad_val)

        resized_width = target_w
        if w_stretch_max > 0:
            resized_width = torch.randint(width - w_stretch_max, width + w_stretch_max, [1]).item()

        h_crop_top = 0
        if h_translate_max > 0:
            h_crop_top = torch.randint(0, h_translate_max * 2, [1]).item()

        w_crop_left = 0
        if resized_width - target_w > 0:
            w_crop_left = torch.randint(0, resized_width - target_w, [1]).item()
        mel = F.interpolate(mel.unsqueeze(0), size=(height + h_translate_max * 2, resized_width))
        mel = mel.squeeze(dim=0)[..., h_crop_top:h_crop_top+target_h, w_crop_left:w_crop_left+target_w]
        return mel

def join(mel_1, mel_2, overlap, interpolate=False):
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


def add_mels(mel_1, mel_2):
    return mel_transform(torch.log(torch.exp(inv_mel_transform(mel_1)) \
                    + torch.exp(inv_mel_transform(mel_2))))

# since this is a log-linear scaling, avoid extreme factors > 0.2 from 1
def scale_volume(mel, factor_min, factor_max, epsilon=0.1):
    if factor_max - factor_min == 0:
        return mel
    with torch.no_grad():
        floor = torch.min(mel)  # keep floor fixed
        ceil = torch.max(mel)
        max_factor = (1. - epsilon - floor) / (ceil - floor)
        factor = torch.rand([1]).item() * (factor_max - factor_min) + factor_min
        # print("         scaling", factor_min, factor_max, max_factor, factor)
        factor = min(max_factor, factor)
        return (mel - floor) * factor + floor

def mel_transform_new(mel_spectrogram):
    mel = (mel_spectrogram + 11.) / 13.
    return torch.clamp(mel, 0, 1)


def inv_mel_transform_new(mel_spectrogram):
    mel = (mel_spectrogram) * 13. - 11.
    return mel

def mel_transform(mel_spectrogram):
    mel = (mel_spectrogram + 11.) / 13. * 2. - 1.
    return torch.clamp(mel, -1, 1)


def inv_mel_transform(mel_spectrogram):
    mel = (mel_spectrogram + 1.) / 2. * 13. - 11.
    return mel

# assumes 3 dimensions
def random_crop(tensor, target_width, std_epsilon=0.2, mean_epsilon=-0.9):
    # if random crop results in uniform values, try again
    width = tensor.shape[2]
    std, mean = -1, mean_epsilon - 1
    if width - target_width <= 0:
        return F.pad(tensor, pad=(0, target_width - width), mode='constant', value=0)
    while std < std_epsilon or mean < mean_epsilon:
        start = torch.randint(width - target_width, [1]).item()
        cropped = tensor[:,:, start:start + target_width]
        std, mean = torch.std_mean(cropped)
    return cropped

# assumes last 2 dimensions are H, W
def pad_mel_channels(tensor, img_size):
    mel_channels = tensor.shape[-2]
    pad_bottom = (img_size - mel_channels) // 2
    pad_top = img_size - mel_channels - pad_bottom
    return F.pad(tensor, (0, 0, pad_bottom, pad_top))

# assumes last 2 dimensions are H, W
def unpad_mel_channels(tensor, mel_channels):
    img_size = tensor.shape[-2]
    pad_bottom = (img_size - mel_channels) // 2
    return torch.narrow(tensor, -2, pad_bottom, mel_channels)

# assumes last 2 dimensions are H, W
def strip_padded(tensor, mel_channels):
    img_size = tensor.shape[-2]
    stripped = unpad_mel_channels(tensor, mel_channels)
    return pad_mel_channels(stripped, img_size)

def noise_normalizer(mel, noise_A, noise_B, margin=0.03, scale_dist=0.15):
    with torch.no_grad():
        mel_mean = torch.mean(mel)
        a_mean = torch.mean(noise_A)
        b_mean = torch.mean(noise_B)
        noise = noise_A
        before_diff = (torch.mean(mel)- torch.mean(noise)).item()
        mean = a_mean
        if mean >= mel_mean - margin:
            noise = noise_B
            mean = b_mean
        if mean >= mel_mean - margin:
            mel = scale_volume(mel, 1., 1. + scale_dist)
            mel_mean = torch.mean(mel)
        if mean >= mel_mean - margin:
            noise = scale_volume(noise, 1. - scale_dist, 1.)
        print("    noise_normalizer before %.2f after %.2f" % (before_diff, (torch.mean(mel) - torch.mean(noise)).item()))
        return mel, noise

def print_and_summarize(*tensors, prefix='', do_print=True):
    if do_print:
        for tensor in tensors:
            print(prefix, tensor.shape, summarize(tensor))
    return tensor


def get_stats(tensor):
    std, mean = torch.std_mean(tensor)
    return torch.min(tensor).item(), mean.item(), std.item(), torch.max(tensor).item()


def summarize(*tensors, include_grad=False, include_shape=False):
    if include_grad:
        strings = ['<{:0.2f} [{:0.2f}/{:0.2f}] {:0.2f}> ({:0.2f} {{{:0.2f}/{:0.2f}}} {:0.2f})'.format(
                *get_stats(t), *get_stats(t.grad)) for t in tensors]
    else:
        if include_shape:
            strings = ['{} <{:0.2f} [{:0.2f}/{:0.2f}] {:0.2f}>'.format(
                    t.detach().cpu().numpy().shape, *get_stats(t)) for t in tensors]
        else:
            strings = ['<{:0.2f} [{:0.2f}/{:0.2f}] {:0.2f}>'.format(
                    *get_stats(t)) for t in tensors]
        
    return ' || '.join(strings)


def wandg_stats(*modules):
    stats = [[], [], [], [], [], [], [], []]
    for m in modules:
        for p in m.parameters():
            vals = (*get_stats(p), *get_stats(p.grad))
            [arr.append(val) for val, arr in zip(vals, stats)]
    return [torch.tensor(x) for x in stats]

# check callbacks on batch end and on epoch end
def print_weights_and_grads(module_dict):
    for name, module in module_dict.items():
        print(name, module)
        for param in module.parameters():
            print('   ', param.shape, summarize(param, include_grad=True), sep='\t\t')
