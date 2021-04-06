import argparse
import os, os.path
import torch
import numpy as np
import soundfile as sf


def load_waveglow(args, parser):
    if not args.from_repo:
        return load_waveglow_from_hub()
    return load_waveglow_from_repo(args, parser)


def load_waveglow_from_hub():
    waveglow = torch.hub.load('nvidia/DeepLearningExamples:torchhub', 'nvidia_waveglow')
    waveglow = waveglow.remove_weightnorm(waveglow)
    waveglow = waveglow.to('cuda')
    waveglow.eval()
    return waveglow, None

def load_waveglow_from_repo(args, parser):
    from inference import load_and_setup_model
    from waveglow.denoiser import Denoiser

    waveglow = load_and_setup_model('WaveGlow', parser, args.waveglow,
                                    args.fp16, args.cpu, forward_is_infer=True)
    if args.denoiser:
        denoiser = None
    else:
        denoiser = Denoiser(waveglow)
        if not args.cpu:
            denoiser.cuda()
    return waveglow, denoiser

def mel2wav(waveglow, mel, chunk_size):
    # split into chunks to avoid overloading GPU memory
    mel = mel.unsqueeze(0)
    chunks = torch.split(mel, chunk_size, dim=2)
    audio = np.zeros([0])
    print('Generating chunks...')
    for i, chunk in enumerate(chunks):
        print('    {} / {}'.format(i, len(chunks)))
        with torch.no_grad():
            generated = waveglow.infer(chunk.to('cuda'))
        audio = np.concatenate((audio, generated[0].data.cpu().numpy()), axis=0)
    return audio


def mels2wavs(parser):
    args, _ = parser.parse_known_args()
    input_dir = args.input_dir
    output_dir = args.output_dir
    chunk_size = args.chunk_size
    waveglow, denoiser = load_waveglow(args, parser)
    for root, _, fnames in sorted(os.walk(input_dir)):
        for fname in fnames:
            path = os.path.join(root, fname)
            mel = torch.tensor(torch.load(path))
            if args.fp16:
                mel = mel.half()
            audio = mel2wav(waveglow, mel, chunk_size)
            out_path = os.path.join(output_dir, os.path.splitext(fname)[0] + '.wav')
            print('Writing audio to', out_path)
            save_soundfile(audio, out_path)


def save_soundfile(audio, filename, samplerate=22050):
    sf.write(filename, audio, samplerate=samplerate)


# add 2 spectrograms together, see if the result makes sense
# DEPRECIATED
def test_combine_mels(mel_long, mel_short, out_file):
    waveglow = torch.hub.load('nvidia/DeepLearningExamples:torchhub', 'nvidia_waveglow')
    waveglow = waveglow.remove_weightnorm(waveglow)
    waveglow = waveglow.to('cuda')
    waveglow.eval()
    # mel_long = torch.load('/home/devin/data/ml/test_ljspeech/wavs/darker-excerpt.mel')
    # mel_short = torch.load('/home/devin/data/ml/test_ljspeech/wavs/03-01-01-01-01-01-01.mel')
    mel_2 = torch.zeros(mel_long.shape)
    # need to exp since these are log-mel spectrograms
    mel_2[:, :mel_short.shape[1]] = torch.exp(mel_short)
    mel = torch.exp(mel_long) * 0.5 + mel_2 * 0.5
    mel = torch.log(mel).unsqueeze(0)
    with torch.no_grad():
        audio = waveglow.infer(mel.to('cuda'))
    audio_out = audio[0].data.cpu().numpy()
    sf.write(out_file, audio_out, samplerate=22050)


"""main"""
if __name__ == '__main__':
    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input_dir', type=str, required=True)
    parser.add_argument('-o', '--output_dir', type=str, required=True)
    parser.add_argument('--chunk_size', type=int, default=80)
    parser.add_argument('--waveglow', type=str, default='voclone/checkpoints/checkpoint_WaveGlow_last.pt')
    parser.add_argument('--fp16', type=bool, default=False)
    parser.add_argument('--cpu', type=bool, default=False)
    parser.add_argument('--denoiser', type=bool, default=True)
    parser.add_argument('--from_repo', type=bool, default=False)
    mels2wavs(parser)
