import math
import random
import python_speech_features
import os
import glob

import numpy as np
import warnings
from scipy.io import wavfile
from matplotlib import pyplot as plt

MAX_NUM_WAVS_PER_CLASS = 2**27 - 1  # ~134M
NOISE_DIR_NAME = 'noise'
SPEECH_DIR_NAME = 'clean_speech'
RANDOM_SEED = 59185


def get_spectrum(wav, win_len=320, win_shift=160, nDFT=320, win_fun=np.hamming):

    wav_np = np.array(wav).flatten()
    wav_np = np.reshape(wav_np, [len(wav_np)])

    # convert int32 to int
    win_len = int(win_len)
    win_shift = int(win_shift)
    nDFT = int(nDFT)

    wav_frame = python_speech_features.sigproc.framesig(sig=wav_np,
                                                        frame_len=win_len,
                                                        frame_step=win_shift,
                                                        winfunc=win_fun)
    wav_fft = np.empty([wav_frame.shape[0], int(win_len / 2 + 1)], dtype=complex)
    for frame in range(wav_frame.shape[0]):
        wav_fft[frame] = np.fft.rfft(a=wav_frame[frame], n=nDFT)
    mag_spectrum = np.abs(wav_fft)
    phase_spectrum = np.arctan2(wav_fft.imag, wav_fft.real)

    return mag_spectrum, phase_spectrum, wav_fft.real, wav_fft.imag


def get_reverberant_sig(anechoic_wav, rir_wav, fs=16e3, max_len_in_sec=None):
    reverberant = np.convolve(anechoic_wav, rir_wav, mode='full')
    if max_len_in_sec is not None:
        max_len = int(max_len_in_sec * fs)
    else:
        max_len = len(reverberant)
    return reverberant[:max_len]


def get_direction_label(filename, num_doa_class, doa_interval, how_many_frames):
    filename = os.path.basename(filename)
    doa = int(filename.split('angle_')[1].split('deg')[0])
    min_doa, max_doa = doa_interval
    num_doa = max_doa - min_doa + 1
    doa_per_class = math.ceil(num_doa/num_doa_class)
    doa_label = np.zeros([how_many_frames, num_doa_class])
    label = math.floor((doa-min_doa)/doa_per_class)
    doa_label[:, label] = 1
    return doa_label


def get_input_specs(anechoic_filename, rir_filename, win_len, win_shift, nDFT, context_window_width, max_len_in_sec=None):
    # read raw wav
    fs, anechoic_wav = wavfile.read(anechoic_filename)
    if fs != 16e3:
        raise Exception('sample rate is not 16k Hz for ' + anechoic_filename)

    anechoic_wav = np.transpose(anechoic_wav / (2 ** 15 - 1))
    if max_len_in_sec is not None and len(anechoic_wav) > max_len_in_sec*fs:
        rand_start_idx = random.randint(0, len(anechoic_wav)-max_len_in_sec*fs)
        anechoic_wav = anechoic_wav[rand_start_idx: rand_start_idx + max_len_in_sec * fs - 1]

    # read rir wav
    fs, rir_wav = wavfile.read(rir_filename)
    if fs != 16e3:
        raise Exception('sample rate is not 16k Hz for ' + rir_filename)
    rir_wav = np.transpose(rir_wav / (2**15-1))

    reverb_wav = list()
    for channel in range(2):
        # for every raw channel, generate reverberant signal
        reverb_wav.append(get_reverberant_sig(anechoic_wav, rir_wav[channel, :], max_len_in_sec))
        # get phase specs from single-channel signal

    reverb_wav = np.array(reverb_wav)
    reverb_wav = reverb_wav / np.max(np.abs(reverb_wav))

    reverb_phase_specs = list()
    for channel in range(2):
        # get phase specs from single-channel signal
        _, _, real_spec, imag_spec = get_spectrum(reverb_wav[channel], win_len, win_shift, nDFT)
        reverb_phase_specs.append(real_spec)
        reverb_phase_specs.append(imag_spec)
    reverb_phase_specs = np.array(reverb_phase_specs)

    _, num_frames, frame_len = reverb_phase_specs.shape
    input_specs = np.zeros([num_frames-context_window_width+1, context_window_width, frame_len, 4])
    for channel in range(4):
        input_specs[:, :, :, channel] = window_spec(reverb_phase_specs[channel], context_window_width)

    return reverb_wav, input_specs


def window_spec(spec, context_window_width):
    num_frames, frame_len = spec.shape
    if num_frames <= context_window_width:
        raise Exception('spec width is smaller than context_window_width')
    windowed_spec = np.zeros([num_frames - context_window_width + 1, context_window_width, frame_len])
    for i in range(context_window_width, num_frames):
        windowed_spec[i-context_window_width, :, :] = spec[i-context_window_width:i, :]
    return windowed_spec


def gen_moving_direct_wav(wav_dir, rir_dir, doa_interval, deg_per_sec, reverb_percent, room_index):

    wav_file_list = glob.glob(os.path.join(wav_dir, "*.wav"))
    min_doa, max_doa = doa_interval
    wav_len_per_deg = int(1 / deg_per_sec * 16e3)
    total_wav_len = (max_doa - min_doa + 1) * wav_len_per_deg
    moving_wav = np.zeros([2, 1])

    wav_file_idx = 0
    anechoic_wav = np.zeros([1, ])

    while len(anechoic_wav) < total_wav_len:
        fs, append_wav = wavfile.read(wav_file_list[wav_file_idx])

        if fs != 16e3:
            raise Exception('fs other than 16k Hz is not supported')

        append_wav = np.transpose(append_wav / (2 ** 15 - 1))
        wav_file_idx = (wav_file_idx + 1) % len(wav_file_list)
        anechoic_wav = np.concatenate([anechoic_wav, append_wav])

    for deg in range(min_doa, max_doa+1):
        rir_filename = 'impulse_reponses_XY_angle_' + str(deg) + 'deg_reverb_' + str(reverb_percent) + 'Percent_ROOM' + str(room_index) + '.wav'
        fs, rir_wav = wavfile.read(os.path.join(rir_dir, rir_filename))
        rir_wav = np.transpose(rir_wav / (2 ** 15 - 1))

        if fs != 16e3:
            raise Exception('fs other than 16k Hz is not supported')

        # generate reverbs for two channels
        start_idx = (deg - min_doa) * wav_len_per_deg
        end_idx = (deg - min_doa + 1) * wav_len_per_deg
        raw_wav = anechoic_wav[start_idx:end_idx]

        reverb_chan_1 = get_reverberant_sig(raw_wav, rir_wav[0, :], max_len_in_sec=1/deg_per_sec)
        reverb_chan_2 = get_reverberant_sig(raw_wav, rir_wav[1, :], max_len_in_sec=1/deg_per_sec)

        moving_wav = np.concatenate([moving_wav, np.array([reverb_chan_1, reverb_chan_2])], axis=1)

    moving_wav = moving_wav / np.max(np.abs(moving_wav))

    return moving_wav


def get_reverb_specs(reverb_wav, win_len, win_shift, nDFT, context_window_width):

    reverb_phase_specs = list()
    for channel in range(2):
        # get phase specs from single-channel signal
        _, _, real_spec, imag_spec = get_spectrum(reverb_wav[channel], win_len, win_shift, nDFT)
        reverb_phase_specs.append(real_spec)
        reverb_phase_specs.append(imag_spec)
    reverb_phase_specs = np.array(reverb_phase_specs)

    _, num_frames, frame_len = reverb_phase_specs.shape
    input_specs = np.zeros([num_frames - context_window_width + 1, context_window_width, frame_len, 4])
    for channel in range(4):
        input_specs[:, :, :, channel] = window_spec(reverb_phase_specs[channel], context_window_width)

    return input_specs


def get_moving_wav_labels(num_frames, win_shift, deg_per_sec, doa_interval):
    fs = 16e3
    min_doa, max_doa = doa_interval
    label = np.zeros([num_frames, max_doa - min_doa + 1])
    idx = np.zeros([num_frames, 1])
    len_per_deg = int(fs / deg_per_sec)

    for i in range(num_frames):
        start_time_idx = i * win_shift
        deg_idx = math.floor(start_time_idx / len_per_deg)
        label[i, deg_idx] = 1
        idx[i] = deg_idx
    pass

    return label, idx


def get_dual_channel_voiced_idx(dual_channel_wav, win_len, win_shift, nDFT, context_window_width, rms_thre):

    xy_spec = get_reverb_specs(dual_channel_wav, win_len, win_shift, nDFT, context_window_width)
    x_spec = xy_spec[:, :, :, :2]
    y_spec = xy_spec[:, :, :, 2:]

    x_rms = np.sqrt(np.mean(x_spec**2, axis=(1, 2, 3)))
    y_rms = np.sqrt(np.mean(y_spec**2, axis=(1, 2, 3)))

    x_rms_idx = x_rms >= rms_thre
    y_rms_idx = y_rms >= rms_thre

    voiced_idx = np.logical_and(x_rms_idx, y_rms_idx)
    voiced_percent = np.mean(voiced_idx)

    return voiced_idx, voiced_percent


if __name__ == '__main__':
    # wav_dir = './noisy_speech'
    # rir_dir = './16kHz'
    # doa_interval = [0, 135]
    # deg_per_sec = 10
    #
    # moving_wav = gen_moving_direct_wav(wav_dir, rir_dir, doa_interval, deg_per_sec)
    # specs = get_reverb_specs(moving_wav, 256, 128, 256, 5)
    # num_frames = specs.shape[0]
    # print(num_frames)
    # label, idx = get_moving_wav_labels(num_frames, 128, deg_per_sec, doa_interval)
    # plt.figure(0)
    #
    # plt.subplot(3, 1, 1)
    # plt.plot(moving_wav[0, :])
    #
    # plt.subplot(3, 1, 2)
    # plt.plot(moving_wav[1, :])
    #
    # plt.subplot(3, 1, 3)
    # plt.plot(idx)
    #
    # plt.show()

    # wavfile.write(filename='./input_test.wav', data=np.transpose(moving_wav), rate=16000)

    # print(moving_wav.shape)

    wav_path = './test1_doa_16kHz/audio_test1_reverb_0Percent_ROOM1.wav'

    fs, reverb_wav = wavfile.read(wav_path)
    reverb_wav = np.transpose(reverb_wav / (2 ** 15 - 1))
    duration = reverb_wav.shape[1] / 16e3

    idx, perc = get_dual_channel_voiced_idx(reverb_wav, 512, 256, 512, 13, 5e-1)
    print('percent: ', perc)
