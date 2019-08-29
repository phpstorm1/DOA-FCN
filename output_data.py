import numpy as np
import python_speech_features as pysp
import scipy.io.wavfile
import os


def reshape_out(spectrum, model_settings):
    spectrum = np.array(spectrum)
    nDFT = int(model_settings['nDFT'] / 2) + 1
    num_frames = int(model_settings['spectrogram_length'])
    num_files = int(spectrum.size / (nDFT*num_frames))
    reshape_spectrum = np.reshape(spectrum.flatten(), [num_files, num_frames, nDFT])
    return reshape_spectrum


def rec_wav(mag_spectrum, phase_spectrum=None, second_mag_spectrum=None, win_len=320, win_shift=160, nDFT=320, win_fun=np.hamming):

    mag = np.array(mag_spectrum)

    if phase_spectrum is not None:
        phase = np.array(phase_spectrum)
        if mag.shape != phase.shape:
            raise Exception("The shape of mag_spectrum and phase_spectrum doesn't match")
        rec_fft = np.multiply(mag, np.exp(1j*phase))
    elif second_mag_spectrum is not None:
        fft_imag = np.array(second_mag_spectrum)
        if mag.shape != fft_imag.shape:
            raise Exception("The shape of mag_spectrum and additional_mag_spectrum doesn't match")
        rec_fft = mag_spectrum + 1.0j * fft_imag
    else:
        raise Exception("Invalid input: both phase_spectrum and additional_mag_spectrym are missing")

    wav_ifft = np.fft.irfft(a=rec_fft, n=nDFT, axis=1)
    wav_ifft = wav_ifft[:, :win_len]

    wav_deframe = pysp.sigproc.deframesig(frames=wav_ifft,
                                          siglen=0,
                                          frame_len=win_len,
                                          frame_step=win_shift,
                                          winfunc=win_fun
                                          )

    # set first frame and last frame to zeros to get rid of the impulse which seems to be caused by STFT
    wav_deframe[0:win_len] = 0
    wav_deframe[-win_len:] = 0

    check_nan = np.isnan(wav_deframe)
    for elem in check_nan:
        if elem:
            raise Exception("Error: NaN in wav_deframe")
    if np.max(abs(wav_deframe)) == 0:
        raise Exception("Error: zeros array for wav_deframe")
    wav_deframe = wav_deframe / np.max(abs(wav_deframe))
    return wav_deframe


def save_wav_file(filename, wav_data, sample_rate):
    """Saves audio data to .wav audio file.

    Args:
      filename: Path to save the file to.
      wav_data: Array of float PCM-encoded audio data.
      sample_rate: Samples per second to encode in the file.
    """
    wav_data = wav_data * (2**15-1)
    wav_data = wav_data.astype(np.int16, order='C')
    scipy.io.wavfile.write(filename, sample_rate, wav_data)

