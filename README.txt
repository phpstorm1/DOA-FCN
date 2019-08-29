- Requirements
	- python 3.5
	- tensorflow 1.13
	- scipy
	- matplotlib
	- python_speech_features
	- glob
	- functools


- Training
	- Run "train.py" to train the model


- Evaluation
	- Run "test_wav_files.py" to evaluate the existing dual-channel reverb wav files
	- Run "test_full_doa.py" to evaluate full-range DOA (0 deg - 135 deg). 


- Notes
	- All wav files have to be in 16k Hz
	- Reverberant wav will be generated on-the-fly when running "test_full_doa.py" with single-channel speech wavs files and rir wav files, whose paths are specified by 'training_speech_dir' and 'rir_data_dir' in "config.json" 
	- The testing code is memory-hungry. Shorten the duration of wavs when Out-of-memory (OOM) error occurs.

