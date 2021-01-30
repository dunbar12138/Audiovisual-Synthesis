# Audiovisual-Synthesis
<b>Unsupervised Any-to-many Audiovisual Synthesis via Exemplar Autoencoders</b>

<a href="https://dunbar12138.github.io/">Kangle Deng</a>, <a href="http://www.cs.cmu.edu/~aayushb/">Aayush Bansal</a>, <a href="http://www.cs.cmu.edu/~deva/">Deva Ramanan</a>

<a href="https://dunbar12138.github.io/projectpage/Audiovisual/index.html"> project page </a>/<a href='http://scs00197.sp.cs.cmu.edu/'> demo </a>/<a href="https://arxiv.org/abs/2001.04463"> arXiv </a>

This repo provides a PyTorch Implementation of our work.

Acknowledgements: This code borrows heavily from <a href='https://github.com/auspicious3000/autovc'>Auto-VC</a> and Tacotron.

### Summary Video

[![](http://img.youtube.com/vi/7BO0-Q3TLfI/0.jpg)](http://www.youtube.com/watch?v=7BO0-Q3TLfI "")

### Demo Video

[![](http://img.youtube.com/vi/t80R2zwXR00/0.jpg)](http://www.youtube.com/watch?v=t80R2zwXR00 "")

## Dependencies

First, make sure ffmpeg installed on your machine.

Then, run: `pip install -r requirements.txt`

## Data

We provide our CelebAudio Dataset at <a href='https://drive.google.com/drive/folders/1_3ulcWKhs3eq2WPzAnFF-LJuSp1aTqwA?usp=sharing'>link</a>.

## Train

### Voice Conversion

Check 'scripts/train_audio.sh' for an example of training a Voice-Conversion model. Make sure directory 'logs' exist.

Generally, run:
```
python train_audio.py --data_path PATH_TO_TRAINING_DATA --experiment_name EXPERIMENT_NAME --save_freq SAVE_FREQ --test_path PATH_TO_TEST_AUDIO --batch_size BATCH_SIZE --save_dir PATH_TO_SAVE_MODEL
```

You can specify any audio data as PATH_TO_TRAINING_DATA, and a small clip of audio as PATH_TO_TEST_AUDIO. For example, the following script trains an audio model for Barack Obama, and use an input clip for test every 2000 iterations. You can find the saved models and test results in the saving directory.

```
python train_audio.py --data_path datasets/celebaudio/BarackObama_01.wav --experiment_name VC_example_run --save_freq 2000 --test_path example/input_3_MartinLutherKing.wav  --batch_size 8 --save_dir ./saved_models/
```

### Audiovisual Synthesis

Check 'scripts/train_audiovisual.sh' for an example of training a Audiovisual-Synthesis model. We usually train an audiovisual model based on a pretrained audio model.

#### 1-stage generation -- video resolution: 256 * 256

Generally, run:
```
python train_audiovisual.py --video_path PATH_TO_TRAINING_DATA --experiment_name EXPERIMENT_NAME --save_freq SAVE_FREQ --test_path PATH_TO_TEST_AUDIO --batch_size BATCH_SIZE --save_dir PATH_TO_SAVE_MODEL --use_256 --load_model LOAD_MODEL_PATH
```

You can specify any audiovisual data as PATH_TO_TRAINING_DATA, and a small clip of audio as PATH_TO_TEST_AUDIO. The following script trains an audiovisual model based on a pre-trained Obama audio model, and use an input clip for test every 2000 iterations. You can find the saved models and test results in the saving directory.

```
python train_audiovisual.py --video_path datasets/video/obama.mp4 --experiment_name Audiovisual_example_run --save_freq 2000 --test_path example/input_3_MartinLutherKing.wav --batch_size 8 --save_dir ./saved_models/ --use_256 --load_model ./saved_models/VC_example_run/Epoch600_Iter00030000.pkl
```


#### 2-stage generation -- video resolution: 512 * 512

If you want the video resolution to be 512 * 512, use the StackGAN-style 2-stage generation.

Generally, run:
```
python train_audiovisual.py --video_path PATH_TO_TRAINING_DATA --experiment_name EXPERIMENT_NAME --save_freq SAVE_FREQ --test_path PATH_TO_TEST_AUDIO --batch_size BATCH_SIZE --save_dir PATH_TO_SAVE_MODEL --residual --load_model LOAD_MODEL_PATH
```



## Test

### Voice Conversion

Check 'scripts/test_audio.sh' for an example of testing a Voice-Conversion model.

To convert a wavfile using a trained model, run:
```
python test_audio.py --model PATH_TO_MODEL --wav_path PATH_TO_INPUT --output_file PATH_TO_OUTPUT
```

You can specify any audio data as PATH_TO_INPUT. For example, the following script converts the input wavfile by use of a pre-trained audio model.

```
python test_audio.py --model ./saved_models/VC_example_run/Epoch600_Iter00030000.pkl --wav_path example/input_1_Trump.wav --output_file ./result.wav
```

### Audiovisual Synthesis

Check 'scripts/test_audiovisual.sh' for an example of testing a Audiovisual-Synthesis model.

#### 1-stage generation -- video resolution: 256 * 256
```
python test_audiovisual.py --load_model PATH_TO_MODEL --wav_path PATH_TO_INPUT --output_file PATH_TO_OUTPUT --use_256 
```

You can specify any audio data as PATH_TO_INPUT. For example, the following script converts the input wavfile by use of a pre-trained audiovisual model.

```
python test_audiovisual.py --load_model ./saved_models/Audiovisual_example_run/Epoch600_Iter00030000.pkl --wav_path example/input_1_Trump.wav  --output_file ./result.mp4 --use_256
```

#### 2-stage generation -- video resolution: 512 * 512
```
python test_audiovisual.py --load_model PATH_TO_MODEL --wav_path PATH_TO_INPUT --output_file PATH_TO_OUTPUT --residual
```


## Tensorboard (Optional)

This repo uses <a href='https://github.com/lanpa/tensorboardX'>TensorboardX</a> to visualize training loss. You can also check test audio results on tensorboard.

Start TensorBoard with ```tensorboard --logdir ./logs```.