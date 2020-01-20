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

We provide our CelebAudio Dataset at <a>link</a>.

## Train

### Voice Conversion

Check 'scripts/train_audio.sh' for an example of training a Voice-Conversion model. Make sure directory 'logs' exist.

Generally, run:
```
python train_audio.py --data_path PATH_TO_TRAINING_DATA --experiment_name EXPERIMENT_NAME --save_freq SAVE_FREQ --test_path_A PATH_TO_TEST_AUDIO --test_path_B PATH_TO_TEST_AUDIO --batch_size BATCH_SIZE --save_dir PATH_TO_SAVE_MODEL
```

### Audiovisual Synthesis

Check 'scripts/train_audiovisual.sh' for an example of training a Audiovisual-Synthesis model. We usually train an audiovisual model based on a pretrained audio model.

#### 1-stage generation -- video resolution: 256 * 256

Generally, run:
```
python train_audiovisual.py --video_path PATH_TO_TRAINING_DATA --experiment_name EXPERIMENT_NAME --save_freq SAVE_FREQ --test_path PATH_TO_TEST_AUDIO --batch_size BATCH_SIZE --save_dir PATH_TO_SAVE_MODEL --use_256 --load_model LOAD_MODEL_PATH
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

### Audiovisual Synthesis

Check 'scripts/test_audiovisual.sh' for an example of testing a Audiovisual-Synthesis model.

#### 1-stage generation -- video resolution: 256 * 256
```
python test_audiovisual.py --load_model PATH_TO_MODEL --wav_path PATH_TO_INPUT --output_file PATH_TO_OUTPUT --use_256 
```

#### 2-stage generation -- video resolution: 512 * 512
```
python test_audiovisual.py --load_model PATH_TO_MODEL --wav_path PATH_TO_INPUT --output_file PATH_TO_OUTPUT --residual
```
