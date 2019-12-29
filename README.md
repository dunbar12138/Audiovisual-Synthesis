# Audiovisual-Synthesis
Unsupervised Any-to-many Audiovisual Synthesis via Exemplar Autoencoders, Kangle Deng, Aayush Bansal, Deva Ramanan, arXiv

This repo provides a PyTorch Implementation of our work.

Acknowledgements: This code borrows heavily from <a href='https://github.com/auspicious3000/autovc'>Auto-VC</a> and Tacotron.

### Dependencies

First, make sure ffmpeg installed on your machine.

Then, run: `pip install -r requirements.txt`

### Data

We provide our CelebAudio Dataset at <a>link</a>.

### Train

#### Voice Conversion

Check 'scripts/train_audio.sh' for training a Voice-Conversion model.

Generally, run:
```
python train_audio.py --data_path PATH_TO_TRAINING_DATA --experiment_name EXPERIMENT_NAME --save_freq SAVE_FREQ --test_path_A PATH_TO_TEST_AUDIO --test_path_B PATH_TO_TEST_AUDIO --batch_size BATCH_SIZE --save_dir PATH_TO_LOG
```

#### Audiovisual Synthesis

Check 'scripts/train_audiovisual.sh' for training a Audiovisual-Synthesis model. We usually train an audiovisual model based on a pretrained audio model.

##### 1-stage generation -- video resolution: 256 * 256

Generally, run:
```
python train_audiovisual.py --video_path PATH_TO_TRAINING_DATA --experiment_name EXPERIMENT_NAME --save_freq SAVE_FREQ --test_path PATH_TO_TEST_AUDIO --batch_size BATCH_SIZE --save_dir PATH_TO_LOG --use_256 --load_model LOAD_MODEL_PATH
```

##### 2-stage generation -- video resolution: 512 * 512

If you want the video resolution to be 512 * 512, use the StackGAN-style 2-stage generation.

Generally, run:
```
python train_audiovisual.py --video_path PATH_TO_TRAINING_DATA --experiment_name EXPERIMENT_NAME --save_freq SAVE_FREQ --test_path PATH_TO_TEST_AUDIO --batch_size BATCH_SIZE --save_dir PATH_TO_LOG --residual --load_model LOAD_MODEL_PATH
```



### Test
