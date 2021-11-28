# Keyword Spotting with Convolutional Recurrent Neural Networks

## Model Architecture
```
_________________________________________________________________
Layer (type)                     Output Shape          Param #
=================================================================
conv_reshape (Reshape)         (None, 59, 13, 1)         0
_________________________________________________________________
conv (Conv2D)                   (None, 8, 7, 32)        3232
_________________________________________________________________
recurrent_reshape (Reshape)      (None, 32, 56)          0
_________________________________________________________________
recurrent1 (GRU)                 (None, 32, 32)         8640
_________________________________________________________________
recurrent2 (GRU)                   (None, 32)           6336
_________________________________________________________________
fc (Dense)                         (None, 64)           2112
_________________________________________________________________
output (Dense)                     (None, 12)            780
=================================================================
Total params: 21,100
Trainable params: 21,100
Non-trainable params: 0
```

The model achieves 92% training accuracy, 91% validation accuracy, and 87%
test accuracy.

## Build Environment

This project was built in a python 3.7.7 virtual environment.

Dependencies can be installed with
    
```pip install -r requirements.txt```

To train the model, run

```python model.py```

To predict a continous audio stream with your microphone, run

```python stream.py```

Please note that for ```stream.py``` to work out of the box, you must have
pulseaudio running as your sound server. Barebones ALSA is supported,
but you will have to specify ```sd.default.device``` yourself.
