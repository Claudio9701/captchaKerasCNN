# Captcha Solver using CNN
Generates captcha images, and trains an CNN to ouput the string showed in a captcha image.

### Specifics
- Implemented in Keras (using TensorFlow backend).
- Each character in the captcha is classified "individually".
- Train / test sets are normalized using 255 as numerator since images RBG channels have a maximum and minimun value of 255 and 0, respectively.

### Dependencies
- TensorFlow or Theano
- Keras
- pandas
- numpy
- captcha

### Does it work?
Not really, current results aren't usable in any online captcha.
