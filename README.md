# lightweight_xception
A modified xception model for pap smear image classification into 4 classes (HSIL, LSIL, NIML, and SCC).
Xception is a deep convolutional neural network architecture inspired by Inception, where Inception modules have been replaced with depthwise separable convolutions.
It has the same number of parameters as Inception V3. (22,910,480 total parameters, Trainable params: 22,855,952, Non-trainable params: 54,528)
It has depthwise-separable convolutions. 
i.e. a spatial convolution performed independently over each channel of an input, followed by a pointwise convolution, i.e. a 1x1 convolution, projecting the channel output by the depthwise convolution onto a new channel space. 
![image](https://github.com/ElbetelTaye/lightweight_xception/assets/119397613/b623bc44-961d-429b-8314-dd5eefcbe376)
In this project, I have used the architecture of the original xception model but have reduced the number of parameters to 402,756 total parameters where 398,532 are trainable and 4,224 are non-trainable.
Below is the modified model architecture of the custom lightweight xception model.

    # Entry Flow
    x = Conv2D(32, (3, 3), strides=(2, 2), activation='relu')(inputs)
    x = BatchNormalization()(x)  # Batch normalization layer
    x = Conv2D(64, (3, 3), strides=(2, 2), activation='relu')(inputs)
    x = BatchNormalization()(x)  # Batch normalization layer
    x = SeparableConv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((3, 3), strides=(2, 2))(x)
   
    # Middle Flow  
    for _ in range(8):
        residual = x
        x = SeparableConv2D(128, (3, 3), activation='relu', padding='same')(x)
        x = BatchNormalization()(x)  # Batch normalization layer
        x = SeparableConv2D(128, (3, 3), activation='relu', padding='same')(x)
        x = BatchNormalization()(x)  # Batch normalization layer
        x = tf.keras.layers.Add()([x, residual])

    # Exit Flow
    x = SeparableConv2D(256, (3, 3), activation='relu', padding='same')(x)
    x = GlobalAveragePooling2D()(x)

