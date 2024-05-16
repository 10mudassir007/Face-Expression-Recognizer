import tensorflow as tf
import matplotlib.pyplot as plt


trainDir = 'train'
testDir = 'test'

train_images = tf.keras.preprocessing.image_dataset_from_directory(
    trainDir,
    seed=123,
    image_size=(48, 48),
    label_mode='categorical',
    shuffle=True,
    color_mode='grayscale'
    )

test_images = tf.keras.preprocessing.image_dataset_from_directory(
    testDir,
    seed=42,
    image_size=(48, 48),
    label_mode='categorical',
    shuffle=True,
    color_mode='grayscale'
    )
print("Train dataset element_spec:", train_images.element_spec)
print("Test dataset element_spec:", test_images.element_spec)

model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(64,(3,3),activation='relu',input_shape=(48, 48,1)),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(32,(3,3),activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Normalization(),
    tf.keras.layers.Dense(128,activation='relu'),
    tf.keras.layers.Dense(7,activation='softmax')
])

model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
print(model.summary())


history = model.fit(train_images,epochs=12,validation_data=test_images)
model.evaluate(test_images)
model.save('cnn1.keras')


# model = tf.keras.Sequential([
#     tf.keras.layers.Conv2D(128,(3,3),activation='relu',input_shape=(48, 48,1)),
#     tf.keras.layers.MaxPooling2D(2,2),
#     tf.keras.layers.Conv2D(64,(3,3),activation='relu'),
#     tf.keras.layers.MaxPooling2D(2,2),
#     tf.keras.layers.Conv2D(64,(3,3),activation='relu'),
#     tf.keras.layers.MaxPooling2D(2,2),
#     tf.keras.layers.Flatten(),
#     tf.keras.layers.BatchNormalization(),
#     tf.keras.layers.Dense(1024,activation='relu',kernel_regularizer=tf.keras.regularizers.l2(0.001)),
#     tf.keras.layers.Dense(7,activation='softmax')
# ])


# model = tf.keras.Sequential([
#     tf.keras.layers.Conv2D(32,(3,3),activation='relu',input_shape=(48, 48,1)),
#     tf.keras.layers.MaxPooling2D(2,2),
#     tf.keras.layers.Conv2D(64,(3,3),activation='relu'),
#     tf.keras.layers.MaxPooling2D(2,2),
#     tf.keras.layers.Conv2D(128,(3,3),activation='relu'),
#     tf.keras.layers.MaxPooling2D(2,2),
#     tf.keras.layers.Flatten(),
#     tf.keras.layers.Dense(512,activation='relu',kernel_regularizer=tf.keras.regularizers.l2(0.001)),
#     tf.keras.layers.Dense(7,activation='softmax')
# ])


# model = tf.keras.Sequential([
#     tf.keras.layers.Conv2D(32,(3,3),activation='relu',input_shape=(48, 48,1)),
#       tf.keras.layers.Normalization(),
#     tf.keras.layers.MaxPooling2D(2,2),
#     tf.keras.layers.Conv2D(64,(3,3),activation='relu'),
#       tf.keras.layers.Normalization(),
#     tf.keras.layers.MaxPooling2D(2,2),
#     tf.keras.layers.Conv2D(128,(3,3),activation='relu'),
#       tf.keras.layers.Normalization(),
#     tf.keras.layers.MaxPooling2D(2,2),
#     tf.keras.layers.Flatten(),
#     tf.keras.layers.Dense(512,activation='relu',kernel_regularizer=tf.keras.regularizers.l2(0.001)),
#       tf.keras.layers.Normalization(),
#     tf.keras.layers.Dense(7,activation='softmax')
# ])
