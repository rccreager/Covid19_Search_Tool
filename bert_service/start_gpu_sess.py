import tensorflow as tf

print("Before Session Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
print(tf.config.experimental.list_physical_devices('GPU'))
sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
print("After Session Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
print(tf.config.experimental.list_physical_devices('GPU'))
