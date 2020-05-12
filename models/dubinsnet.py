import tensorflow as tf

def dubinsNet():
  model = tf.keras.Sequential([
    tf.keras.layers.Input(4),
    tf.keras.layers.Dense(64, kernel_regularizer=tf.keras.regularizers.l2(0.00001), activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(64, kernel_regularizer=tf.keras.regularizers.l2(0.00001), activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(1)
  ])

  return model
