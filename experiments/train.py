import tensorflow as tf
import pandas as pd
from argparse import ArgumentParser
from models.dubinsnet import *

def data_loader():
    pass


def main(args):
    df = pd.read_csv(args.dataset_path)
    target = df.pop(df.columns[4])
    dataset = tf.data.Dataset.from_tensor_slices((df.values, target.values))

    dataset.shuffle(len(df))

    train_size = int(0.8 * len(df))
    train_ds = dataset.take(train_size).batch(256)
    val_ds = dataset.skip(train_size).batch(256)

    model = dubinsNet()
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=args.lr), loss=tf.keras.losses.mean_squared_error,
                  metrics=[tf.keras.metrics.mean_squared_error, tf.keras.metrics.mean_absolute_error])

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath='checkpoints/dubinsnet_{epoch}',
            save_best_only=True,
            monitor='val_loss',
            verbose=1),
        tf.keras.callbacks.TensorBoard(log_dir='log')

    ]

    history = model.fit(train_ds, epochs=args.epochs, validation_data=val_ds, callbacks=callbacks)
    print(history.history)



if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--dataset-path', type=str, default='../Dubin_dataset/dubins_dataset.csv')
    parser.add_argument('--lr', type=float,
                        default=5e-4)
    parser.add_argument('--epochs', type=int,
                        default=250)
    args, _ = parser.parse_known_args()
    main(args)


