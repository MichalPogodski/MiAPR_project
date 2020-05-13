import tensorflow as tf
import pandas as pd
from argparse import ArgumentParser

import inspect
import os
import sys
import datetime

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

from models.dubinsnet import *

def data_loader():
    pass


def main(args):

    path_base = f'results/{datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")}'
    file_writer = tf.summary.create_file_writer(path_base + "/log/metrics")
    file_writer.set_as_default()

    df = pd.read_csv(args.dataset_path)
    target = df.pop(df.columns[4])
    dataset = tf.data.Dataset.from_tensor_slices((df.values, target.values))

    dataset.shuffle(len(df))

    train_size = int(0.8 * len(df))
    train_ds = dataset.take(train_size).batch(32)
    val_ds = dataset.skip(train_size).batch(32)

    model = dubinsNet()

    lr = args.lr
    lr_decay = tf.keras.optimizers.schedules.ExponentialDecay(lr, 2, 0.96)

    def lr_schedule(epoch):
        lr = lr_decay(epoch)
        tf.summary.scalar('learning rate', data=lr, step=epoch)
        return lr

    optimizer = tf.keras.optimizers.Adam()
    model.compile(optimizer=optimizer, loss=tf.keras.losses.mse,
                  metrics=[tf.keras.metrics.mean_squared_error, tf.keras.metrics.mean_absolute_error, tf.keras.metrics.RootMeanSquaredError()])

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=path_base+'/checkpoints/dubinsnet_{epoch}',
            save_best_only=True,
            monitor='val_loss',
            verbose=1),
        tf.keras.callbacks.TensorBoard(log_dir=path_base + '/log'),
        tf.keras.callbacks.LearningRateScheduler(lr_schedule),
        tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                         patience=10, min_delta=1e-2)
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


