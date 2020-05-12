import tensorflow as tf
import pandas as pd
from argparse import ArgumentParser

import inspect
import os
import sys
import datetime

# add parent (root) to pythonpath
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

from models.dubinsnet import *

def data_loader():
    pass


def main(args):

    path_base = f'results/{datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")}'
    df = pd.read_csv(args.dataset_path)
    target = df.pop(df.columns[4])
    dataset = tf.data.Dataset.from_tensor_slices((df.values, target.values))

    dataset.shuffle(len(df))

    train_size = int(0.8 * len(df))
    train_ds = dataset.take(train_size).batch(256)
    val_ds = dataset.skip(train_size).batch(256)

    model = dubinsNet()

    lr = args.lr
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        lr,
        decay_steps=100000,
        decay_rate=0.96,
        staircase=False)

    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
    model.compile(optimizer=optimizer, loss=tf.keras.losses.mean_squared_error,
                  metrics=[tf.keras.metrics.mean_squared_error, tf.keras.metrics.mean_absolute_error])

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=path_base+'/checkpoints/dubinsnet_{epoch}',
            save_best_only=True,
            monitor='val_loss',
            verbose=1),
        tf.keras.callbacks.TensorBoard(log_dir=path_base + '/log')
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


