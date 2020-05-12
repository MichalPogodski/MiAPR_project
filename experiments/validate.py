import tensorflow as tf
import pandas as pd
from argparse import ArgumentParser

import inspect
import os
import sys
import math

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

from models.dubinsnet import *


def main(args):
    df = pd.read_csv(args.dataset_path)
    target = df.pop(df.columns[4])
    dataset = tf.data.Dataset.from_tensor_slices((df.values, target.values))

    train_ds = dataset.batch(len(df))

    model = tf.keras.models.load_model(args.model_path)

    eval_loss = model.evaluate(train_ds)
    print(f'Mean distance: {target.mean()}')
    print(f'MSE: {eval_loss[0]}, RMSE: {math.sqrt(eval_loss[0])}, MAE: {eval_loss[2]}')



if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--dataset-path', type=str, default='../Dubin_dataset/dubins_dataset_validate.csv')
    parser.add_argument('--model-path', type=str, required=True)
    args, _ = parser.parse_known_args()
    main(args)
