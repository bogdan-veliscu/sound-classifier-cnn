import shutil
import json
import os
import torch
import torch.nn as nn
import pandas as pd


class Params:
    def __init__(self, json_path):
        with open(json_path) as f:
            params = json.load(f)
            self.__dict__.update(params)
            self.__dict__["sound_types"] = get_classes(
                os.path.join(os.path.dirname(json_path), "esc50.csv")
            )

    def save(self, json_path):
        with open(json_path, "w") as f:
            params = json.dump(self.__dict__, f, indent=4)

    def update(self, json_path):
        with open(json_path) as f:
            params = json.load(f)
            self.__dict__.update(params)

    @property
    def dict(self):
        return self.__dict__


class RunningAverage:
    def __init__(self):
        self.total = 0
        self.steps = 0

    def update(self, loss):
        self.total += loss
        self.steps += 1

    def __call__(self):
        return self.total / float(self.steps)


def save_checkpoint(state, is_best, split, checkpoint):
    filename = os.path.join(checkpoint, "last{}.pth.tar".format(split))
    if not os.path.exists(checkpoint):
        print("Checkpoint Directory does not exist")
        os.mkdir(checkpoint)
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(
            filename, os.path.join(checkpoint, "model_best_{}.pth.tar".format(split))
        )


def load_checkpoint(checkpoint, model, optimizer=None, parallel=False):
    if not os.path.exists(checkpoint):
        raise Exception("File Not Found Error {}".format(checkpoint))
    checkpoint = torch.load(checkpoint)
    if parallel:
        model.module.load_state_dict(checkpoint["model"])
    else:
        model.load_state_dict(checkpoint["model"])

    if optimizer:
        optimizer.load_state_dict(checkpoint["optimizer"])
    return checkpoint


def initialize_weights(m):
    classname = m.__class__.__name__
    print(classname)
    if classname.find("Linear") != -1:
        nn.init.ones_(m.weight.data)


def get_classes(meta_csv):
    map = {}
    df = pd.read_csv(meta_csv, skipinitialspace=True)
    df2 = df.groupby(["category"]).min().sort_values(by=["target"])

    for index, row in df2.iterrows():
        map[row["target"]] = index

    return map