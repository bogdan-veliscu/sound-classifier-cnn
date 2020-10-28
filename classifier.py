from sound_task.esc_cnn.preprocessing.preprocessingESC import extract_spectrogram
import torch
import torch.nn as nn

from pathlib import Path
import torchaudio
import numpy as np
from PIL import Image
import torchvision

from .models import DenseNet, ResNet, Inception, MobileNet, EffNet
from .utils import load_checkpoint, Params
import os, inspect

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
print("# Arcface path: ", currentdir)


class SoundClassifier:
    def __init__(
        self, config_path=Path(currentdir) / "config" / "esc_mobilenet.json"
    ) -> None:

        self._params = Params(config_path)
        self._sampling_rate = 44100
        self._device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        if self._params.model == "densenet":
            self._model = DenseNet(
                self._params.dataset_name, self._params.pretrained
            ).to(self._device)
        elif self._params.model == "resnet":
            self._model = ResNet(self._params.dataset_name, self._params.pretrained).to(
                self._device
            )
        elif self._params.model == "effnet":
            self._model = EffNet(self._params.dataset_name, self._params.pretrained).to(
                self._device
            )
        elif self._params.model == "mobilenet":
            self._model = MobileNet(
                self._params.dataset_name, self._params.pretrained
            ).to(self._device)
        elif self._params.model == "inception":
            self._model = Inception(
                self._params.dataset_name, self._params.pretrained
            ).to(self._device)

        self._loss_fn = nn.CrossEntropyLoss()
        self._optimizer = torch.optim.Adam(
            self._model.parameters(),
            lr=self._params.lr,
            weight_decay=self._params.weight_decay,
        )
        path = str(
            Path(currentdir) / self._params.checkpoint_dir / "model_best_1.pth.tar"
        )
        print("# checkoint path:", path)
        load_checkpoint(
            path,
            self._model,
            self._optimizer,
        )

    def extract_spectrogram(self, clip):

        num_channels = 3  # color channels for the mel spctogram
        window_sizes = [25, 50, 100]
        hop_sizes = [10, 25, 50]
        centre_sec = 2.5

        specs = []
        for i in range(num_channels):
            window_length = int(round(window_sizes[i] * self._sampling_rate / 1000))
            hop_length = int(round(hop_sizes[i] * self._sampling_rate / 1000))

            clip = torch.Tensor(clip)
            spec = torchaudio.transforms.MelSpectrogram(
                sample_rate=self._sampling_rate,
                n_fft=4410,
                win_length=window_length,
                hop_length=hop_length,
                n_mels=128,
            )(clip)
            eps = 1e-6
            spec = spec.numpy()
            spec = np.log(spec + eps)
            spec = np.asarray(
                torchvision.transforms.Resize((128, 250))(Image.fromarray(spec))
            )
            specs.append(spec)

        return specs

    def infer(self, clip):
        specs = self.extract_spectrogram(clip)

        values = np.array(specs)  # .reshape(-1, 128, 250)
        print("@ values SHAPE:  ", values.shape)
        values = torch.Tensor(values)

        inputs = values.to(self._device)
        outputs = self._model(inputs)

        _, predicted = torch.max(outputs.data, 1)

        return predicted
