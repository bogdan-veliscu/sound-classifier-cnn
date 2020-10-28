from sound_task.esc_cnn.preprocessing.preprocessingESC import extract_spectrogram
import torch
import torch.nn as nn

from pathlib import Path
import torchaudio
import numpy as np
from PIL import Image
import torchvision

import models.densenet
import models.resnet
import models.inception
import models.effnet
import models.mobilenet
import utils
from .utils import load_checkpoint


class SoundClassifier:
    def __init__(self, config_path="") -> None:

        self._params = utils.Params(config_path)
        self._device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        if self._params.model == "densenet":
            self._model = models.densenet.DenseNet(
                self._params.dataset_name, self._params.pretrained
            ).to(self._device)
        elif self._params.model == "resnet":
            self._model = models.resnet.ResNet(
                self._params.dataset_name, self._params.pretrained
            ).to(self._device)
        elif self._params.model == "effnet":
            self._model = models.effnet.EffNet(
                self._params.dataset_name, self._params.pretrained
            ).to(self._device)
        elif self._params.model == "mobilenet":
            self._model = models.mobilenet.MobileNet(
                self._params.dataset_name, self._params.pretrained
            ).to(self._device)
        elif self._params.model == "inception":
            self._model = models.inception.Inception(
                self._params.dataset_name, self._params.pretrained
            ).to(self._device)

        self._loss_fn = nn.CrossEntropyLoss()
        self._optimizer = torch.optim.Adam(
            self._model.parameters(),
            lr=self._params.lr,
            weight_decay=self._params.weight_decay,
        )
        load_checkpoint(Path(self._params.checkpoint_dir) / "model_best_1.pth.tar")

    def extract_spectrogram(self, clip):

        num_channels = 3  # color channels for the mel spctogram
        window_sizes = [25, 50, 100]
        hop_sizes = [10, 25, 50]
        centre_sec = 2.5

        specs = []
        for i in range(num_channels):
            window_length = int(round(window_sizes[i] * self.sampling_rate / 1000))
            hop_length = int(round(hop_sizes[i] * self.sampling_rate / 1000))

            clip = torch.Tensor(clip)
            spec = torchaudio.transforms.MelSpectrogram(
                sample_rate=self.sampling_rate,
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

    def infer(self, clip):
        spec = extract_spectrogram(clip)

        inputs = np.array([spec]).to(self._d)
        outputs = self._model(inputs)

        _, predicted = torch.max(outputs.data, 1)

        return predicted
