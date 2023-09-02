from genericpath import isfile
from lib2to3.fixes.fix_tuple_params import simplify_args
from cv2 import THRESH_OTSU
from numpy import imag
import transformers
import torch
from PIL import Image
import os
import glob
from tqdm import tqdm
import numpy as np


class CLIPRetrieval:
    def __init__(self, pretrained_model_name_or_path: str = None, dtype=torch.float32):
        self.dtype = dtype
        if pretrained_model_name_or_path is not None:
            self.clip_image_processor = transformers.CLIPImageProcessor.from_pretrained(
                pretrained_model_name_or_path
            )
            self.clip_vision_model = (
                transformers.CLIPVisionModelWithProjection.from_pretrained(
                    pretrained_model_name_or_path
                ).to("cuda", dtype=dtype)
            )
            self.clip_vision_model.eval()
            self.clip_vision_model.requires_grad_(False)
        self.image_paths = []
        self.image_features = None

    def save_cache(self, path="."):
        with open(os.path.join(path, "image_paths.txt"), "w") as image_paths_file:
            image_paths_file.writelines(path + "\n" for path in self.image_paths)
        torch.save(self.image_features, os.path.join(path, "image_features.pt"))
        print(
            "save successfully:"
            + os.path.join(path, "image_paths.txt")
            + ","
            + os.path.join(path, "image_features.pt")
        )

    def load_cache(self, path="."):
        with open(os.path.join(path, "image_paths.txt"), "r") as image_paths_file:
            self.image_paths = image_paths_file.readlines()
            self.image_paths = [path.rstrip() for path in self.image_paths]
        self.image_features = torch.load(path + "/image_features.pt").to(
            "cuda", dtype=self.dtype
        )
        assert len(self.image_paths) == self.image_features.shape[0]
        print(
            "load successfully:"
            + os.path.join(path, "image_paths.txt")
            + ","
            + os.path.join(path, "image_features.pt")
        )
        print(self.image_features)

    def add_image(self, image_path):
        if image_path in self.image_paths:
            return
        try:
            image = Image.open(image_path)
        except IOError:
            return
        self.image_paths.append(image_path)

        pixel_values = self.clip_image_processor(
            image, return_tensors="pt"
        ).pixel_values.to("cuda", dtype=self.dtype)
        image_features = self.clip_vision_model(
            pixel_values, output_attentions=False, output_hidden_states=False
        ).image_embeds.to("cuda", dtype=self.dtype)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        if self.image_features is None:
            self.image_features = image_features
        else:
            self.image_features = torch.concat(
                [self.image_features, image_features], dim=0
            ).to("cuda", dtype=self.dtype)
        del pixel_values
        del image_features

    def get_most_similar_image(self, image_path, num: int = 5):
        index = self.image_paths.index(image_path)
        image_features = self.image_features[index]
        # similarity = 100 * (image_features @ self.image_features.T).softmax(dim=-1)
        similarity = 100 * (image_features @ self.image_features.T)
        values, indices = similarity.topk(num)
        return values, [self.image_paths[i] for i in indices]

    def get_most_similar_images(self, topk: int = None, threshold=0):
        similarity = 100 * (self.image_features @ self.image_features.T)
        result = []
        if topk is not None:
            values, indices = similarity.topk(topk)

            for i in range(0, len(self.image_paths)):
                image1_path = self.image_paths[i]
                for j in range(0, topk):
                    image2_path = self.image_paths[indices[i][j]]
                    if (
                        image1_path != image2_path
                        and float(values[i][j]) >= threshold
                        and not np.isclose(float(values[i][j]), 100)
                    ):
                        result.append((image1_path, image2_path, float(values[i][j])))
        else:
            for x, y in torch.nonzero(similarity >= threshold):
                if x == y:
                    continue
                result.append(
                    (self.image_paths[x], self.image_paths[y], similarity[x][y])
                )

        return result

    def add_images_by_directory_path(self, dir_path):
        image_paths = glob.glob(dir_path + "/**", recursive=True)
        print(image_paths)
        for image_path in tqdm(image_paths):
            if os.path.isfile(image_path):
                self.add_image(image_path)

    def remove_image_feature(self, path):
        if path not in self.image_paths:
            print(f"{path} not in cache")
            return
        index = self.image_paths.index(path)
        self.image_paths.pop(index)
        self.image_features = torch.concat(
            [self.image_features[:index], self.image_features[index + 1 :]], dim=0
        )
    def autoremove(self):
        for image_path in self.image_paths:
            if os.path.exists(image_path):
                self.remove_image_feature(image_path)
