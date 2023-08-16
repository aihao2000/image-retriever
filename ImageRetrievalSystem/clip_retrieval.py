from genericpath import isfile
from numpy import imag
import transformers
import torch
from PIL import Image
import os
import glob
from tqdm import tqdm


class CLIPRetrieval:
    def __init__(self, pretrained_model_name_or_path: str = None):
        if pretrained_model_name_or_path is not None:
            self.clip_image_processor = transformers.CLIPImageProcessor.from_pretrained(
                pretrained_model_name_or_path
            )
            self.clip_vision_model = (
                transformers.CLIPVisionModelWithProjection.from_pretrained(
                    pretrained_model_name_or_path
                ).to("cuda:0")
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
        self.image_features = torch.load(path + "/image_features.pt").to("cuda:1")
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
            print(f"{image_path} already in cache")
            return
        try:
            image = Image.open(image_path)
        except IOError:
            return
        self.image_paths.append(image_path)

        pixel_values = self.clip_image_processor(
            image, return_tensors="pt"
        ).pixel_values.to("cuda:0")
        image_features = self.clip_vision_model(
            pixel_values, output_attentions=False, output_hidden_states=False
        ).image_embeds.to("cuda:1")
        image_features /= image_features.norm(dim=-1, keepdim=True)
        if self.image_features is None:
            self.image_features = image_features
        else:
            self.image_features = torch.concat(
                [self.image_features, image_features], dim=0
            ).to("cuda:1")
        del pixel_values
        del image_features

    def get_most_similar_image(self, image_path, num: int = 5):
        index = self.image_paths.index(image_path)
        image_features = self.image_features[index]
        # similarity = 100 * (image_features @ self.image_features.T).softmax(dim=-1)
        similarity = 100 * (image_features @ self.image_features.T)
        vuales, indices = similarity.topk(num)
        return vuales, [self.image_paths[i] for i in indices]

    def add_images_by_directory_path(self, dir_path):
        file_paths = glob.glob(dir_path, recursive=True)
        print(file_paths)
        for path in tqdm(file_paths):
            if os.path.isfile(path):
                self.add_image(path)

    def remove_image_feature(self, path):
        if path not in self.image_paths:
            print(f"{path} not in cache")
            return
        index = self.image_paths.index(path)
        self.image_paths.pop(index)
        self.image_features = torch.concat(
            [self.image_features[:index], self.image_features[index + 1 :]], dim=0
        )
