import transformers
import torch
import math
from PIL import Image
import os
import glob
from tqdm import tqdm
import numpy as np


class Engine:
    def __init__(self):
        self.image_paths = []
        self.image_features = None
        self.init_storage()
        self.init_cache()

    # base function

    def load_model(
        self,
        pretrained_model_name_or_path: str = None,
        dtype=torch.float32,
        device="cpu",
    ):
        self.clip_image_processor = transformers.CLIPImageProcessor.from_pretrained(
            pretrained_model_name_or_path
        )
        self.clip_vision_model = (
            transformers.CLIPVisionModelWithProjection.from_pretrained(
                pretrained_model_name_or_path
            ).to(device, dtype)
        )
        self.clip_vision_model.requires_grad_(False)

        self.clip_text_model = transformers.CLIPTextModelWithProjection.from_pretrained(
            pretrained_model_name_or_path
        ).to(device, dtype)
        self.clip.tokenizer = transformers.CLIPTokenizer.from_pretrained(
            pretrained_model_name_or_path
        )
        self.clip_text_model.requires_grad_(False)

        self.device = device
        self.dtype = dtype

    def get_most_similar_image(self, image_path, num: int = 5):
        index = self.image_paths.index(image_path)
        image_features = self.image_features[index]
        # similarity = 100 * (image_features @ self.image_features.T).softmax(dim=-1)
        similarity = 100 * (
            image_features.to("cuda", dtype=self.dtype)
            @ self.image_features.T.to("cuda", dtype=self.dtype)
        )
        values, indices = similarity.topk(num)
        return values, [self.image_paths[i] for i in indices]

    def get_most_similar_images(
        self, threshold=0, device="cpu", dtype=torch.float32, batch_size=100
    ):
        result = []
        if device == "cpu":
            similarity = 100 * (
                self.image_features.to(device, dtype=dtype).numpy()
                @ self.image_features.T.to(device, dtype=dtype).numpy()
            )
            indices = np.where(similarity >= threshold)
            for x, y in tqdm(zip(list(indices[0]), list(indices[1]))):
                if x == y:
                    continue
                result.append(
                    (self.image_paths[x], self.image_paths[y], similarity[x][y])
                )
        else:
            for i in tqdm(range(len(self.image_features))):
                similarity = 100 * (
                    self.image_features[i].to(device, dtype=self.dtype)
                    @ self.image_features.T.to(device, dtype=self.dtype)
                )
                indexs = torch.where(similarity >= threshold)[0].tolist()
                result.append(
                    [
                        (self.image_paths[i], self.image_paths[index])
                        for index in indexs
                        if index != i
                    ]
                )
        return result

        # n = len(self.image_features)
        # for i in tqdm(range(0, n, batch_size)):
        #     similarity = 100 * (
        #         self.image_features[i * batch_size : min(n, (i + 1) * batch_size)].to(
        #             device, dtype=self.dtype
        #         )
        #         @ self.image_features.T.to(device, dtype=self.dtype)
        #     )
        #     similarity = torch.where(similarity >= threshold)
        #     for row in similarity:
        #         indexs = similarity[row].tolist()
        #         result.append(
        #             [
        #                 (
        #                     self.image_paths[i * batch_size + row],
        #                     self.image_paths[index],
        #                 )
        #                 for index in indexs
        #                 if index != i * batch_size + row
        #             ]
        #         )

    def add_images_by_directory_path(self, dir_path):
        image_paths = glob.glob(dir_path + "/**/*.png", recursive=True) + glob.glob(
            dir_path + "/**/*.jpg", recursive=True
        )
        image_paths = [
            image_path
            for image_path in image_paths
            if image_path not in self.image_paths
        ]
        print(image_paths)
        for image_path in tqdm(image_paths):
            if os.path.isfile(image_path):
                self.add_image(image_path)

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
        image_features = image_features.to("cpu", dtype=self.dtype)
        if self.image_features is None:
            self.image_features = image_features
        else:
            self.image_features = torch.concat(
                [self.image_features, image_features], dim=0
            )
        del pixel_values
        del image_features

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
            if not os.path.exists(image_path):
                print(image_path + "will be remove")
                self.remove_image_feature(image_path)

    # 持久化存储
    def init_storage(self):
        None

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
        self.image_features = torch.load(path + "/image_features.pt").to("cpu")
        assert len(self.image_paths) == self.image_features.shape[0]
        print(
            "load successfully:"
            + os.path.join(path, "image_paths.txt")
            + ","
            + os.path.join(path, "image_features.pt")
        )
        print(self.image_features)

    """
        检索
        返回PIL Image list
    """

    def init_cache(self):
        None

    def search_image_by_text(self, text, num=1):
        self._search_image_by_text(text, num)

    def _search_image_by_text(self, text, num=1):
        input_ids = self.clip_tokenizer(
            [text],
            padding="longest",
            truncation=True,
            return_tensors="pt",
        ).input_ids
        text_embds = self.clip_text_model(input_ids, output_hidden_states=False)
        text_embds /= text_embds.norm(dim=-1, keepdim=True)
        similarity = 100 * (
            text_embds.to(self.device, dtype=self.dtype)
            @ self.image_features.T.to(self.device, dtype=self.dtype)
        )
        values, indices = similarity.topk(num)
        return values, [self.image_paths[i] for i in indices]

    def search_image_by_image(self, image: Image.Image, num=1):
        self._search_image_by_image(image, num)

    def _search_image_by_image(self, image: Image.Image, num=1):
        pixel_values = self.clip_image_processor(
            image, return_tensors="pt"
        ).pixel_values.to("cuda", dtype=self.dtype)
        image_features = self.clip_vision_model(
            pixel_values, output_attentions=False, output_hidden_states=False
        ).image_embeds.to("cuda", dtype=self.dtype)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        image_features = image_features.to("cpu", dtype=self.dtype)

        similarity = 100 * (
            image_features.to("cuda", dtype=self.dtype)
            @ self.image_features.T.to("cuda", dtype=self.dtype)
        )
        values, indices = similarity.topk(num)
        return values, [self.image_paths[i] for i in indices]
