import torch
import clip
import glob
import os
from PIL import Image
from prettytable import PrettyTable


class Engine:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model, self.preprocess = clip.load("ViT-B/32", self.device)
        self.image_features = None
        self.image_paths = []

    def AddImageByDirectoryPath(self, dir_path, image_types=["jpg", "png"]):
        image_paths = []
        for image_type in image_types:
            image_paths += glob.glob(dir_path + "/*." + image_type)
            
        for path in image_paths:
            self.AddImageByFilePath(path)

        self.image_features /= self.image_features.norm(dim=-1, keepdim=True)

    def AddImageByFilePath(self, path):
        self.image_paths.append(path)
        self.AddImage(Image.open(path))

    def AddImage(self, image):
        image = self.preprocess(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            image_feature = self.model.encode_image(image).to(self.device)
            self.image_features = torch.cat((self.image_features, image_feature), dim=0).to(
                self.device) if self.image_features != None else image_feature

    def ImageRetrievalByText(self, text, result_count):
        text = clip.tokenize([text]).to(self.device)
        with torch.no_grad():
            text_feature = self.model.encode_text(text)

        text_feature /= text_feature.norm(dim=-1, keepdim=True)
        similarity = 100.0 * (self.image_features @ text_feature.T).reshape((-1,))
        values, indices = similarity.topk(result_count)
        result = PrettyTable()
        result.field_names = ["路径", "置信度"]

        for i in range(result_count):
            result.add_row([self.image_paths[indices[i]], values[i].item()])
        print(result)
        return [self.image_paths[i] for i in indices]

    def ImageRetrievalByImagePath(self, image_path, result_count):
        return self.ImageRetrievalByImage(Image.open(image_path), result_count)

    def ImageRetrievalByImage(self, image, result_count):
        image = self.preprocess(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            image_feature = self.model.encode_image(image).to(self.device)
        image_feature /= image_feature.norm(dim=-1, keepdim=True)
        similarity = 100.0 * (self.image_features @ image_feature.T).reshape((-1,))
        values, indices = similarity.topk(result_count)
        result = PrettyTable()
        result.field_names = ["路径", "置信度"]

        for i in range(result_count):
            result.add_row([self.image_paths[indices[i]], values[i].item()])
        print(result)
        return [self.image_paths[i] for i in indices]

    def Save(self, path="."):
        image_paths_file = open(path + "/image_paths.txt", "w")
        image_paths_file.writelines(path + "\n" for path in self.image_paths)
        image_paths_file.close()
        torch.save(self.image_features, path + "/image_features.pt")
        print("保存成功:" + path + "/image_paths.txt" + path + "," + "/image_features.pt")

    def Load(self, path="."):
        paths_file = open(path + "/image_paths.txt", "r")
        self.image_paths = paths_file.readlines()
        self.image_paths = [path.rstrip() for path in self.image_paths]
        paths_file.close()
        self.image_features = torch.load(path + "/image_features.pt")
        print("加载成功")
        print(self.image_features)