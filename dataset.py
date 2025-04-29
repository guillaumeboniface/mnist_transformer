import torch
from torch.utils.data import Dataset
import datasets
import torchvision.transforms as transforms

label_to_index = {
    "0": 0,
    "1": 1,
    "2": 2,
    "3": 3,
    "4": 4,
    "5": 5,
    "6": 6,
    "7": 7,
    "8": 8,
    "9": 9,
    "<start>": 10,
    "<end>": 11,
    "<pad>": 12
}

index_to_label = {v: k for k, v in label_to_index.items()}

class MNISTDataset(Dataset):
    def __init__(self, split="train"):
        self.dataset = datasets.load_dataset("ylecun/mnist")[split]
        self.transform = transforms.ToTensor()
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        return self.transform(self.dataset[idx]["image"]), torch.tensor(self.dataset[idx]["label"], dtype=torch.long)

class TiledMNISTDataset(Dataset):
    def __init__(self, split="train", tile_size=4):
        self.dataset = datasets.load_dataset("ylecun/mnist")[split]
        self.tile_size = tile_size
        self.transform = transforms.ToTensor()
        
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        indices = torch.randint(0, len(self.dataset), (4,)).tolist()
        images = []
        labels = []
        for i in indices:
            images.append(self.transform(self.dataset[i]["image"]))
            labels.append(self.dataset[i]["label"])
        tile = torch.cat([torch.cat(images[:2], dim=2), torch.cat(images[2:], dim=2)], dim=1)
        return tile, torch.tensor([label_to_index["<start>"]] + labels, dtype=torch.long), torch.tensor(labels + [label_to_index["<end>"]], dtype=torch.long)

if __name__ == "__main__":
    dataset = MNISTDataset()
    print(dataset[0])
    tiled_dataset = TiledMNISTDataset()
    sample = tiled_dataset[0]
    print(sample[0].shape)
    print(sample[1])
    print(sample[2])
    assert (sample[1][1:] == sample[2][:-1]).all()
        
        