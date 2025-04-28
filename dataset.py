import torch
from torch.utils.data import Dataset
import datasets
import torchvision.transforms as transforms

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
        return tile, torch.tensor(labels, dtype=torch.long)
    
if __name__ == "__main__":
    dataset = MNISTDataset()
    print(dataset[0])
    tiled_dataset = TiledMNISTDataset()
    print(tiled_dataset[0])
    print(tiled_dataset[0][0].shape)
        
        