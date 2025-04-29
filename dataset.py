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

class ScatteredMNISTDataset(Dataset):
    def __init__(self, split="train", max_n=4):
        self.dataset = datasets.load_dataset("ylecun/mnist")[split]
        self.max_n = max_n
        self.transform = transforms.ToTensor()
        self.digit_size = 28  # MNIST digits are 28x28
        self.canvas_size = 128
        
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        n = torch.randint(0, self.max_n + 1, (1,)).item()
        
        canvas = torch.zeros((1, self.canvas_size, self.canvas_size))
        
        positions = []
        labels = []

        # Create a pool to randomly pick positions from (top right corner of the digit)
        trimmed_canvas_size = self.canvas_size - self.digit_size + 1
        pixel_pool = torch.arange(trimmed_canvas_size ** 2)
        # Create a mask to exclude positions that are already occupied
        mask = torch.ones((trimmed_canvas_size ** 2), dtype=torch.bool)
        for _ in range(n):
            idx = torch.randint(0, len(self.dataset), (1,)).item()
            # Randomly pick a position from the pool
            masked_pool = pixel_pool.masked_select(mask)
            pos = torch.randint(0, masked_pool.shape[0], (1,)).item()
            pos = masked_pool[pos]
            x = pos % trimmed_canvas_size
            y = pos // trimmed_canvas_size
            # Place the digit on the canvas
            canvas[:, y:y+self.digit_size, x:x+self.digit_size] = self.transform(self.dataset[idx]["image"])
            positions.append((x, y))
            labels.append(self.dataset[idx]["label"])
            # Update the mask to exclude all the positions that are occupied
            mask.view((trimmed_canvas_size, trimmed_canvas_size))[y:min(y+self.digit_size, trimmed_canvas_size), x:min(x+self.digit_size, trimmed_canvas_size)] = False
        
        # Create input and target sequences
        labels = [label_to_index["<start>"]] + sorted(labels) + [label_to_index["<end>"]]
        labels = labels + [label_to_index["<pad>"]] * max(self.max_n + 2 - len(labels), 0)
        input_seq = torch.tensor(labels[:self.max_n+1], dtype=torch.long)
        target_seq = torch.tensor(labels[1:], dtype=torch.long)
        
        return canvas, input_seq, target_seq

if __name__ == "__main__":
    from torchvision.transforms.functional import to_pil_image
    torch.manual_seed(42)
    # dataset = MNISTDataset()
    # print(dataset[0])
    # tiled_dataset = TiledMNISTDataset()
    # sample = tiled_dataset[0]
    # print(sample[0].shape)
    # print(sample[1])
    # print(sample[2])
    # assert (sample[1][1:] == sample[2][:-1]).all()

    scattered_dataset = ScatteredMNISTDataset()
    for i in range(40):
        sample = scattered_dataset[i]
        print(f"sample {i}: {sample[1]}")
        assert sample[1].shape == sample[2].shape == (5,)
        assert (sample[1][1:] == sample[2][:-1]).all()
        to_pil_image(sample[0]).save(f"samples/scattered_sample_{i}.png")
        