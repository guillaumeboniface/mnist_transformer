from model import MNISTTransformer
from safetensors.torch import load_file
from dataset import TiledMNISTDataset, label_to_index
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

if __name__ == "__main__":
    token_length = 4
    batch_size = 128
    model = MNISTTransformer(16, 64, 4, 3, 13)
    model.load_state_dict(load_file("model/summer-sunset-11/transformer_9.safetensors"))
    model.eval()

    test_dataset = TiledMNISTDataset(split="test")
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    start_sentence = torch.tensor([label_to_index["<start>"]] + [label_to_index["<pad>"]] * token_length, dtype=torch.long)

    accuracy = []
    for batch in tqdm(test_loader):
        image, _, text_output = batch
        if image.shape[0] != batch_size:
            continue
        sentence = start_sentence.repeat(batch_size, 1)
        prediction = []
        for i in range(token_length):
            output = model(sentence, image)
            output = output.argmax(dim=2).select(1, i)
            assert output.shape == (batch_size,)
            sentence[:, i+1] = output
            prediction.append(output)
        prediction = torch.stack(prediction, dim=1)
        accuracy.append((prediction == text_output[:,:-1]).to(torch.float32).mean().item())
    print(f"Accuracy: {sum(accuracy) / len(accuracy)}")
