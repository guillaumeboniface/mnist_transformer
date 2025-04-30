from model import MNISTTransformer
from dataset import ScatteredMNISTDataset, label_to_index, index_to_label
from torch.utils.data import DataLoader
import torch
from torch import nn
import wandb
import itertools
from safetensors.torch import save_file
import os

if __name__ == "__main__":
    torch.manual_seed(42)

    train_dataset = ScatteredMNISTDataset(split="train")
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    
    test_dataset = ScatteredMNISTDataset(split="test")
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)
    
    image_embedding_dim = 64
    text_embedding_dim = 16
    num_heads = 4
    num_layers = 3
    learning_rate = 0.001
    num_epochs = 10
    model = MNISTTransformer(text_embedding_dim, image_embedding_dim, num_heads, num_layers, len(label_to_index), img_size=128, patch_size=16)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    wandb.init(project="mnist-scattered", config={
        "text_embedding_dim": text_embedding_dim,
        "image_embedding_dim": image_embedding_dim,
        "num_heads": num_heads,
        "num_layers": num_layers,
        "learning_rate": learning_rate,
        "model": str(model)
    })

    interval = 100
    for epoch in range(num_epochs):
        epoch_loss = []
        epoch_test_loss = []
        epoch_test_accuracy = []
        for i, (batch, test_batch) in enumerate(zip(train_loader, itertools.cycle(test_loader))):
            model.train()
            image, input_text, output_text = batch

            optimizer.zero_grad()
            output = model(input_text, image)
            loss = criterion(output.transpose(1, 2), output_text)
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                model.eval()
                test_image, test_input_text, test_output_text = test_batch
                test_output = model(test_input_text, test_image)
                test_loss = criterion(test_output.transpose(1, 2), test_output_text)
                test_accuracy = (test_output.argmax(dim=2) == test_output_text).float().mean()
                
            epoch_loss.append(loss.item())
            epoch_test_loss.append(test_loss.item())
            epoch_test_accuracy.append(test_accuracy.item())            
            wandb.log({
                "train_loss": loss.item(),
                "test_loss": test_loss.item(),
                "test_accuracy": test_accuracy.item()
            })

            print(f"\r[Epoch {epoch+1}/{num_epochs}], [Step {i+1}/{len(train_loader)}], Train Loss: {loss.item():.4f}, Test Loss: {test_loss.item():.4f}, Test Accuracy: {test_accuracy.item():.4f}", end="")
            
            if i != 0 and i % interval == 0:
                print(f"\r[Epoch {epoch+1}/{num_epochs}], [Step {i+1}/{len(train_loader)}], Train Loss: {torch.tensor(epoch_loss[-interval:]).mean().item():.4f}, Test Loss: {torch.tensor(epoch_test_loss[-interval:]).mean().item():.4f}, Test Accuracy: {torch.tensor(epoch_test_accuracy[-interval:]).mean().item():.4f}")
            
        print(f"\n[Epoch {epoch+1}/{num_epochs}], Train Loss: {torch.tensor(epoch_loss).mean().item():.4f}, Test Loss: {torch.tensor(epoch_test_loss).mean().item():.4f}, Test Accuracy: {torch.tensor(epoch_test_accuracy).mean().item():.4f}")
        model_path = f"model/{wandb.run.name}/transformer_{epoch}.safetensors"
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        save_file(model.state_dict(), model_path)
                