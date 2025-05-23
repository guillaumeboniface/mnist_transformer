from model import MNISTLocationTransformer
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

    train_dataset = ScatteredMNISTDataset(split="train", location_target=True)
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    
    test_dataset = ScatteredMNISTDataset(split="test", location_target=True)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)
    
    image_embedding_dim = 64
    text_embedding_dim = 16
    num_heads = 4
    num_layers = 3
    learning_rate = 0.001
    num_epochs = 50
    model = MNISTLocationTransformer(text_embedding_dim, image_embedding_dim, num_heads, num_layers, len(label_to_index), img_size=128, patch_size=16)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    wandb.init(project="mnist-location", config={
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
        epoch_test_location_accuracy = []
        for i, (batch, test_batch) in enumerate(zip(train_loader, itertools.cycle(test_loader))):
            model.train()
            image, input_text, output_text, location = batch

            optimizer.zero_grad()
            output, location_pred = model(input_text, image)
            classification_loss = criterion(output.transpose(1, 2), output_text)
            location_loss = criterion(location_pred.transpose(1, 3), location.transpose(1, 2))
            loss = classification_loss + location_loss
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                model.eval()
                test_image, test_input_text, test_output_text, test_location = test_batch
                test_output, test_location_pred = model(test_input_text, test_image)
                test_classification_loss = criterion(test_output.transpose(1, 2), test_output_text)
                test_location_loss = criterion(test_location_pred.transpose(1, 3), test_location.transpose(1, 2))
                test_loss = test_classification_loss + test_location_loss
                test_accuracy = (test_output.argmax(dim=2) == test_output_text).float().mean()
                test_location_accuracy = (test_location_pred.argmax(dim=3) == test_location).float().mean()
                
            epoch_loss.append(loss.item())
            epoch_test_loss.append(test_loss.item())
            epoch_test_accuracy.append(test_accuracy.item())
            epoch_test_location_accuracy.append(test_location_accuracy.item())
            wandb.log({
                "train_loss": loss.item(),
                "test_loss": test_loss.item(),
                "test_accuracy": test_accuracy.item(),
                "test_location_accuracy": test_location_accuracy.item()
            })

            print(f"\r[Epoch {epoch+1}/{num_epochs}], [Step {i+1}/{len(train_loader)}], Train Loss: {loss.item():.4f}, Test Loss: {test_loss.item():.4f}, Test Accuracy: {test_accuracy.item():.4f}, Test Location Accuracy: {test_location_accuracy.item():.4f}", end="")
            
            if i != 0 and i % interval == 0:
                print(f"\r[Epoch {epoch+1}/{num_epochs}], [Step {i+1}/{len(train_loader)}], Train Loss: {torch.tensor(epoch_loss[-interval:]).mean().item():.4f}, Test Loss: {torch.tensor(epoch_test_loss[-interval:]).mean().item():.4f}, Test Accuracy: {torch.tensor(epoch_test_accuracy[-interval:]).mean().item():.4f}, Test Location Accuracy: {torch.tensor(epoch_test_location_accuracy[-interval:]).mean().item():.4f}")
            
        print(f"\n[Epoch {epoch+1}/{num_epochs}], Train Loss: {torch.tensor(epoch_loss).mean().item():.4f}, Test Loss: {torch.tensor(epoch_test_loss).mean().item():.4f}, Test Accuracy: {torch.tensor(epoch_test_accuracy).mean().item():.4f}, Test Location Accuracy: {torch.tensor(epoch_test_location_accuracy).mean().item():.4f}")
        model_path = f"model/{wandb.run.name}/transformer_{epoch}.safetensors"
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        save_file(model.state_dict(), model_path)
                