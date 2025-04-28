from model import TransformerClassifier
from dataset import MNISTDataset
from torch.utils.data import DataLoader
import torch
from torch import nn
import wandb
import itertools

if __name__ == "__main__":
    torch.manual_seed(42)

    train_dataset = MNISTDataset(split="train")
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)

    test_dataset = MNISTDataset(split="test")
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)
    
    embedding_dim = 64
    num_heads = 4
    num_layers = 3
    learning_rate = 0.001
    num_epochs = 10

    model = TransformerClassifier(embedding_dim, num_heads, num_layers, 10)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    wandb.init(project="mnist-transformer", config={
        "embedding_dim": embedding_dim,
        "num_heads": num_heads,
        "num_layers": num_layers,
        "learning_rate": learning_rate,
        "num_epochs": num_epochs
    })

    interval = 100
    for epoch in range(num_epochs):
        epoch_loss = []
        epoch_test_loss = []
        epoch_test_accuracy = []

        for i, (train_batch, test_batch) in enumerate(zip(train_loader, itertools.cycle(test_loader))):
            model.train()
            images, labels = train_batch
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                model.eval()
                test_images, test_labels = test_batch
                test_outputs = model(test_images)
                test_loss = criterion(test_outputs, test_labels)
                test_accuracy = (test_outputs.argmax(dim=1) == test_labels).float().mean()

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
