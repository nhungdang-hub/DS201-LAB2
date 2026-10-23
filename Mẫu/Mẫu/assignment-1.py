import os 
os.makedirs("./checkpoints/assignment-1", exist_ok=True)

import torch
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch import nn, optim
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from tqdm import tqdm

from data_utils.mnist_dataset import MnistDataset, collate_fn
from models.lenet import LeNet

device = "cuda" if torch.cuda.is_available() else "cpu"

def evaluate(dataloader: DataLoader, model: nn.Module) -> dict:
    model.eval()
    predictions = []
    trues = []
    with torch.no_grad():
        for item in tqdm(dataloader, desc="Evaluating"):
            image = item["image"].to(device)
            label = item["label"].to(device)
            output = model(image) 
            output = output.argmax(dim=-1)

            predictions.append(output.item())
            trues.append(label.item())

    return {
        "accuracy": accuracy_score(trues, predictions),
        "precision": precision_score(trues, predictions, average="macro"),
        "recall": recall_score(trues, predictions, average="macro"),
        "f1": f1_score(trues, predictions, average="macro")
    }

if __name__ == "__main__":
    train_dataset = MnistDataset(
        image_path="./dataset/MNIST/train-images-idx3-ubyte",
        label_path="./dataset/MNIST/train-labels-idx1-ubyte"
    )

    test_dataset = MnistDataset(
        image_path="./dataset/MNIST/t10k-images-idx3-ubyte",
        label_path="./dataset/MNIST/t10k-labels-idx1-ubyte"
    )

    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=32,
        shuffle=True,
        collate_fn=collate_fn
    )

    test_dataloader = DataLoader(
        dataset=test_dataset,
        batch_size=1,
        shuffle=True,
        collate_fn=collate_fn
    )

    model = LeNet().to(device)
    loss_fn = nn.CrossEntropyLoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    EPOCHS = 10
    best_score = 0
    best_score_name = "f1"
    for epoch in range(EPOCHS):
        losses = []
        model.train()
        with tqdm(train_dataloader, desc=f"Epoch {epoch+1} - Training") as pb:
            for item in pb:
                image = item["image"].to(device)
                label = item["label"].to(device)
                
                # forward pass
                output = model(image)
                loss = loss_fn(output, label)
                losses.append(loss.item())

                # backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                pb.set_postfix({
                    "loss": np.array(losses).mean()
                })
        
        scores = evaluate(test_dataloader, model)
        for score in scores:
            print(f"\t- {score}: {scores[score]}")
            
        current_score = scores[best_score_name]
        if current_score > best_score:
            best_score = current_score
            torch.save(
                model.state_dict(),
                "./checkpoints/assignment-1/best_model.pth"
            )
