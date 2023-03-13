import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from tqdm.auto import tqdm

num_workers = 4
best_model_path = "model_checkpoints/best_model.pt"


def fit(
    model,
    X,
    y,
    device,
    epochs=10,
    batch_size=64,
    lr=1e-3,
    weight_decay=0,
    valid_size=0.2,
    random_state=0,
    print_acc=False,
):
    # split train and valid sets
    X_train, X_valid, y_train, y_valid = train_test_split(
        X, y, test_size=valid_size, random_state=random_state
    )

    # prepare dataloaders
    train_dataloader = _get_dataloader(X_train, y_train, batch_size=batch_size, shuffle=True)

    # train model
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    best_acc = 0

    for _ in tqdm(range(epochs)):
        train_acc = _train_one_epoch(model, train_dataloader, optimizer, device)
        valid_acc, _ = evaluate_and_get_prediction(model, X_valid, y_valid, device)
        if print_acc:
            print(f"{train_acc=}, {valid_acc=}")

        if valid_acc > best_acc:
            best_acc = valid_acc
            torch.save(model.state_dict(), best_model_path)

    # load the best model back
    model.load_state_dict(torch.load(best_model_path))
    print(f"Best valid accuracy: {best_acc}")
    
    return best_acc


def _get_dataloader(X, y, batch_size, shuffle):
    dataset = TensorDataset(torch.FloatTensor(X), torch.LongTensor(y))
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)


def _train_one_epoch(model, dataloader, optimizer, device):
    model.train()

    loss_fn = nn.CrossEntropyLoss()
    correct = total = 0

    for batch_X, batch_y in dataloader:
        optimizer.zero_grad()

        batch_X = batch_X.to(device)
        batch_y = batch_y.to(device)
        output = model(batch_X)
        
        _, batch_y_pred = torch.max(output, 1)
        correct += (batch_y_pred == batch_y).sum().item()
        total += batch_y.size(0)

        loss = loss_fn(output, batch_y)
        loss.backward()
        optimizer.step()

    return round(correct / total, 5)


def evaluate_and_get_prediction(model, X, y, device, batch_size=256):
    model.eval()

    dataloader = _get_dataloader(X, y, batch_size=batch_size, shuffle=False)

    correct = total = 0
    y_pred = []

    for batch_X, batch_y in dataloader:
        batch_X = batch_X.to(device)
        batch_y = batch_y.to(device)

        with torch.no_grad():
            output = model(batch_X)
            _, batch_y_pred = torch.max(output, 1)

        correct += (batch_y_pred == batch_y).sum().item()
        total += batch_y.size(0)
        y_pred += batch_y_pred.tolist()

    return round(correct / total, 5), y_pred
