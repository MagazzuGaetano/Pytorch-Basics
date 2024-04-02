import torch


def test_loop(dataloader, model, loss_fn, device, metric):
    model.eval()
    num_batches = len(dataloader)
    test_loss = 0

    with torch.no_grad():
        for X, y in dataloader:
            X = X.to(device)
            y = y.to(device)

            pred = model(X)
            test_loss = test_loss + loss_fn(pred, y).item()
            metric.update(pred, y)

    test_loss /= num_batches
    f1 = metric.compute()

    print(f"Test Epoch: \n Avg loss: {test_loss:>8f}, Avg metric: {f1:.2%} \n")

    metric.reset()

    return test_loss, f1
