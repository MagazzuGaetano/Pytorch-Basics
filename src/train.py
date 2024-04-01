def train_loop(dataloader, model, loss_fn, optimizer, device):
    num_batches = len(dataloader)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        pred = model(X.to(device))
        loss = loss_fn(pred, y.to(device))

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 50 == 0:
            loss = loss.item()
            print(f"loss: {loss:>7f}  [{batch:>5d}/{num_batches:>5d}]")
