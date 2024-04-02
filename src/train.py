def train_loop(dataloader, model, loss_fn, optimizer, device, metric, train_print_freq):
    num_batches = len(dataloader)
    model.train()

    train_loss = 0
    for batch, (X, y) in enumerate(dataloader):
        X = X.to(device)
        y = y.to(device)

        pred = model(X)
        loss = loss_fn(pred, y)
        train_loss += loss.item()

        f1 = metric(pred.argmax(1), y.argmax(1))

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % train_print_freq == 0:
            print(
                f"[{batch:>5d}/{num_batches:>5d}] loss: {loss.item():>7f}, metric: {f1:.2%}"
            )

    train_loss /= num_batches
    f1 = metric.compute()

    print(f"Train Epoch: \n Avg loss: {loss:>7f}, Avg metric: {f1:.2%}")

    metric.reset()
