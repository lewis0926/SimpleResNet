import torch.optim.lr_scheduler
import time


def acc_func(logits, labels):
    pred, pred_class_id = torch.max(logits, dim=1)
    return torch.tensor(torch.sum(pred_class_id == labels).item() / len(logits))


def train(model, train_loader, validation_loader, epochs, max_lr, loss_func, optim):
    start_time = time.time()

    optimizer = optim(model.parameters(), max_lr)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr, epochs * len(train_loader))

    epoch_train_losses = []
    epoch_train_accs = []
    epoch_valid_losses = []
    epoch_valid_accs = []

    for epoch in range(epochs):
        model.train()
        train_losses, train_accs = [], []
        valid_losses, valid_accs = [], []

        lrs = []

        for i, (images, labels) in enumerate(train_loader):
            logits = model(images)
            loss = loss_func(logits, labels)
            accuracy = acc_func(logits, labels)
            train_losses.append(loss)
            train_accs.append(accuracy)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            lrs.append(optimizer.param_groups[0]["lr"])
            scheduler.step()

            if (i + 1) % 100 == 0:
                print("Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Acc: {:.4f}"
                      .format(epoch + 1, epochs, i + 1, len(train_loader), loss.item(), accuracy.item()))

        epoch_train_loss = torch.stack(train_losses).mean().item()
        epoch_train_acc = torch.stack(train_accs).mean().item()
        epoch_train_losses.append(epoch_train_loss)
        epoch_train_accs.append(epoch_train_acc)

        print("Avg train loss: {:.4f}, Avg train acc: {:.4f}"
              .format(epoch_train_loss, epoch_train_acc))

        model.eval()
        for images, labels in validation_loader:
            with torch.no_grad():
                logits = model(images)
            loss = loss_func(logits, labels)
            accuracy = acc_func(logits, labels)
            valid_losses.append(loss)
            valid_accs.append(accuracy)

        epoch_valid_loss = torch.stack(valid_losses).mean().item()
        epoch_valid_acc = torch.stack(valid_accs).mean().item()
        epoch_valid_losses.append(epoch_valid_loss)
        epoch_valid_accs.append(epoch_valid_acc)

        print("Avg validation loss: {:.4f}, Avg validation acc: {:.4f}"
              .format(epoch_valid_loss, epoch_valid_acc))

    end_time = time.time()

    results = {"epoch_train_losses": epoch_train_losses,
               "epoch_train_accs": epoch_train_accs,
               "epoch_valid_losses": epoch_valid_losses,
               "epoch_valid_accs": epoch_valid_accs,
               "computation_time": end_time - start_time,
               "learning_rate": lrs}

    return results


def train_constant_lr(model, train_loader, validation_loader, epochs, lr, loss_func, optim):
    start_time = time.time()

    optimizer = optim(model.parameters(), lr)

    epoch_train_losses = []
    epoch_train_accs = []
    epoch_valid_losses = []
    epoch_valid_accs = []

    for epoch in range(epochs):
        model.train()
        train_losses, train_accs = [], []
        valid_losses, valid_accs = [], []

        for i, (images, labels) in enumerate(train_loader):
            logits = model(images)
            loss = loss_func(logits, labels)
            accuracy = acc_func(logits, labels)
            train_losses.append(loss)
            train_accs.append(accuracy)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            if (i + 1) % 100 == 0:
                print("Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Acc: {:.4f}"
                      .format(epoch + 1, epochs, i + 1, len(train_loader), loss.item(), accuracy.item()))

        epoch_train_loss = torch.stack(train_losses).mean().item()
        epoch_train_acc = torch.stack(train_accs).mean().item()
        epoch_train_losses.append(epoch_train_loss)
        epoch_train_accs.append(epoch_train_acc)

        print("Avg train loss: {:.4f}, Avg train acc: {:.4f}"
              .format(epoch_train_loss, epoch_train_acc))

        model.eval()
        for images, labels in validation_loader:
            with torch.no_grad():
                logits = model(images)
            loss = loss_func(logits, labels)
            accuracy = acc_func(logits, labels)
            valid_losses.append(loss)
            valid_accs.append(accuracy)

        epoch_valid_loss = torch.stack(valid_losses).mean().item()
        epoch_valid_acc = torch.stack(valid_accs).mean().item()
        epoch_valid_losses.append(epoch_valid_loss)
        epoch_valid_accs.append(epoch_valid_acc)

        print("Avg validation loss: {:.4f}, Avg validation acc: {:.4f}"
              .format(epoch_valid_loss, epoch_valid_acc))

    end_time = time.time()

    results = {"epoch_train_losses": epoch_train_losses,
               "epoch_train_accs": epoch_train_accs,
               "epoch_valid_losses": epoch_valid_losses,
               "epoch_valid_accs": epoch_valid_accs,
               "computation_time": end_time - start_time}

    return results
