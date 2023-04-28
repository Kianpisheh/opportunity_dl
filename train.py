import numpy as np
import torch

def train(dataloader, model, loss_fn, optimizer):


    N = len(dataloader.dataset)
    model.train()
    avg_loss = 0
    confusion_matrix = np.zeros((4,4)) # row: predicted, col: actual


    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to("cuda", dtype=torch.float), y.to("cuda", dtype=torch.float)

        y = int(y)-102 # match labels

        # Compute prediction error
        pred = model(X)
        y_onehot = torch.zeros(1, 4).to("cuda")
        y_onehot[0,y] = 1.
        loss = loss_fn(pred, y_onehot)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        avg_loss += loss.item()
        loss_val, current = loss.item(), (batch + 1) * len(X)
        # if batch % 10 == 0:
        #     print(f"pred: {pred.tolist()}, label: {y}")
        #     print(f"loss: {loss_val:>7f}  [{current:>5d}/{size:>5d}]")


        # performance
        y_pred = int(pred.argmax().item())
        if y == y_pred:
            confusion_matrix[y,y] += 1
        else:
            confusion_matrix[y_pred, y] += 1
            
        


    return (avg_loss / N), confusion_matrix




# def train_one_epoch(epoch_index, tb_writer):
#     running_loss = 0.
#     last_loss = 0.

#     # Here, we use enumerate(training_loader) instead of
#     # iter(training_loader) so that we can track the batch
#     # index and do some intra-epoch reporting
#     for i, data in enumerate(training_loader):
#         # Every data instance is an input + label pair
#         inputs, labels = data

#         # Zero your gradients for every batch!
#         optimizer.zero_grad()

#         # Make predictions for this batch
#         outputs = model(inputs)

#         # Compute the loss and its gradients
#         loss = loss_fn(outputs, labels)
#         loss.backward()

#         # Adjust learning weights
#         optimizer.step()

#         # Gather data and report
#         running_loss += loss.item()
#         if i % 1000 == 999:
#             last_loss = running_loss / 1000 # loss per batch
#             print('  batch {} loss: {}'.format(i + 1, last_loss))
#             tb_x = epoch_index * len(training_loader) + i + 1
#             tb_writer.add_scalar('Loss/train', last_loss, tb_x)
#             running_loss = 0.

#     return last_loss


# timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
# writer = SummaryWriter('runs/fashion_trainer_{}'.format(timestamp))
# epoch_number = 0

# EPOCHS = 5

# best_vloss = 1_000_000.

# for epoch in range(EPOCHS):
#     print('EPOCH {}:'.format(epoch_number + 1))

#     # Make sure gradient tracking is on, and do a pass over the data
#     model.train(True)
#     avg_loss = train_one_epoch(epoch_number, writer)

#     # We don't need gradients on to do reporting
#     model.train(False)

#     running_vloss = 0.0
#     for i, vdata in enumerate(validation_loader):
#         vinputs, vlabels = vdata
#         voutputs = model(vinputs)
#         vloss = loss_fn(voutputs, vlabels)
#         running_vloss += vloss

#     avg_vloss = running_vloss / (i + 1)
#     print('LOSS train {} valid {}'.format(avg_loss, avg_vloss))

#     # Log the running loss averaged per batch
#     # for both training and validation
#     writer.add_scalars('Training vs. Validation Loss',
#                     { 'Training' : avg_loss, 'Validation' : avg_vloss },
#                     epoch_number + 1)
#     writer.flush()

#     # Track best performance, and save the model's state
#     if avg_vloss < best_vloss:
#         best_vloss = avg_vloss
#         model_path = 'model_{}_{}'.format(timestamp, epoch_number)
#         torch.save(model.state_dict(), model_path)

#     epoch_number += 1
