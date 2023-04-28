import torch
from torch import nn
from torch.utils.data import DataLoader
# import torch.optim as optim


from prepare_data import prepare_opportunity_objects_acc, get_object_usage_feature_vector
from ObjBinaryNN import ObjBinaryNN
from train import train
from MyDataset import MyDataset

data_path = "../data/OpportunityUCIDataset/dataset"

# data prepration and feature extraction
activity_samples = prepare_opportunity_objects_acc(data_path)
activity_samples_features = get_object_usage_feature_vector(activity_samples)

# dataset setup
activity_dataset = MyDataset(activity_samples_features)
data_loader = DataLoader(activity_dataset, batch_size=1)

# training the neural network
model = ObjBinaryNN().to("cuda")

# loss function
loss_fn = nn.CrossEntropyLoss()

# optimizer
# optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)


# training
model.train(True)


EPOCHS = 50

for epoch in range(EPOCHS):
    print('EPOCH {}:'.format(epoch + 1))

    # Make sure gradient tracking is on, and do a pass over the data
    model.train(True)
    avg_loss = train(data_loader, model, loss_fn, optimizer)


    # running_vloss = 0.0
    # for i, vdata in enumerate(validation_loader):
    #     vinputs, vlabels = vdata
    #     voutputs = model(vinputs)
    #     vloss = loss_fn(voutputs, vlabels)
    #     running_vloss += vloss

    print('LOSS train {}'.format(avg_loss))


