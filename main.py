import torch
from torch import nn

from prepare_data import prepare_opportunity_objects_acc, get_object_usage_feature_vector
from ObjBinaryNN import ObjBinaryNN
from train import train
from Dataset import Dataset

data_path = "./data/OpportunityUCIDataset/dataset"

# data prepration and feature extraction
activity_samples = prepare_opportunity_objects_acc(data_path)
activity_samples_features = get_object_usage_feature_vector(activity_samples)

# dataset setup
activity_dataset = Dataset(activity_samples_features)

# training the neural network
model = ObjBinaryNN().to("cuda")

# loss function
loss_fn = nn.CrossEntropyLoss()

# optimizer
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

