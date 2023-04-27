from prepare_data import prepare_opportunity_objects_acc, get_object_usage_feature_vector

data_path = "../data/OpportunityUCIDataset/dataset"


activity_samples = prepare_opportunity_objects_acc(data_path)
activity_samples_features = get_object_usage_feature_vector(activity_samples)

x = 1

