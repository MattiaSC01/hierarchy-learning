import torch
from torch.utils.data import DataLoader
from rhm.datasets.hierarchical import sample_hierarchical_rules, sample_data_from_paths
from rhm.datasets import RandomHierarchyModel


m = 3
num_classes = 4
num_features = 100
num_layers = 3
s = 2
seed = 0

max_dataset_size = 10
Pmax = m ** ((s ** num_layers - 1) // (s - 1)) * num_classes
print(f"pmax: {Pmax}")


all_level_paths, all_level_tuples = sample_hierarchical_rules(
    num_features=num_features,
    num_layers=num_layers,
    m=m,
    num_classes=num_classes,
    s=s,
    seed=seed,
)

for i, path in enumerate(all_level_paths):
    print(f"{i}-th path: {path.shape}")
print()
for i, tup in enumerate(all_level_tuples):
    print(f"{i}-th tuple: {tup.shape}")


g = torch.Generator()
g.manual_seed(seed)
sample_indices = torch.randperm(Pmax, generator=g)[:max_dataset_size]
print(f"\nsample indices: {sample_indices}")


output = sample_data_from_paths(
    samples_indices=sample_indices,
    paths=all_level_paths,
    m=m,
    num_classes=num_classes,
    num_layers=num_layers,
    s=s,
    seed=seed,
    seed_reset_layer=42,
)

x, y, labels = output['x'], output['y'], output['labels']
print(f"x: {x.shape}")
print(f"y: {y}")
print(f"labels:\n{labels}")


dataset = RandomHierarchyModel(
    num_features=num_features,
    m=m,
    num_layers=num_layers,
    num_classes=num_classes,
    s=s,
    input_format='onehot',
    seed=seed,
)

# dl = DataLoader(dataset, batch_size=2, shuffle=False)
# for batch in dl:
#     print(batch)
#     break
