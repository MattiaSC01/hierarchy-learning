import torch
from rhm.datasets.hierarchical import sample_hierarchical_rules, sample_data_from_paths


m = 3
num_classes = 1
num_features = 10
num_layers = 4
s = 2
seed = 0

max_dataset_size = 10
Pmax = m ** ((s ** num_layers - 1) // (s - 1)) * num_classes


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


x, targets = sample_data_from_paths(
    samples_indices=sample_indices,
    paths=all_level_paths,
    m=m,
    num_classes=num_classes,
    num_layers=num_layers,
    s=s,
    seed=seed,
    seed_reset_layer=seed,
)

print(x.shape, targets.shape)
