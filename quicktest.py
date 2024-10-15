import torch

# Example tensor
x = torch.tensor([[1.0, 3.0, 2.0, 7.0, 4.0, 5.0]])

# Set the threshold
threshold = 3.5

# Get the indices of values greater than the threshold
indices = (x > threshold).nonzero(as_tuple=True)[-1]

for i in indices:
    print(int(i))

print("Indices of values greater than the threshold:", indices)
