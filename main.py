import torch as tc
import torchvision as tv
import matplotlib.pyplot as plt
import numpy as np

training_data = tv.datasets.FashionMNIST(
    root='data', train=True, download=True, transform=tv.transforms.ToTensor())

test_data = tv.datasets.FashionMNIST(
    root='data', train=False, download=True, transform=tv.transforms.ToTensor())

# Create data loaders.
batch_size = 64
train_dataloader = tc.utils.data.DataLoader(training_data, batch_size=batch_size, shuffle=True)
test_dataloader = tc.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=True)


class Encoder(tc.nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.flattener = tc.nn.Flatten()
        self.fc_stack = tc.nn.Sequential(
            tc.nn.Linear(28*28, 512),
            tc.nn.ReLU(),
            tc.nn.Linear(512, 512),
            tc.nn.ReLU(),
            tc.nn.Linear(512, 100),
            tc.nn.ReLU()
        )

    def forward(self, x):
        flat = self.flattener(x)
        code = self.fc_stack(flat)
        return code


class Decoder(tc.nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.fc_stack = tc.nn.Sequential(
            tc.nn.Linear(100, 512),
            tc.nn.ReLU(),
            tc.nn.Linear(512, 512),
            tc.nn.Linear(512, 28 * 28 * 1)
        )

    def forward(self, code):
        logits_flat = self.fc_stack(code)
        logits_square = tc.reshape(logits_flat, (-1, 1, 28, 28))
        return logits_square


class AutoEncoder(tc.nn.Module):
    def __init__(self):
        super(AutoEncoder, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self, x):
        code = self.encoder(x)
        decoded_logits = self.decoder(code)
        decoded_probs = tc.nn.Sigmoid()(decoded_logits)
        return decoded_probs

    def encode(self, x):
        return self.encoder(x)

    def decode(self, x):
        decoded_logits = self.decoder(x)
        decode_probabilities = tc.nn.Sigmoid()(decoded_logits)
        return decode_probabilities


device = "cuda" if tc.cuda.is_available() else "cpu"
model = AutoEncoder().to(device)
print(model)

loss_fn = tc.nn.BCELoss()
optimizer = tc.optim.SGD(model.parameters(), lr=1.0)


def train(dataloader, model, loss_fn, optimizer):
    num_training_examples = len(dataloader.dataset)

    for batch, (X, _) in enumerate(dataloader):
        X = X.to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, X)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss = loss.item()
            current_idx = batch_size * (batch-1) + len(X)
            print(f"loss: {loss:>7f}  [{current_idx:>5d}/{num_training_examples:>5d}]")


def test(dataloader, model):
    num_test_examples = len(dataloader.dataset)
    model.eval()
    test_loss, correct = 0, 0
    with tc.no_grad():
        for X, _ in dataloader:
            X = X.to(device)
            pred = model(X)
            test_loss += len(X) * loss_fn(pred, X).item()
    test_loss /= num_test_examples
    print(f"Test Error: Avg loss: {test_loss:>8f} \n")


epochs = 5
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train(train_dataloader, model, loss_fn, optimizer)
    test(test_dataloader, model)

print("Done!")

for X, y in test_dataloader:
    print("Shape of X [N, C, H, W]: ", X.shape)
    print("Shape of y: ", y.shape, y.dtype)
    input_example = X[0]
    output_example = model.forward(tc.unsqueeze(input_example, dim=0)).detach()[0]
    img = np.transpose(np.concatenate([input_example, output_example], axis=-1), axes=[1,2,0])
    img_3channel = np.concatenate([img for _ in range(0,3)], axis=-1)
    plt.imshow(img_3channel)
    plt.show()
    break