import matplotlib.pyplot as plt
import scipy.io as sio
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


# ==================================================
# DATA LOADING
# ==================================================
def load_dataset():

    print("Loading dataset...")

    data = sio.loadmat('CSI_dataset.mat')
    H = data['H_dataset']

    print("Original shape:", H.shape)

    # Fix dimensions
    H = H.transpose(2,0,1)

    print("Transposed shape:", H.shape)

    # Separate real and imaginary parts
    H_real = np.real(H)
    H_imag = np.imag(H)

    H = np.concatenate([H_real, H_imag], axis=2)

    print("Combined shape:", H.shape)

    # Flatten
    X = H.reshape(H.shape[0], -1)

    print("Flattened shape:", X.shape)

    # Normalize
    X = X / np.max(np.abs(X))

    return X.astype(np.float32)


# ==================================================
# AUTOENCODER MODEL
# ==================================================
class Autoencoder(nn.Module):

    def __init__(self):

        super(Autoencoder, self).__init__()

        self.encoder = nn.Sequential(

            nn.Linear(2048,1024),
            nn.ReLU(),

            nn.Linear(1024,512),
            nn.ReLU(),

            nn.Linear(512,256),
            nn.ReLU(),

            nn.Linear(256,128)
        )

        self.decoder = nn.Sequential(

            nn.Linear(128,256),
            nn.ReLU(),

            nn.Linear(256,512),
            nn.ReLU(),

            nn.Linear(512,1024),
            nn.ReLU(),

            nn.Linear(1024,2048)
        )


    def forward(self,x):

        encoded = self.encoder(x)

        decoded = self.decoder(encoded)

        return decoded


# ==================================================
# TRAINING
# ==================================================
def train_model():

    X = load_dataset()

    X_tensor = torch.tensor(X)

    dataset = TensorDataset(X_tensor, X_tensor)

    loader = DataLoader(dataset, batch_size=128, shuffle=True)

    model = Autoencoder()

    optimizer = torch.optim.Adam(model.parameters(), lr=0.0003)

    criterion = nn.MSELoss()

    epochs = 100

    print("Starting training...")

    loss_history = []

    for epoch in range(epochs):

        total_loss = 0

        for batch_x,_ in loader:

            output = model(batch_x)

            loss = criterion(output, batch_x)

            optimizer.zero_grad()

            loss.backward()

            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(loader)

        loss_history.append(avg_loss)

        print(f"Epoch {epoch+1}/{epochs}  Loss: {avg_loss:.6f}")


    torch.save(model.state_dict(),"autoencoder_model.pth")

    print("Model saved.")


    plt.figure()

    plt.plot(loss_history)

    plt.title("Training Loss")

    plt.xlabel("Epoch")

    plt.ylabel("MSE")

    plt.savefig("training_loss.png")

    plt.show()


# ==================================================
# MAIN
# ==================================================
if __name__ == "__main__":

    train_model()