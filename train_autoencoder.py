import numpy as np
import matplotlib.pyplot as plt

# Simple numpy-based autoencoder (no torch dependency)
class SimpleAutoencoder:
    def __init__(self, input_dim, encoding_dim):
        self.input_dim = input_dim
        self.encoding_dim = encoding_dim

        # Initialize weights randomly
        self.encoder_W1 = np.random.randn(input_dim, 512) * 0.01
        self.encoder_b1 = np.zeros(512)
        self.encoder_W2 = np.random.randn(512, 256) * 0.01
        self.encoder_b2 = np.zeros(256)
        self.encoder_W3 = np.random.randn(256, encoding_dim) * 0.01
        self.encoder_b3 = np.zeros(encoding_dim)

        self.decoder_W1 = np.random.randn(encoding_dim, 256) * 0.01
        self.decoder_b1 = np.zeros(256)
        self.decoder_W2 = np.random.randn(256, 512) * 0.01
        self.decoder_b2 = np.zeros(512)
        self.decoder_W3 = np.random.randn(512, input_dim) * 0.01
        self.decoder_b3 = np.zeros(input_dim)

    def relu(self, x):
        return np.maximum(0, x)

    def forward(self, x):
        # Encoder
        h1 = self.relu(np.dot(x, self.encoder_W1) + self.encoder_b1)
        h2 = self.relu(np.dot(h1, self.encoder_W2) + self.encoder_b2)
        encoded = np.dot(h2, self.encoder_W3) + self.encoder_b3

        # Decoder
        h3 = self.relu(np.dot(encoded, self.decoder_W1) + self.decoder_b1)
        h4 = self.relu(np.dot(h3, self.decoder_W2) + self.decoder_b2)
        decoded = np.dot(h4, self.decoder_W3) + self.decoder_b3

        return decoded

    def train_step(self, x, learning_rate=0.001):
        # Forward pass
        h1 = self.relu(np.dot(x, self.encoder_W1) + self.encoder_b1)
        h2 = self.relu(np.dot(h1, self.encoder_W2) + self.encoder_b2)
        encoded = np.dot(h2, self.encoder_W3) + self.encoder_b3

        h3 = self.relu(np.dot(encoded, self.decoder_W1) + self.decoder_b1)
        h4 = self.relu(np.dot(h3, self.decoder_W2) + self.decoder_b2)
        decoded = np.dot(h4, self.decoder_W3) + self.decoder_b3

        # Compute loss (MSE)
        loss = np.mean((decoded - x) ** 2)

        # Simple gradient descent (simplified backprop)
        # In practice, you'd compute proper gradients
        error = decoded - x

        # Update decoder weights (simplified)
        self.decoder_W3 -= learning_rate * np.dot(h4.T, error) / x.shape[0]
        self.decoder_b3 -= learning_rate * np.mean(error, axis=0)

        # Propagate error backward through decoder
        error_h4 = np.dot(error, self.decoder_W3.T) * (h4 > 0)
        self.decoder_W2 -= learning_rate * np.dot(h3.T, error_h4) / x.shape[0]
        self.decoder_b2 -= learning_rate * np.mean(error_h4, axis=0)

        error_h3 = np.dot(error_h4, self.decoder_W2.T) * (h3 > 0)
        self.decoder_W1 -= learning_rate * np.dot(encoded.T, error_h3) / x.shape[0]
        self.decoder_b1 -= learning_rate * np.mean(error_h3, axis=0)

        # Update encoder weights (simplified)
        error_encoded = np.dot(error_h3, self.decoder_W1.T)
        self.encoder_W3 -= learning_rate * np.dot(h2.T, error_encoded) / x.shape[0]
        self.encoder_b3 -= learning_rate * np.mean(error_encoded, axis=0)

        return loss


# ===============================
# STEP 1: LOAD CSI DATASET
# ===============================

data = np.load("channel_dataset.npy")

# The data is already the H matrix
H = data

print("Original CSI shape:", H.shape)


# ===============================
# STEP 2: PREPARE DATA FOR AI
# ===============================

# Convert complex CSI to real values
H_real = np.real(H)
H_imag = np.imag(H)

# Stack real and imaginary parts
H_combined = np.concatenate((H_real, H_imag), axis=-1)

# Flatten channel matrix for neural network
num_samples = H_combined.shape[0]
input_dim = np.prod(H_combined.shape[1:])

X = H_combined.reshape(num_samples, input_dim)

print("Neural network input shape:", X.shape)

# Normalize data
X = (X - np.mean(X, axis=0)) / (np.std(X, axis=0) + 1e-8)


# ===============================
# STEP 3: AUTOENCODER MODEL
# ===============================

encoding_dim = 64   # compression size (this represents CSI feedback bits)

model = SimpleAutoencoder(input_dim, encoding_dim)


# ===============================
# STEP 4: TRAINING SETUP
# ===============================

epochs = 50
learning_rate = 0.01
loss_history = []


# ===============================
# STEP 5: TRAIN AUTOENCODER
# ===============================

print("Training autoencoder...")
for epoch in range(epochs):
    loss = model.train_step(X, learning_rate)
    loss_history.append(loss)

    if epoch % 5 == 0:
        print(f"Epoch {epoch+1}/{epochs}, Loss: {loss:.6f}")


# ===============================
# STEP 6: PLOT TRAINING LOSS
# ===============================

plt.plot(loss_history)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Autoencoder Training Loss")
plt.savefig("training_loss.png")
print("Training loss plot saved.")


# ===============================
# STEP 7: SAVE MODEL
# ===============================

# Save model weights as numpy arrays
np.savez("csi_autoencoder.npz",
         encoder_W1=model.encoder_W1, encoder_b1=model.encoder_b1,
         encoder_W2=model.encoder_W2, encoder_b2=model.encoder_b2,
         encoder_W3=model.encoder_W3, encoder_b3=model.encoder_b3,
         decoder_W1=model.decoder_W1, decoder_b1=model.decoder_b1,
         decoder_W2=model.decoder_W2, decoder_b2=model.decoder_b2,
         decoder_W3=model.decoder_W3, decoder_b3=model.decoder_b3)

print("Autoencoder model saved successfully.")