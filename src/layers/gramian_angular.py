import torch
import matplotlib.pyplot as plt

class GramianAngularFieldPytorch(torch.nn.Module):
    def __init__(self, method='summation'):
        super(GramianAngularFieldPytorch, self).__init__()
        self.method = method

    def min_max_norm(self, X):
        min_val = torch.min(X, dim=-1, keepdim=True)[0]
        max_val = torch.max(X, dim=-1, keepdim=True)[0]
        res = (X - min_val) / (max_val - min_val)
        return res * 2 - 1

    @staticmethod
    def _gasf(X_cos, X_sin):
        X_cos_L = X_cos.unsqueeze(-1)
        X_cos_R = X_cos.unsqueeze(-2)
        X_sin_L = X_sin.unsqueeze(-1)
        X_sin_R = X_sin.unsqueeze(-2)
        X_gasf = torch.matmul(X_cos_L, X_cos_R) - torch.matmul(X_sin_L, X_sin_R)
        return X_gasf

    @staticmethod
    def _gadf(X_cos, X_sin):
        X_sin_L = X_sin.unsqueeze(-1)
        X_cos_R = X_cos.unsqueeze(-2)
        X_cos_L = X_cos.unsqueeze(-1)
        X_sin_R = X_sin.unsqueeze(-2)
        X_gadf = torch.matmul(X_sin_L, X_cos_R) - torch.matmul(X_cos_L, X_sin_R)
        return X_gadf

    def forward(self, X):
        X_cos = self.min_max_norm(X)
        X_sin = torch.sqrt(torch.clamp(1 - X_cos ** 2, 0, 1))
        if self.method in ['s', 'summation']:
            X_new = self._gasf(X_cos, X_sin)
        else:
            X_new = self._gadf(X_cos, X_sin)
        return X_new

# Example usage:
if __name__ == "__main__":
   
    # Generate a sample time series
    time_series = torch.linspace(0, 2 * 3.1416, 100)
    X = torch.sin(2 * time_series)  # Example time series (sine wave)

    # Reshape X to a 2D tensor
    X = X.unsqueeze(0).unsqueeze(0)

    # Perform GAF transformation
    model = GramianAngularFieldPytorch(method='other')
    gaf_image = model(X)

    # Display the original time series and GAF image
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(time_series.numpy(), X[0, 0, :].numpy(), label='Original Time Series')
    plt.title('Original Time Series')
    plt.xlabel('Time')
    plt.ylabel('Amplitude')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.imshow(gaf_image[0, 0, :, :].numpy(), cmap='viridis', origin='upper', extent=[0, 100, 0, 100])
    plt.title('Gramian Angular Field (GAF)')
    plt.xlabel('Time')
    plt.ylabel('Time')
    plt.show()
    z = torch.rand(128,32,64)
    print(model(z).size())
