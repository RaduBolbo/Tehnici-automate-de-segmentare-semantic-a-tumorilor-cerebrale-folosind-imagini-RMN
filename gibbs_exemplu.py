import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, ifft

# Parameters
N = 500  # Number of sample points
T = 1.0 / 800.0  # Sample spacing
x = np.linspace(0.0, N*T, N)
rect = np.where((x >= 0.1) & (x <= 0.4), 1.0, 0.0)  # Rectangular signal

# Fourier Transform
yf = fft(rect)

# Plot the original signal
plt.figure(figsize=(10, 6))
plt.subplot(221)
plt.plot(x, rect)
plt.title('Semnalul original')

# Approximate the signal using a partial sum of the Fourier series
M = 50  # Number of Fourier components to use
approx = np.real(ifft(yf[:M], N))

# Plot the approximation
plt.subplot(222)
plt.plot(x, approx)
plt.title('Reconstrucția, pe baza a M = {} termeni'.format(M))

# Increase M and approximate the signal again
M = 100
approx = np.real(ifft(yf[:M], N))

# Plot the new approximation
plt.subplot(223)
plt.plot(x, approx)
plt.title('Approximation with M = {}'.format(M))

plt.tight_layout()
plt.show()