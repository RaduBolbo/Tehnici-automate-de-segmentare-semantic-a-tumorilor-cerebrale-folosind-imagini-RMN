
'''
import numpy as np

import torch.nn.functional as F
from torchgeometry.losses.one_hot import one_hot

import torch
import cv2



a = np.array([[[[1.2, 1.4, 1.9],[1.5, 1.9, 0.4],[0.2, 0.1, 12]],[[1.2, 1.4, 16],[1.5, 4, 0.4],[0.2, 0.1, 0.12]],[[12, 1.4, 1.6],[1.5, 1.7, 0.4],[0.2, 0.1, 12]]]])
#a = np.uint8(255*np.array([[[[0, 1, 0],[0, 1, 0],[0, 1, 0]],[[1, 0, 0],[1, 0, 0],[1, 0, 0]],[[0, 0, 1],[0, 0, 1],[0, 0, 1]]]]))
##print(a)
#cv2.imshow('a', a[0,:,:,:])
#cv2.waitKey()
#cv2.destroyAllWindows()

#a = np.array([[[0.0, 1.0, 2.0]]])
t = torch.from_numpy(a)
t = F.softmax(t, dim=3)

#print(t)
#print(t.shape)



#print(t.shape)

t = one_hot(t, 3)

#print(t)
'''



import numpy as np
import matplotlib.pyplot as plt

# define a range of probabilities
p = np.linspace(0.01, 0.99, 100)

# calculate cross entropy loss when the true class is 1
loss_positive = -np.log(p)

# calculate cross entropy loss when the true class is 0
loss_negative = -np.log(1 - p)

# plot both scenarios
plt.figure(figsize=(8,6))
plt.plot(p, loss_positive, label='Clasa corectă: 1')
plt.plot(p, loss_negative, label='Clasa corectă: 0')
plt.title('Cross Entropy Loss for different scenarios')
plt.xlabel('Probabilitatea de apartenență la clasă (prezisă)', fontsize=16)
plt.ylabel('Valoarea entropiei încrucișate', fontsize=16)
plt.legend()
plt.show()


