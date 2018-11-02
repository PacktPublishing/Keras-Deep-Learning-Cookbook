import matplotlib.pyplot as plt
import csv
import numpy as np

x = []
y_d_loss = []
y_d_accuracy = []
y_g_loss = []
with open('cartpole_v1_output_aug22.csv','r') as csvfile:
    plots = csv.reader(csvfile, delimiter=',')
    for row in plots:
        x.append(int(row[0]))
        y_d_loss.append(float(row[1]))
        y_d_accuracy.append(float(row[2]))


import matplotlib.pyplot as plt
import numpy as np

f, axarr = plt.subplots(2, sharex=True)

axarr[0].plot(x, y_d_loss)
axarr[0].set_title('CartPole V1 Score')

axarr[1].plot(x, y_d_accuracy,color='r')
axarr[1].set_title('CartPole V1 Epsilon')

plt.subplots_adjust(left=0.11, bottom=0.1, right=0.90, top=0.90, wspace=0.0, hspace=0.4)

plt.show()