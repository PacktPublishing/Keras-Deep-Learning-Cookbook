import matplotlib.pyplot as plt
import csv
import numpy as np

x = []
y_loss = []
y_ae = []
y_mae = []
input_file = "train_1000steps_Sat_Sep_15_01:32:09_2018"
with open('../output/'+ input_file + '.log','r') as csvfile:
    plots = csv.reader(csvfile, delimiter=',')
    for row in plots:
        x.append(int(row[0]))
        y_loss.append(float(row[1]))
        y_ae.append(float(row[2]))
        y_mae.append(float(row[3]))


import matplotlib.pyplot as plt
import numpy as np

f, axarr = plt.subplots(3, sharex=True)

axarr[0].plot(x, y_loss)
axarr[0].set_title('Duel DQN CartPole Episode vs Loss')

axarr[1].plot(x, y_ae,color='r')
axarr[1].set_title('Duel DQN CartPole Episode vs Mean Absolute Error')

axarr[2].plot(x, y_mae,color='r')
axarr[2].set_title('Duel DQN CartPole Episode vs Mean Q')

plt.subplots_adjust(left=0.11, bottom=0.1, right=0.90, top=0.90, wspace=0.0, hspace=0.4)
plt.savefig('../output/' + input_file + '.png')
plt.show()