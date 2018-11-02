import matplotlib.pyplot as plt
import csv


x = []
y_reward = []
input_file = "test_Sat_Sep_15_01:32:12_2018"

with open('../output/' + input_file +'.log','r') as csvfile:
    plots = csv.reader(csvfile, delimiter=',')
    for row in plots:
        x.append(int(row[0]))
        y_reward.append(float(row[1]))


import matplotlib.pyplot as plt
import numpy as np

#f = plt.subplots(2, sharex=True)

plt.plot(x, y_reward)
plt.title('Dueling DQN CartPole V0 Episode vs Reward')
plt.savefig('../output/' + input_file + '.png')
plt.show()
