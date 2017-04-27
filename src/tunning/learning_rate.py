import os
import sys
from random import randint
import numpy as np
import inspect
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

from util import run_test, get_data_4d, get_time
from CNN import CNNModel, train_model, check_valid
from DataHolder import DataHolder
from Config import Config

train_dataset, train_labels, valid_dataset, valid_labels, test_dataset, test_labels = get_data_4d()
my_dataholder = DataHolder(train_dataset,
                           train_labels,
                           valid_dataset,
                           valid_labels,
                           test_dataset,
                           test_labels)


number_of_exp = 10
LR = np.random.random_sample([number_of_exp])
LR = np.append(LR, 0.9)
number_of_exp += 1
LR.sort()
results = []
duration = []
info = []

for i, lr in enumerate(LR):
    print("\n ({0} of {1})".format(i + 1, number_of_exp))
    my_config = Config(tunning=True, learning_rate=lr)
    attrs = vars(my_config)
    config_info = ["%s: %s" % item for item in attrs.items()]
    info.append(config_info)
    my_model = CNNModel(my_config, my_dataholder)
    train_model(my_model, my_dataholder, 10001, 1000, False)
    current_dur = get_time(train_model, 10001)
    score = check_valid(my_model)
    results.append(score)
    duration.append(current_dur)

LR = list(LR)
best_result = max(list(zip(results, LR, duration, info)))
result_string = """In an experiment with {0} learning rate values
the best one is {1} with valid accuracy = {2}.
\nThe training takes {3:.2f} seconds using the following params:
\n{4}""".format(number_of_exp,
                best_result[1],
                best_result[0],
                best_result[2],
                best_result[3])

file = open("learning_rate.txt", "w")
file.write(result_string)
file.close()

plt.plot(LR, results)
plt.xscale('log')
plt.xlabel("learning rate")
plt.ylabel("valid acc")
plt.savefig("learning_rate.png")
plt.clf()

plt.plot(LR, duration)
plt.xscale('log')
plt.xlabel("learning rate")
plt.ylabel("duration (s)")
plt.savefig("learning_rate_du.png")
plt.clf()
