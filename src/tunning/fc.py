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
from CNN import CNNModel, train_model, check_test
from DataHolder import DataHolder
from Config import Config

train_dataset, train_labels, valid_dataset, valid_labels, test_dataset, test_labels = get_data_4d()
my_dataholder = DataHolder(train_dataset,
                           train_labels,
                           valid_dataset,
                           valid_labels,
                           test_dataset,
                           test_labels)

FC = [5, 10, 15, 20, 30, 40, 60, 200]
number_of_exp = len(FC)
results = []
duration = []
info = []

for i, fc in enumerate(FC):
    print("\n ({0} of {1})".format(i + 1, number_of_exp))
    my_config = Config(hidden_nodes_1=3 * fc,
                       hidden_nodes_2=2 * fc,
                       hidden_nodes_3=fc)
    attrs = vars(my_config)
    config_info = ["%s: %s" % item for item in attrs.items()]
    info.append(config_info)
    my_model = CNNModel(my_config, my_dataholder)
    train_model(my_model, my_dataholder, 3, 2, False)
    current_dur = get_time(train_model, 3)
    score = check_test(my_model)
    results.append(score)
    duration.append(current_dur)

best_result = max(list(zip(results, FC, duration, info)))
result_string = """In an experiment with {0} filter sizes
the best one is {1} with test accuracy = {2}.
\nThe training takes {3:.2f} seconds using the following params:
\n{4}""".format(number_of_exp,
                best_result[1],
                best_result[0],
                best_result[2],
                best_result[3])


file = open("fc.txt", "w")
file.write(result_string)
file.close()

plt.plot(FC, results)
plt.xlabel("hidden_nodes_3")
plt.ylabel("score")
plt.savefig("fc.png")

plt.plot(FC, duration)
plt.xlabel("hidden_nodes_3")
plt.ylabel("duration (s)")
plt.savefig("fc_du.png")
