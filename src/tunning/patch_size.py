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

number_of_exp = 5
PATCH_SIZE = [3, 5, 7, 9, 11]
results = []
duration = []
info = []

for i, ps in enumerate(PATCH_SIZE):
    print("\n ({0} of {1})".format(i + 1, number_of_exp))
    my_config = Config(tunning=True, patch_size=ps)
    attrs = vars(my_config)
    config_info = ["%s: %s" % item for item in attrs.items()]
    info.append(config_info)
    my_model = CNNModel(my_config, my_dataholder)
    train_model(my_model, my_dataholder, 10001, 1000, False)
    current_dur = get_time(train_model, 10001)
    score = check_valid(my_model)
    results.append(score)
    duration.append(current_dur)

best_result = max(list(zip(results, PATCH_SIZE, duration, info)))
result_string = """In an experiment with {0} patch sizes
the best one is {1} with valid accuracy = {2}.
\nThe training takes {3:.2f} seconds using the following params:
\n{4}""".format(number_of_exp,
                best_result[1],
                best_result[0],
                best_result[2],
                best_result[3])

file = open("patch_size.txt", "w")
file.write(result_string)
file.close()

plt.plot(PATCH_SIZE, results)
plt.xlabel("patch size")
plt.ylabel("valid acc")
plt.savefig("patch_size.png")
plt.clf()

plt.plot(PATCH_SIZE, duration)
plt.xlabel("patch size")
plt.ylabel("duration (s)")
plt.savefig("patch_size_du.png")
plt.clf()
