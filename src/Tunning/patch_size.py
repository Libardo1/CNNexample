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

from util import run_test, get_data_4d
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

number_of_exp = 5
PATCH_SIZE = [3, 5, 7, 9, 11]
results = []
duration = []
info = []

for i, ps in enumerate(PATCH_SIZE):
    print("\n ({0} of {1})".format(i + 1, number_of_exp))
    my_config = Config(patch_size=ps)
    attrs = vars(my_config)
    config_info = ["%s: %s" % item for item in attrs.items()]
    info.append(config_info)
    my_model = CNNModel(my_config, my_dataholder)
    current_dur = train_model(my_model, my_dataholder)
    score = check_test(my_model)
    results.append(score)
    duration.append(current_dur)

best_result = max(list(zip(results, PATCH_SIZE, info)))
result_string = """In an experiment with {0} patch sizes
the best one is {1} with score = {2}.
\n INFO = {3}""".format(number_of_exp,
                        best_result[1],
                        best_result[0],
                        best_result[2])

file = open("patch_size.txt", "w")
file.write(result_string)
file.close()


plt.plot(PATCH_SIZE, results)
plt.xlabel("patch size")
plt.ylabel("score")
plt.savefig("patch_size.png")

plt.plot(PATCH_SIZE, duration)
plt.xlabel("patch size")
plt.ylabel("duration (s)")
plt.savefig("patch_size_du.png")
