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

print("\n&&&&&&&&& Learning rate &&&&&&&&&&&")

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
    my_config = Config(learning_rate=lr)
    attrs = vars(my_config)
    config_info = ["%s: %s" % item for item in attrs.items()]
    info.append(config_info)
    my_model = CNNModel(my_config, my_dataholder)
    train_model(my_model, my_dataholder, 10001, 1000, False)
    current_dur = get_time(train_model, 10001)
    score = check_test(my_model)
    results.append(score)
    duration.append(current_dur)

LR = list(LR)
best_result = max(list(zip(results, LR, duration, info)))
result_string = """In an experiment with {0} learning rate values
the best one is {1} with test accuracy = {2}.
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
plt.ylabel("test acc")
plt.savefig("learning_rate.png")
plt.clf()

plt.plot(LR, duration)
plt.xscale('log')
plt.xlabel("learning rate")
plt.ylabel("duration (s)")
plt.savefig("learning_rate_du.png")
plt.clf()

learning_rate = best_result[1]

print("\n&&&&&&&&& Decay rate &&&&&&&&&&&")

number_of_exp = 10
DECAY = np.random.random_sample([number_of_exp])
DECAY = np.append(DECAY, 0.96)
number_of_exp += 1
DECAY.sort()
results = []
duration = []
info = []

for i, de in enumerate(DECAY):
    print("\n ({0} of {1})".format(i + 1, number_of_exp))
    my_config = Config(decay_rate=de,
                       learning_rate=learning_rate)
    attrs = vars(my_config)
    config_info = ["%s: %s" % item for item in attrs.items()]
    info.append(config_info)
    my_model = CNNModel(my_config, my_dataholder)
    train_model(my_model, my_dataholder, 10001, 1000, False)
    current_dur = get_time(train_model, 10001)
    score = check_test(my_model)
    results.append(score)
    duration.append(current_dur)

DECAY = list(DECAY)
best_result = max(list(zip(results, DECAY, duration, info)))
result_string = """In an experiment with {0} decay rate values
the best one is {1} with test accuracy = {2}.
\nThe training takes {3:.2f} seconds using the following params:
\n{4}""".format(number_of_exp,
                best_result[1],
                best_result[0],
                best_result[2],
                best_result[3])

file = open("decay_ra.txt", "w")
file.write(result_string)
file.close()

plt.plot(DECAY, results)
plt.xscale('log')
plt.xlabel("decay rate")
plt.ylabel("test acc")
plt.savefig("decay_ra.png")
plt.clf()

plt.plot(DECAY, duration)
plt.xscale('log')
plt.xlabel("decay rate")
plt.ylabel("duration (s)")
plt.savefig("decay_ra_du.png")
plt.clf()

decay_rate = best_result[1]


print("\n&&&&&&&&& Decay steps &&&&&&&&&&&")

DECAY = [40, 80, 100, 150, 230, 300, 450, 600, 800]
number_of_exp = len(DECAY)
results = []
duration = []
info = []

for i, de in enumerate(DECAY):
    print("\n ({0} of {1})".format(i + 1, number_of_exp))
    my_config = Config(steps_for_decay=de,
                       learning_rate=learning_rate,
                       decay_rate=decay_rate)
    attrs = vars(my_config)
    config_info = ["%s: %s" % item for item in attrs.items()]
    info.append(config_info)
    my_model = CNNModel(my_config, my_dataholder)
    train_model(my_model, my_dataholder, 10001, 1000, False)
    current_dur = get_time(train_model, 10001)
    score = check_test(my_model)
    results.append(score)
    duration.append(current_dur)

best_result = max(list(zip(results, DECAY, duration, info)))
result_string = """In an experiment with {0} steps for decay
the best one is {1} with test accuracy = {2}.
\nThe training takes {3:.2f} seconds using the following params:
\n{4}""".format(number_of_exp,
                best_result[1],
                best_result[0],
                best_result[2],
                best_result[3])

file = open("decay_st.txt", "w")
file.write(result_string)
file.close()

plt.plot(DECAY, results)
plt.xlabel("steps for decay")
plt.ylabel("test acc")
plt.savefig("decay_st.png")
plt.clf()

plt.plot(DECAY, duration)
plt.xlabel("steps for decay")
plt.ylabel("duration (s)")
plt.savefig("decay_st_du.png")
plt.clf()

steps_for_decay = best_result[1]


print("\n&&&&&&&&& Batch size &&&&&&&&&&&")

BATCH_SIZE = [40, 80, 120, 150, 230, 300, 450, 600]
number_of_exp = len(BATCH_SIZE)
results = []
duration = []
info = []

for i, ba in enumerate(BATCH_SIZE):
    print("\n ({0} of {1})".format(i + 1, number_of_exp))
    my_config = Config(steps_for_decay=steps_for_decay,
                       learning_rate=learning_rate,
                       decay_rate=decay_rate,
                       batch_size=ba)
    attrs = vars(my_config)
    config_info = ["%s: %s" % item for item in attrs.items()]
    info.append(config_info)
    my_model = CNNModel(my_config, my_dataholder)
    train_model(my_model, my_dataholder, 10001, 1000, False)
    current_dur = get_time(train_model, 10001)
    score = check_test(my_model)
    results.append(score)
    duration.append(current_dur)

best_result = max(list(zip(results, BATCH_SIZE, duration, info)))
result_string = """In an experiment with {0} batch sizes
the best one is {1} with test accuracy = {2}.
\nThe training takes {3:.2f} seconds using the following params:
\n{4}""".format(number_of_exp,
                best_result[1],
                best_result[0],
                best_result[2],
                best_result[3])

file = open("batch_size.txt", "w")
file.write(result_string)
file.close()

plt.plot(BATCH_SIZE, results)
plt.xlabel("batch size")
plt.ylabel("test acc")
plt.savefig("batch_size.png")
plt.clf()

plt.plot(BATCH_SIZE, duration)
plt.xlabel("batch size")
plt.ylabel("duration (s)")
plt.savefig("batch_size_du.png")
plt.clf()

batch_size = best_result[1]

print("\n&&&&&&&&& Patch size &&&&&&&&&&&")

number_of_exp = 5
PATCH_SIZE = [3, 5, 7, 9, 11]
results = []
duration = []
info = []

for i, ps in enumerate(PATCH_SIZE):
    print("\n ({0} of {1})".format(i + 1, number_of_exp))
    my_config = Config(patch_size=ps,
                       steps_for_decay=steps_for_decay,
                       learning_rate=learning_rate,
                       decay_rate=decay_rate,
                       batch_size=batch_size)
    attrs = vars(my_config)
    config_info = ["%s: %s" % item for item in attrs.items()]
    info.append(config_info)
    my_model = CNNModel(my_config, my_dataholder)
    train_model(my_model, my_dataholder, 10001, 1000, False)
    current_dur = get_time(train_model, 10001)
    score = check_test(my_model)
    results.append(score)
    duration.append(current_dur)

best_result = max(list(zip(results, PATCH_SIZE, duration, info)))
result_string = """In an experiment with {0} patch sizes
the best one is {1} with test accuracy = {2}.
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
plt.ylabel("test acc")
plt.savefig("patch_size.png")
plt.clf()

plt.plot(PATCH_SIZE, duration)
plt.xlabel("patch size")
plt.ylabel("duration (s)")
plt.savefig("patch_size_du.png")
plt.clf()

patch_size = best_result[1]

print("\n&&&&&&&&& Filter &&&&&&&&&&&")

FILTER1 = range(1, 17)
number_of_exp = len(FILTER1)
results = []
duration = []
info = []

for i, fi in enumerate(FILTER1):
    print("\n ({0} of {1})".format(i + 1, number_of_exp))
    my_config = Config(num_filters_1=fi,
                       num_filters_2=2 * fi,
                       patch_size=patch_size,
                       steps_for_decay=steps_for_decay,
                       learning_rate=learning_rate,
                       decay_rate=decay_rate,
                       batch_size=batch_size)
    attrs = vars(my_config)
    config_info = ["%s: %s" % item for item in attrs.items()]
    info.append(config_info)
    my_model = CNNModel(my_config, my_dataholder)
    train_model(my_model, my_dataholder, 10001, 1000, False)
    current_dur = get_time(train_model, 10001)
    score = check_test(my_model)
    results.append(score)
    duration.append(current_dur)

best_result = max(list(zip(results, FILTER1, duration, info)))
result_string = """In an experiment with {0} filter sizes
the best one is {1} with test accuracy = {2}.
\nThe training takes {3:.2f} seconds using the following params:
\n{4}""".format(number_of_exp,
                best_result[1],
                best_result[0],
                best_result[2],
                best_result[3])


file = open("filter1.txt", "w")
file.write(result_string)
file.close()

plt.plot(FILTER1, results)
plt.xlabel("filter1")
plt.ylabel("test acc")
plt.savefig("filter1.png")
plt.clf()

plt.plot(FILTER1, duration)
plt.xlabel("filter1")
plt.ylabel("duration (s)")
plt.savefig("filter1_du.png")
plt.clf()

num_filters_1 = best_result[1]
num_filters_2 = 2 * best_result[1]
print(num_filters_1, num_filters_2)

print("\n&&&&&&&&& Fully Connected &&&&&&&&&&&")

FC = [5, 10, 15, 20, 30, 40, 60, 200]
number_of_exp = len(FC)
results = []
duration = []
info = []

for i, fc in enumerate(FC):
    print("\n ({0} of {1})".format(i + 1, number_of_exp))
    my_config = Config(hidden_nodes_1=3 * fc,
                       hidden_nodes_2=2 * fc,
                       hidden_nodes_3=fc,
                       num_filters_1=num_filters_1,
                       num_filters_2=num_filters_2,
                       patch_size=patch_size,
                       steps_for_decay=steps_for_decay,
                       learning_rate=learning_rate,
                       decay_rate=decay_rate,
                       batch_size=batch_size)
    attrs = vars(my_config)
    config_info = ["%s: %s" % item for item in attrs.items()]
    info.append(config_info)
    my_model = CNNModel(my_config, my_dataholder)
    train_model(my_model, my_dataholder, 10001, 1000, False)
    current_dur = get_time(train_model, 10001)
    score = check_test(my_model)
    results.append(score)
    duration.append(current_dur)

best_result = max(list(zip(results, FC, duration, info)))
result_string = """In an experiment with {0} fully connected sizes
the best one is {1} with test accuracy = {2}.
\nThe training takes {3:.2f} seconds using the following params:
\n{4}""".format(number_of_exp,
                best_result[1],
                best_result[0],
                best_result[2],
                best_result[3])


file = open("final.txt", "w")
file.write(result_string)
file.close()

plt.plot(FC, results)
plt.xlabel("hidden_nodes_3")
plt.ylabel("test acc")
plt.savefig("fc.png")
plt.clf()

plt.plot(FC, duration)
plt.xlabel("hidden_nodes_3")
plt.ylabel("duration (s)")
plt.savefig("fc_du.png")
plt.clf()
