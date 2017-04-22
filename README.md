# CNN example

This project is my first attempt to create a CNN in tensorflow. I use the [TF-recomm](https://github.com/songgc/TF-recomm) dataset to create some predictions. In the folder Tutorials you can find a basic script to run the model.

### Requirements

* Tensorflow 
* Numpy


## Example

```
$ cd src
$ bash download.sh
$ python3 svd.py -s 20000


>> Start training
step  batch_error  test_error  elapsed_time
  0   7.86%        10.02%*    0.33(s)
400   88.57%        91.32%*    0.15(s)
800   87.86%        93.31%*    0.13(s)
1200   85.71%        94.14%*    0.12(s)

&&&&&&&&& #training steps = 1201 &&&&&&&&&&&
training time: 0:0:2:58 (DAYS:HOURS:MIN:SEC)

&&&&&&&&& For TensorBoard visualization type &&&&&&&&&&&

tensorboard  --logdir=logs/22-04-2017_19-15-40

check_test =  0.9414
check_valid =  0.8869
Prediction = J
Real label = J

```
