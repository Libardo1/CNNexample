# CNN example

This project is my first attempt to create a CNN in tensorflow. I use the [notMNIST](http://yaroslavvb.blogspot.com.br/2011/09/notmnist-dataset.html) dataset to create some predictions. In the folder Tutorials you can find a basic script to run the model.

### Requirements

* Tensorflow 
* Numpy
* Matplotlib


## Example

```
$ cd src
$ bash download.sh
$ python3 CNN.py


>> Start training
step  batch_acc  valid_acc  elapsed_time
  0   10.87%        10.00%*    0.26(s)
150   80.00%        81.84%*    0.23(s)
300   86.09%        86.72%*    0.24(s)


&&&&&&&&& #training steps = 301 &&&&&&&&&&&
training time: 0:0:1:22 (DAYS:HOURS:MIN:SEC)

&&&&&&&&& For TensorBoard visualization type &&&&&&&&&&&

tensorboard  --logdir=logs/26-04-2017_14-10-41

check_valid =  0.8672
Prediction = H
Real label = H

```
