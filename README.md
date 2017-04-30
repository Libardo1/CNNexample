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
  0   10.00%        12.17%*    1.10(s)
1000   90.43%        89.48%*    0.08(s)
2000   92.17%        90.70%*    0.08(s)
3000   91.30%        91.13%*    0.08(s)
4000   90.87%        91.27%*    0.08(s)
5000   92.17%        91.25%    0.08(s)
6000   93.91%        91.45%*    0.08(s)
7000   95.22%        91.82%*    0.09(s)
8000   93.04%        91.37%    0.08(s)
9000   97.39%        91.75%    0.08(s)
10000   95.65%        91.70%    0.08(s)
11000   93.91%        91.97%*    0.09(s)
12000   96.52%        91.76%    0.10(s)
13000   96.96%        91.84%    0.10(s)
14000   98.26%        91.89%    0.11(s)
15000   96.09%        92.01%*    0.08(s)
16000   97.83%        91.89%    0.07(s)
17000   94.35%        91.75%    0.09(s)
18000   94.78%        91.93%    0.07(s)
19000   98.26%        91.98%    0.08(s)
20000   94.78%        92.01%    0.08(s)
21000   96.96%        92.08%*    0.09(s)
22000   96.09%        91.93%    0.10(s)
23000   98.26%        91.75%    0.08(s)
24000   97.83%        92.11%*    0.08(s)
25000   95.65%        92.03%    0.08(s)
26000   95.65%        91.99%    0.09(s)
27000   98.26%        91.91%    0.07(s)
28000   97.83%        92.04%    0.07(s)
29000   95.22%        91.80%    0.08(s)
30000   96.09%        91.92%    0.08(s)
31000   97.83%        92.06%    0.08(s)
32000   98.70%        92.04%    0.09(s)
33000   96.96%        92.06%    0.08(s)
34000   97.39%        92.24%*    0.10(s)
35000   98.70%        92.02%    0.10(s)
36000   98.26%        92.49%*    0.09(s)
37000   99.57%        92.07%    0.08(s)
38000   97.83%        92.25%    0.08(s)
39000   97.39%        91.91%    0.08(s)
40000   96.96%        92.28%    0.08(s)

&&&&&&&&& #training steps = 40004 &&&&&&&&&&&
training time: 0:1:7:14 (DAYS:HOURS:MIN:SEC)

&&&&&&&&& For TensorBoard visualization type &&&&&&&&&&&

tensorboard  --logdir=logs/30-04-2017_18-43-10

check_valid =  0.9249
check_test =  0.9623
Prediction = H
Real label = H
```
