# -*- coding: utf-8 -*-  
import numpy as np
import paddle as paddle
import paddle.dataset.mnist as mnist
import paddle.fluid as fluid
import paddle.fluid.framework as framework
from PIL import Image
from paddle.fluid.framework import Program

def multilayer_perceptron(input):
    hidden1 = fluid.layers.fc(input=input, size=100, act='relu')
    hidden2 = fluid.layers.fc(input=hidden1, size=100, act='relu')
    fc = fluid.layers.fc(input=hidden2, size=10, act='softmax')
    return fc

def convolutional_neural_network(input):
    conv1 = fluid.layers.conv2d(input=input,
                                num_filters=32,
                                filter_size=3,
                                stride=1,use_cudnn=False)

    pool1 = fluid.layers.pool2d(input=conv1,
                                pool_size=2,
                                pool_stride=1,
                                pool_type='max')

    conv2 = fluid.layers.conv2d(input=pool1,
                                num_filters=64,
                                filter_size=3,
                                stride=1,use_cudnn=False)

    pool2 = fluid.layers.pool2d(input=conv2,
                                pool_size=2,
                                pool_stride=1,
                                pool_type='max')

    fc = fluid.layers.fc(input=pool2, size=10, act='softmax')
    return fc

def minist_classfication_network():
    image = fluid.layers.data(name='image', shape=[1, 28, 28], dtype='float32')
    label = fluid.layers.data(name='label', shape=[1], dtype='int64')

    #model = multilayer_perceptron(image)
    model = convolutional_neural_network(image)

    cost = fluid.layers.cross_entropy(input=model, label=label)
    avg_cost = fluid.layers.mean(cost)

    optimizer = fluid.optimizer.AdamOptimizer(learning_rate=0.001)
    optimizer.minimize(avg_cost)


startup_program = framework.Program()
train_program = framework.Program()

test_reader = paddle.batch(mnist.test(), batch_size=128)


place = fluid.CPUPlace()
exe = fluid.Executor(place)

with framework.program_guard(train_program, startup_program):
    image = fluid.layers.data(name='image', shape=[1, 28, 28], dtype='float32')
    label = fluid.layers.data(name='label', shape=[1], dtype='int64')
    #model = multilayer_perceptron(image)
    model = convolutional_neural_network(image)



    cost = fluid.layers.cross_entropy(input=model, label=label)
    avg_cost = fluid.layers.mean(cost)
    acc = fluid.layers.accuracy(input=model, label=label)

    optimizer = fluid.optimizer.AdamOptimizer(learning_rate=0.001)
    optimizer.minimize(avg_cost)

feeder = fluid.DataFeeder(place=place, feed_list=[image, label])
#load param
isTrain = True
if isTrain:
    param_path = "./build/breakpoint/"
    fluid.io.load_persistables(executor=exe, dirname=param_path,main_program=train_program)
else:
    exe.run(startup_program)
    train_reader = paddle.batch(mnist.train(), batch_size=128)
    for pass_id in range(1):
        for batch_id, data in enumerate(train_reader()):
            train_cost, train_acc = exe.run(program=train_program,
                                            feed=feeder.feed(data),
                                            fetch_list=[avg_cost, acc])
            if batch_id % 100 == 0:
                print('Pass:%d, Batch:%d, Cost:%0.5f, Accuracy:%0.5f' %
                (pass_id, batch_id, train_cost[0], train_acc[0]))

#load program
test_program = train_program.clone(for_test=True)



test_accs = []
test_costs = []
for batch_id, data in enumerate(test_reader()):
#    print data
    test_cost,test_acc = exe.run(program=test_program, feed=feeder.feed(data),fetch_list=[avg_cost,acc])
    test_accs.append(test_acc[0])
    test_costs.append(test_cost[0])
test_cost = (sum(test_costs) / len(test_costs))
test_acc = (sum(test_accs) / len(test_accs))
#print('Cost:%0.5f' % (test_cost))
print('Cost:%0.5f, Accuracy:%0.5f' % (test_cost, test_acc))

