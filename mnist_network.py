import numpy as np
import paddle as paddle
import paddle.dataset.mnist as mnist
import paddle.fluid as fluid
import paddle.fluid.framework as framework
from PIL import Image


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
    acc = fluid.layers.accuracy(input=model, label=label)


    optimizer = fluid.optimizer.AdamOptimizer(learning_rate=0.001)
    optimizer.minimize(avg_cost)


def save_program_desc():
    startup_program = framework.Program()
    train_program = framework.Program()
   
    with framework.program_guard(train_program, startup_program):
        minist_classfication_network()

    with open("startup_program", "w") as f:
        f.write(startup_program.desc.serialize_to_string())
    with open("main_program", "w") as f:
        f.write(train_program.desc.serialize_to_string())
    #print train_program
save_program_desc()

