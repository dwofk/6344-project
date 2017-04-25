import numpy as np
import tensorflow as tf
from PIL import Image
import os

def prelu(x, alpha, name):
    pos = tf.nn.relu(x, name)
    neg = alpha * (x - abs(x)) * 0.5
    
    return pos + neg

def buildModel(image):
    # conv1
    with tf.variable_scope('conv1') as scope:
        kernel = tf.Variable(tf.random_normal([5, 5, 3, 32], stddev=0.05), name='weights')
        conv = tf.nn.conv2d(image, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.Variable(tf.zeros([32]), name='biases')
        pre_activation = tf.nn.bias_add(conv, biases)
        #conv1 = tf.nn.relu(pre_activation, name=scope.name)
        alpha = tf.get_variable('alpha', pre_activation.get_shape()[-1], initializer=tf.constant_initializer(0.0), dtype=tf.float32)
        conv1 = prelu(pre_activation, alpha, name=scope.name)

    with tf.variable_scope('conv2') as scope:
        kernel = tf.Variable(tf.random_normal([5, 5, 32, 3], stddev=0.05), name='weights')
        conv = tf.nn.conv2d(conv1, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.Variable(tf.zeros([3]), name='biases')
        pre_activation = tf.nn.bias_add(conv, biases)
        #conv1 = tf.nn.relu(pre_activation, name=scope.name)
        alpha = tf.get_variable('alpha', pre_activation.get_shape()[-1], initializer=tf.constant_initializer(0.0), dtype=tf.float32)
        conv2 = prelu(pre_activation, alpha, name=scope.name)
##
##    with tf.variable_scope('conv2') as scope:
##        kernel = tf.Variable(tf.random_normal([1, 1, 32, 5], stddev=0.05), name='weights')
##        #kernel = _variable_with_weight_decay('weights', shape=[1, 1, 32, 5], stddev=0.05, wd=0.0)
##        conv = tf.nn.conv2d(conv1, kernel, [1, 1, 1, 1], padding='SAME')
##        biases = tf.Variable(tf.zeros([5]), name='biases')
##        pre_activation = tf.nn.bias_add(conv, biases)
##        alpha = tf.get_variable('alpha', pre_activation.get_shape()[-1], initializer=tf.constant_initializer(0.0), dtype=tf.float32)
##        conv2 = prelu(pre_activation, alpha, name=scope.name)
##
##    with tf.variable_scope('conv3') as scope:
##        kernel = tf.Variable(tf.random_normal([3, 3, 5, 5], stddev=0.05), name='weights')
##        #kernel = _variable_with_weight_decay('weights', shape=[3, 3, 5, 5], stddev=0.05, wd=0.0)
##        conv = tf.nn.conv2d(conv2, kernel, [1, 1, 1, 1], padding='SAME')
##        biases = tf.Variable(tf.zeros([5]), name='biases')
##        pre_activation = tf.nn.bias_add(conv, biases)
##        alpha = tf.get_variable('alpha', pre_activation.get_shape()[-1], initializer=tf.constant_initializer(0.0), dtype=tf.float32)
##        conv3 = prelu(pre_activation, alpha, name=scope.name)
##
##    with tf.variable_scope('conv4') as scope:
##        kernel = tf.Variable(tf.random_normal([1, 1, 5, 32], stddev=0.05), name='weights')
##        #kernel = _variable_with_weight_decay('weights', shape=[1, 1, 5, 32], stddev=0.05, wd=0.0)
##        conv = tf.nn.conv2d(conv3, kernel, [1, 1, 1, 1], padding='SAME')
##        biases = tf.Variable(tf.zeros([32]), name='biases')
##        pre_activation = tf.nn.bias_add(conv, biases)
##        alpha = tf.get_variable('alpha', pre_activation.get_shape()[-1], initializer=tf.constant_initializer(0.0), dtype=tf.float32)
##        conv4 = prelu(pre_activation, alpha, name=scope.name)
##
##    with tf.variable_scope('conv5') as scope:
##        kernel = tf.Variable(tf.random_normal([9, 9, 32, 3], stddev=0.05), name='weights')
##        #kernel = _variable_with_weight_decay('weights', shape=[9, 9, 32, 3], stddev=0.05, wd=0.0)
##        conv = tf.nn.conv2d(conv4, kernel, [1, 1, 1, 1], padding='SAME')
##        biases = tf.Variable(tf.zeros([3]), name='biases')
##        pre_activation = tf.nn.bias_add(conv, biases)
##        alpha = tf.get_variable('alpha', pre_activation.get_shape()[-1], initializer=tf.constant_initializer(0.0), dtype=tf.float32)
##        conv5 = prelu(pre_activation, alpha, name=scope.name)
##
##    return conv5
    return conv2

def setFromImage(image):
    return [image, np.flipud(image), np.fliplr(image)]

def fileList(filepath):
    filelist = []
    directories = os.listdir(filepath)
    for directory in directories:
        filenames = os.listdir(filepath + directory)
        fullfilenames = [filepath + directory + '\\' + x for x in filenames]
        filelist.extend(fullfilenames)
    return filelist

with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
    currentDirectory = 'S:\\Dropbox (Personal)\\Assignments\\6.344\\project\\6344-project\\exposure_cnn\\'
    trainshape = (32, 32, 1, 3)
    x = tf.placeholder('float32', shape=trainshape)
    l = tf.placeholder('float32', shape=trainshape)
    y = buildModel(x)
    loss = tf.sqrt(tf.reduce_sum(tf.square(tf.subtract(l, y))))
    tf.summary.scalar('loss', loss)
    optimizer = tf.train.GradientDescentOptimizer(0.00000005)
    train_op = optimizer.minimize(loss)
    
    input_filepath = 'phos\\0\\'
    label_filepath = 'phos\\plus_2\\'
    inputs = fileList(input_filepath)
    #labels = fileList(label_filepath)
    #inputs = os.listdir(input_filepath)
    #labels = os.listdir(label_filepath)

    #with tf.Graph().as_default():
    init_op = tf.global_variables_initializer()
    saver = tf.train.Saver()
    sess.run(init_op)
    
    feed_dict = {}
    step = 0
    print('Starting')
    for input_file in inputs:
        fileparts = input_file.split('\\')
        scene = fileparts[-2]
        file = fileparts[-1] # the last element of this split should be the file name
        label_file = label_filepath + scene + '\\' + file
        if (os.path.exists(label_file)):
            im_in = Image.open(input_file)
            h, w = im_in.size
            im_in = np.asarray(im_in).astype('float32')
            im_in = im_in.reshape([h, w, 1, 3]);
            #im_inputs = setFromImage(im_in)

            im_la = Image.open(label_file)
            h, w = im_la.size
            im_la = np.asarray(im_la).astype('float32')
            im_la = im_la.reshape([h, w, 1, 3]);
            #im_labels = setFromImage(im_la)
            _, loss_value = sess.run([train_op, loss], feed_dict={x:im_in, l:im_la})
            if (step % 100 == 0):
                print("Step: {}, Loss: {}".format(step, loss_value))
            step = step + 1
        else:
            print('From {}: {} does not exist'.format(input_file, label_file))

    saver.save(sess, currentDirectory + 'model.ckpt') # save model checkpoint
    #im = Image.open('input.bmp')
    #h, w = im.size
    #im_te = np.asarray(im).astype('float32')
    #im_in =  im_te.reshape([h, w, 1, 3]);
    

    

    
    
    

    
    #im_out = y.eval(session=sess,feed_dict={x:im_in})
    #im_out = im_out.reshape([h, w, 3]).astype('uint8')
    #print(np.shape(im_out))

#im_o = Image.fromarray(im_out)

#im_o.save('output.bmp')
