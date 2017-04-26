import numpy as np
import tensorflow as tf
from PIL import Image
import os

# create a prelu image
def prelu(x, alpha, name):
    pos = tf.nn.relu(x, name)
    neg = alpha * (x - abs(x)) * 0.5
    
    return pos + neg

# build the net: structure fully defined within this function
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

# currently not used: do some simple transformations to get more mileage out of an image
def setFromImage(image):
    return [image, np.flipud(image), np.fliplr(image)]

# specialized to current file structure: images should be within 2 layers of folders
# extract full list of image file names to use
def fileList(filepath):
    filelist = []
    directories = os.listdir(filepath)
    for directory in directories:
        filenames = os.listdir(filepath + directory)
        fullfilenames = [filepath + directory + '\\' + x for x in filenames]
        filelist.extend(fullfilenames)
    return filelist

# train on all 32x32 image pieces once
# loadModel = bool: True if a loadName is passed in, to load a model from memory, False if initializing new model
# saveName is the location to save the new model
def runTrain(loadModel, learning_rate_init, loadName='', saveName=''):
    with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
        currentDirectory = 'S:\\6344-project\\exposure_cnn\\'
        trainshape = (32, 32, 1, 3)
        norm = 32*32*1*3
        x = tf.placeholder('float32', shape=trainshape)
        l = tf.placeholder('float32', shape=trainshape)
        y = buildModel(x)
        loss = tf.sqrt(tf.reduce_sum(tf.square(tf.subtract(l, y)))/norm) # RMSE
        tf.summary.scalar('loss', loss)

        #learning_rate_init = 0.000001
        #momentum = 0.9
        momentum = 0.5
        decay_steps = 10000
        decay_rate = 0.9
        global_step = tf.Variable(0, trainable=False)
        learning_rate = learning_rate_init#tf.train.exponential_decay(learning_rate_init, global_step, decay_steps, decay_rate)
        #optimizer = tf.train.GradientDescentOptimizer(learning_rate)
        optimizer = tf.train.MomentumOptimizer(learning_rate, momentum, use_nesterov=True)
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

        if loadModel:
            saver.restore(sess, loadName)
            print('Load model from ' + loadName)
        else:
            sess.run(init_op)
            print('Initializing fresh variables')

        step = 0
        feed_dict = {}
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
                #print(global_step.eval(session=sess))
                #step = global_step.eval(session=sess)
                if (step % 100 == 0):
                    print("Step: {}, Loss: {}, Rate: {}".format(step, loss_value, learning_rate))
                #global_step = global_step + 1
                step = step + 1
            else:
                print('From {}: {} does not exist'.format(input_file, label_file))
        
        saver.save(sess, currentDirectory + saveName) # save model checkpoint
    #im = Image.open('input.bmp')
    #h, w = im.size
    #im_te = np.asarray(im).astype('float32')
    #im_in =  im_te.reshape([h, w, 1, 3]);
    
# process a single image of any size using the provided net saved at modelName
def processImage(modelName, imageName):
    im = Image.open(imageName)
    print(im.size)
    h, w = im.size
    im_te = np.asarray(im).astype('float32')
    im_in =  im_te.reshape([h, w, 1, 3]);
    
    x = tf.placeholder('float32', np.shape(im_in))
    print(np.shape(im_in))
    print(x)
    y = buildModel(x)

    
    with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
        saver = tf.train.Saver()
        saver.restore(sess, modelName)
        print('Load model from ' + modelName)
        
        im_out = y.eval(session=sess,feed_dict={x:im_in})
        im_out = im_out.reshape([w, h, 3]).clip(0, 255).astype('uint8')
    im_o = Image.fromarray(im_out)
    im_o.save('output.bmp')


modelPath = 'model\\'
modelName = modelPath + 'model.ckpt'
# Begin a new model
#runTrain(False, 0.000001, saveName=modelName)

'''
for i in range(5):
    loadModel = modelPath + modelName
    modelPath = 'model{}\\'.format(i)
    if not os.path.exists(modelPath):
        os.makedirs(modelPath)
    runTrain(True, loadName=loadModel, saveName=modelPath+modelName)
'''
# Update an existing model
runTrain(True, 0.0000007, loadName=modelName, saveName=modelName)


# Run an image through the net
'''
scene = 15
exposure = 0
phos_path = 'C:\\Users\\vysarge\\Documents\\hdr_dataset\\Phos2_3MP\\Phos2_scene{}\\'.format(scene)
test_name = 'Phos2_uni_sc{}_{}.png'.format(scene, exposure)
processImage(modelName, phos_path + test_name)
'''
