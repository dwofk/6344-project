import numpy as np
import tensorflow as tf
from PIL import Image
import os
import random
import cv2

device_name = "/cpu:0"
#device_name = "/gpu:0"
ycrcb = True
customImage = True
customName = 'ChineseGarden.png'
input_w = 3000
input_h = 2000
custom_w = 870 #4032
custom_h = 578 #3024
kernel_size = 1
training = False
plusLabel = True
#os_slash = '\\'
os_slash = '/'

currentDirectory = '/home/vysarge/Documents/repos/6344-project/exposure_cnn/'
#currentDirectory = 'S:\\6344-project\\exposure_cnn\\'

# create a prelu image
def prelu(x, alpha, name):
    pos = tf.nn.relu(x, name)
    neg = alpha * (x - abs(x)) * 0.5
    
    return pos + neg

# build the net: structure fully defined within this function
def buildModel(image, isTraining):
    # conv1
    with tf.variable_scope('conv1') as scope:
        kernel = tf.Variable(tf.random_normal([kernel_size, kernel_size, 3, 32], stddev=0.05), name='weights')
        conv = tf.nn.conv2d(image, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.Variable(tf.zeros([32]), name='biases')
        pre_activation = tf.nn.bias_add(conv, biases)
        #norm1 = tf.contrib.layers.batch_norm(pre_activation, is_training=isTraining)
        #conv1 = tf.nn.relu(norm1, name='conv1')
        #conv1 = tf.nn.relu(pre_activation, name=scope.name)
        alpha = tf.Variable(tf.zeros(pre_activation.get_shape()[-1]), name='alpha');
        #alpha = tf.get_variable('alpha', pre_activation.get_shape()[-1], initializer=tf.constant_initializer(0.0), dtype=tf.float32)
        conv1 = prelu(pre_activation, alpha, name=scope.name)

    with tf.variable_scope('conv2') as scope:
        kernel = tf.Variable(tf.random_normal([1, 1, 32, 3], stddev=0.05), name='weights')
        conv = tf.nn.conv2d(conv1, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.Variable(tf.zeros([3]), name='biases')
        pre_activation = tf.nn.bias_add(conv, biases)
        #alpha = tf.Variable(tf.zeros(pre_activation.get_shape()[-1]), name='alpha');
        #conv1 = tf.nn.relu(pre_activation, name=scope.name)
        #norm2 = tf.contrib.layers.batch_norm(pre_activation, is_training=isTraining)
        alpha = tf.get_variable('alpha', pre_activation.get_shape()[-1], initializer=tf.constant_initializer(0.0), dtype=tf.float32)
        conv2 = prelu(pre_activation, alpha, name=scope.name)
        #norm2 = tf.contrib.layers.batch_norm(pre_activation, is_training=isTraining)
        #conv2 = tf.nn.relu(norm2, name='conv2')
    return conv2

def buildChannelModel(image, isTraining):
    # conv1
    with tf.variable_scope('conv1') as scope:
        kernel = tf.Variable(tf.random_normal([1, 1, 1, 32], stddev=0.05), name='weights')
        conv = tf.nn.conv2d(image, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.Variable(tf.zeros([32]), name='biases')
        pre_activation = tf.nn.bias_add(conv, biases)
        alpha = tf.Variable(tf.zeros(pre_activation.get_shape()[-1]), name='alpha');
        conv1 = prelu(pre_activation, alpha, name=scope.name)

    with tf.variable_scope('conv2') as scope:
        kernel = tf.Variable(tf.random_normal([1, 1, 32, 1], stddev=0.05), name='weights')
        conv = tf.nn.conv2d(conv1, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.Variable(tf.zeros([1]), name='biases')
        pre_activation = tf.nn.bias_add(conv, biases)
        alpha = tf.get_variable('alpha', pre_activation.get_shape()[-1], initializer=tf.constant_initializer(0.0), dtype=tf.float32)
        conv2 = prelu(pre_activation, alpha, name=scope.name)
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
        fullfilenames = [filepath + directory + os_slash + x for x in filenames]
        filelist.extend(fullfilenames)
    random.shuffle(filelist)
    return filelist


# train on all 32x32 image pieces once
# loadModel = bool: True if a loadName is passed in, to load a model from memory, False if initializing new model
# plusExposure = bool: True if higher-exposure labels should be used, False if lower-exposure labels should be used
# saveName is the location to save the new model
def runTrain(x, y, saver, loadModel, plusExposure, learning_rate, momentum, loadName='', saveName=''):
    config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)
    with tf.device(device_name):
    #with tf.device("/cpu:0"):
        trainshape = (32, 32, 1, 3)
        norm = 32*32*1*3
        l = tf.placeholder('float32', shape=trainshape)
        sqdiff = tf.square(tf.subtract(l, y))
        y_sqdiff = sqdiff[:,:,:,0]
        cr_sqdiff = sqdiff[:,:,:,1]*10
        cb_sqdiff = sqdiff[:,:,:,2]*10
        print(sqdiff)
        reduced = tf.reduce_sum(y_sqdiff + cr_sqdiff + cb_sqdiff)/norm
        print(reduced)
        loss = tf.sqrt(reduced) # RMSE
        tf.summary.scalar('loss', loss)

        global_step = tf.Variable(0, trainable=False)
        #optimizer = tf.train.GradientDescentOptimizer(learning_rate)
        optimizer = tf.train.MomentumOptimizer(learning_rate, momentum)#, use_nesterov=True)
        #optimizer = tf.train.AdagradOptimizer(learning_rate)
        train_op = optimizer.minimize(loss)
        
        #input_filepath = 'phos' + os_slash + '0' + os_slash
        #label_filepath = 'phos\\plus_2\\'
        input_filepath = 'C:\\Users\\vysarge\\Documents\\hdr_dataset\\empapatches\\0\\'
        if (plusExposure):
            label_filepath = 'C:\\Users\\vysarge\\Documents\\hdr_dataset\\empapatches\\plus_4\\'
        else:
            label_filepath = 'C:\\Users\\vysarge\\Documents\\hdr_dataset\\empapatches\\minus_4\\'
        
        inputs = fileList(input_filepath)

        #with tf.Graph().as_default():
        
    with tf.Session(config=config) as sess:
        if loadModel:
            init_op = tf.global_variables_initializer()
            sess.run(init_op)
            saver.restore(sess, loadName + '\\' + 'model.ckpt')
            print('Load model from ' + loadName)
        else:
            init_op = tf.global_variables_initializer()
            sess.run(init_op)
            print('Initializing fresh variables')

        step = 0
        feed_dict = {}
        sum_loss = 0
        for input_file in inputs:
            fileparts = input_file.split('\\')
            scene = fileparts[-2]
            file = fileparts[-1] # the last element of this split should be the file name
            label_file = label_filepath + scene + '\\' + file
            if (os.path.exists(label_file)):
                im_in = cv2.imread(input_file)
                im_in = cv2.cvtColor(im_in, cv2.COLOR_RGB2YCR_CB)
                #im_in = cv2.cvtColor(im_in, cv2.COLOR_RGB2HSV)
                h, w, d = np.shape(im_in)
                im_in = np.asarray(im_in).astype('float32')
                im_in = im_in.reshape([h, w, 1, 3]);
                #im_inputs = setFromImage(im_in)

                im_la = cv2.imread(label_file)
                im_la = cv2.cvtColor(im_la, cv2.COLOR_RGB2YCR_CB)
                #im_la = cv2.cvtColor(im_la, cv2.COLOR_RGB2HSV)
                h, w, d = np.shape(im_la)
                im_la = np.asarray(im_la).astype('float32')
                im_la = im_la.reshape([h, w, 1, 3]);
                #im_labels = setFromImage(im_la)
                _, loss_value = sess.run([train_op, loss], feed_dict={x:im_in, l:im_la})
                #print(global_step.eval(session=sess))
                #step = global_step.eval(session=sess)
                sum_loss = sum_loss + loss_value
                if (step % 10000 == 0):
                    avg_loss = sum_loss/10000
                    sum_loss = 0
                    print("Step: {}, Loss: {}, Rate: {}".format(step, avg_loss, learning_rate))
                #global_step = global_step + 1
                step = step + 1
            else:
                print('From {}: {} does not exist'.format(input_file, label_file))

        if not os.path.exists(currentDirectory+saveName):
            os.makedirs(currentDirectory+saveName)
        saver.save(sess, currentDirectory + saveName + '\\' + 'model.ckpt') # save model checkpoint
    #im = Image.open('input.bmp')
    #h, w = im.size
    #im_te = np.asarray(im).astype('float32')
    #im_in =  im_te.reshape([h, w, 1, 3]);

def runTrainChannel(channel, x, y, saver, loadModel, plusExposure, learning_rate, momentum, loadName='', saveName=''):
    config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)
    with tf.device(device_name):
    #with tf.device("/cpu:0"):
        trainshape = (32, 32, 1, 1)
        norm = 32*32*1*1
        l = tf.placeholder('float32', shape=trainshape)
        sqdiff = tf.square(tf.subtract(l, y))
        reduced = tf.reduce_sum(sqdiff)/norm
        loss = tf.sqrt(reduced) # RMSE
        tf.summary.scalar('loss', loss)
        
        currentDirectory = 'S:\\6344-project\\exposure_cnn\\'

        global_step = tf.Variable(0, trainable=False)
        #optimizer = tf.train.GradientDescentOptimizer(learning_rate)
        optimizer = tf.train.MomentumOptimizer(learning_rate, momentum)#, use_nesterov=True)
        #optimizer = tf.train.AdagradOptimizer(learning_rate)
        train_op = optimizer.minimize(loss)
        
        #input_filepath = 'phos\\0\\'
        #label_filepath = 'phos\\plus_2\\'
        input_filepath = 'C:\\Users\\vysarge\\Documents\\hdr_dataset\\empapatches\\0\\'
        if (plusExposure):
            label_filepath = 'C:\\Users\\vysarge\\Documents\\hdr_dataset\\empapatches\\plus_4\\'
        else:
            label_filepath = 'C:\\Users\\vysarge\\Documents\\hdr_dataset\\empapatches\\minus_4\\'
        
        inputs = fileList(input_filepath)

        #with tf.Graph().as_default():
        
    with tf.Session(config=config) as sess:
        if loadModel:
            init_op = tf.global_variables_initializer()
            sess.run(init_op)
            saver.restore(sess, loadName + '\\' + 'model.ckpt')
            print('Load model from ' + loadName)
        else:
            init_op = tf.global_variables_initializer()
            sess.run(init_op)
            print('Initializing fresh variables')

        step = 0
        feed_dict = {}
        sum_loss = 0
        for input_file in inputs:
            fileparts = input_file.split('\\')
            scene = fileparts[-2]
            file = fileparts[-1] # the last element of this split should be the file name
            label_file = label_filepath + scene + '\\' + file
            if (os.path.exists(label_file)):
                im_in = cv2.imread(input_file)
                if (ycrcb):
                    im_in = cv2.cvtColor(im_in, cv2.COLOR_RGB2YCR_CB)
                im_in = im_in[:,:,channel]
                h, w = np.shape(im_in)
                im_in = np.asarray(im_in).astype('float32')
                im_in = im_in.reshape([h, w, 1, 1]);
                #im_inputs = setFromImage(im_in)

                im_la = cv2.imread(label_file)
                if (ycrcb):
                    im_la = cv2.cvtColor(im_la, cv2.COLOR_RGB2YCR_CB)
                im_la = im_la[:,:,channel]
                #im_la = cv2.cvtColor(im_la, cv2.COLOR_RGB2HSV)
                h, w = np.shape(im_la)
                im_la = np.asarray(im_la).astype('float32')
                im_la = im_la.reshape([h, w, 1, 1]);
                #im_labels = setFromImage(im_la)
                _, loss_value = sess.run([train_op, loss], feed_dict={x:im_in, l:im_la})
                #print(global_step.eval(session=sess))
                #step = global_step.eval(session=sess)
                sum_loss = sum_loss + loss_value
                if (step % 1000 == 0):
                    avg_loss = sum_loss/1000
                    sum_loss = 0
                    print("Step: {}, Loss: {}, Rate: {}".format(step, avg_loss, learning_rate))
                #global_step = global_step + 1
                step = step + 1
            else:
                print('From {}: {} does not exist'.format(input_file, label_file))

        if not os.path.exists(currentDirectory+saveName):
            os.makedirs(currentDirectory+saveName)
        saver.save(sess, currentDirectory + saveName + '\\' + 'model.ckpt') # save model checkpoint
    #im = Image.open('input.bmp')
    #h, w = im.size
    #im_te = np.asarray(im).astype('float32')
    #im_in =  im_te.reshape([h, w, 1, 3]);

def train(modelName, epochs):
    assert(epochs>0, 'Must train for at least one epoch!')
    config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)
    with tf.device(device_name):
        trainshape = (32, 32, 1, 3)
        norm = 32*32*1*3
        x = tf.placeholder('float32', shape=trainshape)
        y = buildModel(x, True)
        print(y)
        
        saver = tf.train.Saver()
        learning_rate = 0.000003
        momentum = 0.9
        runTrain(x, y, saver, False, plusLabel, learning_rate, momentum, loadName='', saveName=modelName+'0')
        for i in range(epochs-1):
            loadName = modelName+'{}'.format(i)
            saveName = modelName+'{}'.format(i+1)
            learning_rate = learning_rate / 2
            runTrain(x, y, saver, True, plusLabel, learning_rate, momentum, loadName=loadName, saveName=saveName)

def trainChannel(modelName, epochs):
    assert(epochs>0, 'Must train for at least one epoch!')
    config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)
    with tf.device(device_name):
        trainshape = (32, 32, 1, 1)
        norm = 32*32*1*1
        x = tf.placeholder('float32', shape=trainshape)
        y = buildChannelModel(x, True)
        print(y)
        
        saver = tf.train.Saver()
        learning_rate = 0.0000003
        momentum = 0.9
        runTrainChannel(0, x, y, saver, False, True, learning_rate, momentum, loadName='', saveName=modelName+'0y')
        runTrainChannel(1, x, y, saver, False, True, learning_rate, momentum, loadName='', saveName=modelName+'0cr')
        runTrainChannel(2, x, y, saver, False, True, learning_rate, momentum, loadName='', saveName=modelName+'0cb')
        for i in range(epochs-1):
            loadName = modelName+'{}'.format(i)
            saveName = modelName+'{}'.format(i+1)
            learning_rate = learning_rate / 2
            runTrainChannel(0, x, y, saver, True, True, learning_rate, momentum, loadName=loadName+'y', saveName=saveName+'y')
            runTrainChannel(1, x, y, saver, True, True, learning_rate, momentum, loadName=loadName+'cr', saveName=saveName+'cr')
            runTrainChannel(2, x, y, saver, True, True, learning_rate, momentum, loadName=loadName+'cb', saveName=saveName+'cb')


# process a single image of any size using the provided net saved at modelName
def processImage(x, y, modelName, imageName, outputName):
    #im = Image.open(imageName)
    #im = im.convert('YCbCr')
    
    im = cv2.imread(imageName)
    if not customImage:
        im = cv2.resize(im, (input_w, input_h), interpolation=cv2.INTER_AREA)
    if (ycrcb):
        im = cv2.cvtColor(im, cv2.COLOR_RGB2YCR_CB)
    #im = cv2.cvtColor(im, cv2.COLOR_RGB2HSV)
    
    print(np.shape(im))
    h, w, d = np.shape(im)
    im_te = np.asarray(im).astype('float32')
    im_in =  im_te.reshape([h, w, 1, 3]);

    
    
    with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
        saver = tf.train.Saver()
        init_op = tf.global_variables_initializer()
        sess.run(init_op)
        saver.restore(sess, modelName)
        print('Load model from ' + modelName)
        
        im_out = y.eval(session=sess,feed_dict={x:im_in})
        im_o = im_out.reshape([h, w, 3]).clip(0, 255).astype('uint8')
    if (ycrcb):
        im_o = cv2.cvtColor(im_o, cv2.COLOR_YCR_CB2RGB)
    #im_o = cv2.cvtColor(im_out, cv2.COLOR_HSV2RGB)
    
    cv2.imwrite(currentDirectory + outputName, im_o)
    print("Wrote to {}".format(currentDirectory + outputName))


modelPath = 'model' + os_slash
modelName = modelPath + 'model'
# train a new model
if (training):
    train(modelName, 9)
#trainChannel(modelName, 3)


# Run an image through the net
if not training and not customImage:
    modelPath = 'model' + os_slash
    modelName = modelPath + 'model8' + os_slash + 'model.ckpt'

    config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)
    with tf.device(device_name):
        x = tf.placeholder('float32', shape=(input_h, input_w, 1, 3))
        y = buildModel(x, True)

    image_dir = 'inputs'
    output_dir = 'outputs'
    for directory in os.listdir(image_dir):
        if not os.path.exists(currentDirectory+output_dir+os_slash+directory):
            os.makedirs(currentDirectory+output_dir+os_slash+directory)
        for image_file in os.listdir(image_dir + os_slash + directory):
            processImage(x, y, modelName, image_dir+os_slash+directory+os_slash+image_file, output_dir+os_slash+directory+os_slash+image_file)
        print(currentDirectory+output_dir+os_slash+directory)
else:
    modelPath = 'model' + os_slash
    modelName = modelPath + 'model8' + os_slash + 'model.ckpt'
    
    config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)
    with tf.device(device_name):
        x = tf.placeholder('float32', shape=(custom_h, custom_w, 1, 3))
        y = buildModel(x, True)

    image_file = 'inputs/' + customName
    output_file = 'outputs/' + customName
    processImage(x, y, modelName, image_file, output_file)

#processImage(x, y, modelName, 'input.jpg', 'output.jpg')
#processImage(x, y, modelName, 'input2.jpg', 'output2.jpg')
#scene = 15
#exposure = 0
#phos_path = 'C:\\Users\\vysarge\\Documents\\hdr_dataset\\Phos2_3MP\\Phos2_scene{}\\'.format(scene)

#test_name = 'Phos2_uni_sc{}_{}.png'.format(scene, exposure)
#processImage(modelName, phos_path + test_name)


