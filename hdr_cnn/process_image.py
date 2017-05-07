import numpy as np
import tensorflow as tf
from PIL import Image
import os
import imageio

# build the net: structure fully defined within this function
def buildModel(image, isTraining):
    # conv1
    with tf.variable_scope('conv1') as scope:
        kernel = tf.Variable(tf.random_normal([3, 3, 3, 64], stddev=0.05), name='weights')
        conv = tf.nn.conv2d(image, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.Variable(tf.zeros([64]), name='biases')
        pre_activation = tf.nn.bias_add(conv, biases)
	norm = tf.contrib.layers.batch_norm(pre_activation, is_training=isTraining)
        conv1 = tf.nn.relu(norm, name=scope.name)

    with tf.variable_scope('conv2') as scope:
        kernel = tf.Variable(tf.random_normal([1, 1, 64, 3], stddev=0.05), name='weights')
        conv = tf.nn.conv2d(conv1, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.Variable(tf.zeros([3]), name='biases')
        pre_activation = tf.nn.bias_add(conv, biases)
	norm = tf.contrib.layers.batch_norm(pre_activation, is_training=isTraining)
        conv2 = tf.nn.relu(norm, name=scope.name)

    with tf.variable_scope('conv3') as scope:
        kernel = tf.Variable(tf.random_normal([1, 1, 3, 3], stddev=0.05), name='weights')
        conv = tf.nn.conv2d(conv2, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.Variable(tf.zeros([3]), name='biases')
        pre_activation = tf.nn.bias_add(conv, biases)
	norm = tf.contrib.layers.batch_norm(pre_activation, is_training=isTraining)
        conv3 = tf.nn.relu(norm, name=scope.name)

    with tf.variable_scope('conv4') as scope:
        kernel = tf.Variable(tf.random_normal([1, 1, 3, 3], stddev=0.05), name='weights')
        conv = tf.nn.conv2d(conv3, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.Variable(tf.zeros([3]), name='biases')
        pre_activation = tf.nn.bias_add(conv, biases)
	norm = tf.contrib.layers.batch_norm(pre_activation, is_training=isTraining)
        conv4 = tf.nn.relu(norm, name=scope.name)

    with tf.variable_scope('conv5') as scope:
        kernel = tf.Variable(tf.random_normal([1, 1, 3, 3], stddev=0.05), name='weights')
        conv = tf.nn.conv2d(conv4, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.Variable(tf.zeros([3]), name='biases')
        pre_activation = tf.nn.bias_add(conv, biases)
	norm = tf.contrib.layers.batch_norm(pre_activation, is_training=isTraining)
        conv5 = tf.nn.relu(norm, name=scope.name)
    return conv5

# currently not used: do some simple transformations to get more mileage out of an image
def setFromImage(image):
    return [image, np.flipud(image), np.fliplr(image)]

# specialized to current file structure: images should be within 2 layers of folders
# extract full list of image file names to use
def fileList(filepath):
    filelist = os.listdir(filepath)
    for i in range(len(filelist)):
        filelist[i] = filepath+filelist[i]
    return filelist

# train on all 32x32 image pieces once
# loadModel = bool: True if a loadName is passed in, to load a model from memory, False if initializing new model
# saveName is the location to save the new model
def runTrain(loadModel, learning_rate_init, loadName='', saveName=''):
    with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
        trainshape = (1, 64, 64, 3)
        norm = 1*64*64*3
        x = tf.placeholder('float32', shape=trainshape)
        l = tf.placeholder('float32', shape=trainshape)
        y = buildModel(x, True)
        loss = tf.sqrt(tf.reduce_sum(tf.square(tf.subtract(l, y)))/norm) # RMSE
        tf.summary.scalar('loss', loss)

        momentum = 0.5
        decay_steps = 10000
        decay_rate = 0.9
        global_step = tf.Variable(0, trainable=False)
        learning_rate = learning_rate_init
        optimizer = tf.train.MomentumOptimizer(learning_rate, momentum, use_nesterov=True)
        train_op = optimizer.minimize(loss)
        
        input_filepath = '/mnt/6344-project-data/_Image_patches/'
        label_filepath = '/mnt/6344-project-data/_HDR_patches/'
        inputs = fileList(input_filepath)
       
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
            fileparts = input_file.split('/')
            img_name = fileparts[-1].split('.')[0]
            label_file = label_filepath + img_name + '.hdr'
            im_in = imageio.imread(input_file)
            h, w, c = im_in.shape
            im_in = np.asarray(im_in).astype('float32')
            im_in = im_in.reshape([1, h, w, 3]);

            im_la = imageio.imread(label_file)
            h, w, c = im_la.shape
            im_la = np.asarray(im_la).astype('float32')
            im_la = im_la.reshape([1, h, w, 3]);

            _, loss_value = sess.run([train_op, loss], feed_dict={x:im_in, l:im_la})

	    if (step % 100 == 0):
      		print("Step {} of {}, Loss: {}, Rate: {}".format(step, len(inputs), loss_value, learning_rate))
            step = step + 1

        
        saver.save(sess, saveName) # save model checkpoint

    
# process a single image of any size using the provided net saved at modelName
def processImage(modelName, imageName, savePath):
    im = imageio.imread(imageName+'.jpg')
    h, w, c = im.shape
    im_te = np.asarray(im).astype('float32')
    im_in =  im_te.reshape([1, h, w, 3]);
    x = tf.placeholder('float32', np.shape(im_in))
    y = buildModel(x, False)

    
    with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
        saver = tf.train.Saver()
        saver.restore(sess, modelName)
        print('Load model from ' + modelName)
        
        im_out = y.eval(session=sess,feed_dict={x:im_in})
        im_out = im_out.reshape([w, h, 3])
        imageio.imwrite(savePath+'.hdr', im_out)



modelPath = '/mnt/6344-project-models/'
if not os.path.exists(modelPath):
	os.makedirs(modelPath)

# Begin a new model
#runTrain(False, 0.000001, saveName=modelPath+'model.ckpt')


for i in range(5):
    runModelPath = modelPath+'model'+str(i)+'/'
    if not os.path.exists(runModelPath):
        os.makedirs(runModelPath)
    if i == 0:
	runTrain(False, 0.000001, saveName=runModelPath+'model.ckpt')
    else:
        loadPath = modelPath+'model'+str(i-1)+'/'
	runTrain(True, loadName=loadModel, saveName=runModelPath+modelName)
#'''
# Update an existing model
#runTrain(True, 0.0000007, loadName=modelName, saveName=modelName)


# Run an image through the net

#path = '/mnt/6344-project-data/'
#save_path = '/mnt/6344-project-results/'
#img_name = 'AgiaGalini_resized'
#if not os.path.exists(save_path):
#	os.makedirs(save_path)
#processImage(modelPath+'model.ckpt', path+img_name,save_path+img_name)

