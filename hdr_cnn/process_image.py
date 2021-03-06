import numpy as np
import tensorflow as tf
from PIL import Image
import os
import imageio
import argparse

# build the net: structure fully defined within this function
def buildModel(image, isTraining):
    # conv1
    with tf.variable_scope('conv1') as scope:
    	kernel = tf.Variable(tf.random_normal([3, 3, 1, 64], stddev=0.05), name='weights')
    	conv = tf.nn.conv2d(image, kernel, [1, 1, 1, 1], padding='SAME')
    	biases = tf.Variable(tf.zeros([64]), name='biases')
    	pre_activation = tf.nn.bias_add(conv, biases)
    	norm = tf.contrib.layers.batch_norm(pre_activation, is_training=isTraining)
    	conv1 = tf.nn.relu(norm, name='conv1')

    with tf.variable_scope('conv2') as scope:
    	kernel = tf.Variable(tf.random_normal([1, 1, 64, 1], stddev=0.05), name='weights')
    	conv = tf.nn.conv2d(conv1, kernel, [1, 1, 1, 1], padding='SAME')
    	biases = tf.Variable(tf.zeros([1]), name='biases')
    	pre_activation = tf.nn.bias_add(conv, biases)
    	norm1 = tf.contrib.layers.batch_norm(pre_activation, is_training=isTraining)
    	conv2 = tf.nn.relu(norm1, name='conv2')

    with tf.variable_scope('conv3') as scope:
    	kernel = tf.Variable(tf.random_normal([1, 1, 1, 1], stddev=0.05), name='weights')
    	conv = tf.nn.conv2d(conv2, kernel, [1, 1, 1, 1], padding='SAME')
    	biases = tf.Variable(tf.zeros([1]), name='biases')
    	pre_activation = tf.nn.bias_add(conv, biases)
    	norm2 = tf.contrib.layers.batch_norm(pre_activation, is_training=isTraining)
    	conv3 = tf.nn.relu(norm2, name='conv3')

    with tf.variable_scope('conv4') as scope:
    	kernel = tf.Variable(tf.random_normal([1, 1, 1, 1], stddev=0.05), name='weights')
    	conv = tf.nn.conv2d(conv3, kernel, [1, 1, 1, 1], padding='SAME')
    	biases = tf.Variable(tf.zeros([1]), name='biases')
    	pre_activation = tf.nn.bias_add(conv, biases)
    	norm3 = tf.contrib.layers.batch_norm(pre_activation, is_training=isTraining)
    	conv4 = tf.nn.relu(norm3, name='conv4')

    with tf.variable_scope('conv5') as scope:
    	kernel = tf.Variable(tf.random_normal([1, 1, 1, 1], stddev=0.05), name='weights')
    	conv = tf.nn.conv2d(conv4, kernel, [1, 1, 1, 1], padding='SAME')
    	biases = tf.Variable(tf.zeros([1]), name='biases')
    	pre_activation = tf.nn.bias_add(conv, biases)
    	norm4 = tf.contrib.layers.batch_norm(pre_activation, is_training=isTraining)
    	conv5 = tf.nn.relu(norm4, name='conv5')
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
def runTrain(channel, x,y,saver,loadModel, learning_rate_init, loadName='', saveName=''):
    with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
        trainshape = (1, 64, 64, 1)
        norm = 1*64*64*1
        l = tf.placeholder('float32', shape=trainshape)
        loss = tf.sqrt(tf.reduce_sum(tf.square(tf.subtract(l, y)))/norm) # RMSE
        tf.summary.scalar('loss', loss)

        momentum = 0.9
        decay_steps = 10000
        decay_rate = 0.9
        global_step = tf.Variable(0, trainable=False)
        learning_rate = learning_rate_init
        optimizer = tf.train.MomentumOptimizer(learning_rate, momentum)
        train_op = optimizer.minimize(loss)
        
        input_filepath = '/mnt/6344-project-data/_Image_patches/'
        label_filepath = '/mnt/6344-project-data/_HDR_patches/'
        inputs = fileList(input_filepath)
        init_op = tf.initialize_all_variables()
	sess.run(init_op)
        if loadModel:
            saver.restore(sess, loadName)
            print('Load model from ' + loadName)
        #else:
            #sess.run(init_op)
            #print('Initializing fresh variables')

        step = 0
        feed_dict = {}
        for input_file in inputs:
            fileparts = input_file.split('/')
            img_name = fileparts[-1].split('.')[0]
            label_file = label_filepath + img_name + '.hdr'
            im_in = imageio.imread(input_file)
            h, w, c = im_in.shape
            im_in = np.asarray(im_in).astype('float32')
	    im_in = im_in[:,:,channel]
            im_in = im_in.reshape([1, h, w, 1]);

            im_la = imageio.imread(label_file)
            h, w, c = im_la.shape
            im_la = np.asarray(im_la).astype('float32')
	    im_la = im_la[:,:,channel]
            im_la = im_la.reshape([1, h, w, 1]);

            _, loss_value = sess.run([train_op, loss], feed_dict={x:im_in, l:im_la})

	    if (step % 100 == 0):
      		print("Step {} of {}, Loss: {}, Rate: {}".format(step, len(inputs), loss_value, learning_rate))
            step = step + 1

        #print(tf.all_variables())
        saver.save(sess, saveName) # save model checkpoint

    
# process a single image of any size using the provided net saved at modelName
def processImage(modelName, imageName, savePath):
    im = imageio.imread(imageName+'.jpg')
    h, w, c = im.shape
    im_te = np.asarray(im).astype('float32')
    im_in =  im_te.reshape([1, h, w, 3]);
    x = tf.placeholder('float32', (1,h,w,1))
    y = buildModel(x, False)

    im_out = np.zeros([h,w,3])
    with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
        saver = tf.train.Saver()
        saver.restore(sess, modelName+'model_r.ckpt')
        print('Load model from ' + modelName)
      	r = im_in[:,:,:,0]
	r = r.reshape([1,h,w,1])
       	out = y.eval(session=sess,feed_dict={x:r})
	im_out[:,:,0] = out.reshape([h,w])
        saver.restore(sess, modelName+'model_g.ckpt')
        print('Load model from ' + modelName)
      	g = im_in[:,:,:,1]
	g = g.reshape([1,h,w,1])
	out = y.eval(session=sess,feed_dict={x:g})
	im_out[:,:,1]= out.reshape([h,w])
        saver.restore(sess, modelName+'model_b.ckpt')
        print('Load model from ' + modelName)
      	b = im_in[:,:,:,2]
	b = b.reshape([1,h,w,1])
	out = y.eval(session=sess,feed_dict={x:b})
	im_out[:,:,2] =out.reshape([h,w])
        #im_out = im_out.reshape([h, w, 3])
	
        imageio.imwrite(savePath+'.hdr', np.asarray(im_out).astype('float32'))



modelPath = '/mnt/6344-project-models/'
if not os.path.exists(modelPath):
	os.makedirs(modelPath)

# Begin a new model
#runTrain(False, 0.000001, saveName=modelPath+'model.ckpt')
def train():
	trainshape = (1, 64, 64, 1)
	x = tf.placeholder('float32', shape=trainshape)
	y = buildModel(x, True)
	saver = tf.train.Saver()
	lr = 0.00001
	for i in range(1):
    		runModelPath = modelPath+'model'+str(i)+'/'
    		if not os.path.exists(runModelPath):
        		os.makedirs(runModelPath)
   	 	if i == 0:
			runTrain(0, x,y, saver, False, lr, saveName=runModelPath+'model_r.ckpt')
			runTrain(1, x,y, saver, False, lr, saveName=runModelPath+'model_g.ckpt')
			runTrain(2, x,y, saver, False, lr, saveName=runModelPath+'model_b.ckpt')
    		else:
        		loadPath = modelPath+'model'+str(i-1)+'/'
			lr = .9*lr
			runTrain(0, x,y, saver, True, lr, loadName=loadPath+'model_r.ckpt', saveName=runModelPath+'model_r.ckpt')
			runTrain(1, x,y, saver, True, lr, loadName=loadPath+'model_g.ckpt', saveName=runModelPath+'model_g.ckpt')
			runTrain(2, x,y, saver, True, lr, loadName=loadPath+'model_b.ckpt', 	saveName=runModelPath+'model_b.ckpt')
def test():		
# Run an image through the net
	path = '/mnt/6344-project-data/resized_imgs/'
	save_path = '/mnt/6344-project-results/'
	img_name = 'Cafe'
	if not os.path.exists(save_path):
		os.makedirs(save_path)
	processImage(modelPath+'model0/', path+img_name,save_path+img_name)


parser = argparse.ArgumentParser()
parser.add_argument('--train', action='store_true')
parser.add_argument('--test', action='store_true')
args = parser.parse_args()

if args.train:
	print('Training')
	train()
if args.test:
	print('Testing')
	test()
