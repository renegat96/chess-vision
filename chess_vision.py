"""
@authors Adam Stanford-Moore and Hristo Stoyanov
@date June 7, 2019
Stanford Universtiy CS230 Final Project
"""
import math
import argparse
import numpy as np
import random
import tensorflow as tf
from matplotlib.image import imread

from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.layers import Dense, Flatten, Input, AlphaDropout, Activation
from tensorflow.keras.models import Model
from tensorflow.contrib.image import rotate
import glob
import matplotlib.pyplot as plt
from time import time
#import sklearn.metrics as metrics
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
import sys
from keras.callbacks import Callback

import cv2

OUTFOLDER = "testing607"
class_names = ['bb', 'bk', 'bn', 'bp', 'bq', 'br',
               'empty',
               'wb', 'wk', 'wn', 'wp', 'wq', 'wr']
class_weights = {
        0:  1.2, #'bb'
        1:  1.2, #'bk'
        2:  1, #'bn'
        3:  1  , #'bp'
        4:  1  , #'bq'
        5:  1  , #'br'
        6:  1  , #'empty'
        7:  1.2, #'wb'
        8:  1.2, #'wk'
        9:  1, #'wn'
        10: 1  , #'wp'
        11: 1  , #'wq'
        12: 1    #'wr'
}
#MAX_TRAIN_NUM = 400 #limit number of train images per class
#MAX_TEST_NUM = 20 #limit number of test images per class
PATH_TO_DATA = '/home/ubuntu/cs230/Chess ID Public Data'

mnist = tf.keras.datasets.mnist

#Inside folderPath is the subdirectory 'train' and 'dev' inside which are folders 'bb','bn',etc
def load_set(folderPath,datasets,MAX_NUM,train_percent=80,dev_percent=10, augment_arr = False):
    for index,subdir in enumerate(class_names):
        path = folderPath + "/" + subdir + "/*"
        files = glob.glob(path)
        x, y = [], []
        for j,myFile in enumerate(files):
            if j > MAX_NUM: break
            image = load_img(myFile, target_size=(224,224))
            image = img_to_array(image)
            image = preprocess_input(image)
            x.append(image)
            y.append(index)
        if len(x) == 0: continue #in case no pawns/empty
        the_split = split(x,y,train_percent,dev_percent)    
        if augment_arr == True:
            the_split[0][0],the_split[0][1] = augment(the_split[0][0],the_split[0][1]) #augmenting train data only
        for a in range(3):
            for b in range(2):
                if type(datasets[a][b]) == type(None):
                    datasets[a][b] = the_split[a][b]
                else:
                    datasets[a][b] = np.concatenate((datasets[a][b],the_split[a][b]))
                #print(datasets[a][b].shape,the_split[a][b].shape)
            
def load_data(folderPath):
    datasets = np.array([[None,None],[None,None],[None,None]]) #train x,y ; dev x,y ; test x,y   

    #load_set(folderPath + '/output_train/',datasets,MAX_NUM=1500,train_percent=80,dev_percent=10)
    #print(datasets[0][0].shape,datasets[0][1].shape)
    #print("Done with loading training data", len(datasets[0][1]))
    
    #load_set(folderPath + '/output_test/',datasets,MAX_NUM=20000,train_percent=80,dev_percent=10)
    #print(datasets[1][0].shape,datasets[1][1].shape)
    #print("Done with loading testing data", len(datasets[1][1]))

    load_set('data/',datasets,MAX_NUM = 100000,train_percent=80,dev_percent=10,augment_arr = True)
    #print(datasets[2][0].shape,datasets[2][1].shape)
    print("Done with loading custom data", len(datasets[2][1]))

    #load_set('/home/ubuntu/adam_sorted/',datasets,MAX_NUM = 100000,train_percent=80,dev_percent=10,augment_arr = True)
    #print(datasets[2][0].shape,datasets[2][1].shape)
    #print("Done with loading custom data", len(datasets[2][1]))

    #load_set('/home/ubuntu/last_data_sorted_nice/',datasets,MAX_NUM = 100000,train_percent=80,dev_percent=10,augment_arr = True)
    #print(datasets[2][0].shape,datasets[2][1].shape)
    #print("Done with loading custom data", len(datasets[2][1]))

    return (datasets[0][0],datasets[0][1]),(datasets[1][0],datasets[1][1]),(datasets[2][0],datasets[2][1])

def augment(x_train, y_train):
    print("augmenting % images" % len(x_train))
    rot90 = np.rot90(x_train, k=1, axes=(1, 2))
    fl = np.flip(x_train, axis=1)
    #rot180 = np.rot90(x_train, k=2, axes=(1, 2))
    x = np.concatenate((x_train, rot90, fl))
    y = np.concatenate((y_train, y_train, y_train))
    return (x, y)

def shuffle(x,y):
    indices = np.arange(len(y))
    x = np.array([x[i] for i in indices])
    y = np.array([y[i] for i in indices])
    return x,y 

# split data into train/dev/test
def split(x_data,y_data,train_percent=80,dev_percent=10):
    indices = np.arange(len(y_data))
    np.random.shuffle(indices)
    train_split = int(train_percent/100.0 * len(indices)) #90% train
    dev_split = int((train_percent+dev_percent)/100 * len(indices))   #10% dev
    train_indices = indices[:train_split]
    dev_indices = indices[train_split:dev_split]
    test_indices = indices[dev_split:]
    x_train = np.array([x_data[i] for i in train_indices])
    y_train = np.array([y_data[i] for i in train_indices])
    x_dev = np.array([x_data[i] for i in dev_indices])
    y_dev = np.array([y_data[i] for i in dev_indices])
    x_test = np.array([x_data[i] for i in test_indices])
    y_test = np.array([y_data[i] for i in test_indices])
    return [[x_train,y_train],[x_dev,y_dev],[x_test,y_test]]

def revert_preprocess(x):
    img = np.copy(x)
    # revert to real image from pre-processing
    mean = [103.939, 116.779, 123.68]
    img[..., 0] += mean[0]
    img[..., 1] += mean[1]
    img[..., 2] += mean[2]
    # BGR -> RGB
    img = img[..., ::-1]
    return img.astype(int)

def save_data(x, y, dirname):
    for i in range(len(x)):
        img = revert_preprocess(x[i])
        label = y[i]
        cv2.imwrite("%s/%s/%d.png" % (dirname, class_names[label], i), img)

np.random.seed(230) #for reproducibility
(x_train,y_train),(x_dev,y_dev),(x_test,y_test) = load_data(PATH_TO_DATA)
x_train,y_train = shuffle(x_train,y_train)
x_dev,y_dev = shuffle(x_dev,y_dev)
x_test,y_test = shuffle(x_test,y_test)

save_data(x_dev, y_dev, 'dev_data')

print("X_train",x_train.shape)
print("Y_train",y_train.shape)
for i in range(len(class_names)):
    print(class_names[i], np.count_nonzero(y_train == i))

print("X_dev",x_dev.shape)
print("y_dev",y_dev.shape)
for i in range(len(class_names)):
    print(class_names[i], np.count_nonzero(y_dev == i))

print("X_test",x_test.shape)
print("y_test",y_test.shape)
for i in range(len(class_names)):
    print(class_names[i], np.count_nonzero(y_test == i))




def parseArgs():
    parser = argparse.ArgumentParser(description='Train a chess vision model')
    parser.add_argument('--layers', type=int, help='an integer for the accumulator', dest='layers')
    parser.add_argument('--batch-size', type=int, dest='batch_size')
    parser.add_argument('--epochs', type=int, dest='epochs')
    #parser.add_argument('--dropout', type=float, dest='dropout')

    args = parser.parse_args()
    return (args.layers, args.batch_size, args.epochs) #, args.dropout)


LAYERS, BATCH_SIZE, EPOCHS= parseArgs()
print("Layers",LAYERS, "batch:",BATCH_SIZE, "Epochs",EPOCHS)


class Metrics(Callback):
    #def __init__(self, val_data):#, batch_size = 20):
    #    super().__init__()
    #    self.validation_data = val_data
    #    #self.batch_size = batch_size
    def __init__(self, train_data,val_data):
        self.validation_data = val_data
        self.train_data = train_data
    def on_train_begin(self, logs={}):
        self.train_f1s = [] 
        self.val_f1s = []
        self.val_class_f1s = []
        #self.val_recalls = []
        #self.val_precisions = []
 
    def on_epoch_end(self, epoch, logs={}):
        #print(self.validation_data)
        val_predict_all = self.model.predict(self.validation_data[0])
        val_predict = np.argmax(val_predict_all,axis=1)
        val_targ = self.validation_data[1]
        train_targ = self.train_data[1]
        train_predict = np.argmax(self.model.predict(self.train_data[0]),axis=1)
        
        self.val_class_f1s.append(f1_score(val_targ, val_predict,average=None))
        _val_f1 = np.mean(self.val_class_f1s[-1])
        
        self.train_f1s.append(f1_score(train_targ, train_predict,average='macro'))
        #_val_recall = recall_score(val_targ, val_predict,average='macro')
        #_val_precision = precision_score(val_targ, val_predict,average='macro')
        self.val_f1s.append(_val_f1)
        #self.val_recalls.append(_val_recall)
        #self.val_precisions.append(_val_precision)
        #print (" — val_f1: %f — val_precision: %f — val_recall %f" %(_val_f1, _val_precision, _val_recall))
        return
    def on_batch_begin(self, batch, logs={}):
        return
    def on_batch_end(self, batch, logs={}):
        return
    def on_train_end(self, logs={}):
        return
    def on_epoch_begin(self, epoch, logs={}):
        return
    def on_train_batch_begin(self, batch, logs=None):
        return 
    def on_train_batch_end(self, batch, logs=None):
        return
    def on_test_batch_begin(self, batch, logs=None):
        return
    def on_test_batch_end(self, batch, logs=None):
        return 
metrics = Metrics((x_train,y_train),(x_dev,y_dev))


model = tf.keras.applications.ResNet50(include_top=True,
                                             weights='imagenet')
#model.summary()
last_layer = model.get_layer('avg_pool').output
#x = Flatten(name='flatten')(last_layer)
#print(class_names)
"""
out = Dense(256,
            kernel_initializer=tf.keras.initializers.he_normal(seed=230*230),
            activation=tf.nn.relu,
            name='custom_relu_1')(last_layer)
out = AlphaDropout(DROPOUT)(out)
"""
"""
out = Dense(256,
            kernel_initializer=tf.keras.initializers.he_normal(seed=230*231),
            activation=tf.nn.relu,
            name='custom_relu_2')(last_layer)
out = AlphaDropout(DROPOUT)(out)
"""
out = Dense(len(class_names),
            kernel_initializer=tf.keras.initializers.TruncatedNormal(mean=0.0, stddev=0.05, seed=None),
            activation=tf.nn.softmax,
            name='output_layer')(last_layer)
custom_resnet_model = Model(inputs=model.input, outputs=out)

for l in custom_resnet_model.layers[:-LAYERS]: 
    l.trainable = False
#custom_resnet_model.summary()

op = tf.keras.optimizers.Adam()
model = custom_resnet_model
model.compile(optimizer=op,
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'],)

history = model.fit(x_train, y_train, validation_data=(x_dev,y_dev),
         batch_size=BATCH_SIZE, epochs=EPOCHS, #class_weight=class_weights,
         shuffle=True, callbacks=[metrics])

print("F1",metrics.val_f1s)
plt.plot(range(1,EPOCHS+1),metrics.train_f1s,label='Train F1 score')
plt.plot(range(1,EPOCHS+1),metrics.val_f1s,label='Dev F1 score')
#plt.plot(range(1,EPOCHS+1),metrics.val_recalls,label='average recall')
#plt.plot(range(1,EPOCHS+1),metrics.val_precisions,label='average precision')
plt.xlabel("Epoch")
plt.legend()
plt.savefig(OUTFOLDER + "/f1_per_epoch_batch%d_layers%d_epochs%d" % (BATCH_SIZE,LAYERS,EPOCHS))
plt.close()

FMTS = ['C0','C1','C2','C3','C4','C5','k','C0--','C1--','C2--','C3--','C4--','C5--']
f1_class_list = np.array(metrics.val_class_f1s).T
for i in range(len(f1_class_list)):
    plt.plot(range(1,EPOCHS+1),f1_class_list[i],FMTS[i],label=class_names[i])
plt.xlim(-1,EPOCHS)
plt.xlabel("Epoch")
plt.ylabel("F1 Score")
plt.title("F1 Score per Class")
plt.legend(loc='upper left')
plt.savefig(OUTFOLDER + "/f1_all_classes_batch%d_layers%d_epochs%d" % (BATCH_SIZE,LAYERS,EPOCHS))
plt.close()

# Plot training & validation accuracy values
plt.plot(range(1,EPOCHS+1),history.history['acc'])
plt.plot(range(1,EPOCHS+1),history.history['val_acc'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.savefig(OUTFOLDER + "/train_model_acc_batch%d_layers%d_epochs%d" % (BATCH_SIZE,LAYERS,EPOCHS))
plt.close()

# Plot training & validation loss values
plt.plot(range(1,EPOCHS+1),history.history['loss'])
plt.plot(range(1,EPOCHS+1),history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.savefig(OUTFOLDER + "/train_model_loss_batch%d_layers%d_epochs%d" % (BATCH_SIZE,LAYERS,EPOCHS))
plt.close()

#model.evaluate(x_train, y_train)
model.evaluate(x_dev, y_dev)
#model.evaluate(x_custom, y_custom)

predictions = model.predict(x_dev)
y_pred = np.argmax(predictions,axis=1)
f1 = f1_score(y_dev, y_pred, average=None)
f1_mean = np.mean(f1)
print("F1 per class", f1)
print("Total F1 = %.4f" % np.mean(f1))

predictions_test = model.predict(x_test)
y_pred_test = np.argmax(predictions_test,axis=1)
f1_test = f1_score(y_test, y_pred_test, average=None)
f1_mean_test = np.mean(f1_test)
print("F1 per class; Test", f1_test)
print("Total F1; test = %.4f" % np.mean(f1_test))

plt.bar(class_names, f1, align='center', alpha=0.5)
#plt.xticks(y_pos, objects)
plt.ylabel('F1 score')
plt.title('F1 score per class')
plt.savefig(OUTFOLDER + '/train_f1_score_batch%d_layers%d_epochs%d' % (BATCH_SIZE,LAYERS,EPOCHS))
plt.close()

f=open("F1_SCORE_RESULTS.txt", "a+")
f.write("F1 %.3f f1_test %.3f layers %d Batch_size %d epochs %d\n" %  (f1_mean,f1_mean_test,LAYERS,BATCH_SIZE,EPOCHS))
f.close()

f = open("F1_MAX.txt","r")
lines = f.readlines()
max_so_far = 0 if len(lines) == 0 else float(lines[0].split()[1])
f.close()

if max_so_far == None or f1_mean > max_so_far:
    f2 = open("F1_MAX.txt", "w+")
    f2.write("F1 %.3f layers %d Batch_size %d epochs %d \n" % (f1_mean,LAYERS,BATCH_SIZE,EPOCHS))
    f2.close()
    model.save("best_model_f1%d_batch%d_layers%d_epochs%d.h5" % (f1_mean*100,BATCH_SIZE,LAYERS,EPOCHS))


#confusion matrix
def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    fig, ax = plt.subplots(figsize=(8, 8))
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes)
    ax.set_title(title,size = 20)
    ax.set_ylabel('True label',size = 20)
    ax.set_xlabel('Predicted label',size = 20)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor",size = 18)
    plt.setp(ax.get_yticklabels(),size = 18)

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    name = OUTFOLDER + "/confusion_matrix_batch%d_layers%d_epochs%d_f1%d" %     (BATCH_SIZE,LAYERS,EPOCHS,f1_mean_test*100)
    if normalize:
        name = name + "_norm"
    plt.savefig(name)
    plt.close()
    return ax

np.set_printoptions(precision=2)

# Plot non-normalized confusion matrix
plot_confusion_matrix(y_test, y_pred_test, classes=class_names,
                      title='Confusion matrix, without normalization')

# Plot normalized confusion matrix
plot_confusion_matrix(y_test, y_pred_test, classes=class_names, normalize=True,
                      title='Normalized confusion matrix')

def plot_image(i, predictions_array, true_label, img):
  predictions_array, true_label, img = predictions_array[i], true_label[i], img[i]
  img = revert_preprocess(img)
  plt.grid(False)
  plt.xticks([])
  plt.yticks([])
  
  plt.imshow(img, cmap=plt.cm.binary)

  predicted_label = np.argmax(predictions_array)
  if predicted_label == true_label:
    color = 'blue'
  else:
    color = 'red'
  
  plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
                                100*np.max(predictions_array),
                                class_names[true_label]),
                                color=color)

def plot_value_array(i, predictions_array, true_label):
  predictions_array, true_label = predictions_array[i], true_label[i]
  plt.grid(False)
  plt.xticks([])
  plt.yticks([])
  thisplot = plt.bar(range(len(class_names)), predictions_array, color="#777777")
  plt.ylim([0, 1]) 
  predicted_label = np.argmax(predictions_array)
 
  thisplot[predicted_label].set_color('red')
  thisplot[true_label].set_color('blue')

# Plot the first X test images, their predicted label, and the true label
# Color correct predictions in blue, incorrect predictions in red
num_rows = 10
num_cols = 3
num_images = num_rows*num_cols
plt.figure(figsize=(2*2*num_cols, 2*num_rows))
for i in range(num_images):
  plt.subplot(num_rows, 2*num_cols, 2*i+1)
  plot_image(i, predictions, y_test, x_test) #prediction_test    #takes in probability array not class predictions
  plt.subplot(num_rows, 2*num_cols, 2*i+2)
  plot_value_array(i, predictions,  y_test)
plt.savefig(OUTFOLDER + "/examples_batch%d_layers%d_epochs%d_f1%d" % (BATCH_SIZE,LAYERS,EPOCHS,f1_mean_test*100))
plt.close()


mislabelled = [[] for _ in range(13)] #stored for every class, array of indexes where mislabelled
for i,p in enumerate(y_pred):
    true = y_dev[i]
    if p != true:
        mislabelled[true].append(i)

num_rows = 13
num_cols = 5
num_images = num_rows*num_cols
plt.figure(figsize=(2*2*num_cols, 2*num_rows))
for c in range(13):
    i = num_cols*2*c
    for j in range(min(num_cols,len(mislabelled[c]))):
        plt.subplot(num_rows, 2*num_cols, i+j+1)
        plot_image(mislabelled[c][j], predictions, y_dev, x_dev)
plt.savefig(OUTFOLDER + "/misslabelled_batch%d_layers%d_epochs%d_f1%d" % (BATCH_SIZE,LAYERS,EPOCHS,f1_mean_test*100))
plt.close()
