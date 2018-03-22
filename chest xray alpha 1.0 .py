
"""Part 1 - Data Preprocessing"""

# Importing the libraries
import numpy as np
import pandas as pd
from sklearn.datasets import load_files
from glob import glob
import os
from keras.utils import np_utils
import matplotlib.pyplot as plt
from keras.preprocessing import image
from tqdm import tqdm


def path_to_tensor(img_path):
    img = image.load_img('./Image Dataset/Image total/' + str(img_path), target_size = (299, 299)) #TODO: finalize the path name
    x = image.img_to_array(img)
    return np.expand_dims(x, axis = 0)

def paths_to_tensor(img_paths):
    list_of_tensors = [path_to_tensor(img_path) for img_path in tqdm(img_paths)]
    return np.vstack(list_of_tensors)              



"""Part 2 - Training fine tunned models"""

# Importing the libraries
from keras.models import Model
from keras.applications.mobilenet import MobileNet
from keras.preprocessing import image
from keras.layers import Dense, GlobalAveragePooling2D, AveragePooling2D, BatchNormalization, Dropout, Flatten
from keras import backend as K
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from keras.initializers import glorot_normal
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras import applications # Better alternative for applications
from keras import optimizers
from keras.layers import Conv2D, MaxPooling2D


base_model = MobileNet(include_top = False, weights = None)
model = Sequential()
model.add(AveragePooling2D((4,4), input_shape = (299, 299, 3)))
model.add(BatchNormalization())
model.add(base_model)
model.add(GlobalAveragePooling2D())
model.add(Dropout(0.5))
model.add(Dense(1024))
model.add(Dense(14, activation = 'sigmoid'))

model.summary()
    
# Compiling the model
model.compile(optimizer = 'rmsprop', 
              loss = 'binary_crossentropy',
              metrics = ['accuracy'])

# Loading the top model's weight
model.load_weights('./saved models/weights.best.MobileNet_new_localization.hdf5')


"""Part 4 - Predicting new values"""


def chest_xray_model():
    base_model = MobileNet(include_top = False, weights = None)
    model = Sequential()
    model.add(AveragePooling2D((4,4), input_shape = (299, 299, 3))) #TODO: later change this to a number
    model.add(BatchNormalization())
    model.add(base_model)
    model.add(GlobalAveragePooling2D())
    model.add(Dropout(0.5))
    model.add(Dense(64))
    model.add(Dense(14, activation = 'sigmoid'))

    # Compiling the model
    model.compile(optimizer = 'rmsprop', 
                  loss = 'binary_crossentropy',
                  metrics = ['accuracy'])
    
    model.load_weights('./saved models/weights.best.MobileNet_new.hdf5')
    
    return model

# Write a function that takes a path to an image as input and returns the chest illness that is predicted by the model.
def chest_xray_predict(img_path):
    # Converting a image file path to a tensor
    img_width, img_height = 299, 299
    img = image.load_img(str(img_path), target_size = (img_width, img_height))
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis = 0)
    img = img.astype('float32')
    
    # Prediction
    prediction = chest_xray_model().predict(img)
    
    # Chest Ilness names
    illness_names = ['Cardiomegaly', 'Emphysema', 'Effusion', 'Hernia', 'Infiltration','Mass', 'Nodule', 
                     'Atelectasis', 'Pneumothorax', 'Pleural_Thickening','Pneumonia', 'Fibrosis', 'Edema', 
                     'Consolidation'] 

    # Making a dictionary of illness name and their prediction percentage
    prediction_dict = {}
    for index, i in enumerate(prediction[0]):
        prediction_dict[illness_names[index]] = i
    
    # Sorting the prediction from top to bottom
    import operator
    sorted_prediction = sorted(prediction_dict.items(), key=operator.itemgetter(1), reverse=True)
    
    return sorted_prediction

# Prediction - option 1
#prediction = chest_xray_predict('./Image Dataset/Image total/00006703_000.png')

#
def prediction_xray_chest(img_path, top_model_number):
    
    prediction = chest_xray_predict(img_path)
    
    for index, i in enumerate(prediction):
        print(index, i[0], "with: %", "%4.5f" % (i[1] *100), 'probability out of 14 illnesses')
        if index == (top_model_number - 1):
            break

# Prediction - option 2        
#prediction_xray_chest(img_path = './Image Dataset/Image total/00000978_000.png', top_model_number = 3)


# The best prediction function
def prediction_xray_chest(img_path):
    
    prediction = chest_xray_predict(img_path)
    
    dic = {}
    print("Top three prediction:")
    for index, i in enumerate(prediction):
        dic[index] = i
        #return index + 1, i[0]
        
        if index == (3 - 1):
            break
    return dic


def scan_chest_xray(path):
    # Getting the prediction
    result = prediction_xray_chest(path)
    top_3_results = []
    for index, i in enumerate(result.values()): 
        #top_3_results.append((str(index + 1), str(i[0])))
        top_3_results.append( str(i[0]))
    top_3_results = pd.DataFrame(top_3_results)
    top_3_results = top_3_results.rename(columns={0:''}, index = {0: '1', 1: '2', 2: '3'})
      
    # Left side of plot  
    fig = plt.figure(figsize = (8, 4))
    ax = fig.add_subplot(121)
    fig.subplots_adjust(top = 0.85)
    ax.set_title("Top 3 prediction for chest Xray scan:")
    ax.text(0.3, 0.6, top_3_results, horizontalalignment = 'left', verticalalignment = 'top', fontsize =12, bbox=dict(facecolor='gray', alpha=0.07))
    
    # Right side of plot
    ax2 = fig.add_subplot(122)
    img = image.load_img(path)
    ax2.imshow(img)
    
    # Eliminating the axis
    ax.set_frame_on(False)
    ax.get_xaxis().tick_bottom()
    ax.axes.get_yaxis().set_visible(False)
    ax.axes.get_xaxis().set_visible(False)
    
    plt.show()
    
# ==============================
# Choose your image
path = './Image Dataset/Image total/00000001_000.png'
path = './pneumonia.jpg'
# ==============================

# Predinction of the xray scan
scan_chest_xray(path)


"""Writing Localization function"""

# Importing the libraries
from keras.applications.mobilenet import preprocess_input
from keras.preprocessing import image
from keras.models import Model
import scipy
import cv2
import sys

# Preprocessing the images
def pretrained_path_to_tensor(img_path):
    # Loading the image
    img = image.load_img(img_path, target_size = (299, 299))
    
    # Converting the image to tensor
    x = image.img_to_array(img)
    
    # Converting 3D tensor to 4D tensor
    x = np.expand_dims(x, axis = 0)
    
    # Converting RGB to BGR / Subtract mean ImageNet pixel
    return preprocess_input(x)
    
# The model
def get_mobilenet():
    
    # Defining the fientuned MobileNet model 
    base_model = MobileNet(include_top = False, weights = None)
    
    model = Sequential()
    model.add(AveragePooling2D((4,4), input_shape = (299, 299, 3)))
    model.add(BatchNormalization())
    
    x = model.output
    x = base_model(x)
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.5)(x)
    x = Dense(1024)(x)
    prediction = Dense(14, activation = 'sigmoid')(x)
    
    model = Model(inputs = model.input, outputs = prediction)
   
    # Loading the weights
    model.load_weights('./saved models/weights.best.MobileNet_new_localization.hdf5')
    
    # Getting the AMP layer weight
    all_amp_layer_weights = model.layers[-1].get_weights()[0]
    
    # Extracting the wanted output
    mobilenet_model = Model(inputs = model.input, outputs = (model.layers[-5].get_output_at(-1), model.layers[-1].output))
    
    return mobilenet_model, all_amp_layer_weights

def get_model():
    
    base_model = MobileNet(include_top = False, weights = None).layers
    
    model = Sequential()
    model.add(AveragePooling2D((4,4), input_shape = (299, 299, 3)))
    model.add(BatchNormalization())
    model.layers.insert(2, MobileNet(include_top = False, weights = None).layers)
    model.add(GlobalAveragePooling2D())
    model.add(Dropout(0.5))
    model.add(Dense(1024))
    model.add(Dense(14, activation = 'sigmoid'))
    

# Class activation map (CAM)
def mobilenet_CAM(img_path, model, all_amp_layer_weights):
    # Getting filtered images from last convolutional layer + model prediction output
    last_conv_output, pred_vec = model.predict(pretrained_path_to_tensor(img_path)) # (1, 3, 3, 1024)
    
    # Converting the dimension of last convolutional layer to 3 x 3 x 1024     
    last_conv_output = np.squeeze(last_conv_output)
    
    # Model's prediction (between 1 and 999 which are inclusive)
    pred = np.argmax(pred_vec)
    
    # Bilinear upsampling (resize each image to size of original image)
    # TODO:
    mat_for_mult = scipy.ndimage.zoom(last_conv_output, (99 + 2/3, 99 + 2/3, 1), order = 1)  # dim: 299 x 299 x 1024  
    #mat_for_mult = scipy.ndimage.zoom(last_conv_output, (81 + 1/3, 81 + 1/3, 1), order = 1).shape  # dim: 299 x 299 x 1024  
    
    # Getting the AMP layer weights
    amp_layer_weights = all_amp_layer_weights[:, pred] # dim: (1024,)    
    
    # CAM for object class that is predicted to be in the image
    final_output = np.dot(mat_for_mult.reshape((299, 299, 1024)), amp_layer_weights).reshape(299, 299) # dim: 224 x 224

    # Return class activation map (CAM)
    return final_output, pred


def plot_CAM(img_path, ax, model, all_amp_layer_weights):
    # Loading the image / resizing to 224x224 / Converting BGR to RGB
    im = cv2.resize(cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB), (299, 299))
    
    # Plotting the image
    ax.imshow(255-im.squeeze(), cmap=plt.cm.gray, vmin=0, vmax=255)
    
    # Getting the class activation map
    CAM, pred = mobilenet_CAM(img_path, model, all_amp_layer_weights)
    
    CAM = (CAM - CAM.min()) / (CAM.max() - CAM.min())
    
    # Plotting the class activation map
    ax.imshow(CAM, cmap = plt.cm.jet, alpha = 0.5, interpolation='nearest', vmin=0, vmax=1)

"""
def plot_CAM(img_path, ax, model, all_amp_layer_weights):
    # Loading the image / resizing to 224x224 / Converting BGR to RGB
    im = cv2.resize(cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB), (299, 299))
    
    # Plotting the image
    ax.imshow(im, alpha = 0.5)
    
    # Getting the class activation map
    CAM, pred = mobilenet_CAM(img_path, model, all_amp_layer_weights)
    
    # Plotting the class activation map
    ax.imshow(CAM, cmap = 'nipy_spectral', alpha = 0.5)"""

   
if __name__ == '__main__':
    mobilenet_model, all_amp_layer_weights = get_mobilenet()
    fig, ax = plt.subplots()
    CAM = plot_CAM(path, ax, mobilenet_model, all_amp_layer_weights)
    plt.show()
      
for i in os.listdir('./Image Dataset/Image total/')[10:20]:
    fig, ax = plt.subplots()
    CAM = plot_CAM('./Image Dataset/Image total/' + str(i), ax, mobilenet_model, all_amp_layer_weights)
    plt.show()


