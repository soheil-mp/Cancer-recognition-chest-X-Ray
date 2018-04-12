
# Importing the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import os
import scipy
import cv2
import sys
import time

# Importing the keras libraries and packages
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
from keras.applications.mobilenet import preprocess_input
from keras.preprocessing import image
from keras.models import Model


def get_model_architect():
    
    # Defining the architect of CNN
    base_model = MobileNet(include_top = False, weights = None, input_shape = (299, 299, 3))
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    final_layer = Dense(14, activation = 'sigmoid')(x)
    
    # Connecting the two model
    model = Model(base_model.input, final_layer)
    
    # Summary of model
    #model.summary()
    
    # Loading the weights
    model.load_weights('./weights.best.MobileNet_new_2.hdf5')

    # Compiling the model
    model.compile(optimizer = 'rmsprop', 
                  loss = 'binary_crossentropy',
                  metrics = ['accuracy'])
    
    return model


def get_prediction(img_path):
    
    # Converting a image file path to a tensor
    img_width, img_height = 299, 299
    img = image.load_img(str(img_path), target_size = (img_width, img_height))
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis = 0)
    img = img.astype('float32')
    
    # Prediction
    prediction = get_model_architect().predict(img)
    
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



# Predicting only the Top-3 
def prediction_xray_chest(img_path):
    
    prediction = get_prediction(img_path)
    
    dic = {}
    for index, i in enumerate(prediction):
        dic[index] = i
        #return index + 1, i[0]
        
        if index == (3 - 1):
            break
    return dic


 
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

    
# Model
def get_model():
    
    # Defining the architect of CNN
    base_model = MobileNet(include_top = False, weights = None, input_shape = (299, 299, 3))
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    final_layer = Dense(14, activation = 'sigmoid')(x)
    
    # Connecting the two model
    model = Model(base_model.input, final_layer)
    
    # Summary of model
    #model.summary()
    
    # Loading the weights
    model.load_weights('./weights.best.MobileNet_new_2.hdf5')
    
    # Getting the AMP layer weight
    all_amp_layer_weights = model.layers[-1].get_weights()[0]
    
    # Extracting the wanted output
    mobilenet_model = Model(inputs = model.input, outputs = (model.layers[-3].output, model.layers[-1].output))
    
    return mobilenet_model, all_amp_layer_weights

    

def mobilenet_CAM(img_path, model, all_amp_layer_weights):
    # Getting filtered images from last convolutional layer + model prediction output
    last_conv_output, pred_vec = model.predict(pretrained_path_to_tensor(img_path)) # (1, 3, 3, 1024)
    
    # Converting the dimension of last convolutional layer to 3 x 3 x 1024     
    last_conv_output = np.squeeze(last_conv_output)
    
    # Model's prediction (between 1 and 999 which are inclusive)
    pred = np.argmax(pred_vec)
    
    # Bilinear upsampling (resize each image to size of original image)
    mat_for_mult = scipy.ndimage.zoom(last_conv_output, (29.9, 29.9, 1), order = 1)  # dim: 299 x 299 x 1024  
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
    ax.imshow(1 - CAM, cmap = plt.cm.jet, alpha = 0.5, interpolation='nearest', vmin=0, vmax=1)

    
    
def scan_chest_xray(path):
    
    # Getting the prediction
    print("============================================")
    print("Prediction process has been started... (1/2)")
    start = time.time()
    result = prediction_xray_chest(path)
    top_3_results = []
    for index, i in enumerate(result.values()): 
        #top_3_results.append((str(index + 1), str(i[0])))
        top_3_results.append(str(i[0]))
    top_3_results = pd.DataFrame(top_3_results)
    top_3_results = top_3_results.rename(columns={0:''}, index = {0: '1', 1: '2', 2: '3'})
    end = time.time()
    print("Prediction has been complete in {:.3} seconds!".format(end - start))
      
    # Left side of plot  
    print("Localizing in the x-ray image... (2/2)")
    start = time.time()
    fig = plt.figure(figsize = (8, 4))
    ax = fig.add_subplot(121)
    fig.subplots_adjust(top = 0.85)
    ax.set_title("Top 3 prediction for chest Xray scan:")
    ax.text(0.3, 0.6, top_3_results, horizontalalignment = 'left', verticalalignment = 'top', fontsize =12, bbox=dict(facecolor='gray', alpha=0.07))
    
    # Right side of plot
    ax2 = fig.add_subplot(122)
    mobilenet_model, all_amp_layer_weights = get_model()
    im = cv2.resize(cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB), (299, 299))
    CAM, pred = mobilenet_CAM(path, mobilenet_model, all_amp_layer_weights)
    CAM = (CAM - CAM.min()) / (CAM.max() - CAM.min())
    ax2.imshow(255-im.squeeze(), cmap=plt.cm.gray, vmin=0, vmax=255)
    ax2.imshow(1 - CAM, cmap = plt.cm.jet, alpha = 0.5, interpolation='nearest', vmin=0, vmax=1)
    end = time.time()
    print("Plotting has been complete in {:.3} seconds!".format(end - start))
    print("============================================")

    # Eliminating the axis
    ax.set_frame_on(False)
    ax.get_xaxis().tick_bottom()
    ax.axes.get_yaxis().set_visible(False)
    ax.axes.get_xaxis().set_visible(False)
    
    plt.show()


    #path = './Pneumothorax.jpeg'
    # path = './Image Dataset/Image total/00000003_001.png'
    
if __name__ == '__main__':
    # Choose your image
    path = input("Please insert the x-ray image path ->")
    path = str(path)
    
    # Scan the x-ray imaged
    scan_chest_xray(path)
    
    
    