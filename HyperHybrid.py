import os
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.utils import shuffle
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import DenseNet121
from keras.applications.vgg16 import VGG16
from keras.models import Model
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, Flatten, concatenate
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold
from keras.wrappers.scikit_learn import KerasClassifier
from tensorflow.keras.optimizers import Adam
from keras.preprocessing import image
from sklearn.model_selection import train_test_split
from keras.utils import np_utils
from tensorflow.keras.callbacks import EarlyStopping
from keras.models import Sequential
import keras
from tensorflow.keras import layers
import glob
import matplotlib.pyplot as plt


def create_hybrid_model(unit):
	inputs_2 = keras.Input(shape=(224, 224, 3), name="img")
	# intialize VGG16
	vgg = VGG16(input_tensor=inputs_2, weights='imagenet', include_top=False) 
	# set trainable false for using pre-trained model
	for layer in vgg.layers:
		layer.trainable = False
	# initialize Densenet
	resnet = DenseNet121(input_tensor=inputs_2, weights='imagenet', include_top=False)
	for layer in resnet.layers:
		layer.trainable = False
	# concate two models
	mergedOutput = concatenate([vgg.output, resnet.output])
	mergedOutput = keras.layers.GlobalAveragePooling2D()(mergedOutput)

	x = layers.Dense(units=unit, activation='relu')(mergedOutput)
	x = layers.Dense(256, activation="relu")(mergedOutput)

	prediction = Dense(27, activation='softmax')(x)
	# intialize model 
	model = Model(inputs=vgg.input, outputs=prediction)
	# compile the hybrid model, this is the output layer
	model.compile(loss="categorical_crossentropy",optimizer='adam',metrics=['accuracy'])
	return model

# reading images from dataset
def get_files(directory):
  if not os.path.exists(directory):
    return 0
  count=0
  for current_path,dirs,files in os.walk(directory):
  # find the number of classes
    for dr in dirs:
      count+= len(glob.glob(os.path.join(current_path,dr+"/*")))
  return count
  

# augmentation, return class names
def DataGenerator():
	train ='train'
	validation_dir = 'test'
	train_samples=get_files(train)
	num_classes=len(glob.glob(train+"/*")) # get number of classes from current directory
	test_samples=get_files(validation_dir)

	print(num_classes,"Classes")
	print(train_samples,"Train images")
	print(test_samples,"Test images")

	train_datagen=ImageDataGenerator(rescale=1./255,
	                                	shear_range=0.2,
	                                    zoom_range=0.2,
	                                    horizontal_flip=True,
										vertical_flip=True)
	test_datagen=ImageDataGenerator(rescale=1./255)

	batch_size =64
	# augmentation for the images
	train_generator =train_datagen.flow_from_directory(train,target_size=(224, 224),batch_size=batch_size)
	test_generator=test_datagen.flow_from_directory(validation_dir,shuffle=True,target_size=(224, 224),batch_size=batch_size)

	class_dict =train_generator.class_indices
	print(class_dict)      
	
	li = list(class_dict.keys())
	print(li)
	return train_generator,test_generator,li


def process():
	train_generator,test_generator,class_name=DataGenerator()
	# get train / test data and store images as X_train / test and class name as Y_train / test
	X_train, y_train = next(train_generator)
	X_test, y_test = next(test_generator)
	
	# Hyperparameter Tuning
	unit=[5,6,10,11,12,15]
	batch_sizes = [16, 32, 64]

	param_grid = {'unit':unit,'batch_size': batch_sizes}
	kf = KFold(n_splits=5, shuffle=True)  # Define the number of folds you want to use

	hybrid_model = KerasClassifier(build_fn=create_hybrid_model, epochs=1, verbose=0)
	# give input as search space defines above
	grid_search = GridSearchCV(estimator=hybrid_model, param_grid=param_grid, cv=kf)
	grid_result = grid_search.fit(X_train, y_train)  # X & y_train are your training data

	best_params = grid_result.best_params_
	print(best_params)
	best_model = grid_result.best_estimator_
	print(best_model)
	

	para=grid_result.best_params_
	BestBatchSize=para["batch_size"]
	BestUnit=para["unit"]
	print(BestBatchSize)
	print(BestUnit)

	means = grid_result.cv_results_['mean_test_score']
	stds = grid_result.cv_results_['std_test_score']
	params = grid_result.cv_results_['params']
	print(means)
	print(stds)
	print(params)
	
	
	MeanValue=[]
	STDValue=[]
	ParamValue=[]

	for mean in means:
		MeanValue.append(mean)	
	for std in stds:
		STDValue.append(std)	
	for param in params:
		ParamValue.append("BS:"+str(param["batch_size"])+",Unit:"+str(param["unit"]))
	print(ParamValue)
	print(STDValue)
	print(MeanValue)
	
	acc = MeanValue
	alc = ParamValue
	colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#8c564b"]
	explode = (0.1, 0, 0, 0, 0)  

	fig = plt.figure()
	plt.bar(alc,acc,color=colors)
	plt.xlabel('Parameter')
	plt.ylabel('Value')
	plt.title('MeanValue')
	fig.savefig('results/MeanValue.png') 
	plt.pause(5)
	plt.show(block=False)
	plt.close()

	acc = STDValue
	alc = ParamValue
	colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#8c564b"]
	explode = (0.1, 0, 0, 0, 0)
	fig = plt.figure()
	plt.bar(alc,acc,color=colors)
	plt.xlabel('Parameter')
	plt.ylabel('Value')
	plt.title('STDValue')
	fig.savefig('results/STDValue.png') 
	plt.pause(5)
	plt.show(block=False)
	plt.close()

	model=create_hybrid_model(BestUnit)

	early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

	# define the hybrid model with best unit size & batch size
	history = model.fit(train_generator, validation_data=test_generator, epochs=40, shuffle=True, batch_size=BestBatchSize, callbacks=[early_stopping])

	# predict and return the class for test images
	predictions = model.predict(test_generator)
	predicted_labels = np.argmax(predictions, axis=1)

	model.save('results/hybrid_model.h5')

	acc = history.history['accuracy']
	val_acc = history.history['val_accuracy']
	loss = history.history['loss']
	val_loss = history.history['val_loss']
	epochs = range(1, len(loss) + 1)

	plt.plot(epochs, acc, color='green', label='Training Accuracy')
	plt.plot(epochs, val_acc, color='blue', label='Validation Accuracy')
	plt.title('Training and Validation Accuracy')
	plt.ylabel('Accuracy')
	plt.xlabel('Epoch')
	plt.legend()
	plt.savefig("results/acc.png") 
	plt.show()
	
	plt.figure()
	plt.plot(epochs, loss, color='pink', label='Training Loss')
	plt.plot(epochs, val_loss, color='red', label='Validation Loss')
	plt.title('Training and Validation Loss')
	plt.xlabel('Epoch')
	plt.ylabel('Loss')
	plt.legend()
	plt.savefig("results/loss.png") 
	plt.show()

	data = {
    'Epochs': epochs,
    'Training Accuracy': acc,
    'Validation Accuracy': val_acc,
    'Training Loss': loss,
    'Validation Loss': val_loss
	}
	df = pd.DataFrame(data)
	print(df)

process()