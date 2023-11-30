#import required packages 
import tensorflow
from keras.models import load_model
import numpy as np
from tensorflow.keras.utils import load_img
import os
from tensorflow.keras.utils import img_to_array
from googletrans import Translator
from flask import Flask, render_template, jsonify, request
from keras.preprocessing import image
import glob
from tensorflow.keras.preprocessing.image import ImageDataGenerator

app = Flask(__name__, template_folder='template')

        # load learned hybrid model from the directory
model = load_model('results/hybrid_model.h5')

#  function returns  index page
@app.route('/')
def index_view():
    return render_template('index.html')



#Allow files with extension png, jpg and jpeg
ALLOWED_EXT = set(['JPG' ,'jpg', 'jpeg' , 'png'])
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in ALLOWED_EXT
# read and get the images and class names from image folder           
def get_files(directory):
  if not os.path.exists(directory):
    return 0
  count=0
  for current_path,dirs,files in os.walk(directory):
    for dr in dirs:
      count+= len(glob.glob(os.path.join(current_path,dr+"/*")))
  return count

# augmentation as describe HyperHybrid.py file please refer  
def DataGenerator():
	train ='train'
	validation_dir = 'test'
	train_samples =get_files(train)
	num_classes=len(glob.glob(train+"/*"))
	test_samples=get_files(validation_dir)
	print(num_classes,"Classes")
	print(train_samples,"Train images")
	print(test_samples,"Test images")
	train_datagen=ImageDataGenerator(rescale=1./255,
	                                   shear_range=0.2,
	                                   zoom_range=0.2,
	                                   horizontal_flip=True)
	test_datagen=ImageDataGenerator(rescale=1./255)
	
	
	
	
	img_width,img_height =224,224
	input_shape=(img_width,img_height,3)
	batch_size =64
	train_generator =train_datagen.flow_from_directory(train,target_size=(img_width,img_height),batch_size=batch_size)
	test_generator=test_datagen.flow_from_directory(validation_dir,shuffle=True,target_size=(img_width,img_height),batch_size=batch_size)
	
	
	class_dict =train_generator.class_indices
	print(class_dict)      
	
	li = list(class_dict.keys())
	print(li)
	return train_generator,test_generator,li

# this is the function for prediction, it gets the class name from predict.py
# and according to class name it return the precautions required.
# each line represents the precautions for each plant type given in the dataset folder
@app.route('/predict',methods=['GET','POST'])
def predict():
	train_generator,test_generator,class_name=DataGenerator()
	rem=[{'r0':"The uploaded leaf is Diseased. Apple scab is a fungal disease that affects apple trees, causing leaf spots, premature leaf drop, and fruit scabs. Here are some remedies for apple scab:",'r1':"Proper sanitation - Remove fallen leaves and infected fruit and dispose of them properly.",'r2':"Fungicide sprays - Apply fungicide sprays starting from early spring, and follow the instructions on the label.",'r3':"Cultural controls - Choose apple varieties that are resistant to apple scab and maintain proper air circulation by pruning the trees.",'r4':"Avoid overhead irrigation - Water the trees at the base of the trunk to avoid moistening the leaves.",'r5':"Improve soil health - Provide proper fertilization and maintain healthy soil conditions to promote the growth of strong and resistant trees.",'r6':"It's important to follow a regular spray schedule and practice good cultural practices to prevent apple scab."},
        {'r0':"The uploaded leaf is Healthy"},{'r0':"The uploaded leaf is Diseased. Cedar apple rust is a member of the family Pucciniaceae, a group of fungi that contains many species that usually require two or more hosts to complete the life cycle. Some remedies are :",'r1':"Fungicide sprays, such as triadimefon or myclobutanil, can be utilized to safeguard against the disease and effectively manage its proliferation.",'r2':"Crop rotation: Planting apple trees in a different location every year can help reduce the amount of rust spores in the soil and reduce the risk of infection.",'r3':"Pruning and removing infected leaves: Regularly pruning and removing infected leaves from apple trees can reduce the spread of the disease and improve tree health.",'r4':"Maintaining a healthy tree: Keeping your apple trees healthy by providing adequate water, nutrients, and sunlight will improve their resistance to disease.",'r5':"Removing cedar trees: If possible, removing cedar trees in the vicinity of apple trees can greatly reduce the risk of infection.",'r6':"It's important to monitor your apple trees regularly and take prompt action to prevent and control Apple Cedar Apple Rust to protect their health and productivity."},
		{'r0':"The uploaded leaf is Healthy"},
		{'r0':"The uploaded leaf is Diseased", 'r1':"Spot disease"},
		{'r0':"The uploaded leaf is Healthy"},
		{'r0':"The uploaded leaf is Healthy"},
		{'r0':"The uploaded leaf is Diseased \nCercospora leaf spot and gray leaf spot are fungal diseases that affect corn plants. Here are some remedies for Cercospora leaf spot and gray leaf spot:",'r1':"Crop rotation: Planting corn in a different location every year can reduce the amount of fungal spores in the soil and reduce the risk of infection.",'r2':"Fungicide sprays: Applying fungicide sprays, such as chlorothalonil or azoxystrobin, can help prevent and control the spread of the disease.",'r3':"Sanitation: Regularly removing infected leaves and debris from the field can help reduce the spread of the disease.",'r4':"Resistant varieties: Planting corn varieties that are resistant to Cercospora leaf spot and Gray leaf spot can help reduce the risk of infection.",'r5':"Proper irrigation: Providing adequate water to the plants without allowing the leaves to stay wet for extended periods of time can help reduce the risk of infection.",'r6':"It's important to monitor your corn crops regularly and take prompt action to prevent and control Cercospora leaf spot and Gray leaf spot to protect their health and productivity."},
		{'r1':"The uploaded leaf is Diseased \nCrop rotation: Avoid planting corn in the same field for several years in a row. Instead, rotate with other crops that are not in the grass family.",'r2':"Resistant varieties: Planting corn varieties that have resistance to Northern Leaf Blight can reduce the impact of the disease.",'r3':"Fungicide application: Using fungicides specifically formulated to control Northern Leaf Blight can help prevent the spread of the disease. It's best to apply fungicides preventively, before the symptoms of the disease appear.",'r4':"Proper irrigation: Overhead irrigation can contribute to the spread of Northern Leaf Blight by providing an environment that is favorable for the fungus to grow. Drip irrigation or watering at the base of the plant can help reduce the spread of the disease.",'r5':"Sanitation: Remove and destroy infected plant debris from the previous season to reduce the number of spores and spores that can start new infections in the following season.",'r6':"It's important to implement a combination of these measures to effectively manage Northern Leaf Blight in corn crops."},
		{'r0':"The uploaded leaf is Diseased \nCommon rust is a fungal disease that affects corn plants. Here are some remedies for common rust:",'r1':"Crop rotation: Avoid planting corn in the same field year after year, as the fungal spores from the previous crop can survive in the soil and infect the new crop.",'r2':"Use resistant varieties: Plant corn varieties that have resistance to common rust.",'r3':"Fungicide application: Apply fungicides at the appropriate growth stage of the crop, as per the recommendation of a plant pathologist.",'r4':"Proper plant nutrition: Ensure that the plants receive adequate nutrients, especially potassium, which helps the plant to develop a thicker cuticle, making it less susceptible to fungal diseases.",'r5':"Sanitation: Remove and destroy infected leaves and debris to prevent the buildup and spread of the disease.",'r6':"Irrigation management: Proper irrigation management, such as avoiding overhead irrigation, can reduce the spread of fungal spores and increase the health of the plants.",'r7':"It's important to note that these remedies may not work in every situation, and a combination of these strategies may be necessary for effective disease management."},
		{'r0':"The uploaded leaf is Healthy"},
        {'r0':"The uploaded leaf is Diseased \nEarly blight is a common disease that affects potatoes and can cause significant damage to the leaves, stems, and tubers of the plant. To control and prevent early blight, there are several cultural practices that can be implemented:",'r1':"Sanitation: Maintaining a clean garden or field is essential for reducing the spread of early blight. Remove and dispose of all infected leaves and plant debris to prevent the spread of the disease.",'r2':"Crop rotation: Avoid planting potatoes in the same location year after year, as this can lead to a buildup of disease fragments in the soil.",'r3':"Proper irrigation: Avoid overhead watering, as this can spread the disease and increase the severity of infection. Instead, use drip irrigation or water the plants at the base of the plant.",'r4':"Fungicide sprays: Spraying fungicides at regular intervals during the growing season can help to control the spread of early blight. Be sure to follow the instructions on the fungicide label carefully.",'r5':"Resistant varieties: Consider planting potato varieties that are known to be resistant to early blight.",'r6':"In addition to these cultural practices, it's important to monitor the garden or field regularly and take action as soon as symptoms of early blight are noticed. Early detection and treatment can help to reduce the severity of the disease and minimize crop losses."},
		{'r0':"The uploaded leaf is Diseased \nLate blight is a serious and potentially devastating disease that affects potatoes and tomatoes. To control and prevent late blight, there are several cultural practices that can be implemented:",'r1':"Sanitation: Maintaining a clean garden or field is essential for reducing the spread of late blight. Remove and dispose of all infected leaves and plant debris to prevent the spread of the disease.",'r2':"Crop rotation: Avoid planting potatoes and tomatoes in the same location year after year, as this can lead to a buildup of disease fragments in the soil.",'r3':"Proper irrigation: Avoid overhead watering, as this can spread the disease and increase the severity of infection. Instead, use drip irrigation or water the plants at the base of the plant.",'r4':"Fungicide sprays: Spraying fungicides at regular intervals during the growing season can help to control the spread of late blight. Be sure to follow the instructions on the fungicide label carefully.",'r5':"Fungicide sprays: Spraying fungicides at regular intervals during the growing season can help to control the spread of late blight. Be sure to follow the instructions on the fungicide label carefully.",'r6':"In addition to these cultural practices, it's important to monitor the garden or field regularly and take action as soon as symptoms of late blight are noticed. Early detection and treatment can help to reduce the severity of the disease and minimize crop losses."},
		{'r0':"The uploaded leaf is Healthy"},
        {'r0':"The uploaded leaf is Healthy"},
        {'r0':"The uploaded leaf is Diseased \nPowdery mildew is a common fungal disease that affects a variety of plants, including squash. To control and prevent powdery mildew, there are several cultural practices that can be implemented:",'r1':"Sanitation: Maintaining a clean garden or field is essential for reducing the spread of powdery mildew. Remove and dispose of all infected leaves and plant debris to prevent the spread of the disease.",'r2':"Proper spacing: Allow enough space between plants to promote good air circulation and reduce the humidity levels in the garden or field.",'r3':"Proper irrigation: Avoid overhead watering, as this can spread the disease and increase the severity of infection. Instead, use drip irrigation or water the plants at the base of the plant.",'r4':"Fungicide sprays: Spraying fungicides at regular intervals during the growing season can help to control the spread of powdery mildew. Be sure to follow the instructions on the fungicide label carefully.",'r5':"Resistant varieties: Consider planting squash varieties that are known to be resistant to powdery mildew.",'r6':"In addition to these cultural practices, it's important to monitor the garden or field regularly and take action as soon as symptoms of powdery mildew are noticed. Early detection and treatment can help to reduce the severity of the disease and minimize crop losses."},
		{'r0':"The uploaded leaf is Healthy"},
        {'r0':"The uploaded leaf is Diseased \nEarly blight is a fungal disease that affects tomato plants, causing leaf yellowing and defoliation, and can also affect the fruit. Here are some remedies for early blight in tomatoes:",'r1':"Crop rotation: Planting tomatoes in a different location each year can help reduce the buildup of the fungus in the soil.",'r2':"Sanitation: Clean up any plant debris in and around the tomato plants to reduce the number of spores of the fungus present.",'r3':"Resistant varieties: Choose tomato varieties that are resistant to early blight.",'r4':"Proper watering: Watering the tomato plants at the base of the plant and avoiding overhead watering can help reduce the spread of the fungus.",'r5':"Fungicides: Fungicides containing chlorothalonil, mancozeb, or copper can be used as a preventative measure or to control early blight.",'r6':"Stake or cage the plants: Properly staking or caging the tomato plants can increase air flow and reduce humidity, which can help prevent the development of early blight.",'r7':"Remove infected leaves: Regularly removing any infected leaves and disposing of them far from the tomato plants can help reduce the spread of the fungus.",'r8':"It's important to follow good cultural practices, such as providing adequate spacing between plants, avoiding overcrowding, and providing adequate sunlight, water, and nutrients to the plants, as this can also help reduce the severity of early blight. Additionally, it is recommended to avoid overhead watering and to take measures to reduce the spread of the fungus, such as using clean tools and avoiding working with the plants when they are wet."},
		{'r0':"The uploaded leaf is Diseased \nSeptoria leaf spot is a fungal disease that affects tomato plants, causing small, circular, brown or black spots on the leaves. Here are some remedies for septoria leaf spot in tomatoes:",'r1':"Crop rotation: Planting tomatoes in a different location each year can help reduce the buildup of the fungus in the soil.",'r2':"Sanitation: Clean up any plant debris in and around the tomato plants to reduce the number of fragments of the fungus present.",'r3':"Resistant varieties: Choose tomato varieties that are resistant to septoria leaf spot.",'r4':"Proper watering: Watering the tomato plants at the base of the plant and avoiding overhead watering can help reduce the spread of the fungus.",'r5':"Fungicides: Fungicides containing chlorothalonil, mancozeb, or copper can be used as a preventative measure or to control septoria leaf spot.",'r6':"Stake or cage the plants: Properly staking or caging the tomato plants can increase air flow and reduce humidity, which can help prevent the development of septoria leaf spot.",'r7':"Remove infected leaves: Regularly removing any infected leaves and disposing of them far from the tomato plants can help reduce the spread of the fungus.",'r8':"It's important to follow good cultural practices, such as providing adequate spacing between plants, avoiding overcrowding, and providing adequate sunlight, water, and nutrients to the plants, as this can also help reduce the severity of septoria leaf spot. Additionally, it is recommended to avoid overhead watering and to take measures to reduce the spread of the fungus, such as using clean tools and avoiding working with the plants when they are wet."},
		{'r0':"The uploaded leaf is Healthy"},
		{'r0':"The uploaded leaf is Diseased \nTomato bacterial spot is a common problem in tomato plants that is caused by the bacterium Xanthomonas campestris. Here are some remedies for tomato bacterial spot:",'r1':"Crop rotation: Planting tomatoes in a different location each year can help reduce the buildup of bacteria in the soil.",'r2':"Sanitation: Clean up any plant debris in and around the tomato plants to reduce the number of bacteria present.",'r3':"Resistant varieties: Choose tomato varieties that are resistant to bacterial spot.",'r4':"Avoid overhead watering: Watering the tomato plants from below, such as through drip irrigation, can help reduce the spread of the bacteria.",'r5':"Copper sprays: Copper-based fungicides can be used as a preventative measure to help reduce the severity of bacterial spot.",'r6':"Remove infected leaves: Regularly removing any infected leaves and disposing of them far from the tomato plants can help reduce the spread of the bacteria.",'r7':"It's important to follow good cultural practices, such as providing adequate spacing between plants, avoiding overcrowding, and providing adequate sunlight, water, and nutrients to the plants, as this can also help reduce the severity of bacterial spot. Additionally, it is recommended to avoid overhead watering and to take measures to reduce the spread of bacteria, such as using clean tools and avoiding working with the plants when they are wet."},
		{'r0':"The uploaded leaf is Diseased \nLate blight is a fungal disease that affects tomato and potato plants, causing rapid wilting, yellowing, and blackening of the leaves and stems. Here are some remedies for late blight in tomatoes:",'r1':"Crop rotation: Planting tomatoes in a different location each year can help reduce the buildup of the fungus in the soil.",'r2':"Resistant varieties: Choose tomato varieties that are resistant to late blight.",'r3':"Proper watering: Watering the tomato plants at the base of the plant and avoiding overhead watering can help reduce the spread of the fungus.",'r4':"Fungicides: Fungicides containing mefenoxam or mancozeb can be used as a preventative measure or to control late blight.",'r5':"Stake or cage the plants: Properly staking or caging the tomato plants can increase air flow and reduce humidity, which can help prevent the development of late blight.",'r6':"Remove infected plants: Remove and destroy any infected plants to prevent the spread of the fungus.",'r7':"Sanitation: Clean up any plant debris in and around the tomato plants to reduce the number of fragments of the fungus present.",'r8':"It's important to follow good cultural practices, such as providing adequate spacing between plants, avoiding overcrowding, and providing adequate sunlight, water, and nutrients to the plants, as this can also help reduce the severity of late blight. Additionally, it is recommended to avoid overhead watering and to take measures to reduce the spread of the fungus, such as using clean tools and avoiding working with the plants when they are wet.",'r9':"Note: Late blight can be a serious disease and it is important to take preventive measures to protect your tomato plants from infection. If you live in an area with a high risk of late blight, it may be best to avoid growing susceptible tomato and potato varieties altogether."},		
		{'r0':"The uploaded leaf is Diseased \nTomato mosaic virus (ToMV) is a viral disease that affects tomato plants, causing mottling, yellowing, stunting, and reduced yields. There is no cure for ToMV, and infected plants should be removed and destroyed to prevent the spread of the virus. Here are some remedies for ToMV in tomatoes:",'r1':"Use resistant varieties: Some tomato varieties have been developed that are resistant to ToMV, so it's a good idea to choose these varieties if you are planting in an area with a history of ToMV.",'r2':"Prevent the spread of the virus: ToMV is transmitted by several types of insects, as well as through contaminated seed and tools, so it's important to take measures to reduce the spread of the virus, such as using insecticides to control insect populations, planting only disease-free seed, and disinfecting tools between uses.",'r3':"Avoid spreading the virus: If you are working with infected plants, be sure to disinfect your tools and wash your hands and clothing before working with healthy plants to avoid spreading the virus.",'r4':"Remove infected plants: As soon as you suspect that a plant may be infected with ToMV, remove it from your garden and destroy it.",'r5':"Practice good cultural practices: Providing adequate sunlight, water, and nutrients, and avoiding overcrowding, can help keep your tomato plants healthy and less susceptible to ToMV and other diseases.",'r6':"It's important to monitor your tomato plants regularly for signs of ToMV, such as mottling, yellowing, stunting, and reduced yields, and to take action as soon as you suspect that a plant may be infected. The earlier you detect and remove infected plants, the less chance there is for the virus to spread to healthy plants."},
		{'r0':"The uploaded leaf is Diseased \nTomato yellow leaf curl virus (TYLCV) is a serious viral disease that affects tomato plants, causing yellowing, curling, and stunting of the leaves, as well as reduced yields and quality of the fruit. Unfortunately, there is no cure for TYLCV, and infected plants should be removed and destroyed to prevent the spread of the virus. Here are some remedies for TYLCV in tomatoes:",'r1':"Use resistant varieties: Some tomato varieties have been developed that are resistant to TYLCV, so it's a good idea to choose these varieties if you are planting in an area with a history of TYLCV.",'r2':"Prevent the spread of the virus: The whitefly, a tiny insect, is the primary vector of TYLCV, so it's important to take measures to reduce whitefly populations, such as using yellow sticky traps, removing weeds and volunteer tomatoes that can serve as alternate hosts, and using insecticidal soap or other insecticides to control whiteflies.",'r3':"Avoid spreading the virus: If you are working with infected plants, be sure to disinfect your tools and wash your hands and clothing before working with healthy plants to avoid spreading the virus.",'r4':"Remove infected plants: As soon as you suspect that a plant may be infected with TYLCV, remove it from your garden and destroy it.",'r5':"Practice good cultural practices: Providing adequate sunlight, water, and nutrients, and avoiding overcrowding, can help keep your tomato plants healthy and less susceptible to TYLCV and other diseases.",'r6':"It's important to monitor your tomato plants regularly for signs of TYLCV, such as yellowing, curling, and stunting of the leaves, and to take action as soon as you suspect that a plant may be infected. The earlier you detect and remove infected plants, the less chance there is for the virus to spread to healthy plants."},
		{'r0':"The uploaded leaf is Diseased \nTomato leaf mold is a fungal disease that causes yellow spots on the leaves, which can eventually turn brown and become covered in a fuzzy, gray mold. Here are some remedies for tomato leaf mold:",'r1':"Crop rotation: Planting tomatoes in a different location each year can help reduce the buildup of the fungus in the soil.",'r2':"Proper spacing: Providing adequate spacing between plants can increase air flow and reduce humidity, which can help prevent the development of leaf mold.",'r3':"Resistant varieties: Choose tomato varieties that are resistant to leaf mold.",'r4':"Proper watering: Watering the tomato plants at the base of the plant and avoiding overhead watering can help reduce the spread of the fungus.",'r5':"Fungicides: Fungicides containing chlorothalonil or mancozeb can be used as a preventative measure or to control leaf mold.",'r6':"Stake or cage the plants: Properly staking or caging the tomato plants can increase air flow and reduce humidity, which can help prevent the development of leaf mold.",'r7':"Remove infected leaves: Regularly removing any infected leaves and disposing of them far from the tomato plants can help reduce the spread of the fungus.",'r8':"It's important to follow good cultural practices, such as providing adequate sunlight, water, and nutrients to the plants, as this can also help reduce the severity of leaf mold. Additionally, it is recommended to avoid overhead watering and to take measures to reduce the spread of the fungus, such as using clean tools and avoiding working with the plants when they are wet."},
		{'r0':"The uploaded leaf is Healthy"},
		{'r0':"The uploaded leaf is Diseased \nBlack rot is a common disease that can cause significant damage to the fruit and vine. To control and prevent black rot, there are several cultural practices that can be implemented:",'r1':"Sanitation: Maintaining a clean vineyard is essential for reducing the spread of black rot. Remove and dispose of all infected fruit and leaves to prevent the spread of the disease.",'r2':"Crop rotation: Avoid planting grapes in the same location year after year, as this can lead to a buildup of disease spores in the soil.",'r3':"Pruning: Proper pruning can help to promote good air circulation in the vineyard, which can reduce the spread of black rot.",'r4':"Fungicide sprays: Spraying fungicides at regular intervals during the growing season can help to control the spread of black rot. Be sure to follow the instructions on the fungicide label carefully.",'r5':"Resistant varieties: Consider planting grape varieties that are known to be resistant to black rot.",'r6':"In addition to these cultural practices, it's important to monitor the vineyard regularly and take action as soon as symptoms of black rot are noticed. Early detection and treatment can help to reduce the severity of the disease and minimize crop losses."},
		
		]
	
	#the following code is for choosing the image from folder and read the image
	if request.method == 'POST':
		file = request.files['file']
		print(allowed_file(file.filename))
		if file and allowed_file(file.filename):
			filename = file.filename
			file_path = os.path.join('static/testimages', filename)
			file.save(file_path)
			image_path = file_path
			# convert the image to required dimension and some preprocessing steps also done
			new_img = image.load_img(image_path, target_size=(224, 224))
			# convert resize image to array
			img = image.img_to_array(new_img)
			# convert as numpy array
			img = np.expand_dims(img, axis=0)
			# normalize the image feature dividing by 255
			img = img/255
			print("Following is our prediction:")
			# call the model and predict function img (pre-processed) is image input given by user
			prediction = model.predict(img)
			# get the class values number
			d = prediction.flatten()
			print(d)
			j = d.max()
			print(j)
			# following get the class name and get the remedy 
			for index,item in enumerate(d):
				if item == j:
					print(index)
					pred_class = class_name[index]
					remedy=rem[index]
			return render_template('predict.html', pred_class=pred_class,prob=90, remedy=remedy,user_image = file_path)
		else:
			return "Unable to read the file. Please check file extension"
	return render_template('predict.html')




if __name__ == '__main__':
    app.run(debug=True,use_reloader=False, port=5000)
    
    