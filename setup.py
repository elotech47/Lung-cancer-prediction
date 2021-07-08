import streamlit as st  
#from gradcam_02 import GradCAM
from gradCam import GradCAM
import cv2
from PIL import Image, ImageEnhance
import numpy as np 
import pandas as pd 
import os
import csv
import keras
from tensorflow.keras.models import load_model
from imutils import paths
import imutils
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import Normalize
# from keras.utils import CustomObjectScope
# from keras.initializers import glorot_uniform
from keras import backend as K
import matplotlib.pyplot as plt

with open("style.css") as f:
    st.markdown('<style>{}</style>'.format(f.read()), unsafe_allow_html=True)

def getList(dict): 
    return [*dict] 
@st.cache(allow_output_mutation=True)
def loadModel():
    # with CustomObjectScope({'GlorotUniform': glorot_uniform()}):
    
    model = load_model('./Models/lung_cancer_model.h5')
    print("Model Loaded Succesfully")
    #print(model.summary())
    return model

#print(loadModel().summary)

#@st.cache(allow_output_mutation=True)
def Diagnose(image):
    global label
    #model,session = model_upload()
    model= loadModel()
    image = cv2.cvtColor(np.float32(image), cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, ( 224, 224))
    #cv2.imshow("image", image)
    data = []
    data.append(image)
    # convert the data and labels to NumPy arrays while scaling the pixel
    # intensities to the range [0, 255]
    data = np.array(data) / 255.0
    #K.set_session(session)
    pred = model.predict(data)
    print(pred)
    return pred
   

    # st.write(alt.Chart(df).mark_bar().encode(
    # x=alt.X('Status', sort=None),
    # y='Percentage',))

def main():

    """An AI Diagnostic app for detecting Lung CANCER from X-ray Scan images"""
    #image
    from PIL import Image
    img = Image.open("logo2.png")
    st.image(img, width=300,caption="")
    st.title("LUNG CANCER PREDICTOR")
    st.info("An AI-Based lung cancer diagnostics using deep learning algorithm")
    st.text("Project built by Christopher Divine.")

    activities = ["upload", "Questions", "Welcome", "About"]
    choice = st.sidebar.selectbox("Select Activty",activities)

    if choice == "Welcome":
        st.subheader("Welcome AI Based DIagnostic Tool")
        img = Image.open("image.jpeg")
        st.image(img, width=600,caption="Deeplearning based Screening")
        st.write("This program is a computer application that uses Deep Neural Network to diagnose for Lung Cancer using X_ray images of patients")
        #st.write("One of the major issues we have combacting the corona virus pandemic is the slow testing process")
        st.write("Here we created this easy to use deep learning program with 97 percent testing accuracy.")
        st.write( "This program would support doctors when detecting this type of cancer")
        #st.write("This programs uses Deep Transfer learning and X-ray images to detect lung cancer from chest X-ray radiograph.")
        st.write("")
        st.write("")        
        st.subheader("User guide")
        st.write("Select Upload from the select activity dropdown from the left side of the page")
        st.write("Upload the X-ray image taken of the patient")
        st.write("Click on Diagnose")
        st.write("The prediction comes with percentage of certainty and a heatmap that shows the areas in the X-ray Affected")

    if choice == 'upload':
        st.subheader("Upload X-ray Image")
        image_file = st.file_uploader("Upload Image",type=['jpg','png','jpeg'])
        
        if image_file is not None:
            Sampleimage = Image.open(image_file).convert('RGB')
            st.text("X_ray Image")
            st.success("Successful loaded X-ray Image")
            is_check = st.checkbox("Display Image")
            if is_check:
                st.image(Sampleimage,width=300)

        name = st.text_input("Enter name of patient-diagnosis")
        if st.button("Diagnose"):
            pred = Diagnose(Sampleimage)
            i = np.argmax(pred[0])
            cancer = pred[0][0]
            normal = pred[0][1]
            data = (np.around([cancer, normal],decimals = 2))
            Cancer = data[0]
            normalR = data[1]
            st.info("Here is the Diagnosis")
            if Cancer >= normal:
                imageID, label, prob = [1, "Cancer", Cancer*100]
                st.write("Lung Cancer Suspected with {} certainty".format(prob))
                label = "{}: {:.2f}%".format(label, prob )
                st.write("[INFO] {}".format(label))
            else:
                imageID, label, prob = [1, "normal", normal*100]
                st.write("Normal Condition Suspected with {} certainty".format(prob))
                label = "{}: {:.2f}%".format(label, prob)
                st.write("[INFO] {}".format(label))
            my_dict = {"cancer":Cancer,"normal":normalR}
            df = pd.DataFrame(list(my_dict.items()),columns = ['Status','Percentage']) 
                # Get a color map
            my_cmap = cm.get_cmap('jet')
            
            # Get normalize function (takes data in range [vmin, vmax] -> [0, 1])
            my_norm = Normalize(vmin=0, vmax=100)
            fig, ax = plt.subplots()
            plt.bar("Status", "Percentage", data = df, color = my_cmap(my_norm(data)))
            plt.xlabel("Status")
            plt.ylabel("Percentage")
            plt.title("Percentage of Status")
            st.pyplot(fig)
            
            if name == None:
                st.error("Please fill in patient name and diagnosis seperated with {}".format("-"))
            else:
                name_save = "{}.jpeg".format(name)
                nameDetailed = "{}_detailed.jpeg".format(name)
                model = loadModel()
                cam = GradCAM(model, i)
                Reimage = cv2.cvtColor(np.float32(Sampleimage), cv2.COLOR_BGR2RGB)
                Reimage = cv2.resize(Reimage, ( 224, 224))
                ReimageGrad = np.array(Reimage)/255.0
                ReimageGrad = np.expand_dims(ReimageGrad, axis=0)
                imageArray = np.array(Sampleimage)
                orig = cv2.cvtColor(imageArray, cv2.COLOR_BGR2RGB)
                orig =cv2.resize(orig, ( 224, 224))
                cam_image = ReimageGrad#cv2.imread(imagepath)
                gradImageOutput = cam.compute_heatmap(cam_image)#, label, name)
                #st.write("[INFO]: Showing GradCam Heatmap") # and then overlay heatmap on top of the image
                st.success("Check below for Detailed Diagnosis analysis")
                heatmap = cv2.resize(gradImageOutput, (orig.shape[1], orig.shape[0]))
                (heatmap, output) = cam.overlay_heatmap(heatmap, orig, alpha=0.5)

                #draw the predicted label on the output image
                cv2.rectangle(output, (0, 0), (340, 40), (0, 0, 0), -1)
                cv2.putText(output, label, (10, 25), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (255, 255, 255), 2)
                #display the original image and resulting heatmap and output image
                #to our screen
                cv2.imwrite(name_save,output)
                outputCompare = np.hstack([orig, heatmap, output])
                outputCompare = imutils.resize(outputCompare, height=900, width = 900)
                #cv2.imshow("Output", outputCompare)
                cv2.imwrite(nameDetailed,outputCompare)
                #if st.button("Investigate"):
                explainedImage = Image.open(nameDetailed)
                st.image(explainedImage,width=900)
                #st.write("investigated")
                #cv2.waitKey(0)

    if choice == "About":
        st.subheader("About Lung Cancer DIAGNOSTICS")
        st.write("This programs uses Deep Transfer learning and CT scans images to detect Lung cancer from chest X-ray radiograph.")
        img2 = Image.open("image.jpeg")
        st.text("VGG16 Architecture")
        st.image(img2, width=500,caption="Photo Credit: https://www.researchgate.net/figure/Fig-A1The-standard-VGG-16-network-architecture-as-proposed-in-32-Note-that-only_fig3_322512435")
    
    if choice == "Questions":
        st.header("Please Answer the following questions for detailed investigations")
        fullname = "fullname"
        fullname = st.text_input("Enter Your Full Name", "Enter Here...")
        fullname = fullname.title()
        age = []
        age = st.text_input("Enter your Age", "Enter Age Here")
        age = age.title()
        phone = st.text_input("Enter your Phone Number", "Enter Here")
        phone = phone.title()
        Data = [fullname, age, phone]
        if st.button("Show Data"):
            st.text("FullName: {}".format(Data[0]))
            st.text("Age:{}".format(Data[1]))
            st.text("Phone:{}".format(Data[2]))
        if st.button("Save Data"):
            if os.path.isfile('PatientsDetails/PatientsDetails.csv'):
                with open('PatientsDetails/PatientsDetails.csv', 'a+', newline = "") as csvFile:
                        writer = csv.writer(csvFile)#, delimiter=',')
        #writer.writerow([i for i in heading])
                        writer.writerows([Data])#FFFFFF#FFFFFF
                        csvFile.close()
            else:
                with open('PatientsDetails/PatientsDetails.csv', 'a+', newline = "") as csvFile:
                        writer = csv.writer(csvFile)#, delimiter=',')
                        #writer.writerow([i for i in heading])
                        writer.writerows([Data])#FFFFFF#FFFFFF
                        csvFile.close()


            df = pd.read_csv("PatientsDetails/PatientsDetails.csv")


    st.write("")
    st.write("")
    st.write("")
    st.write("")
    st.write("")
    st.write("")
    st.write("")
    st.write("")
    st.write("")
    st.write("copyright: Christopher Divine ")

if __name__ == '__main__':
		main()	