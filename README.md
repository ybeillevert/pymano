# Pymano
Welcome to my project for the Datascientist course from Datascientest.
The objective is to build a model that can recognize a word written by hand.<br/>
To train the model, we used the IAM Handwriting Database that contains images of handwritten words<br/>
The final product should contain:
- the code a the model
- the code of a streamlit application for a presentation
- a report of ~20 pages

This project was done in ~30 days and I was in a team of 3 people.

## My role in the team
To help the teamwork, I took the responsability of the project leader. My mission were:
- Setup a notion project to share documents, guide and planning
- Manage the planning on notion
- Organise team meetings 
- Writing guides to help my teammate to setup their environment (guides are in the [guides](/guides/) folder)
- Providing help to my teammate on technical or datascience related questions

On top on that, I was in charge of
- building a Convolutional Recurrent Neuronal Network (CRNN)
- contributing to the report 
- developing the code to test the model with a canvas and a picture taken from the camera

My teammates were in charge of building a transfer learning model from a VGG16, writing the report and build the streamlit web site.

## Reading suggestions
First, I suggest to to check the Powerpoint presentation here [presentation](/documents/Pymano-Prez.pptx). 

Then, you can check the code written in the CRNN_... files along with the [src/modules](/src/modules) folder

And I you really like the subject, you can check our report here [report](/documents/Pymano_Rapport.pptx).

## Src folder structure

### src/data folder
The [src/data](/src/data/) folder contains meta-information for each picture of a word

### src/images folder
After the CRNN model was trained, we want to play a little with it. So we each wrote a text by hand and took a picture of it. Then I cut each word into smaller images and give all this images to the model. The result was not so good as we had a CER of 62%

### src/models folder
It contains a backup of the CRNN model and its metric after each epoch

### src/modules folder
It contains some useful functions that were taken out from the notebook to help readability.

### src/streamlit
It contains the code of the streamlit app. It's main purpose here is to allow you to test the model. To do that, you just have to run the following commands

    pip install streamlit
    pip install streamlit-drawable-canvas==0.7.0
    cd .\src\streamlit
    streamlit run pymano.py
    
## Disclaimer
The code in the src folder are here for presentation purpose and are not meant to be run for several reasons :
- The IAM database contains more than 2Go of images and Github is limited to 2Go. So if you want to train the model, you'll have to download them manually
- The training of the CRNN take 6 hours for 20 epoc