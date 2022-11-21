#____IMPORT LIBRAIRIES________________________________________________
import streamlit as st
import base64
import re
import os
import numpy as np
import time, cv2
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

import tensorflow as tf
from keras.applications.vgg16 import VGG16
from keras.applications.vgg19 import VGG19
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dense, Flatten
from keras.layers import Dense, Dropout, Flatten, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from keras.models import Model, Sequential
from tensorflow.keras.layers import Input
from keras import backend as K
from tensorflow.keras.layers import Input, Dense
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import img_to_array
from keras.applications.vgg16 import preprocess_input
from keras import layers
from keras import models
from PIL import Image




from streamlit_option_menu import option_menu
import graphviz as graphviz
#_____________________________________________________________________


#____IMPORT DATAFRAME________________________________________________
df = pd.read_csv("words.csv", index_col=0, sep=";")
df = df[df["word_seg"] == "ok"]
df = df.drop(columns=["word_seg","graylevel","bounding_box_x","bounding_box_y","bounding_box_w","bounding_box_h","gram_tag"])
df = df[~df["path"].isin(['./images/words/a01/a01-117/a01-117-05-02.png','./images/words/r06/r06-022/r06-022-03-05.png'])] # images in error
df=df.head(500)
#_____________________________________________________________________





#SIDEBAR______________________________________________________________

with st.sidebar:
    menu = option_menu("Menu Principal", ["Accueil", "Présentation des données", "CRNN", "Transfer Learning", "À vous de jouer!"],icons=['house-fill', "bar-chart-line", 'diagram-3', 'diagram-3-fill','camera-fill'],menu_icon="app-indicator", default_index=0,
                         styles={
        "container": {"padding": "5!important", "background-color": "black"},
        "icon": {"color": "pink", "font-size": "25px"}, 
        "nav-link": {"font-size": "16px", "text-align": "left","margin":"0px", "--hover-color": "#eee"},
        "nav-link-selected": {"background-color": "pink","color": "white"}
    }
    )

#_____________________________________________________________________







#______________________________________________________________________
#ACCUEIL_______________________________________________________________
if menu == 'Accueil':
    #TITRES________________x___x___x________________
    title = """  <p style=font-family:Impact;color:White;font-size:45px;" > Création d'un modèle d'oscérisation</p>"""

    st.markdown(title, unsafe_allow_html=True)

    #________

    subtitle = """  <p style=font-family:Noteworthy;color:Pink;font-size:20px;" > Utilisation du Deep Learning pour créer un modèle de reconnaissance de texte.</p>"""

    st.markdown(subtitle, unsafe_allow_html=True)

    #IMAGE________________x___x___x________________ 
    img = Image.open("Re.jpg") 
  
    st.image(img, width=600) 

    #TEXTE________________x___x___x________________


    st.write("La reconnaissance optique de caractères, ou Optical Character Recognition (OCR), comprends des procédés d’extraction de texte à partir d’images de texte.")
    st.write("Cette méthode permet de récupérer du texte à partir d’une image et de le sauvegarder dans un fichier exploitable. Ces techniques sont connues notamment pour la lecture et le traitement des chèques en banque, mais aussi pour la numérisation des archives tels que les actes de naissances, de mariage etc...")
    st.write("Nous pourrions aussi imaginer une application d’assistance aux personnes malvoyantes, saisie automatique de données...etc les applications sont multiples.")
#_____________________________________________________________________










#______________________________________________________________________
#_____PRESENTATION DES DONNÉES_________________________________________

elif menu=="Présentation des données":
    title = """  <p style=font-family:Noteworthy;color:Pink;font-size:40px;" > Présentation des données</p>"""
    st.markdown(title, unsafe_allow_html=True)

    st.markdown(
        "Le jeu de données de l’IAM est découpé en 4 sous ensembles:  \n" 
        "- forms  \n" 
        "- lines  \n" 
        "- sentences  \n" 
        "- et words.")

    st.markdown("Notre objectif étant de créer un modèle de reconnaissance de lettres, nous avons fait le choix de n’utiliser que le sous-ensemble **words**.") 

    st.write("Ce sous-ensemble est constitué des images correspondants aux mots ainsi que du fichier words.txt contenant des informations sur chaque images")
    
    df = df.head()
    st.table(df)
    image = Image.open("wordsdata.png")
    st.image(image, caption="Exemple d'image de la base de données")
    
    
 






    
elif menu=="CRNN":
    title = """  <p style=font-family:Noteworthy;color:Pink;font-size:40px;" > Convolutionnal Recurrent Neural Network</p>"""
    st.markdown(title, unsafe_allow_html=True)
    st.write("Le CRNN est la combinaison d'un réseau neuronal convolutif, d'un réseau neuronal récurrent, ainsi que la fonction de perte d'une classification temporelle.")
    st.write("Ce mix de modèle permet d'obtenir un seul modèle permettant d'extraire les informations pertinentes (CNN), de les classifier au mieux (RNN) et de résoudre la problématique d'alignement des caractères.")
    st.write("Voici comment fonctionne un modèle CRNN:")
    st.graphviz_chart('''
    digraph {
        run -> intr
        intr -> runbl
        runbl -> run
        run -> kernel
        kernel -> zombie
        kernel -> sleep
        kernel -> runmem
        sleep -> swap
        swap -> runswap
        runswap -> new
        runswap -> runmem
        new -> runmem
        sleep -> runmem
    }
''')
    
    
    
    
    
    
    
elif menu =="Transfer Learning":
    title = """  <p style=font-family:Noteworthy;color:Pink;font-size:40px;" > Transfer Learning</p>"""
    st.markdown(title, unsafe_allow_html=True)


elif menu=="À vous de jouer!":
    picture = st.camera_input("Take a picture")

    if picture:
        st.image(picture)
    
    
    from pym_dataframe import get_clean_dataframe
    from pym_image_preprocessing import preprocess
    from pym_models import ctc_loss, build_cnn_rnn
    from pym_encoding import encode_labels, greedy_decoder
    from pym_drawing_reader import PymDrawingReader