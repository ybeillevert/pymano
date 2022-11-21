#____IMPORT LIBRAIRIES________________________________________________
import streamlit as st
import numpy as np
import cv2
import matplotlib.pyplot as plt

import tensorflow as tf
from PIL import Image
from streamlit_drawable_canvas import st_canvas

from modules.pym_dataframe import get_clean_dataframe
from modules.pym_encoding import greedy_decoder

from streamlit_option_menu import option_menu
import graphviz as graphviz
#_____________________________________________________________________






#____IMPORT DATAFRAME________________________________________________
df=get_clean_dataframe()

df=df.head(2000)

#_____________________________________________________________________





#SIDEBAR______________________________________________________________

with st.sidebar:
    menu = option_menu("Menu", ["Accueil", "Pymano", "Présentation des données", "Transfer Learning", "CRNN", "Canvas", "À vous de jouer!"],icons=["house-fill", "blockquote-left", "bar-chart-line", "diagram-3", 'diagram-3-fill', 'pencil','camera-fill'],menu_icon="app-indicator", default_index=0,
                         styles={
        "container": {"padding": "5!important", "background-color": "white"},
        "icon": {"color": "palevioletred", "font-size": "25px"}, 
        "nav-link": {"font-size": "16px", "text-align": "left","margin":"0px", "--hover-color": "#eee"},
        "nav-link-selected": {"background-color": "palevioletred","color": "white"}
    }
    )

#_____________________________________________________________________







#______________________________________________________________________
#ACCUEIL_______________________________________________________________
if menu == 'Accueil':
    #TITRES________________x___x___x________________
    title = """  <p style=font-family:Impact;color:slategrey;font-size:60px;" > Création d'un modèle d'oscérisation</p>"""

    st.markdown(title, unsafe_allow_html=True)

    #________

    subtitle = """  <p style=font-family:Noteworthy;color:teal;font-size:20px;" > Utilisation du Deep Learning pour créer un modèle de reconnaissance de texte.</p>"""

    st.markdown(subtitle, unsafe_allow_html=True)

    #IMAGE________________x___x___x________________ 
    img = Image.open("Re.jpg") 
  
    st.image(img, width=600) 

    #TEXTE________________x___x___x________________


    st.write("La reconnaissance optique de caractères, ou Optical Character Recognition (OCR), est un procédé d’extraction de texte à partir d’images de texte.")
    st.write("Cette méthode permet de récupérer du texte à partir d’une image et de le sauvegarder dans un fichier exploitable. Ces techniques sont connues notamment pour la lecture et le traitement des chèques en banque, ou encore pour la numérisation des archives tels que les actes de naissances ou de mariages, utiles en cas de recherche généalogique. ")
    st.write("Nous pourrions aussi imaginer une application d’assistance aux personnes malvoyantes, saisie automatique de données...")
    st.write("Les applications sont multiples.")
#_____________________________________________________________________


#______________________________________________________________________
#PYMANO_______________________________________________________________

elif menu == 'Pymano':
    #TITRES________________x___x___x________________
    title = """  <p style=font-family:Noteworthy;color:palevioletred;font-size:40px;" > Le projet Pymano</p>"""
    st.markdown(title, unsafe_allow_html=True)
    subtitle = """  <p style=font-family:Noteworthy;color:teal;font-size:20px;" > Contexte</p>"""
    st.markdown(subtitle, unsafe_allow_html=True)
    st.write("Ce projet s'inscrit dans la formation Data Scientist proposée par l'organisme **DataScientest**. ")
    st.write("Le projet **Pymano** est né d'une volonté de faciliter le traitement et le stockage de documents")
    st.write("Le domaine de l'oscérisation n'est pas récent, car c'est Gustav Tauschek qui en **1929**, a créé la première machine d'oscérisation. Bien entendu, c'est un domaine qui a suscité beaucoup d'intérêt et qui a su évoluer depuis.")
    st.image("ocr.jpg")
    st.write("Notre choix s'est porté sur ce projet car il revêt un certain challenge.  \n"
             "En effet, c'est un problème complexe, utilisant le **Deep Learning** pour de la reconnaissance de caractère. Nous avons eu un peu plus de deux mois pour finir ce projet, mais nous aurions pu y passer bien plus de temps, à paufiner, à chercher de meilleures performances, d'autres méthodes, affiner les modèles... ") 
    subtitle = """  <p style=font-family:Noteworthy;color:teal;font-size:20px;" > Objectif du projet</p>"""
    st.markdown(subtitle, unsafe_allow_html=True)
    
    st.write("L’objectif de ce projet est donc d’utiliser un algorithme de Deep Learning pour **reconnaître** les caractères manuscrits issues du dataset IAM. Le but étant de pouvoir, une fois le modèle entrainé, faire reconnaitre un mot quelconque écrit par un écrivain lambda.")








#______________________________________________________________________
#_____PRESENTATION DES DONNÉES_________________________________________

elif menu=="Présentation des données":
    title = """  <p style=font-family:Noteworthy;color:palevioletred;font-size:40px;" > Présentation des données</p>"""
    st.markdown(title, unsafe_allow_html=True)

    st.markdown(
        "Le jeu de données de l’IAM est découpé en 4 sous ensembles:  \n" 
        "- forms  \n" 
        "- lines  \n" 
        "- sentences  \n" 
        "- et words.")

    st.markdown("Notre objectif étant de créer un modèle de reconnaissance de texte, et nous avons fait le choix de n’utiliser que le sous-ensemble **words**.") 
    st.image("image9.png", caption="Image de la base de données brute")
    st.write("Ce sous-ensemble est constitué de plus de 96 000 images de mots ainsi que du fichier words.txt contenant des informations sur chaque image.")
    
    df1 = df.head()
    st.table(df1)
    image = Image.open("wordsdata.png")
    st.image(image, caption="Exemple d'image de la base de données")
    st.write(" ")
    st.write(" ")
    
    m = st.markdown("""
<style>
div.stButton > button:first-child {
    background-color: #db7093;
    color:#FFFFFF;
}
div.stButton > button:hover {
    background-color: #f4c1d1;
    color:#FFFFFF;
    }
</style>""", unsafe_allow_html=True)

    
    if st.button('Images'):
        plt.figure(figsize=(12,5))
        j=1
        for i in np.random.randint(0, len(df), size=[5]):
            plt.subplot(2,4,j)
            st.image(df.path[i], caption = df["transcription_word"][i])            
    
    else:
        st.write('*Cliquez pour afficher aléatoirement des photos du dataset*')
    
    
    st.write(" ")
    st.write(" ")
    
    st.write("Une analyse des données a été bien entendu effectuée afin d'obtenir le dataset le plus **propre** possible")
    st.write("Nous ne détaillerons pas toutes les actions menées ici, seulement d'un exemple de nettoyage du dataset.")
    st.write("En effet, nous avons étudié l'état de **segmentation** des mots, et il s'est avéré qu'il y avait beaucoup d'erreur de semgentation")
    st.image("segmentation.png", caption="Analyse de la segmentation des données")
    st.write("Nous n'avons donc pris que les mots dont la segmentation était **'ok'**.")
    
    
    
    
elif menu =="Transfer Learning":
    title = """  <p style=font-family:Noteworthy;color:palevioletred;font-size:40px;" > Transfer Learning</p>"""
    st.markdown(title, unsafe_allow_html=True)
    st.write("Nous avons essayé une approche de l'oscérisaiton par le **Transfer learning**.")
    st.write("Le principe du Transfer learning est la capacité à utiliser des connaissances **existantes**, développées pour la résolution de problématiques données, pour résoudre une nouvelle problématique.")
    st.image("tl.png")
    st.write("Pour cela, nous avons utilser le modèle de Karen Simonyan et Andrew Zisserman: le $VGG16$.") 
    
    st.write("La structure VGG16 se découpe en plusieurs couches de convolution, alternées par des couches de Max pooling.")
    
    col1, col2, col3 = st.columns(3)

    with col1:
        st.write('')

    with col2:
        st.image("Architecture.png",caption="Schéma de l'architecture du VGG16.",width=500)

    with col3:
        st.write(' ')  
        
    st.write("Après divers essais pour tenter d'améliorer au mieux le modèle nous avons retenu le modèle suivant:")
   
    col1, col2, col3 = st.columns(3)

    with col1:
        st.write('')

    with col2:
        st.graphviz_chart('''
    digraph {
        VGG16 -> "Couche Flatten"
        "Couche Flatten" -> "Couche Dense à 500 neurones"
        "Couche Dense à 500 neurones" -> "Couche Dropout à 0.5"
        "Couche Dropout à 0.5" -> "Couche Dense à 250 neurones"
        "Couche Dense à 250 neurones" -> "Couche de Prediction Softmax"
    }
''')

    with col3:
        st.write(' ')  
    
    subtitle = """  <p style=font-family:Noteworthy;color:palevioletred;font-size:20px;" > Résultats et discussion.</p>"""

    st.markdown(subtitle, unsafe_allow_html=True)
    st.write("Nous avons entraîné le modèle sur le plus de données posssible par notre ordinateur, et avons atteint un maximum d'entrainement sur 20 000 données et une epoch.")
    st.write("Les résultats n'ayant pas été concluants, nous avons entraîné notre modèle sur 2 000 données et 20 epochs")
    st.write("Voici les courbes des différentes métriques: la loss et l'accuracy.")
    
    
    m = st.markdown("""
<style>
div.stButton > button:first-child {
    background-color: #db7093;
    color:#FFFFFF;
}
div.stButton > button:hover {
    background-color: #f4c1d1;
    color:#FFFFFF;
    }
</style>""", unsafe_allow_html=True)

    
    if st.button('Résultats'):
              
        col1, col2, col3 = st.columns(3)

        with col1:
            st.image('accuracy.png', caption = "Accuracy du modèle entrainé sur 2 000 données et 20 epochs", width=500)

        with col2:
            st.write(" ")

        with col3:
            st.image("loss.png",caption="Loss du modèle entrainé sur 2 000 données et 20 epochs ",width=500)
        st.write("Nous obtenons une loss de 2.24 et une accuracy de 0.21 sur le jeu de test")
   
    

    
        st.write("Nous avons aussi calculé la CER qui est littéralement la Character Error Rate.")
        st.image("CER.png", caption = "Formule du calcul de la CER", width= 400)
        st.markdown(
            "Avec :  \n" 
            "- Sub : Nombre de substitutions  \n" 
            "- Sup : Nombre de suppressions  \n" 
            "- Ins : Nombre d’insertions  \n" 
            "- N : Nombre de caractères dans le texte de référence.")
        st.write("Nous obtenons un CER de 83% avec ce modèle de Transfer Learning entraîné sur 2 000 données seulement")
        st.write("Cela signifie que 83% des mots ont été mal retranscrits. Ce n'est pas très concluant en terme de performance du modèle, mais nous avons de même 17% des mots qui sont reconnus")
    else:
        st.write('*Cliquez pour afficher les résultats*')


        
elif menu=="CRNN":
    title = """  <p style=font-family:Noteworthy;color:palevioletred;font-size:40px;" > Convolutionnal Recurrent Neural Network</p>"""
    st.markdown(title, unsafe_allow_html=True)
    
    st.write("Le réseau Neuronal Convolutif Récurrent ou CRNN est une combinaison de deux réseau de neurones:  \n"
"- Réseau Neuronal Convolutif  ou CNN  \n"
"- Réseau Neuronal Récurrent ou RNN  \n"
"La fonction de perte utilisée est :  \n"
"Classification Temporelle Connectioniste ou CTC")
    

    st.write("Ce mix de modèle permet d'obtenir un seul modèle permettant d'extraire les informations pertinentes (CNN), de les classifier au mieux (RNN) et de résoudre la problématique d'alignement des caractères.")
    
    subtitle = """  <p style=font-family:Noteworthy;color:palevioletred;font-size:20px;" > Le modèle.</p>"""
    st.markdown(subtitle, unsafe_allow_html=True)
    
    st.write("Voici comment fonctionne un modèle CRNN:")
    
    st.write("Un CNN se décompose en 4 opérations qui permettent d’extraire les features :  \n"
             "-Convolution  \n"
             "-Batch normalization  \n"
             "-Fonction d’activation Leaky ReLU  \n"
             "-Max-pooling  \n")
    st.write("S'ajoute à cela deux RNN. Le RNN est un réseau de neurone capable de reconnaitre des patterns dans une séquence de données (ici nos patterns sont les caractères).")
    
    subtitle = """  <p style=font-family:Noteworthy;color:palevioletred;font-size:20px;" > Le CNN pour l'extraction de features.</p>"""
    st.markdown(subtitle, unsafe_allow_html=True)
   
    st.write("- il découpe l’image en fragments \n" 
             "- et effectue des opérations de convolution sur chaque fragments.")
    
    st.image("image15.png", caption= "Schéma de l'action du CNN")
    
    subtitle = """  <p style=font-family:Noteworthy;color:palevioletred;font-size:20px;" > Le RNN pour la classification.</p>"""
    st.markdown(subtitle, unsafe_allow_html=True)
    
    st.write("Il lit chaque fragment et construit une table de probabilité pour chaque caractères.")
    st.image("image24.png",caption="Mécanisme de fragmentation d'une image")
    
    subtitle = """  <p style=font-family:Noteworthy;color:palevioletred;font-size:20px;" > CTC pour le décodage.</p>"""
    st.markdown(subtitle, unsafe_allow_html=True)
    
    st.write("Permet de fusionner les caractères les plus probable pour former une prédiction")
    st.image("image25.png", caption ="Fragmentation avec les caractères blancs")

         
    subtitle = """  <p style=font-family:Noteworthy;color:palevioletred;font-size:20px;" > Résultats et discussion.</p>"""
    st.markdown(subtitle, unsafe_allow_html=True)
    
    m = st.markdown("""
<style>
div.stButton > button:first-child {
    background-color: #db7093;
    color:#FFFFFF;
}
div.stButton > button:hover {
    background-color: #f4c1d1;
    color:#FFFFFF;
    }
</style>""", unsafe_allow_html=True)

    
    if st.button('Résultats'):
    
        st.image("image18.png", caption="Résultats du modèle CRNN")
        st.write("Nous pouvons observer les courbes ci-dessus qui décrivent l'évolution de la CER selon les epochs, et constatons que nous avons une CER aux alentours de 8%.")
        st.write("Ceci est un très bon résultat")

    else:
        st.write('*Cliquez pour afficher les résultats*')
    
        
elif menu=="Canvas":    
    title = """  <p style=font-family:Noteworthy;color:palevioletred;font-size:40px;" > Écrivez un mot de votre choix </p>"""

    # use only the 0.7.0 version !!!
    # pip install streamlit-drawable-canvas==0.7.0
    # or if you want to reinstall : pip install --force-reinstall --no-deps streamlit-drawable-canvas==0.7.0
    canvas_result = st_canvas(       
        fill_color="rgba(255, 165, 0, 0.3)",  # Fixed fill color with some opacity
        stroke_width=8,
        stroke_color='#000',
        background_color='#fff',
        height=240,
        width=640,
        drawing_mode='freedraw',
        update_streamlit = False
    )
    
    if st.button('Oscérisation'):
        if canvas_result.image_data is not None:
            cv2.imwrite(f"test_canvas.jpg",  canvas_result.image_data)

            st.write ("Décodage en cours...")

            imgSize = (32, 128) 
            model = tf.keras.models.load_model('cnn_rnn_20.h5', compile=False)
            img = cv2.imread('./test_canvas.jpg', cv2.IMREAD_GRAYSCALE)
            t, img = cv2.threshold(img,120,255, type = cv2.THRESH_BINARY)
            img = cv2.resize(img, (128,32))/255
            img = tf.expand_dims(img, -1)
            img = tf.expand_dims(img, 0)
            pred = model(img)
            predictions = greedy_decoder(pred)[0]
            st.write("Le décodage traduit:",predictions)
        
elif menu=="À vous de jouer!":
    picture = st.camera_input("Prenez une photo!")

    if picture:
        st.image(picture)
        
        with open ('test.jpg','wb') as file:
              file.write(picture.getbuffer())  

        st.write ("Décodage en cours...")

        img_path="test.jpg"
        
        
        imgSize = (32, 128) 
        model = tf.keras.models.load_model('cnn_rnn_20.h5', compile=False)
        img = cv2.imread('./test.jpg', cv2.IMREAD_GRAYSCALE)
        t, img = cv2.threshold(img,130,255, type = cv2.THRESH_BINARY)
        img = cv2.resize(img, (128,32))/255
        img = tf.expand_dims(img, -1)
        img = tf.expand_dims(img, 0)
        pred = model(img)
        predictions = greedy_decoder(pred)[0]
        st.write("Le décodage traduit:",predictions)
