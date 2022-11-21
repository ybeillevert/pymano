# 3 - Librairies et Notebook

# Charger une nouvelle librairie

Dans le cas présent je vais essayer d’utiliser open-cv qui est la librairie qui permet de faire des opérations sur des images.

- Chercher sur google le nom du package. Par ex: j’ai cherché “opencv python package name” sur google et j’ai trouvé “opencv-python’
- Inclure la commande suivante *! pip install nompackage -q*
    
    ![Untitled](3%20-%20Librairies%20et%20Notebook%208340cd8bdc2841898c2a50a1c86ccf8f/Untitled.png)
    
    - -q permet de ne pas afficher les messages relatifs à l’installation
- Vous pouvez utiliser la librairie
    
    ![Untitled](3%20-%20Librairies%20et%20Notebook%208340cd8bdc2841898c2a50a1c86ccf8f/Untitled%201.png)
    
    # Utiliser un notebook dans un autre notebook
    
    Il suffit juste d’exécuter la commande suivante en haut de votre notebook
    
    ```jsx
    %run "LE_NOM_DU_NOTEBOOK.ipynb”
    ```
    
    Techniquement c’est comme si vous aviez copié-collé tout le contenu de l’autre notebook puis exécuté celui-ci