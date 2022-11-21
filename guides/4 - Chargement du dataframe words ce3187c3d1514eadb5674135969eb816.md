# 4 - Chargement du dataframe words

### Génération des csv

Pour charger le dataframe de words dans votre notebook , vous devez d’abord vous assurer que vous avez bien le csv généré.

Pour cela tapez la série de commandes suivante:

```python
from modules.pym_csv import build_words #Chargement du module
build_words() # Chargement du csv
```

Normalement, vous devriez avoir le csv words.csv dans le dossier data

### Chargement du dataframe

```python
import csv
df = pd.read_csv('./data/words.csv', index_col=0)
```