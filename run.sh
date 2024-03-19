#!/bin/bash

# Chemin vers le fichier Python
chemin_python="main.py"

# Vérifier si le fichier Python existe
if [ -f "$chemin_python" ]; then
    # Exécuter le fichier Python
    python "$chemin_python"
else
    echo "Le fichier $chemin_python n'existe pas."
fi