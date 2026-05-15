Chapitre IV — Instructions communes
IV.1 Règles générales

Votre projet doit être écrit en Python 3.10 ou ultérieur.
Votre projet doit respecter le standard de codage flake8.
Vos fonctions doivent gérer les exceptions gracieusement pour éviter les plantages. Utilisez des blocs try-except pour gérer les erreurs potentielles. Préférez les gestionnaires de contexte pour les ressources comme les fichiers ou connexions. Si votre programme plante à cause d'exceptions non gérées lors de la revue, il sera considéré comme non fonctionnel.
Toutes les ressources (ex. handles de fichiers, connexions réseau) doivent être correctement gérées pour éviter les fuites. Utilisez des gestionnaires de contexte quand c'est possible.
Votre code doit inclure des annotations de types pour les paramètres de fonctions, les types de retour et les variables (en utilisant le module typing). Utilisez mypy pour la vérification statique des types. Toutes les fonctions doivent passer mypy sans erreurs.
Incluez des docstrings dans les fonctions et classes en suivant PEP 257 (style Google ou NumPy) pour documenter l'objectif, les paramètres et les retours.

IV.2 Makefile
Incluez un Makefile dans votre projet pour automatiser les tâches courantes. Il doit contenir les règles suivantes :

install : Installe les dépendances du projet avec pip, uv, pipx ou tout autre gestionnaire de paquets.
run : Exécute le script principal du projet.
debug : Lance le script principal en mode debug via le débogueur intégré de Python (pdb).
clean : Supprime les fichiers temporaires ou caches (__pycache__, .mypy_cache) pour garder l'environnement propre.
lint : Exécute flake8 . et mypy . --warn-return-any --warn-unused-ignores --ignore-missing-imports --disallow-untyped-defs --check-untyped-defs
lint-strict (optionnel) : Exécute flake8 . et mypy . --strict

IV.3 Directives supplémentaires

Créez des programmes de test pour vérifier la fonctionnalité du projet (non soumis ni évalués). Utilisez des frameworks comme pytest ou unittest pour les tests unitaires, en couvrant les cas limites.
Incluez un fichier .gitignore pour exclure les artefacts Python.
Il est recommandé d'utiliser des environnements virtuels (venv ou conda) pour l'isolation des dépendances lors du développement.

IV.3.1 Exigences supplémentaires

Toutes les classes doivent utiliser pydantic pour la validation.
Vous pouvez utiliser les paquets numpy et json.
L'utilisation de dspy (ou tout paquet similaire) est totalement interdite, y compris pytorch, huggingface, transformers, outlines, etc.
Vous devez utiliser le modèle suivant :

Qwen/Qwen3-0.6B (par défaut)
Vous pouvez utiliser d'autres modèles tant que votre projet fonctionne avec Qwen/Qwen3-0.6B.


La fonction à appeler doit être choisie par le LLM, pas par des heuristiques ou autre magie médiévale.
Il est interdit d'utiliser des méthodes ou attributs privés du paquet llm_sdk.
Vous devez créer un environnement virtuel et installer les paquets numpy et pydantic avec uv. Pour utiliser llm_sdk, vous pouvez le copier dans le même répertoire que celui contenant src.
Le correcteur, ainsi que la moulinette, se contenteront de lancer uv sync.
Toutes les erreurs doivent être gérées gracieusement. Votre programme ne doit jamais planter inopinément et doit toujours fournir des messages d'erreur clairs.

IV.3.2 Utilisation
Votre programme doit être lancé avec la commande suivante (où src est le dossier contenant vos fichiers) :
bashuv run python -m src [--functions_definition <fichier_definition>] [--input <fichier_input>] [--output <fichier_output>]
Par défaut, le programme lit les fichiers d'entrée depuis data/input/ et écrit la sortie dans data/output/. Exemple :
bashuv run python -m src \
  --functions_definition data/input/functions_definition.json \
  --input data/input/function_calling_tests.json \
  --output data/output/function_calls.json

Chapitre V — Partie obligatoire
V.1 Résumé
Dans ce projet, vous créerez un outil de function calling qui traduit des prompts en langage naturel en appels de fonctions structurés. Pour une question telle que « Quel est la somme de 40 et 2 ? », votre solution ne doit pas retourner 42, mais fournir :

Le nom de la fonction : fn_add_numbers
Les arguments : {"a": 40, "b": 2}

Votre implémentation doit utiliser le décodage contraint pour garantir une sortie JSON valide à 100 %, assurant une fiabilité quasi parfaite même avec un modèle de 0,5 milliard de paramètres.
V.2 Fichiers d'entrée
Votre solution traitera deux fichiers d'entrée situés dans le répertoire data/input/ :
function_calling_tests.json : contient un tableau JSON de prompts en langage naturel que votre système doit traiter.
json[
  {"prompt": "What is the sum of 2 and 3?"},
  {"prompt": "What is the sum of 265 and 345?"},
  {"prompt": "Greet shrek"},
  {"prompt": "Greet john"},
  {"prompt": "Reverse the string 'hello'"},
  ...
]
functions_definition.json : contient les fonctions disponibles que votre système peut appeler. Chaque fonction comprend : nom, noms et types des arguments, type de retour, description.
json[
  {
    "name": "fn_add_numbers",
    "description": "Add two numbers together and return their sum.",
    "parameters": {"a": {"type": "number"}, "b": {"type": "number"}},
    "returns": {"type": "number"}
  },
  {
    "name": "fn_greet",
    "description": "Generate a greeting message for a person by name.",
    "parameters": {"name": {"type": "string"}},
    "returns": {"type": "string"}
  },
  ...
]

⚠️ Ces exemples établissent le niveau de complexité attendu. Cependant, votre solution sera testée avec des prompts et des ensembles de fonctions différents. Vous devez implémenter une gestion correcte des erreurs JSON pour les fichiers d'entrée, car ils peuvent contenir du JSON invalide ou être absents.

V.3 Interaction avec le LLM
V.3.1 Le SDK LLM
Fourni avec ce projet, vous trouverez une classe wrapper Small_LLM_Model dans le paquet llm_sdk que vous pouvez utiliser pour interagir avec le LLM.
Le SDK fournit plusieurs méthodes essentielles :

get_logits_from_input_ids(input_ids: Tensor) -> Tensor — Prend un tenseur input_ids et retourne les logits bruts après appel au modèle LLM.
get_path_to_vocabulary_json() -> str — Retourne le chemin vers un fichier JSON contenant la correspondance structurée entre input_ids et tokens.
encode(text: str) -> List[int] — Encode une chaîne de texte en sa liste correspondante d'IDs de tokens.
decode(token_ids: List[int]) -> str (optionnel) — Décode une liste d'IDs de tokens en chaîne de texte.

V.3.2 Le Pipeline de génération
Le processus de génération LLM suit ces étapes :

Prompt : Votre question en langage naturel
Tokenisation : Le texte est découpé en sous-mots (tokens). Contrairement à un simple découpage par mots, les tokeniseurs incluent souvent des espaces précédents, gèrent la ponctuation et découpent les mots en composants plus petits via des algorithmes comme BPE ou SentencePiece.
Input IDs : Les tokens sont convertis en IDs numériques que le modèle comprend.
Traitement LLM : Le modèle traite ces nombres via son réseau de neurones.
Logits : Le modèle produit des scores de probabilité pour chaque token suivant possible.
Sélection du token : Le token suivant est choisi en fonction de ces probabilités, généralement celui avec le score le plus élevé. C'est à cette étape que le décodage contraint peut être appliqué pour restreindre les choix de tokens et garantir que les sorties suivent une structure spécifique (JSON valide à 100 %).


⚠️ Ce processus se répète token par token. Chaque token généré est ajouté au prompt, et les étapes 2 à 6 se répètent jusqu'à ce que la réponse complète soit générée.

V.3.3 Comprendre le Décodage Contraint
Les modèles de langage génèrent du texte un token à la fois. À chaque étape, le modèle produit une distribution de probabilités (logits) sur tous les tokens suivants possibles.
Le décodage contraint intervient dans ce processus en modifiant les logits avant la sélection du token :

Le modèle produit des logits pour tous les tokens possibles.
Vous identifiez quels tokens maintiendraient une structure JSON valide et la conformité au schéma attendu.
Vous mettez à -inf les logits des tokens invalides (ceux qui briseraient le schéma ou la structure).
Vous échantillonnez uniquement parmi les tokens valides restants.

Dans ce projet, le décodage contraint doit non seulement garantir un JSON syntaxiquement valide, mais aussi imposer un schéma spécifique. Par exemple, si le champ JSON "firmware" ne peut prendre que quelques valeurs prédéfinies, le décodeur doit restreindre la sélection du token à ces options autorisées. Cela garantit que chaque token généré maintient une validité structurelle et sémantique, assurant un JSON 100 % récupérable et toujours parsable sans erreurs.

⚠️ Votre solution ne doit PAS se reposer sur le modèle produisant spontanément du JSON correct à partir d'un prompt. Soumettre au modèle les définitions de fonctions en espérant une sortie structurée n'est pas fiable, et ce n'est pas la compétence que nous attendons de vous ici.


💡 Réfléchissez à la façon dont vous pouvez utiliser le fichier JSON du vocabulaire pour établir la correspondance entre les tokens et leurs représentations en chaîne. C'est crucial pour déterminer quels tokens sont valides à chaque étape de génération.

V.4 Format du fichier de sortie
Votre programme produira un seul fichier JSON : data/output/function_calling_results.json.
Pour chaque prompt, ajoutez un objet JSON à ce fichier. Chaque objet du tableau doit contenir exactement les clés suivantes :

prompt (string) : La requête originale en langage naturel
name (string) : Le nom de la fonction à appeler
parameters (object) : Tous les arguments requis avec les types corrects

Exemple de sortie :
json[
  {
    "prompt": "What is the sum of 2 and 3?",
    "name": "fn_add_numbers",
    "parameters": {"a": 2.0, "b": 3.0}
  },
  {
    "prompt": "Reverse the string 'hello'",
    "name": "fn_reverse_string",
    "parameters": {"s": "hello"}
  }
]
Règles de validation :

Le fichier doit être du JSON valide (pas de virgules finales, pas de commentaires)
Les clés et types doivent correspondre exactement au schéma dans functions_definition.json
Aucune clé supplémentaire ni prose ne sont autorisées dans la sortie
Tous les arguments requis doivent être présents
Les types des arguments doivent correspondre à la définition de la fonction (number, string, boolean, etc.)


⚠️ Les fichiers d'entrée fournis peuvent changer lors de la revue entre pairs. Ne codez pas en dur des solutions basées sur les exemples fournis.

V.5 Performance et Fiabilité
Votre implémentation doit atteindre :

Précision quasi parfaite : 90 %+ de sélection correcte de fonction et d'extraction d'arguments
JSON valide à 100 % : Chaque sortie doit être parsable et conforme au schéma
Vitesse raisonnable : Traiter tous les prompts de test en moins de 5 minutes sur du matériel standard
Gestion robuste des erreurs : Gérer gracieusement les entrées malformées, les fichiers manquants et les cas limites


💡 Le modèle Qwen3-0.6B n'a que 500 millions de paramètres, pourtant avec un décodage contraint approprié, il peut atteindre une fiabilité comparable à des modèles bien plus grands. Cela démontre la puissance du guidage structurel par rapport à la taille brute du modèle.

V.6 Tester votre implémentation
Pour vérifier que votre solution fonctionne correctement :

Assurez-vous que les fichiers d'entrée se trouvent dans le répertoire data/input/
Lancez : uv run python -m src [--functions_definition ...] [--input ...] [--output ...]
Vérifiez que output/function_calling_results.json est créé
Validez la structure et le contenu JSON
Vérifiez que les noms de fonctions et les types d'arguments correspondent aux définitions


⚠️ Testez avec divers cas limites : chaînes vides, grands nombres, caractères spéciaux, mauvais types, prompts ambigus, et fonctions avec plusieurs paramètres.


Chapitre VI — Exigences du README
Un fichier README.md doit être fourni à la racine de votre dépôt Git. Son objectif est de permettre à quiconque n'est pas familier avec le projet (pairs, encadrants, recruteurs, etc.) de comprendre rapidement de quoi il s'agit, comment le lancer et où trouver plus d'informations.
Le README.md doit inclure au minimum :

La toute première ligne doit être en italique et lire : This project has been created as part of the 42 curriculum by <login1>[, <login2>[, <login3>[...]]]
Une section "Description" présentant clairement le projet, son objectif et un aperçu général.
Une section "Instructions" contenant toutes les informations pertinentes sur la compilation, l'installation et/ou l'exécution.
Une section "Resources" listant les références classiques liées au sujet (documentation, articles, tutoriels, etc.), ainsi qu'une description de l'utilisation de l'IA — en précisant pour quelles tâches et quelles parties du projet.

Pour ce projet, le README.md doit également inclure :

Explication de l'algorithme : Décrivez votre approche de décodage contraint en détail
Décisions de conception : Expliquez les choix clés de votre implémentation
Analyse de performance : Discutez de la précision, de la vitesse et de la fiabilité de votre solution
Difficultés rencontrées : Documentez les difficultés rencontrées et comment vous les avez résolues
Stratégie de test : Décrivez comment vous avez validé votre implémentation
Exemple d'utilisation : Fournissez des exemples clairs d'exécution de votre programme


💡 Votre README doit être rédigé en anglais.

Chapitre VIII — Soumission et revue par les pairs
Soumettez votre travail dans votre dépôt Git comme d'habitude. Seul le travail dans votre dépôt sera examiné lors de la soutenance. N'hésitez pas à vérifier deux fois les noms de vos fichiers pour vous assurer qu'ils sont corrects.
Votre dépôt doit contenir :

Le répertoire src/ avec votre implémentation
pyproject.toml et uv.lock pour la gestion des dépendances
Le répertoire llm_sdk/ (copié depuis le paquet fourni)
Le répertoire data/input/ avec les fichiers de test (pour la démonstration)
README.md avec une documentation complète
Tous les fichiers supplémentaires nécessaires à l'exécution de votre solution

⚠️ N'incluez pas le répertoire output/ dans votre dépôt. Il sera généré lors de la revue par les pairs.


#Feuille de suivi — call me maybe

Étape 1 — Setup du projet
 Structure des dossiers (src/, data/input/, llm_sdk/) — dossiers présents
 Initialiser uv à la racine, installer numpy et pydantic (pyproject.toml + uv.lock racine manquants)
 Makefile
 .gitignore
Étape 2 — Parsing des fichiers d'entrée
 Lire et valider functions_definition.json
 Lire et valider function_calling_tests.json
 Modéliser avec des classes Pydantic
Étape 3 — Prise en main du SDK
 Tester encode() et get_logits_from_input_ids()
 Explorer le fichier vocabulaire JSON
 Construire un prompt qui décrit les fonctions au LLM
Étape 4 — Sélection de la fonction
 Le LLM choisit le nom de la fonction à appeler
 Constrained decoding pour restreindre aux noms valides
Étape 5 — Extraction des arguments (cœur du projet)
 Mapping token → caractère depuis le vocabulaire
 Implémenter le constrained decoding pour forcer un JSON valide
 Respecter les types (number, string, bool…)
Étape 6 — Écriture de la sortie
 Générer function_calling_results.json
 Gérer l'argument --output
Étape 7 — CLI et robustesse
 Point d'entrée python -m src avec argparse
 Gestion propre de toutes les erreurs (pas de crash)
Étape 8 — Qualité du code
 flake8 et mypy sans erreurs
 Docstrings sur toutes les fonctions
Étape 9 — README
 Toutes les sections obligatoires + explication du constrained decoding