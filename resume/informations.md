# C'est quoi un token:

- Un LLM ne reçoit jamais du texte brut, mais une séquence d'entiers.
- La tokenisation est l'étape qui transforme une chaîne de caractères en liste d'IDs numériques. L'inverse s'appelle le décodage.
- Les tokens ne sont pas des mots, ce sont des sous-mots, le mot "unhappiness" peut devenir ["un", "happi", "ness"] — trois tokens. À contrario, le mot "user" peut-être un seul token, et sur la version gpt-4o il équivaut à 1428.

---

Byte Pair Encoding (BPE):

- C'est un algorithme:

1) Commence avec un vocabulaire de caractères individuels (a, b, c, ..., 0-9, etc.)
2) Compte les paires de symboles adjacents les plus fréquentes dans ton corpus
3) Fusionne la paire la plus fréquente en un nouveau symbole
4) Répète jusqu'à atteindre la taille de vocabulaire voulue (~150 000 pour Qwen)

Résultat : les mots très fréquents ont un token unique, les mots rares sont découpés en morceaux connus. Le vocabulaire ne peut jamais être "out-of-vocabulary" parce qu'au pire on revient aux octets individuels.

---

Le Ġ et ▁:

- Dans le fichier vocab.json tu verras des tokens comme "Ġworld" ou "▁world". Ce préfixe signifie "espace avant ce mot". Donc "world" et "Ġworld" sont deux tokens différents avec deux IDs différents.

- Quand tu feras du constrained decoding sur du JSON (étape 5), tu devras savoir si le token "true" correspond à l'ID de "true" ou de "Ġtrue" — sinon tu bloques la mauvaise chose.

---

# Logits:

Un LLM (comme GPT, Qwen, etc.) génère du texte en prédisant le prochain token à chaque étape.

À partir d’une séquence de tokens en entrée, le modèle produit un vecteur de logits : un score pour chaque token du vocabulaire (souvent des dizaines ou centaines de milliers).

Les logits ne sont pas des probabilités. Ce sont des scores bruts internes qui indiquent à quel point chaque token est compatible avec le contexte.

Exemple :

Input:
"Bonjour, comment"

Logits (simplifié):
"ça"     → 12.4
"tu"     → 5.1
"merci"  → 0.3
"chien"  → -8.2

Le token avec le plus grand logit est celui jugé le plus probable.

Les logits peuvent être négatifs, très grands, ou non bornés. Ils n’ont pas d’interprétation directe en probabilité.

---

Softmax:

Pour transformer les logits en probabilités, on applique la fonction softmax.

- Les gros scores deviennent probables
- Les petits scores deviennent quasi nuls
- La somme des probabilités = 1

Cela permet d’obtenir une distribution de probabilité sur tous les tokens.

---

Choix du token:

Il existe plusieurs méthodes pour choisir le prochain token :

- Greedy decoding : on prend le token avec le plus grand logit (argmax)
- Sampling : on tire aléatoirement selon les probabilités
- Temperature : contrôle la créativité
  - T < 1 → plus déterministe
  - T > 1 → plus aléatoire

---

Constrained decoding:

Le constrained decoding consiste à forcer le modèle à respecter une structure (ex: JSON, code).

Pour cela, on interdit certains tokens en modifiant les logits :

logits[invalid_tokens] = -inf

Pourquoi ça marche :

exp(-inf) = 0 → probabilité nulle

Donc ces tokens ne peuvent pas être choisis.

---

Résultat final:

next_token = argmax(logits)

On choisit donc le meilleur token parmi ceux autorisés uniquement.

---

# Structure de vocab.json:

Le fichier retourné par get_path_to_vocab_file() est un dictionnaire JSON qui représente le vocabulaire du modèle.

Il contient une correspondance entre les tokens et leurs identifiants numériques (IDs).

Exemple :

{
  "!": 0,
  "\"": 1,
  "#": 2,
  ...
  "Hello": 9906,
  "Ġworld": 1917
}

Cela signifie :

- Les clés = tokens (chaînes de caractères)
- Les valeurs = IDs (entiers)

Donc le mapping est :

token (string) → token_id (int)

---

Pourquoi ce format ?

Un LLM ne travaille pas avec du texte directement.

Il transforme toujours le texte en IDs avant de le traiter :

"Hello world" → [9906, 1917]

Et à l’inverse, après génération :

[9906, 1917] → "Hello world"

Le vocabulaire sert donc de table de conversion entre texte et représentation numérique.

---

Créer le mapping inverse:

Dans la plupart des cas, on a besoin de faire l’inverse :

id → token

Pour cela :

import json

with open(model.get_path_to_vocab_file()) as f:
    vocab = json.load(f)

id_to_token = {v: k for k, v in vocab.items()}

Exemples :

id_to_token[9906] → "Hello"
id_to_token[1917] → "Ġworld"

---

Pourquoi c’est crucial pour le constrained decoding ?

Le constrained decoding consiste à forcer le modèle à générer uniquement du texte valide selon des règles (ex : JSON, code, DSL).

Pour cela, à chaque étape de génération, il faut savoir :

- quels tokens sont autorisés
- quels tokens sont interdits
- comment ils correspondent à la structure attendue

Exemple :

On génère :
{"name": "

À ce moment-là, le modèle doit continuer avec un nom valide.

On peut alors filtrer le vocabulaire :

- garder uniquement les tokens compatibles (ex: lettres, fn_, etc.)
- mettre tous les autres à -inf dans les logits

logits[invalid_tokens] = -inf

Sans le mapping id → token, il est impossible de savoir ce que représente un token ID, donc impossible de filtrer correctement.

---

Attention aux tokens multi-caractères:

Un token ne correspond pas forcément à un seul caractère.

Exemples :

"fn_add"   → peut être 1 seul token
"Hello"    → 1 token
"Ġworld"   → 1 token (inclut un espace)

Cela implique que :

- un token peut ajouter plusieurs caractères en une seule étape
- la génération ne progresse pas caractère par caractère mais par blocs

---

Conséquence pour le constrained decoding:

Quand un token est accepté :

- il peut avancer la position de plusieurs caractères dans la sortie
- il peut modifier fortement la structure en une seule étape

Donc le système doit :

- vérifier la validité après chaque token complet
- gérer la structure attendue en fonction des chaînes produites, pas des caractères individuel

---

# Construire un prompt pour le LLM:

Un LLM ne “sait” rien au sens classique. Il ne possède pas de mémoire explicite ni de compréhension logique des fonctions comme un programme. Il fonctionne uniquement en complétant du texte : il prédit le prochain token le plus probable à partir du contexte fourni.

👉 Donc ton objectif n’est pas de lui donner des instructions strictes, mais de construire un contexte tel que la suite de texte la plus probable corresponde exactement à ce que tu veux (ex : un nom de fonction + ses arguments).

---

Format d’un bon prompt (function calling):

Il n’existe pas de format universel, mais un bon prompt contient généralement :

- un contexte clair (rôle du modèle)
- la liste des fonctions disponibles
- la description des fonctions et de leurs paramètres
- la requête utilisateur
- un début de réponse partiellement écrit (très important)

---

Exemple :

You are a function-calling assistant.
Your task is to select the correct function and extract arguments from the user request.

Available functions:
- fn_add_numbers(a: number, b: number) -> number
  Description: Add two numbers together and return their sum.
- fn_greet(name: string) -> string
  Description: Generate a greeting message for a person by name.
- fn_reverse_string(s: string) -> string
  Description: Reverse a string.

User request: What is the sum of 40 and 2?

Response (JSON):
{"name": "

---

Pourquoi ça marche ?

Un LLM est un modèle qui prédit le prochain token.

Donc si tu termines le prompt par :
{"name": "

tu forces la distribution des tokens possibles vers une continuation structurée (JSON + nom de fonction).

Le modèle va alors naturellement produire :
- un nom de fonction plausible
- parmi celles présentes dans le prompt
- car c’est statistiquement la suite la plus cohérente

👉 C’est le principe du few-shot prompting :
tu montres une structure, et le modèle la complète.

---

Pourquoi la fin du prompt est critique:

Le dernier token du prompt influence fortement la prédiction suivante.

Si tu finis par :
{"name": "

alors le modèle est “coincé” dans une structure JSON et la continuation la plus probable devient un champ de fonction valide.

---

Prompt dynamique:

Dans un vrai système, le prompt n’est pas écrit à la main.

Il est généré automatiquement à partir des définitions de fonctions disponibles.

Exemple :

def build_prompt(prompt: str, definitions: list[DefinitionTest]) -> str:
    functions_str = "\n".join(
        f"- {d.name}({', '.join(f'{k}: {v.type}' for k, v in d.parameters.items())}) -> {d.returns.type}\n"
        f"  Description: {d.description}"
        for d in definitions
    )

    return f\"\"\"You are a function-calling assistant.

Available functions:
{functions_str}

User request: {prompt}

Response (JSON):
{{"name": \"\"\""


Résumé:

- un LLM ne manipule pas du texte brut mais des IDs numériques
- la tokenisation transforme le texte en tokens
- un token peut être un mot, une partie de mot ou un caractère
- le même mot peut être découpé différemment selon le tokenizer

---

# Byte Pair Encoding (BPE):

Résumé:

- BPE construit le vocabulaire en fusionnant les séquences fréquentes
- les mots fréquents deviennent souvent un seul token
- les mots rares sont découpés en sous-parties connues
- permet de limiter les tokens inconnus

---

# Le Ġ et ▁:

Résumé:

- Ġ et ▁ représentent généralement un espace avant le token
- "world" et "Ġworld" sont deux tokens différents
- important pour le constrained decoding et les formats structurés

---

# Logits:

Résumé:

- le modèle produit un score (logit) pour chaque token possible
- les logits représentent la plausibilité d’un token dans le contexte
- le token avec le plus grand logit est généralement le plus probable
- les logits ne sont pas des probabilités

---

# Softmax:

Résumé:

- softmax transforme les logits en probabilités
- les probabilités sont comprises entre 0 et 1
- leur somme vaut toujours 1
- les gros logits deviennent plus probables

---

# Choix du token:

Résumé:

- greedy decoding = prend le token le plus probable
- sampling = choisit aléatoirement selon les probabilités
- temperature contrôle le niveau de créativité
- le choix du token influence toute la suite de la génération

---

# Constrained decoding:

Résumé:

- force le modèle à respecter une structure (JSON, code, etc.)
- les tokens invalides sont bloqués avec -inf
- un token interdit obtient une probabilité nulle
- permet de garantir des sorties valides

---

# Structure de vocab.json:

Résumé:

- vocab.json contient le mapping token → ID
- on inverse souvent le mapping pour obtenir ID → token
- indispensable pour comprendre les tokens générés
- utilisé pour filtrer les tokens pendant le constrained decoding

---

# Tokens multi-caractères:

Résumé:

- un token peut représenter plusieurs caractères
- la génération se fait par blocs de texte
- accepter un token peut faire avancer plusieurs caractères d’un coup
- important pour les contraintes structurées

---

# Construire un prompt pour le LLM:

Résumé:

- un LLM ne “comprend” pas, il complète du texte
- le prompt sert à orienter les probabilités de tokens
- la structure du prompt influence directement la sortie
- terminer une structure (ex JSON) guide fortement la génération
- few-shot prompting = montrer un exemple à compléter
- le prompt peut être généré dynamiquement à partir des fonctions disponibles