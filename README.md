# visage_id — reconnaissance faciale pour le back-office

Package Python d'**inférence uniquement** : il ne contient aucun code d'entraînement et n'a besoin ni de PyTorch ni de GPU.
Dépendances : `numpy`, `opencv-python-headless` (4.11), `pillow`. Les deux modèles ONNX (~14 Mo) sont dans le package.

Chaîne : détection de visage (YuNet) → alignement → empreinte de 512 nombres (ArcFace MobileFaceNet) → similarité avec le
**gabarit** de la personne attendue. Précision mesurée : 99,3 % sur LFW (voir `docs/BENCHMARK_RECO.md`).

> **Guide pour brancher le package au backend : [`INTEGRATION_BACKEND.md`](INTEGRATION_BACKEND.md)** (étapes, tables, routes, erreurs, sécurité).

## Périmètre : ce que le package fait, et ce qu'il ne fait pas

**Il fait une seule chose** : à partir d'une image et d'un gabarit (ou d'une photo de référence), dire si c'est la même personne
(`verifier`, `verifier_photo`), plus l'enrôlement (`creer_gabarit`) et le stockage du gabarit.

**Il ne fait pas, et ne fera pas** (c'est le travail du backend ou un projet à part) : détecter qu'une photo ou une vidéo est présentée
à la place d'une vraie personne (anti-usurpation), limiter les essais ou bloquer un compte, gérer les étapes de connexion, les droits ou
les sessions, capturer l'image de la caméra, piloter une LED. Le package n'a aucun état : il répond à chaque appel indépendamment.
Les pistes d'amélioration du projet sont dans `docs/AMELIORATIONS.md`.

## Installation

```bash
# Depuis GitHub (le plus simple) : les modèles ONNX sont dans le dépôt, rien d'autre à télécharger
uv add "visage-id @ git+https://github.com/ESPI-TeamSTARRS/visage-id.git"       # adaptez l'URL au dépôt réel
# ou :  pip install "git+https://github.com/ESPI-TeamSTARRS/visage-id.git"
```

Dépôt privé : le poste ou le build Docker doit avoir accès (jeton GitHub ou clé SSH). Python ≥ 3.12 (testé en 3.13, sur Windows).

Variante : à partir d'un fichier `.whl` (12,8 Mo, modèles inclus) : `uv build --wheel` dans ce dossier, puis
`uv add ./dist/visage_id-0.1.0-py3-none-any.whl`.

Pas de service supplémentaire : le package est appelé en direct dans le backend.

## Utilisation

```python
from visage_id import (Reconnaisseur, decoder_image, gabarit_en_octets, gabarit_depuis_octets,
                       AucunVisage, ImageInvalide, GabaritInvalide)

reco = Reconnaisseur()            # UNE fois au démarrage du backend (charge les modèles), puis réutiliser

# Enrôlement : 3 à 10 photos variées de la personne (octets JPEG/PNG reçus du réseau)
gabarit = reco.creer_gabarit([decoder_image(octets) for octets in photos])   # AucunVisage si aucun visage
en_base = gabarit_en_octets(gabarit)                                         # 2 069 octets -> colonne bytea

# Connexion (1:1) : l'identifiant tapé désigne la personne, on compare à SON gabarit
verdict = reco.verifier(decoder_image(octets_capture), gabarit_depuis_octets(en_base))
verdict.reconnu        # bool : score >= reco.seuil
verdict.visage_trouve  # False si la caméra ne voit aucun visage
verdict.score          # similarité cosinus (à NE PAS renvoyer au navigateur, voir Sécurité)
```

**Plus simple, sans gabarit :** `reco.verifier_photo(image_camera, photo_reference)` compare directement la capture à une photo
fournie par le backend (`AucunVisage` si la photo de référence n'a pas de visage). Moins rapide (la référence est retraitée à chaque
appel) et moins fiable qu'un gabarit moyenné sur plusieurs photos ; le gabarit évite aussi de conserver des photos (données biométriques).

Un exemple complet FastAPI (enrôlement + vérification, testé avec le wheel installé) : `exemples/api_fastapi.py`.
Autre mode : `reco.identifier(image, {"sam": gabarit, ...})` → « qui est-ce ? » (1:N) avec nom et `confiance` ; prévu pour une démo,
moins sûr que le 1:1 quand le nombre de personnes grandit.

### Erreurs (toutes héritent de `ValueError`, sans donnée biométrique dans le message)

| Erreur | Quand | Réponse HTTP conseillée |
|---|---|---|
| `ImageInvalide` | fichier vide, illisible, ni JPEG ni PNG, > 10 Mo, > 4096 px de côté | 422 |
| `AucunVisage` | à l'enrôlement, aucune photo ne contient de visage | 422 |
| `GabaritInvalide` | gabarit corrompu **ou calculé avec un autre modèle** | 409 : réenrôler la personne |

## Mise en production : ce qu'il faut savoir

**Performance et concurrence.** Mesuré sur le wheel installé (Python 3.13, CPU, portable) : 11 ms par vérification pour une image de 250 px, 26 ms pour du 640x480.
`Reconnaisseur` est utilisable depuis plusieurs threads mais un verrou **sérialise** les calculs : pour plus de débit, lancez plusieurs
*processus* (workers uvicorn/gunicorn), un `Reconnaisseur` par processus (~110 Mo de mémoire chacun, mesuré). Utilisez des endpoints `def`
(pool de threads) pour ne pas bloquer la boucle asynchrone.

**Stockage.** Le gabarit (2 069 octets) est lié au modèle qui l'a créé. Si le modèle change dans une version future, les anciens gabarits
sont refusés (`GabaritInvalide`) au lieu de donner des résultats faux en silence : prévoir un réenrôlement. Ne **pas** stocker les photos :
seul le gabarit est nécessaire.

**Seuil.** `SEUIL_PAR_DEFAUT = 0.23` (0,1 % de faux accepts sur LFW). LFW est plus facile que la vraie vie : **recalibrez avec des
photos de vos utilisateurs** avant la mise en production (augmenter le seuil = moins de faux accepts, plus de faux rejets).
Réglable : `Reconnaisseur(seuil=0.30)`.

### Bon usage des résultats

- **Ne renvoyez jamais `score` au navigateur** : il permettrait de « grimper » vers une image acceptée par essais successifs. Renvoyez un booléen.
- **Données biométriques (RGPD, catégorie particulière)** : le gabarit en est une. Consentement explicite, finalité unique, suppression à la
  demande ou au départ de la personne, chiffrement de la colonne au repos si possible, ne jamais journaliser image ni gabarit.
- Les images reçues sont limitées (JPEG/PNG, 10 Mo, 4096 px) et leurs dimensions lues dans l'en-tête avant décodage, contre les
  « bombes de décompression ». Gardez aussi une limite de taille de requête au niveau de la passerelle.

### Docker (non testé ici : Docker n'était pas démarré)

```dockerfile
FROM python:3.13-slim
COPY visage_id-0.1.0-py3-none-any.whl /tmp/
RUN pip install --no-cache-dir /tmp/visage_id-0.1.0-py3-none-any.whl && rm /tmp/*.whl
```

`opencv-python-headless` ne nécessite pas de bibliothèque graphique (pas de `libGL`). Le package n'écrit rien sur le disque : compatible
avec un système de fichiers en lecture seule et un utilisateur non-root. Si votre `uv.lock` est strict, ajoutez le wheel comme dépendance
locale (`uv add ./visage_id-...whl`).

## Licences

Les poids ArcFace (InsightFace) sont limités à la **recherche non commerciale** (acceptable en projet scolaire ; à remplacer par SFace,
Apache 2.0, ~0,2 point de précision en moins, pour un usage commercial). YuNet : MIT. Détails : `docs/BENCHMARK_RECO.md` et `NOTICE.md`.

## Développement

```bash
uv sync && uv run pytest        # 32 tests
```

Les modèles ont été choisis après un benchmark sur LFW (`docs/BENCHMARK_RECO.md`). Ne modifiez pas `MODELE_ID` sans réenrôler les gabarits.
