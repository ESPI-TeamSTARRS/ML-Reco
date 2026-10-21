# Brancher `visage_id` au backend — guide pour l'équipe DEV

Objectif : après la saisie de l'identifiant, vérifier avec la caméra que la personne est bien celle du compte, **puis** autoriser le mot de passe.
Le package fait uniquement « ce visage est-il celui de cette personne ? ». Tout le reste (étapes, blocages, LED, droits, anti-usurpation éventuelle) est hors du package : c'est le backend ou un autre projet.

> **Ce qui a été exécuté et ce qui ne l'a pas été.** Les appels à `visage_id` ci-dessous sont ceux de l'exemple `exemples/api_fastapi.py`,
> exécuté avec le wheel installé (Python 3.13). Le reste (tables, routes, cas d'usage, JavaScript) est un **squelette à adapter**,
> non exécuté contre BaseCamp : le dépôt BaseCamp n'a pas été modifié.

## 1. Installer

```bash
cd apps/backend/auth                           # le service concerné (voir §2)
uv add "visage-id @ git+https://github.com/ESPI-TeamSTARRS/ML-Reco.git"
```

Les deux modèles ONNX (~14 Mo) sont dans le dépôt : l'installation les apporte, aucun téléchargement ensuite.
Dépôt privé : donner l'accès au poste de développement et au build Docker (jeton GitHub en secret de build, ou clé SSH).
Variante hors réseau : un fichier `.whl` (`uv build --wheel`, puis `uv add ./visage_id-0.1.0-py3-none-any.whl`).
Python ≥ 3.12 (votre 3.13 convient). 4 dépendances seulement (numpy, opencv headless, pillow, le package). Pas de PyTorch, pas de GPU.
Comme pour le reste de BaseCamp (« aucune bibliothèque partagée »), c'est une dépendance du seul service qui l'utilise.

## 2. Où le mettre

Proposition : dans le service **`auth`** (c'est de l'authentification, il a déjà les comptes). À valider avec le skill `architecte` : un nouveau
service ne se justifie que si 2 critères sur 4 sont remplis.

Respect de la Clean Architecture et de « un fichier = une classe » :

```
auth/app/<tranche>/
  application/ports/reconnaissance_faciale.py     # l'interface (Protocol)
  application/connexion_visage.py                 # le cas d'usage (étapes, blocages)
  infrastructure/visage_id_reconnaissance.py      # l'adaptateur : SEUL fichier qui importe visage_id
  presentation/api/routes_visage.py               # les routes
```

Le domaine et l'application ne connaissent pas `visage_id` : les tests se font avec un faux (voir §7).

## 3. L'adaptateur (le seul code qui touche le package)

```python
# infrastructure/visage_id_reconnaissance.py
from visage_id import Reconnaisseur, decoder_image, gabarit_depuis_octets, gabarit_en_octets


class VisageIdReconnaissance:
    """Implémente le port ReconnaissanceFaciale. Une instance pour toute la durée de vie du processus."""

    def __init__(self, seuil: float) -> None:
        self._reco = Reconnaisseur(seuil=seuil)  # charge les modèles (~0,1 s) : à faire UNE fois

    def enroler(self, photos: list[bytes]) -> bytes:
        """3 à 10 photos (octets JPEG/PNG) -> gabarit sérialisé (2 069 octets) à stocker."""
        return gabarit_en_octets(self._reco.creer_gabarit([decoder_image(p) for p in photos]))

    def verifier(self, capture: bytes, gabarit: bytes) -> bool:
        verdict = self._reco.verifier(decoder_image(capture), gabarit_depuis_octets(gabarit))
        return verdict.reconnu  # JAMAIS le score vers le client (voir §8)
```

Création au démarrage (lifespan FastAPI) et injection : `app.state.reconnaissance = VisageIdReconnaissance(seuil=settings.face_threshold)`.
Le seuil vient de la configuration (`FACE_THRESHOLD`, défaut `0.23`), pas du code : il se règle sans redéployer.

**Aucun état fonctionnel en mémoire** : `Reconnaisseur` ne garde que les modèles chargés (une ressource, comme un pool de connexions).
Gabarits, étapes et compteurs d'essais vont en base.

## 4. Les données

```sql
ALTER TABLE users ADD COLUMN face_template BYTEA NULL;        -- gabarit de 2 069 octets ; NULL = visage non enregistré
-- (chiffrer la colonne au repos si l'infra le permet : c'est une donnée biométrique)

CREATE TABLE login_attempts (                                  -- un parcours de connexion
  id            UUID PRIMARY KEY,
  user_ref      TEXT NOT NULL,                                 -- identifiant saisi (peut ne pas exister : voir §8)
  step          TEXT NOT NULL,                                 -- 'face' | 'password' | 'done' | 'failed'
  expires_at    TIMESTAMPTZ NOT NULL                           -- ex. 2 minutes : l'identifiant est « remis à vide » après
);
CREATE TABLE login_lockouts (
  user_ref      TEXT PRIMARY KEY,
  failures      INT NOT NULL DEFAULT 0,
  lockouts      INT NOT NULL DEFAULT 0,                        -- nombre de blocages déjà subis (pour l'escalade)
  locked_until  TIMESTAMPTZ NULL
);
```

## 5. Le déroulé de connexion

| # | Appel | Le backend fait | Réponse |
|---|---|---|---|
| 1 | `POST /auth/identifiant` `{identifiant}` | Refuse si `locked_until > now` (423 + durée restante). Sinon crée une ligne `login_attempts` (`step='face'`). **Crée la ligne même si l'identifiant n'existe pas.** | `{tentative_id}` |
| 2 | `POST /auth/visage` (multipart : `tentative_id`, `photo`) | Exige `step='face'` et non expiré. Charge `face_template`, appelle `verifier`. Si reconnu → `step='password'`. Sinon → échec (voir plus bas). | `{"reconnu": true/false}` seulement |
| 3 | `POST /auth/mot-de-passe` `{tentative_id, mot_de_passe}` | Exige `step='password'`. Mot de passe valide → `step='done'`, émet le jeton existant. Sinon → échec de connexion. | jeton, ou 401 |

**Échec au visage** (consigne) : l'identifiant est vidé (`step='failed'`), `failures += 1`. 1ʳᵉ fois : on peut recommencer. 2ᵉ échec : `locked_until = now + 5 min`,
`lockouts += 1`, `failures = 0`. *« Puis etc. »* n'est pas précisé dans la consigne : hypothèse = durée doublée à chaque blocage consécutif
(5, 10, 20 min…). **À confirmer.** Le compteur est par identifiant (ajouter aussi un compteur par IP à la passerelle).
**Mot de passe faux** : `step='failed'`, retour à l'étape 1 (identifiant et caméra remis à zéro).

Squelette du cas d'usage (application, sans FastAPI ni `visage_id`) :

```python
class ConnexionVisage:
    def __init__(self, reconnaissance: ReconnaissanceFaciale, tentatives, comptes, blocages, horloge): ...

    def verifier_visage(self, tentative_id: str, capture: bytes) -> bool:
        t = self._tentatives.charger(tentative_id)           # lève si inconnue, expirée, ou step != 'face'
        compte = self._comptes.par_identifiant(t.user_ref)    # peut être None ou sans face_template
        reconnu = False
        if compte and compte.face_template:
            try:
                reconnu = self._reconnaissance.verifier(capture, compte.face_template)
            except (ImageInvalide, GabaritInvalide):
                reconnu = False                               # voir §6 pour la distinction côté HTTP
        if reconnu:
            self._tentatives.passer_a(t, "password")
        else:
            self._tentatives.echouer(t)
            self._blocages.enregistrer_echec(t.user_ref)      # applique 1 essai puis blocage
        return reconnu
```

## 6. Les erreurs du package et les réponses HTTP

Toutes héritent de `ValueError`, sans donnée biométrique dans le message (journalisables).

| Exception | Cause | Réponse conseillée |
|---|---|---|
| `ImageInvalide` | fichier vide/illisible, ni JPEG ni PNG, > 10 Mo, > 4096 px | **422** « image invalide » (ne compte pas comme un échec de visage) |
| `AucunVisage` | à l'enrôlement : aucune photo ne contient de visage | **422** « aucun visage détecté » |
| `GabaritInvalide` | gabarit corrompu ou calculé avec un autre modèle | **409** + réenrôlement de la personne (journaliser) |

À la vérification, « aucun visage dans la capture » n'est pas une exception : `verifier` renvoie `reconnu=False` (c'est un échec normal).

## 7. Tests

- **Cas d'usage** : un faux `ReconnaissanceFaciale` (`verifier → True/False`) suffit : étapes, expiration, blocage 1 essai puis 5 min, escalade.
- **Adaptateur** (un test d'intégration) : enrôler 3 photos d'une même personne, vérifier une 4ᵉ (reconnu) et une photo d'une autre personne (refusée). Il faut de vraies photos de visages : des photos de coéquipiers consentants, **pas** dans le dépôt.
- Le package a ses propres tests (`uv run pytest` dans `package/`, 32 tests) : inutile de les refaire.

## 8. Sécurité : ce que le backend DOIT faire

1. **Le visage est un 2ᵉ facteur** : mot de passe toujours exigé après.
2. **Ne jamais renvoyer le score** (ni « presque reconnu ») au client : il permettrait de trouver par essais une image acceptée. Booléen seulement.
3. **Même réponse** pour « identifiant inexistant » et « visage non reconnu » ; l'étape 1 crée une tentative dans les deux cas.
4. **Limiter les essais** (§5) et la taille des requêtes (≤ 10 Mo, à la passerelle aussi).
5. **Journaliser** : identifiant, résultat, horodatage. **Jamais** l'image ni le gabarit.
6. **RGPD** : donnée biométrique → consentement explicite, finalité unique, suppression du gabarit sur demande ou départ, pas de conservation des photos (seul le gabarit est nécessaire).
7. Le jeton de session n'est émis qu'après le mot de passe (étape 3), jamais après le visage seul.

## 9. Capturer la photo (à décider : le package ne s'en occupe pas)

**Option A — le navigateur capture et envoie (la plus rapide, recommandée pour démarrer).** La page de connexion utilise la caméra de l'ordinateur
qui affiche la page. Si ce navigateur est sur le PC hôte, la consigne « caméra sur le PC hôte » est respectée.

```js
// Squelette (non exécuté) : prendre UNE image, l'envoyer, puis ÉTEINDRE la caméra (exigé par la consigne)
const flux = await navigator.mediaDevices.getUserMedia({ video: { width: 640, height: 480 } });
const video = document.createElement("video"); video.srcObject = flux; await video.play();
const canvas = document.createElement("canvas"); canvas.width = 640; canvas.height = 480;
canvas.getContext("2d").drawImage(video, 0, 0);
flux.getTracks().forEach((t) => t.stop());                         // caméra éteinte
const photo = await new Promise((r) => canvas.toBlob(r, "image/jpeg", 0.85));
const form = new FormData(); form.append("tentative_id", tentativeId); form.append("photo", photo, "capture.jpg");
const reponse = await fetch("/auth/visage", { method: "POST", body: form });
```

**Piège important :** `getUserMedia` ne fonctionne qu'en **HTTPS ou sur `localhost`**. Sur le réseau RESILIENCE-G, une page servie en `http://192.168.x.x`
verra la caméra **refusée** par le navigateur. Il faut du HTTPS (la passerelle Traefik peut le fournir avec un certificat local) ou ouvrir la page via `localhost` sur le PC hôte.

**Option B — un agent sur le PC hôte capture (si la caméra doit être lue hors navigateur).** Docker Desktop sous Windows expose mal une webcam aux
conteneurs ; il faudrait un petit processus natif sur l'hôte qui prend l'image et appelle le backend. Plus de travail : à n'envisager que si l'option A est refusée.

Conseil qualité : visage de face, bien éclairé, pas de contre-jour. 1 image par tentative suffit ; si le taux de faux rejets est trop haut en pratique, envoyer 2 ou 3 images
et accepter dès qu'une est reconnue (en gardant le même nombre d'essais décompté).

## 10. Enrôlement (gestion des habilitations)

Page d'administration : l'admin choisit l'utilisateur, la webcam prend **5 à 10 images variées** (de face, léger tour de tête, expression, avec/sans lunettes si porté),
envoyées à `POST /admin/users/{id}/face` → `enroler(photos)` → `UPDATE users SET face_template = ...`. Droits : réservé aux administrateurs.
Révocation : `face_template = NULL`. Réenrôler après un changement d'apparence marqué ou de modèle (`GabaritInvalide`).
5 photos variées valent mieux que 30 quasi identiques.

## 11. Exploitation

- **Charge** : 26 ms par vérification (640x480, CPU), ~110 Mo par processus, un verrou interne sérialise les calculs. Pour plus de débit : plusieurs workers (processus), pas plus de threads. Endpoints `def` (pool de threads), pas `async def`.
- **Docker** (non testé ici, le démon n'était pas démarré) : `pip install` du wheel dans `python:3.13-slim` ; aucune bibliothèque système nécessaire (OpenCV *headless*) ; le package n'écrit rien sur disque (compatible `read_only`, non-root).
- **Santé** : si `Reconnaisseur()` s'instancie sans erreur au démarrage, les modèles sont chargés ; échouer vite (ne pas démarrer) si le fichier est absent.
- **Mises à jour du package** : tout changement de `MODELE_ID` rend les gabarits existants invalides (voulu) : prévoir un réenrôlement.

## 12. À régler avant d'ouvrir à de vrais utilisateurs

- **Seuil** : `0.23` vient d'un jeu de test public (0,1 % de faux accepts). Le recalibrer avec des photos réelles de vos utilisateurs et d'inconnus (voir `docs/AMELIORATIONS.md`).
- **Autres points à régler** (licence des poids, tests Docker/Linux...) : `docs/AMELIORATIONS.md`.

## 13. Dépannage rapide

| Symptôme | Cause probable |
|---|---|
| La caméra ne s'ouvre pas dans le navigateur | page en HTTP (hors `localhost`) : passer en HTTPS (§9) |
| Toujours « non reconnu » | mauvaise lumière / contre-jour ; gabarit créé avec peu de photos variées ; seuil trop haut |
| Quelqu'un d'autre est reconnu | seuil trop bas : l'augmenter (ex. `0.30`) et mesurer |
| `GabaritInvalide` | gabarit d'un autre modèle ou corrompu : réenrôler |
| `ImageInvalide` | fichier ni JPEG ni PNG, trop lourd, ou trop grand |
| 1ʳᵉ requête lente | normal : le premier calcul est plus lent ; créer le `Reconnaisseur` au démarrage |
