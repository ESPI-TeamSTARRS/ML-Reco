"""Exemple d'intégration de visage_id dans un backend FastAPI (illustration, pas du code de production tel quel).

À adapter : la « base » est ici un dictionnaire en mémoire ; en vrai, une colonne binaire (bytea) par utilisateur, avec
l'authentification, les droits et la limitation de débit de votre backend. Lancer : uvicorn exemples.api_fastapi:app
"""
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, HTTPException, UploadFile

from visage_id import (
    AucunVisage,
    GabaritInvalide,
    ImageInvalide,
    Reconnaisseur,
    decoder_image,
    gabarit_depuis_octets,
    gabarit_en_octets,
)
from visage_id.images import TAILLE_MAX_OCTETS

BASE: dict[str, bytes] = {}  # identifiant -> gabarit sérialisé (remplacer par la base de données)


@asynccontextmanager
async def cycle_de_vie(app: FastAPI):
    app.state.reco = Reconnaisseur()  # UNE fois au démarrage : charge les modèles (~0,1 s)
    yield


app = FastAPI(lifespan=cycle_de_vie)


def _lire(fichier: UploadFile):
    """Lit au plus la taille permise (jamais un upload sans limite en mémoire), puis décode en toute sécurité."""
    return decoder_image(fichier.file.read(TAILLE_MAX_OCTETS + 1))


# Endpoints `def` (et non `async def`) : FastAPI les exécute dans un pool de threads, ce qui convient à du calcul CPU.
@app.post("/visages/{identifiant}", status_code=201)
def enroler(identifiant: str, photos: list[UploadFile] = File(...)):
    """Enrôlement : 3 à 10 photos de la personne -> gabarit stocké."""
    if not 3 <= len(photos) <= 10:
        raise HTTPException(422, "Envoyez entre 3 et 10 photos.")
    try:
        gabarit = app.state.reco.creer_gabarit([_lire(p) for p in photos])
    except ImageInvalide:
        raise HTTPException(422, "Image invalide.") from None
    except AucunVisage:
        raise HTTPException(422, "Aucun visage détecté.") from None
    BASE[identifiant] = gabarit_en_octets(gabarit)
    return {"enrole": True}


@app.post("/visages/{identifiant}/verification")
def verifier(identifiant: str, photo: UploadFile = File(...)):
    """1:1 — « cette photo est-elle bien la personne `identifiant` ? ». Ne renvoie PAS le score (voir README)."""
    try:
        image = _lire(photo)
        octets = BASE.get(identifiant)
        if octets is None:  # même réponse qu'un visage non reconnu : ne révèle pas si l'identifiant existe
            return {"reconnu": False}
        verdict = app.state.reco.verifier(image, gabarit_depuis_octets(octets))
    except ImageInvalide:
        raise HTTPException(422, "Image invalide.") from None
    except GabaritInvalide:
        raise HTTPException(409, "Gabarit à renouveler.") from None
    return {"reconnu": verdict.reconnu}
