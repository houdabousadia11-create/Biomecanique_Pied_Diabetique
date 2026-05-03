# ============================================================
# PROJET AdaptStep / Digi'Feet
# Architecture complète : Blocs 0 → 1a → 1b → 2 → 3 → 4
# ============================================================
#
# Ordre d'exécution :
#   Bloc 1a : conversion brute tensions → kPa (SANS filtre)
#   Bloc 0  : surveillance qualité capteur sur signal brut
#   Bloc 1b : filtrage Butterworth sur signal validé
#   Bloc 2  : contrôle qualité postural
#   Bloc 3  : calibration initiale du seuil personnalisé S0
#   Bloc 4  : modèle bayésien adaptatif
# ============================================================

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Any, List, Optional
import numpy as np
import numpy.typing as npt
import os
from scipy.signal import butter, filtfilt

try:
    from reportlab.lib.pagesizes import A4
    from reportlab.platypus import (
        SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image
    )
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.lib import colors
    from reportlab.pdfgen import canvas as rl_canvas
    from reportlab.pdfgen import canvas
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.enums import TA_CENTER
except ImportError:
    print("Warning: ReportLab non installé. Les PDFs ne pourront pas être générés.")


# ============================================================
# UTILITAIRES COMMUNS (internes, partagés entre tous les blocs)
# ============================================================

def _entete(c, width, height, titre_principal, sous_titre, logo_path):
    """Dessine l'en-tête commun à tous les rapports Digi'Feet."""
    if logo_path and os.path.exists(logo_path):
        c.drawImage(
            logo_path, 40, height - 95,
            width=170, preserveAspectRatio=True, mask="auto"
        )
    c.setFont("Helvetica-Bold", 18)
    c.drawString(230, height - 60, titre_principal)
    c.setFont("Helvetica", 11)
    c.drawString(230, height - 80, sous_titre)
    c.drawString(230, height - 98, f"Date : {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}")
    return height - 135


def _titre_section(c, texte, y, width):
    """Titre de section avec ligne de séparation."""
    c.setFont("Helvetica-Bold", 13)
    c.drawString(40, y, texte)
    c.line(40, y - 5, width - 40, y - 5)
    return y - 25


def _texte(c, texte, x, y, taille=10, gras=False):
    """Texte simple sur le canvas."""
    font = "Helvetica-Bold" if gras else "Helvetica"
    c.setFont(font, taille)
    c.drawString(x, y, texte)


def _nouvelle_page(c, y, limite=120):
    """Crée une nouvelle page si l'espace restant est insuffisant."""
    if y < limite:
        c.showPage()
        return A4[1] - 60
    return y


def _pied_de_page(c, width):
    """Pied de page standard."""
    c.setFont("Helvetica-Oblique", 7)
    c.drawString(
        40, 20,
        "Document généré automatiquement - Aide à l'interprétation des données capteurs, "
        "sans valeur de diagnostic médical."
    )


def _barre_score(c, score, x, y, largeur=120, hauteur=10):
    """
    Dessine une barre de progression colorée pour visualiser un score 0-1.
    Vert >= 0.7, Orange >= 0.4, Rouge < 0.4.
    """
    c.setFillColorRGB(0.85, 0.85, 0.85)
    c.rect(x, y, largeur, hauteur, fill=1, stroke=0)
    if score >= 0.70:
        c.setFillColorRGB(0.18, 0.65, 0.33)
    elif score >= 0.40:
        c.setFillColorRGB(0.95, 0.60, 0.10)
    else:
        c.setFillColorRGB(0.80, 0.15, 0.15)
    c.rect(x, y, largeur * score, hauteur, fill=1, stroke=0)
    c.setFillColorRGB(0, 0, 0)
    c.setStrokeColorRGB(0.5, 0.5, 0.5)
    c.rect(x, y, largeur, hauteur, fill=0, stroke=1)


def _tableau(c, lignes, colonnes_x, y, hauteur_ligne=16, gras_entete=True):
    """
    Dessine un tableau léger (pas de bordures complètes, juste ligne d'en-tête).
    lignes : liste de tuples, première ligne = en-tête.
    colonnes_x : liste des positions x de chaque colonne.
    """
    if gras_entete and lignes:
        c.setFont("Helvetica-Bold", 9)
        for i, cel in enumerate(lignes[0]):
            c.drawString(colonnes_x[i], y, cel)
        y -= 4
        c.line(40, y, colonnes_x[-1] + 120, y)
        y -= hauteur_ligne - 4
        lignes = lignes[1:]
    c.setFont("Helvetica", 8.5)
    for row in lignes:
        for i, cel in enumerate(row):
            c.drawString(colonnes_x[i], y, str(cel))
        y -= hauteur_ligne
    return y


# ============================================================
# CONFIGURATIONS
# ============================================================

@dataclass
class SensorCalibrationReference:
    """
    Référence de calibration initiale par capteur.
    sigma_ref_par_capteur[i] est l'écart-type de bruit mesuré lors
    de la mise en service du capteur i (en kPa, sur signal brut non filtré).
    Doit être établi sur une fenêtre stable de N échantillons sans appui variable.
    """
    sigma_ref_par_capteur: npt.NDArray   # shape (n_capteurs,), unité kPa
    n_capteurs: int = 0

    def __post_init__(self):
        self.sigma_ref_par_capteur = np.asarray(
            self.sigma_ref_par_capteur, dtype=float
        )
        self.n_capteurs = len(self.sigma_ref_par_capteur)

    @classmethod
    def depuis_defaut(cls, n_capteurs: int, sigma_ref_kpa: float = 5.0):
        """
        Crée une référence par défaut si la calibration individuelle
        n'est pas disponible. sigma_ref_kpa = valeur typique littérature
        pour FSR en appui statique (3–8 kPa).
        """
        return cls(
            sigma_ref_par_capteur=np.full(n_capteurs, sigma_ref_kpa)
        )


@dataclass
class SensorQualityConfig:
    """
    Configuration du Bloc 0 : Surveillance qualité capteur.
    Conforme au doc : w1 = 0.5, w2 = 0.5, alpha_bruit = 2.0.
    """
    alpha_bruit: float = 2.0                    # coefficient de tolérance bruit (doc : 2.0)
    w1_bruit: float = 0.5                       # poids score bruit  (doc : w1 = 0.5)
    w2_coherence: float = 0.5                   # poids score cohérence (doc : w2 = 0.5)
    delta_temp_seuil_symetrie: float = 2.2      # seuil asymétrie thermique IWGDF (°C)
    pente_fsr_seuil_drift: float = 0.8          # pente FSR inter-fenêtres (kPa/fenêtre)
    n_fenetres_pente: int = 10                  # nb de fenêtres pour le calcul de pente
    sc_zone_seuil_bloquage: float = 0.4         # SC_z < 0.4 → blocage mise à jour
    sc_zone_seuil_reduction: float = 0.7        # SC_z < 0.7 → poids réduit


@dataclass
class PreprocessingConfig:
    """Configuration du Bloc 1 : Acquisition et prétraitement."""
    fsr_a_coeffs: npt.NDArray    # coefficients a_i par capteur (loi puissance)
    fsr_b_coeffs: npt.NDArray    # coefficients b_i par capteur (loi puissance)
    ntc_a: float                 # coefficient A de Steinhart-Hart
    ntc_b: float                 # coefficient B de Steinhart-Hart
    ntc_c: float                 # coefficient C de Steinhart-Hart
    butter_order: int = 2
    cutoff_freq_hz: float = 20.0
    sampling_rate_hz: float = 50.0
    vcc: float = 3.3             # tension d'alimentation (V)
    pullup_resistor: float = 10000.0  # résistance de référence du pont diviseur (Ω)


# ============================================================
# BLOC 1a — CONVERSION BRUTE (tensions → kPa, SANS filtrage)
# ============================================================

class SignalConverter:
    """
    Bloc 1a : Conversion des signaux bruts en grandeurs physiques.
    Opère sur les signaux RAW (non filtrés) pour permettre au Bloc 0
    d'évaluer le bruit réel du capteur avant filtrage.
    """

    def __init__(self, config: PreprocessingConfig):
        self.config = config

    def fsr_to_pressure_raw(
        self,
        voltages: npt.NDArray,
        capteur_id: int
    ) -> npt.NDArray:
        """
        Conversion tension → pression via pont diviseur + loi puissance.
        R_FSR(k) = R_ref × (Vcc / Vout(k) − 1)
        P_i(k)   = a_i × R_FSR(k)^b_i
        """
        vcc = self.config.vcc
        r_ref = self.config.pullup_resistor
        # Évite division par zéro
        resistance = r_ref * (vcc / (voltages + 1e-9) - 1.0)
        resistance = np.clip(resistance, 1.0, None)  # résistance > 0
        a = self.config.fsr_a_coeffs[capteur_id]
        b = self.config.fsr_b_coeffs[capteur_id]
        return a * (resistance ** b)

    def ntc_to_temperature(self, resistances: npt.NDArray) -> npt.NDArray:
        """
        Conversion résistance → température (°C) via Steinhart-Hart.
        1/T = A + B·ln(R) + C·(ln(R))³
        T(°C) = T(K) − 273.15
        """
        log_r = np.log(np.clip(resistances, 1.0, None))
        temp_k = 1.0 / (
            self.config.ntc_a
            + self.config.ntc_b * log_r
            + self.config.ntc_c * (log_r ** 3)
        )
        return temp_k - 273.15

    def run_bloc1a(
        self,
        raw_voltages_fsr: npt.NDArray,
        raw_resistances_ntc: npt.NDArray
    ) -> Dict[str, Any]:
        """
        Fonction principale du Bloc 1a.
        Retourne les pressions brutes (non filtrées) et les températures.
        """
        n_samples, n_fsr = raw_voltages_fsr.shape
        pressures_raw_kpa = np.zeros((n_samples, n_fsr))

        for i in range(n_fsr):
            pressures_raw_kpa[:, i] = self.fsr_to_pressure_raw(
                raw_voltages_fsr[:, i], i
            )

        temperatures_c = self.ntc_to_temperature(raw_resistances_ntc)

        return {
            "pressures_raw_kpa": pressures_raw_kpa,   # non filtrées → pour Bloc 0
            "temperatures_c": temperatures_c,
            "bloc1a_ok": True,
        }


# ============================================================
# BLOC 0 — SURVEILLANCE QUALITÉ CAPTEUR
# Opère sur les pressions BRUTES (sortie Bloc 1a)
# ============================================================

class SensorQualityMonitor:
    """
    Bloc 0 : Détection de dérives capteurs et calcul des scores de confiance.
    Travaille sur les pressions brutes non filtrées pour capter le vrai bruit FSR.
    Conforme au doc : f_bruit = max(0, 1 − σ_bruit,i / (α_bruit × σ_ref,i))
                      SC_i = w1 × f_bruit + w2 × f_coh   (w1=w2=0.5)
    Maintient un historique inter-fenêtres pour la tendance FSR (n_fenetres_pente).
    """

    def __init__(
        self,
        config: SensorQualityConfig,
        calibration_ref: SensorCalibrationReference
    ):
        self.config = config
        self.calibration_ref = calibration_ref
        # Historique des pressions moyennes par capteur sur les N dernières fenêtres
        # Utilisé pour calculer la tendance FSR inter-fenêtres (doc 0.2)
        self._historique_moyennes: List[npt.NDArray] = []

    def compute_fsr_noise(
        self,
        pressures_raw_kpa: npt.NDArray,
        capteur_id: int
    ) -> float:
        """
        Calcule l'écart-type de bruit du signal FSR brut sur la fenêtre.
        σ_bruit,i = std(P_i(k), k=1..N)
        """
        signal = pressures_raw_kpa[:, capteur_id]
        return float(np.std(signal, ddof=1))

    def compute_f_bruit(self, sigma_bruit: float, sigma_ref: float) -> float:
        """
        Normalisation bruit (doc) :
        f_bruit = max(0, 1 − σ_bruit,i / (α_bruit × σ_ref,i))
        """
        denom = self.config.alpha_bruit * sigma_ref
        if denom <= 0:
            return 0.0
        return float(max(0.0, 1.0 - sigma_bruit / denom))

    def compute_tendance_fsr_intercapteur(self, capteur_id: int) -> float:
        """
        Calcule la pente FSR inter-fenêtres sur les n_fenetres_pente dernières fenêtres.
        Pente positive et soutenue → dérive capteur probable.
        """
        n = self.config.n_fenetres_pente
        if len(self._historique_moyennes) < 2:
            return 0.0
        historique = self._historique_moyennes[-n:]
        moyennes = np.array([h[capteur_id] for h in historique])
        if len(moyennes) < 2:
            return 0.0
        # Régression linéaire simple pour la pente
        x = np.arange(len(moyennes), dtype=float)
        pente = float(np.polyfit(x, moyennes, 1)[0])
        return pente

    def compute_cross_coherence(
        self,
        pente_fsr: float,
        delta_temp_sym: float
    ) -> Dict[str, Any]:
        """
        Cohérence croisée FSR × NTC (doc section 0.2).
        Vecteur d'état v_z = (ΔP_z > ε_P, ΔT_z > ε_T) simplifié ici
        par la comparaison de la pente FSR et de l'asymétrie thermique.

        Règles de décision :
          - ΔT_sym > 2.2°C et pente FSR > seuil → dérive capteur
          - ΔT_sym > 2.2°C et pente FSR ≤ seuil → évolution clinique
          - ΔT_sym ≤ 2.2°C → normal
        """
        if delta_temp_sym > self.config.delta_temp_seuil_symetrie:
            if pente_fsr > self.config.pente_fsr_seuil_drift:
                return {"diagnostic": "dérive_capteur", "f_coh": 0.2}
            else:
                return {"diagnostic": "évolution_clinique", "f_coh": 0.8}
        return {"diagnostic": "normal", "f_coh": 1.0}

    def compute_sensor_confidence(
        self,
        f_bruit: float,
        f_coh: float
    ) -> float:
        """
        Score de confiance combiné (doc) :
        SC_i = w1 × f_bruit(σ_bruit,i) + w2 × f_coh(D_z)
        avec w1 = w2 = 0.5
        """
        sc = self.config.w1_bruit * f_bruit + self.config.w2_coherence * f_coh
        return float(np.clip(sc, 0.0, 1.0))

    def run_bloc0(
        self,
        pressures_raw_kpa: npt.NDArray,
        temperatures_c: npt.NDArray,
        temperatures_c_symetrique: npt.NDArray,
        sensor_zones: npt.NDArray
    ) -> Dict[str, Any]:
        """
        Fonction principale du Bloc 0.

        Entrées :
            pressures_raw_kpa         : pressions brutes non filtrées (n_samples, n_capteurs)
            temperatures_c            : températures converties de ce pied (n_samples, n_ntc) ou vecteur
            temperatures_c_symetrique : températures du pied symétrique pour comparaison
            sensor_zones              : zone anatomique de chaque capteur

        Sortie :
            result contenant sensor_confidences, zone_confidences, actions, bloc0_ok
        """
        n_capteurs = pressures_raw_kpa.shape[1]

        # Mise à jour de l'historique inter-fenêtres
        moyennes_fenetre = np.mean(pressures_raw_kpa, axis=0)
        self._historique_moyennes.append(moyennes_fenetre)
        if len(self._historique_moyennes) > self.config.n_fenetres_pente * 2:
            self._historique_moyennes = self._historique_moyennes[
                -self.config.n_fenetres_pente:
            ]

        # Asymétrie thermique inter-pieds (scalaire moyen)
        t_moy = float(np.mean(temperatures_c))
        t_sym = float(np.mean(temperatures_c_symetrique))
        delta_temp_sym = abs(t_moy - t_sym)

        sensor_confidences = []
        zone_confidences: Dict[str, float] = {}
        zones_unique = np.unique(sensor_zones)

        for zone in zones_unique:
            mask = sensor_zones == zone
            capteurs_zone_idx = np.where(mask)[0]
            confidences_zone = []

            for idx in capteurs_zone_idx:
                sigma_bruit = self.compute_fsr_noise(pressures_raw_kpa, idx)
                sigma_ref = float(
                    self.calibration_ref.sigma_ref_par_capteur[idx]
                )

                f_bruit = self.compute_f_bruit(sigma_bruit, sigma_ref)

                pente_fsr = self.compute_tendance_fsr_intercapteur(idx)
                coherence = self.compute_cross_coherence(pente_fsr, delta_temp_sym)
                f_coh = coherence["f_coh"]

                sc = self.compute_sensor_confidence(f_bruit, f_coh)
                sensor_confidences.append(sc)
                confidences_zone.append(sc)

            zone_confidences[str(zone)] = (
                float(np.mean(confidences_zone)) if confidences_zone else 1.0
            )

        # Décisions par zone (doc section 0.3)
        actions = {}
        for zone, sc_z in zone_confidences.items():
            if sc_z < self.config.sc_zone_seuil_bloquage:
                actions[zone] = {
                    "action": "bloquer_mise_a_jour",
                    "alerte": "forte",
                    "score": round(sc_z, 3)
                }
            elif sc_z < self.config.sc_zone_seuil_reduction:
                actions[zone] = {
                    "action": "reduire_poids",
                    "alerte": "moderee",
                    "score": round(sc_z, 3)
                }
            else:
                actions[zone] = {
                    "action": "normal",
                    "alerte": "aucune",
                    "score": round(sc_z, 3)
                }

        return {
            "sensor_confidences": [round(s, 3) for s in sensor_confidences],
            "zone_confidences": zone_confidences,
            "actions": actions,
            "delta_temp_sym": delta_temp_sym,
            "bloc0_ok": True,
        }


def generer_pdf_bloc0(result, logo_path="image.png", filename=None):
    """
    Génère un rapport PDF cliniquement exploitable pour le Bloc 0.

    Sections :
      1. Résumé décisionnel global
      2. Explication simplifiée (pour le clinicien / patient)
      3. Scores de confiance par capteur (avec barres visuelles)
      4. Synthèse par zone anatomique + décision
      5. Détail des alertes et causes probables
      6. Données transmises aux blocs suivants
    """
    try:
        nom_fichier = filename if filename else f"Rapport_Bloc0_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
        c = rl_canvas.Canvas(nom_fichier, pagesize=A4)
        width, height = A4

        actions = result["actions"]
        zone_confidences = result["zone_confidences"]
        sensor_confidences = result["sensor_confidences"]

        scores = list(zone_confidences.values())
        score_global = float(np.mean(scores)) if scores else 0.0
        alertes = [v["alerte"] for v in actions.values()]
        n_bloquees = sum(1 for a in alertes if a == "forte")
        n_moderees = sum(1 for a in alertes if a == "moderee")

        if n_bloquees > 0:
            decision = "ALERTE CRITIQUE — MISE À JOUR BLOQUÉE"
            conclusion = (
                f"{n_bloquees} zone(s) bloquée(s). "
                "Les données de ces zones ne doivent pas alimenter les blocs suivants."
            )
        elif n_moderees > 0:
            decision = "ALERTE MODÉRÉE — POIDS RÉDUIT"
            conclusion = (
                f"{n_moderees} zone(s) en surveillance. "
                "Les données sont utilisées avec un poids diminué dans les calculs."
            )
        else:
            decision = "QUALITÉ CAPTEUR NORMALE"
            conclusion = "Tous les capteurs présentent une qualité suffisante. Aucune action requise."

        def interp_score(sc):
            if sc >= 0.70:
                return "Bon"
            elif sc >= 0.40:
                return "Moyen — surveillance recommandée"
            else:
                return "Faible — alerte"

        def interp_action(action):
            mapping = {
                "bloquer_mise_a_jour": "Données exclues du calcul",
                "reduire_poids": "Données pondérées à la baisse",
                "normal": "Données intégrées normalement",
            }
            return mapping.get(action, action)

        y = _entete(c, width, height, "Rapport Digi'Feet — Bloc 0",
                    "Surveillance Qualité Capteur & Score de Confiance", logo_path)

        y = _titre_section(c, "1. Résumé décisionnel global", y, width)
        _texte(c, f"Décision : {decision}", 55, y, 11, gras=True)
        y -= 18
        _texte(c, f"Score de confiance global : {score_global:.2f} / 1.00", 55, y, 11)
        y -= 16
        _texte(c, f"Conclusion : {conclusion}", 55, y, 10)
        y -= 16
        _texte(c, f"Nombre de zones analysées : {len(zone_confidences)}", 55, y, 10)
        y -= 16
        _texte(c, f"Nombre de capteurs analysés : {len(sensor_confidences)}", 55, y, 10)
        y -= 30

        y = _titre_section(c, "2. Explication simplifiée", y, width)
        lignes_expl = [
            "Ce bloc évalue la fiabilité de chaque capteur FSR (pression) et NTC (température).",
            "Il calcule un score de confiance (0 à 1) en combinant à poids égaux (50/50) :",
            "  • le niveau de bruit du signal par rapport à la référence de calibration individuelle,",
            "  • la cohérence entre la pente FSR et l'asymétrie thermique inter-pieds.",
            "Un score faible signale une dérive probable ; les données concernées",
            "sont alors exclues ou pondérées pour ne pas fausser les blocs suivants.",
        ]
        c.setFont("Helvetica", 10)
        for ligne in lignes_expl:
            c.drawString(55, y, ligne)
            y -= 15
        y -= 15

        y = _nouvelle_page(c, y, limite=200)
        y = _titre_section(c, "3. Scores de confiance par capteur", y, width)

        entete_capt = ["N°", "Score", "Visualisation (0 → 1)", "Interprétation"]
        col_x_capt = [55, 135, 185, 355]

        c.setFont("Helvetica-Bold", 9)
        for i, h in enumerate(entete_capt):
            c.drawString(col_x_capt[i], y, h)
        y -= 4
        c.line(40, y, width - 40, y)
        y -= 14

        c.setFont("Helvetica", 8.5)
        for idx, sc in enumerate(sensor_confidences):
            if y < 80:
                _pied_de_page(c, width)
                c.showPage()
                y = height - 60
            c.drawString(col_x_capt[0], y, f"Capteur {idx + 1}")
            c.drawString(col_x_capt[1], y, f"{sc:.3f}")
            _barre_score(c, sc, col_x_capt[2], y - 2, largeur=150, hauteur=10)
            c.drawString(col_x_capt[3], y, interp_score(sc))
            y -= 18
        y -= 15

        y = _nouvelle_page(c, y, limite=160)
        y = _titre_section(c, "4. Synthèse par zone anatomique", y, width)

        entete_zone = ["Zone", "Score moyen", "Visualisation", "Décision", "Alerte"]
        col_x_zone = [55, 130, 175, 340, 500]

        c.setFont("Helvetica-Bold", 9)
        for i, h in enumerate(entete_zone):
            c.drawString(col_x_zone[i], y, h)
        y -= 4
        c.line(40, y, width - 40, y)
        y -= 14

        c.setFont("Helvetica", 8.5)
        for zone, sc in zone_confidences.items():
            if y < 80:
                _pied_de_page(c, width)
                c.showPage()
                y = height - 60
            act = actions[zone]
            c.drawString(col_x_zone[0], y, zone.replace("_", " ").capitalize())
            c.drawString(col_x_zone[1], y, f"{sc:.3f}")
            _barre_score(c, sc, col_x_zone[2], y - 2, largeur=150, hauteur=10)
            c.drawString(col_x_zone[3], y, interp_action(act["action"]))
            alerte = act["alerte"]
            if alerte == "forte":
                c.setFillColorRGB(0.80, 0.10, 0.10)
            elif alerte == "moderee":
                c.setFillColorRGB(0.90, 0.50, 0.05)
            else:
                c.setFillColorRGB(0.10, 0.55, 0.20)
            c.drawString(col_x_zone[4], y, alerte.capitalize())
            c.setFillColorRGB(0, 0, 0)
            y -= 18
        y -= 15

        y = _nouvelle_page(c, y, limite=140)
        y = _titre_section(c, "5. Détail des alertes et causes probables", y, width)

        zones_alertees = {z: v for z, v in actions.items() if v["alerte"] != "aucune"}

        if not zones_alertees:
            _texte(c, "Aucune alerte détectée. Tous les capteurs sont dans les seuils normaux.", 55, y, 10)
            y -= 20
        else:
            c.setFont("Helvetica", 10)
            for zone, act in zones_alertees.items():
                if y < 80:
                    _pied_de_page(c, width)
                    c.showPage()
                    y = height - 60
                sc_zone = zone_confidences[zone]
                _texte(c, f"Zone {zone.replace('_', ' ').capitalize()} "
                           f"(score : {sc_zone:.3f}) :", 55, y, 10, gras=True)
                y -= 15
                if act["alerte"] == "forte":
                    cause = (
                        "Score < 0.40. Causes probables : bruit FSR excessif par rapport "
                        "à la référence de calibration, dérive thermique marquée, ou capteur défaillant."
                    )
                else:
                    cause = (
                        "Score 0.40–0.70. Causes probables : bruit FSR modéré, asymétrie "
                        "thermique légère, ou début de dérive. Surveillance recommandée."
                    )
                if len(cause) > 90:
                    c.drawString(65, y, cause[:90])
                    y -= 13
                    c.drawString(65, y, cause[90:])
                else:
                    c.drawString(65, y, cause)
                y -= 20

        y = _nouvelle_page(c, y, limite=120)
        y = _titre_section(c, "6. Données transmises aux blocs suivants", y, width)

        _texte(c, "Les informations ci-dessous sont passées aux Blocs 1b, 2, 3, 4 :", 55, y, 10)
        y -= 18

        transmissions = [
            ("zone_confidences",          "Score moyen de confiance par zone (pondération)"),
            ("actions[zone]['action']",   "Décision par zone : bloquer / réduire_poids / normal"),
            ("sensor_confidences",        "Score individuel de chaque capteur (traçabilité)"),
        ]

        c.setFont("Helvetica", 9)
        for cle, desc in transmissions:
            c.setFont("Helvetica-Bold", 9)
            c.drawString(55, y, f"• {cle}")
            c.setFont("Helvetica", 9)
            c.drawString(255, y, f"→ {desc}")
            y -= 16

        _pied_de_page(c, width)
        c.save()
        print(f"[SUCCESS] PDF Bloc 0 généré : {nom_fichier}")

    except Exception as e:
        print(f"[ERROR] Impossible de générer le PDF Bloc 0 : {e}")
        raise


# ============================================================
# BLOC 1b — FILTRAGE BUTTERWORTH
# Opère sur les pressions brutes converties par le Bloc 1a,
# APRÈS validation qualité par le Bloc 0.
# ============================================================

class SignalFilter:
    """
    Bloc 1b : Filtrage passe-bas Butterworth d'ordre 2 à 20 Hz.
    Applique le filtre sur les pressions brutes converties.
    Les capteurs dont la zone est bloquée par le Bloc 0 sont mis à zéro.
    """

    def __init__(self, config: PreprocessingConfig):
        self.config = config
        b, a = butter(
            config.butter_order,
            config.cutoff_freq_hz / (config.sampling_rate_hz / 2),
            btype='low',
            analog=False
        )
        self._b = b
        self._a = a

    def apply_lowpass(self, signal: npt.NDArray) -> npt.NDArray:
        """Filtre passe-bas Butterworth (filtfilt → phase nulle)."""
        return filtfilt(self._b, self._a, signal)

    def run_bloc1b(
        self,
        pressures_raw_kpa: npt.NDArray,
        result_bloc0: Dict[str, Any],
        sensor_zones: npt.NDArray
    ) -> Dict[str, Any]:
        """
        Fonction principale du Bloc 1b.

        Entrées :
            pressures_raw_kpa : pressions brutes converties (sortie Bloc 1a)
            result_bloc0      : résultat du Bloc 0 (actions par zone)
            sensor_zones      : zone anatomique de chaque capteur

        Sortie :
            pressures_kpa     : pressions filtrées, capteurs bloqués mis à 0
            bloc1b_ok         : booléen
        """
        n_samples, n_capteurs = pressures_raw_kpa.shape
        pressures_kpa = np.zeros_like(pressures_raw_kpa)

        actions = result_bloc0.get("actions", {})

        for i in range(n_capteurs):
            zone = str(sensor_zones[i])
            action = actions.get(zone, {}).get("action", "normal")

            if action == "bloquer_mise_a_jour":
                # Capteur bloqué : on transmet zéro (sera ignoré en aval)
                pressures_kpa[:, i] = 0.0
            else:
                pressures_kpa[:, i] = self.apply_lowpass(pressures_raw_kpa[:, i])

        return {
            "pressures_kpa": pressures_kpa,
            "bloc1b_ok": True,
        }


def generer_pdf_bloc1(result, logo_path="image.png", filename=None):
    """
    Génère un rapport PDF cliniquement exploitable pour le Bloc 1.

    Sections :
      1. Résumé du prétraitement
      2. Explication simplifiée (clinicien / patient)
      3. Statistiques détaillées des signaux FSR (pression)
      4. Statistiques détaillées des signaux NTC (température)
      5. Vérification de cohérence des données
      6. Données transmises aux blocs suivants
    """
    try:
        nom_fichier = filename if filename else f"Rapport_Bloc1_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
        c = rl_canvas.Canvas(nom_fichier, pagesize=A4)
        width, height = A4

        pressures_kpa = result["pressures_kpa"]
        temperatures_c = result["temperatures_c"]

        n_samples, n_capteurs = pressures_kpa.shape
        n_ntc = temperatures_c.shape[1] if temperatures_c.ndim == 2 else 1
        duree_estimee_s = n_samples / 50.0

        p_mean_par_capteur = np.mean(pressures_kpa, axis=0)
        p_std_par_capteur = np.std(pressures_kpa, axis=0)
        p_min_par_capteur = np.min(pressures_kpa, axis=0)
        p_max_par_capteur = np.max(pressures_kpa, axis=0)

        p_global_mean = float(np.mean(pressures_kpa))
        p_global_std = float(np.std(pressures_kpa))
        p_global_min = float(np.min(pressures_kpa))
        p_global_max = float(np.max(pressures_kpa))

        if temperatures_c.ndim == 2:
            t_mean_par_sonde = np.mean(temperatures_c, axis=0)
            t_std_par_sonde = np.std(temperatures_c, axis=0)
            t_min_par_sonde = np.min(temperatures_c, axis=0)
            t_max_par_sonde = np.max(temperatures_c, axis=0)
        else:
            t_mean_par_sonde = np.array([float(np.mean(temperatures_c))])
            t_std_par_sonde = np.array([float(np.std(temperatures_c))])
            t_min_par_sonde = np.array([float(np.min(temperatures_c))])
            t_max_par_sonde = np.array([float(np.max(temperatures_c))])

        t_global_mean = float(np.mean(temperatures_c))
        t_asymetrie = float(np.max(t_mean_par_sonde) - np.min(t_mean_par_sonde))

        alertes_coherence = []
        if p_global_max > 600:
            alertes_coherence.append(
                f"Pression max ({p_global_max:.1f} kPa) dépasse 600 kPa — vérifier l'étalonnage FSR"
            )
        if p_global_min < -5:
            alertes_coherence.append(
                f"Pression min ({p_global_min:.1f} kPa) négative — vérifier la tension d'alimentation"
            )
        if t_global_mean < 20 or t_global_mean > 40:
            alertes_coherence.append(
                f"Température moyenne ({t_global_mean:.1f} °C) hors plage normale (20–40 °C)"
            )
        if t_asymetrie > 3.0:
            alertes_coherence.append(
                f"Asymétrie thermique entre sondes ({t_asymetrie:.1f} °C) > 3 °C"
            )
        bruit_moyen_kpa = float(np.mean(p_std_par_capteur))
        if bruit_moyen_kpa > 20:
            alertes_coherence.append(
                f"Bruit FSR moyen élevé ({bruit_moyen_kpa:.1f} kPa) — filtrage insuffisant"
            )
        elif bruit_moyen_kpa > 15:
            alertes_coherence.append(
                f"Bruit FSR moyen modéré ({bruit_moyen_kpa:.1f} kPa) — surveiller la stabilité"
            )

        statut_coherence = "OK" if not alertes_coherence else f"{len(alertes_coherence)} point(s) à vérifier"

        def qualite_filtrage(bruit):
            if bruit < 5:
                return "Excellent"
            elif bruit < 10:
                return "Bon"
            elif bruit < 15:
                return "Acceptable"
            else:
                return "À surveiller"

        y = _entete(c, width, height, "Rapport Digi'Feet : Bloc 1",
                    "Acquisition et Prétraitement des Signaux", logo_path)

        y = _titre_section(c, "1. Résumé du prétraitement", y, width)
        _texte(c, "Statut : Signal prétraité avec succès (1a conversion + Bloc 0 + 1b filtrage)", 55, y, 11, gras=True)
        y -= 18
        _texte(c, f"Nombre d'échantillons traités : {n_samples}  ({duree_estimee_s:.1f} s à 50 Hz)", 55, y, 10)
        y -= 16
        _texte(c, f"Capteurs FSR : {n_capteurs}  /  Sondes NTC : {n_ntc}", 55, y, 10)
        y -= 16
        _texte(c, f"Cohérence des données : {statut_coherence}", 55, y, 10, gras=(len(alertes_coherence) > 0))
        y -= 30

        y = _titre_section(c, "2. Explication simplifiée", y, width)
        lignes_expl = [
            "Ce bloc transforme les signaux bruts de la semelle en grandeurs physiques exploitables :",
            "  • Bloc 1a : tensions FSR → pression (kPa) via pont diviseur et loi puissance.",
            "  • Bloc 0 intercalé : évaluation qualité sur signal brut, calcul SC_z.",
            "  • Bloc 1b : filtrage passe-bas Butterworth 2e ordre à 20 Hz sur les signaux validés.",
            "  • Résistances NTC → température (°C) via équation de Steinhart-Hart.",
        ]
        c.setFont("Helvetica", 10)
        for ligne in lignes_expl:
            c.drawString(55, y, ligne)
            y -= 15
        y -= 15

        y = _nouvelle_page(c, y, limite=180)
        y = _titre_section(c, "3. Statistiques par capteur FSR (pression, kPa)", y, width)

        entete_fsr = ["Capteur", "Moyenne", "Écart-type", "Min", "Max", "Qualité filtrage"]
        col_x_fsr = [40, 100, 160, 215, 260, 315]

        c.setFont("Helvetica-Bold", 9)
        for i, h in enumerate(entete_fsr):
            c.drawString(col_x_fsr[i], y, h)
        y -= 4
        c.line(40, y, width - 40, y)
        y -= 14

        c.setFont("Helvetica", 8.5)
        for idx in range(n_capteurs):
            if y < 80:
                _pied_de_page(c, width)
                c.showPage()
                y = height - 60
            c.drawString(col_x_fsr[0], y, f"FSR {idx + 1}")
            c.drawString(col_x_fsr[1], y, f"{p_mean_par_capteur[idx]:.2f}")
            c.drawString(col_x_fsr[2], y, f"{p_std_par_capteur[idx]:.2f}")
            c.drawString(col_x_fsr[3], y, f"{p_min_par_capteur[idx]:.2f}")
            c.drawString(col_x_fsr[4], y, f"{p_max_par_capteur[idx]:.2f}")
            c.drawString(col_x_fsr[5], y, qualite_filtrage(p_std_par_capteur[idx]))
            y -= 16

        y -= 5
        c.line(40, y + 12, width - 40, y + 12)
        c.setFont("Helvetica-Bold", 9)
        c.drawString(col_x_fsr[0], y, "Global")
        c.drawString(col_x_fsr[1], y, f"{p_global_mean:.2f}")
        c.drawString(col_x_fsr[2], y, f"{p_global_std:.2f}")
        c.drawString(col_x_fsr[3], y, f"{p_global_min:.2f}")
        c.drawString(col_x_fsr[4], y, f"{p_global_max:.2f}")
        c.drawString(col_x_fsr[5], y, qualite_filtrage(bruit_moyen_kpa))
        y -= 25

        y = _nouvelle_page(c, y, limite=160)
        y = _titre_section(c, "4. Statistiques par sonde NTC (température, °C)", y, width)

        entete_ntc = ["Sonde", "Moyenne", "Écart-type", "Min", "Max", "Interprétation"]
        col_x_ntc = [40, 100, 160, 215, 260, 315]

        c.setFont("Helvetica-Bold", 9)
        for i, h in enumerate(entete_ntc):
            c.drawString(col_x_ntc[i], y, h)
        y -= 4
        c.line(40, y, width - 40, y)
        y -= 14

        c.setFont("Helvetica", 8.5)
        for idx in range(len(t_mean_par_sonde)):
            if y < 80:
                _pied_de_page(c, width)
                c.showPage()
                y = height - 60
            t_m = t_mean_par_sonde[idx]
            if t_m < 20:
                interp_ntc = "Trop basse : vérifier capteur"
            elif t_m < 25:
                interp_ntc = "Froide : pied froid ou environnement frais"
            elif t_m < 33:
                interp_ntc = "Normale"
            elif t_m < 37:
                interp_ntc = "Chaude : effort ou inflammation légère"
            else:
                interp_ntc = "Élevée : vérifier (inflammation ?)"
            c.drawString(col_x_ntc[0], y, f"NTC {idx + 1}")
            c.drawString(col_x_ntc[1], y, f"{t_m:.2f}")
            c.drawString(col_x_ntc[2], y, f"{t_std_par_sonde[idx]:.2f}")
            c.drawString(col_x_ntc[3], y, f"{t_min_par_sonde[idx]:.2f}")
            c.drawString(col_x_ntc[4], y, f"{t_max_par_sonde[idx]:.2f}")
            c.drawString(col_x_ntc[5], y, interp_ntc)
            y -= 16

        y -= 10
        _texte(c, f"Asymétrie thermique inter-sondes : {t_asymetrie:.2f} °C", 55, y, 9,
               gras=(t_asymetrie > 3.0))
        y -= 25

        y = _nouvelle_page(c, y, limite=140)
        y = _titre_section(c, "5. Vérification de cohérence des données", y, width)

        checks = [
            ("Plage pression FSR [0 - 600 kPa]", p_global_max <= 600 and p_global_min >= -5),
            ("Température plantaire [20 - 40 °C]", 20 <= t_global_mean <= 40),
            ("Asymétrie thermique < 3 °C", t_asymetrie < 3.0),
            ("Bruit FSR résiduel < 15 kPa", bruit_moyen_kpa < 15),
        ]

        if not alertes_coherence:
            _texte(c, "Aucune anomalie détectée. Données cohérentes avec une acquisition normale.", 55, y, 10)
            y -= 18

        c.setFont("Helvetica", 9)
        for nom_check, ok in checks:
            symbole = "OK" if ok else "!!"
            if ok:
                c.setFillColorRGB(0.10, 0.55, 0.20)
            else:
                c.setFillColorRGB(0.80, 0.10, 0.10)
            c.drawString(55, y, symbole)
            c.setFillColorRGB(0, 0, 0)
            c.drawString(75, y, nom_check)
            y -= 15

        if alertes_coherence:
            y -= 5
            c.setFont("Helvetica-Bold", 9)
            c.drawString(55, y, "Points à vérifier :")
            y -= 14
            c.setFont("Helvetica", 9)
            for alerte in alertes_coherence:
                if y < 80:
                    _pied_de_page(c, width)
                    c.showPage()
                    y = height - 60
                if len(alerte) > 95:
                    c.drawString(65, y, "• " + alerte[:93])
                    y -= 12
                    c.drawString(75, y, alerte[93:])
                else:
                    c.drawString(65, y, "• " + alerte)
                y -= 15
        y -= 15

        y = _nouvelle_page(c, y, limite=120)
        y = _titre_section(c, "6. Données transmises aux blocs suivants", y, width)

        _texte(c, "Les variables ci-dessous sont passées aux Blocs 2, 3, 4 :", 55, y, 10)
        y -= 18

        transmissions = [
            ("pressures_kpa",
             f"Matrice ({n_samples} × {n_capteurs}) : pressions filtrées en kPa"),
            ("temperatures_c",
             f"Matrice ({n_samples} × {n_ntc}) : températures converties en °C"),
            ("bloc1b_ok",
             "Booléen : True si le prétraitement s'est terminé sans erreur"),
        ]

        c.setFont("Helvetica", 9)
        for cle, desc in transmissions:
            c.setFont("Helvetica-Bold", 9)
            c.drawString(55, y, f"• {cle}")
            c.setFont("Helvetica", 9)
            c.drawString(200, y, f"→ {desc}")
            y -= 16

        _pied_de_page(c, width)
        c.save()
        print(f"[SUCCESS] PDF Bloc 1 généré : {nom_fichier}")

    except Exception as e:
        print(f"[ERROR] Impossible de générer le PDF Bloc 1 : {e}")
        raise


# ============================================================
# BLOC 2 — CONTRÔLE DE QUALITÉ POSTURAL
# ============================================================
# Modification minimale : ajout de sensor_confidence_by_zone
# en entrée de bloc2_postural_quality_control pour que le résultat
# transmis aux blocs 3 et 4 contienne déjà SC_z prêt à l'emploi.
# Les fonctions internes compute_cop, compute_window_indicators,
# evaluate_postural_stability, slow_update_reference sont inchangées.
# ============================================================

@dataclass
class PosturalConfig:
    """Paramètres réglables du Bloc 2."""
    pression_activation_kpa: float = 5.0
    d_max_absolu_mm: float = 15.0
    delta_surface_max: float = 0.10
    delta_pression_totale_max: float = 0.15
    n_init: int = 5
    lambda_ref: float = 0.98
    w_min: float = 0.30


@dataclass
class PosturalReference:
    """Référence posturale propre au patient."""
    cop_ref: Optional[npt.NDArray] = None
    sigma_cop: Optional[float] = None
    learning_cops: list = field(default_factory=list)

    def is_ready(self) -> bool:
        return self.cop_ref is not None and self.sigma_cop is not None


def compute_cop(pressures_kpa, sensor_positions_mm, activation_threshold_kpa):
    """Calcule le Centre of Pressure pour chaque échantillon temporel."""
    pressures_kpa = np.asarray(pressures_kpa, dtype=float)
    sensor_positions_mm = np.asarray(sensor_positions_mm, dtype=float)
    active_mask = pressures_kpa > activation_threshold_kpa
    active_pressures = np.where(active_mask, pressures_kpa, 0.0)
    total_pressure = np.sum(active_pressures, axis=1)
    cop = np.full((pressures_kpa.shape[0], 2), np.nan)
    valid = total_pressure > 0
    cop[valid, 0] = (
        np.sum(active_pressures[valid] * sensor_positions_mm[:, 0], axis=1)
        / total_pressure[valid]
    )
    cop[valid, 1] = (
        np.sum(active_pressures[valid] * sensor_positions_mm[:, 1], axis=1)
        / total_pressure[valid]
    )
    return cop, active_mask


def compute_window_indicators(pressures_kpa, sensor_positions_mm, sensor_area_cm2, config: PosturalConfig):
    """Calcule les indicateurs posturaux d'une fenêtre de mesure."""
    cop, active_mask = compute_cop(pressures_kpa, sensor_positions_mm, config.pression_activation_kpa)
    cop_mean = np.nanmean(cop, axis=0)
    cop_distances = np.linalg.norm(cop - cop_mean, axis=1)
    d_cop_mean = np.nanmean(cop_distances)
    n_active = np.sum(active_mask, axis=1)
    surface_contact = n_active * sensor_area_cm2
    surface_mean = np.mean(surface_contact)
    surface_ref = surface_mean if surface_mean > 0 else 1e-9
    delta_surface = (np.max(surface_contact) - np.min(surface_contact)) / surface_ref
    pression_totale = np.sum(np.where(active_mask, pressures_kpa, 0.0), axis=1)
    pression_mean = np.mean(pression_totale)
    pression_ref = pression_mean if pression_mean > 0 else 1e-9
    delta_pression_totale = (np.max(pression_totale) - np.min(pression_totale)) / pression_ref
    return {
        "cop": cop,
        "cop_mean": cop_mean,
        "d_cop_mean": d_cop_mean,
        "surface_mean_cm2": surface_mean,
        "delta_surface": delta_surface,
        "pression_totale_mean_kpa": pression_mean,
        "delta_pression_totale": delta_pression_totale,
    }


def update_learning_reference(reference: PosturalReference, cop_mean, config: PosturalConfig):
    """Ajoute une fenêtre à la phase d'apprentissage de la référence posturale."""
    reference.learning_cops.append(np.asarray(cop_mean, dtype=float))
    if len(reference.learning_cops) >= config.n_init:
        cops = np.vstack(reference.learning_cops)
        reference.cop_ref = np.mean(cops, axis=0)
        distances = np.linalg.norm(cops - reference.cop_ref, axis=1)
        sigma = np.std(distances, ddof=1)
        reference.sigma_cop = max(sigma, 1e-6)
    return reference


def evaluate_postural_stability(indicators, reference: PosturalReference, config: PosturalConfig):
    """Décide si une fenêtre est posturalement acceptable."""
    d_cop = indicators["d_cop_mean"]
    delta_surface = indicators["delta_surface"]
    delta_pressure = indicators["delta_pression_totale"]
    absolute_ok = (
        d_cop < config.d_max_absolu_mm
        and delta_surface < config.delta_surface_max
        and delta_pressure < config.delta_pression_totale_max
    )
    if reference.is_ready():
        distance_to_ref = np.linalg.norm(indicators["cop_mean"] - reference.cop_ref)
        patient_ok = distance_to_ref < 2.0 * reference.sigma_cop
        d_cop_norm = distance_to_ref / (2.0 * reference.sigma_cop)
    else:
        patient_ok = True
        d_cop_norm = d_cop / config.d_max_absolu_mm
    surface_norm = delta_surface / config.delta_surface_max
    pressure_norm = delta_pressure / config.delta_pression_totale_max
    w_post = 1.0 - np.mean([
        min(d_cop_norm, 1.0),
        min(surface_norm, 1.0),
        min(pressure_norm, 1.0),
    ])
    w_post = float(np.clip(w_post, 0.0, 1.0))
    accepted = absolute_ok and patient_ok and w_post >= config.w_min
    return {
        "accepted": accepted,
        "w_post": w_post,
        "absolute_ok": absolute_ok,
        "patient_ok": patient_ok,
        "d_cop_norm": d_cop_norm,
    }


def slow_update_reference(reference: PosturalReference, session_cop_mean, config: PosturalConfig):
    """Met à jour lentement la référence posturale après une session valide."""
    if not reference.is_ready():
        return reference
    session_cop_mean = np.asarray(session_cop_mean, dtype=float)
    reference.cop_ref = (
        config.lambda_ref * reference.cop_ref
        + (1.0 - config.lambda_ref) * session_cop_mean
    )
    return reference


def bloc2_postural_quality_control(
    pressures_kpa,
    sensor_positions_mm,
    sensor_area_cm2,
    reference: PosturalReference,
    sensor_confidence_by_zone: Optional[Dict[str, float]] = None,  # ← ajout Bloc 0
    config: Optional[PosturalConfig] = None
):
    """
    Fonction principale du Bloc 2.

    Entrées :
        pressures_kpa            : données de pression filtrées du Bloc 1b
        sensor_positions_mm      : positions x, y des capteurs
        sensor_area_cm2          : surface élémentaire d'un capteur
        reference                : référence posturale du patient
        sensor_confidence_by_zone: scores SC_z du Bloc 0 (transmis aux blocs suivants)

    Sorties :
        result    : indicateurs, décision, w_post, sensor_confidence_by_zone inclus
        reference : éventuellement mise à jour
    """
    if config is None:
        config = PosturalConfig()
    if sensor_confidence_by_zone is None:
        sensor_confidence_by_zone = {}

    indicators = compute_window_indicators(
        pressures_kpa=pressures_kpa,
        sensor_positions_mm=sensor_positions_mm,
        sensor_area_cm2=sensor_area_cm2,
        config=config
    )
    stability = evaluate_postural_stability(
        indicators=indicators,
        reference=reference,
        config=config
    )
    if not reference.is_ready() and stability["absolute_ok"]:
        reference = update_learning_reference(
            reference=reference,
            cop_mean=indicators["cop_mean"],
            config=config
        )

    result = {
        "indicators": indicators,
        "stability": stability,
        "w_post": stability["w_post"],
        "window_accepted": stability["accepted"],
        "reference_ready": reference.is_ready(),
        "cop_reference": reference.cop_ref,
        "sigma_cop": reference.sigma_cop,
        "sensor_confidence_by_zone": sensor_confidence_by_zone,  # ← propagé
    }
    return result, reference


def afficher_resultats_bloc2(result):
    """Affichage simplifié des résultats du Bloc 2."""
    indicators = result["indicators"]
    stability = result["stability"]

    print("\n" + "=" * 55)
    print("              RÉSULTATS DU BLOC 2")
    print("        Contrôle de qualité postural")
    print("=" * 55)

    print("\n1) Décision générale")
    if result["window_accepted"]:
        print("   Fenêtre ACCEPTÉE")
        print("   Les données peuvent être utilisées pour la suite.")
    else:
        print("   Fenêtre REJETÉE")
        print("   Les données ne doivent pas être utilisées pour le calcul du seuil.")

    print("\n2) Score de stabilité posturale")
    print(f"   w_post = {result['w_post']:.2f} / 1.00")
    if result["w_post"] >= 0.70:
        print("   Interprétation : posture stable")
    elif result["w_post"] >= 0.30:
        print("   Interprétation : posture moyenne / à surveiller")
    else:
        print("   Interprétation : posture instable")

    print("\n3) Indicateurs calculés")
    print(f"   CoP moyen X : {indicators['cop_mean'][0]:.2f} mm")
    print(f"   CoP moyen Y : {indicators['cop_mean'][1]:.2f} mm")
    print(f"   Déplacement moyen du CoP : {indicators['d_cop_mean']:.2f} mm")
    print(f"   Surface moyenne de contact : {indicators['surface_mean_cm2']:.2f} cm²")
    print(f"   Variation de surface : {indicators['delta_surface'] * 100:.2f} %")
    print(f"   Pression totale moyenne : {indicators['pression_totale_mean_kpa']:.2f} kPa")
    print(f"   Variation pression totale : {indicators['delta_pression_totale'] * 100:.2f} %")

    print("\n4) Vérification des critères")
    print(f"   Critères absolus respectés : {stability['absolute_ok']}")
    print(f"   Critère patient respecté : {stability['patient_ok']}")

    print("\n5) Référence patient")
    if result["reference_ready"]:
        print("   Référence patient prête.")
        print(f"   CoP référence : {result['cop_reference']}")
        print(f"   Sigma CoP : {result['sigma_cop']:.4f}")
    else:
        print("   Référence patient pas encore prête.")
        print("   Il faut encore plusieurs fenêtres stables pour l'apprentissage.")

    print("\n6) Données utiles pour les blocs suivants")
    print(f"   window_accepted = {result['window_accepted']}")
    print(f"   w_post = {result['w_post']:.3f}")
    print(f"   sensor_confidence_by_zone = {result['sensor_confidence_by_zone']}")

    print("=" * 55 + "\n")


def generer_pdf_bloc2(result, logo_path="logo_act.png", filename=None):
    """
    Génère un PDF structuré et lisible des résultats du Bloc 2.
    Le document est pensé pour être compréhensible à la fois par un professionnel
    et par un patient.
    """
    indicators = result["indicators"]
    stability = result["stability"]

    nom_fichier = filename if filename else f"Rapport_Bloc2_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"

    c = canvas.Canvas(nom_fichier, pagesize=A4)
    width, height = A4

    def titre_section(texte, y):
        c.setFont("Helvetica-Bold", 13)
        c.drawString(40, y, texte)
        c.line(40, y - 5, width - 40, y - 5)
        return y - 25

    def texte_normal(texte, x, y, taille=10):
        c.setFont("Helvetica", taille)
        c.drawString(x, y, texte)

    def nouvelle_page_si_besoin(y, limite=120):
        if y < limite:
            c.showPage()
            return height - 60
        return y

    def bool_oui_non(valeur):
        return "Oui" if valeur else "Non"

    def interpretation_score(w_post):
        if w_post >= 0.70:
            return "Posture stable : mesure de bonne qualité."
        elif w_post >= 0.30:
            return "Posture moyenne : mesure exploitable avec prudence."
        else:
            return "Posture instable : mesure peu fiable."

    def cause_probable_rejet():
        causes = []
        if indicators["d_cop_mean"] >= 15:
            causes.append("déplacement du centre de pression trop important")
        if indicators["delta_surface"] >= 0.10:
            causes.append("variation de surface de contact trop élevée")
        if indicators["delta_pression_totale"] >= 0.15:
            causes.append("variation de pression totale trop élevée")
        if result["w_post"] < 0.30:
            causes.append("score postural trop faible")
        if not causes:
            return "aucune cause principale évidente ; vérifier les données d'entrée"
        return ", ".join(causes)

    if os.path.exists(logo_path):
        c.drawImage(logo_path, 40, height - 95, width=170, preserveAspectRatio=True, mask="auto")

    c.setFont("Helvetica-Bold", 18)
    c.drawString(230, height - 60, "Rapport Digi'Feet — Bloc 2")
    c.setFont("Helvetica", 11)
    c.drawString(230, height - 80, "Contrôle de qualité postural")
    c.drawString(230, height - 98, f"Date : {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}")
    y = height - 135

    y = titre_section("1. Résumé simple", y)
    decision = "ACCEPTÉE" if result["window_accepted"] else "REJETÉE"
    texte_normal(f"Décision : fenêtre {decision}", 55, y, 11)
    y -= 18
    texte_normal(f"Score de stabilité posturale : {result['w_post']:.2f} / 1.00", 55, y, 11)
    y -= 18
    texte_normal(f"Interprétation : {interpretation_score(result['w_post'])}", 55, y, 10)
    y -= 18
    if result["window_accepted"]:
        texte_normal("Conclusion simple : les données peuvent être transmises aux blocs suivants.", 55, y, 10)
    else:
        texte_normal("Conclusion simple : les données ne doivent pas être utilisées pour calculer le seuil.", 55, y, 10)
        y -= 18
        texte_normal(f"Cause probable : {cause_probable_rejet()}.", 55, y, 10)
    y -= 35

    y = titre_section("2. Explication simplifiée", y)
    texte_normal("Cette étape vérifie si la personne est restée suffisamment stable pendant la mesure.", 55, y)
    y -= 16
    texte_normal("Si la posture varie trop, les pressions sous le pied peuvent être faussées.", 55, y)
    y -= 16
    texte_normal("Le résultat permet donc de décider si la mesure est fiable ou non.", 55, y)
    y -= 35

    y = titre_section("3. Indicateurs mesurés", y)
    lignes = [
        ("CoP moyen X", f"{indicators['cop_mean'][0]:.2f} mm", "Position moyenne latérale du centre d'appui"),
        ("CoP moyen Y", f"{indicators['cop_mean'][1]:.2f} mm", "Position moyenne longitudinale du centre d'appui"),
        ("Déplacement moyen du CoP", f"{indicators['d_cop_mean']:.2f} mm", "Oscillation moyenne de la posture"),
        ("Surface moyenne", f"{indicators['surface_mean_cm2']:.2f} cm²", "Surface du pied en contact avec la semelle"),
        ("Variation surface", f"{indicators['delta_surface'] * 100:.2f} %", "Variation de la surface pendant la mesure"),
        ("Pression totale moyenne", f"{indicators['pression_totale_mean_kpa']:.2f} kPa", "Somme moyenne des pressions mesurées"),
        ("Variation pression totale", f"{indicators['delta_pression_totale'] * 100:.2f} %", "Variation globale de l'appui"),
    ]

    c.setFont("Helvetica-Bold", 9)
    c.drawString(55, y, "Paramètre")
    c.drawString(210, y, "Valeur")
    c.drawString(310, y, "Signification")
    y -= 12
    c.line(55, y, width - 55, y)
    y -= 15

    c.setFont("Helvetica", 8.5)
    for nom, valeur, sens in lignes:
        c.drawString(55, y, nom)
        c.drawString(210, y, valeur)
        c.drawString(310, y, sens)
        y -= 17
    y -= 20

    y = titre_section("4. Critères de validation", y)
    criteres = [
        ("Déplacement CoP < 15 mm", indicators["d_cop_mean"] < 15),
        ("Variation surface < 10 %", indicators["delta_surface"] < 0.10),
        ("Variation pression totale < 15 %", indicators["delta_pression_totale"] < 0.15),
        ("Critère patient spécifique", stability["patient_ok"]),
        ("Critères absolus globaux", stability["absolute_ok"]),
    ]
    for nom, ok in criteres:
        texte_normal(f"{nom} : {bool_oui_non(ok)}", 55, y, 10)
        y -= 16
    y -= 20

    y = titre_section("5. Données transmises aux blocs suivants", y)
    texte_normal(f"window_accepted = {result['window_accepted']}", 55, y, 10)
    y -= 16
    texte_normal(f"w_post = {result['w_post']:.3f}", 55, y, 10)
    y -= 16
    texte_normal("Si window_accepted vaut False, la fenêtre doit être rejetée.", 55, y, 10)
    y -= 16
    texte_normal("Si elle vaut True, w_post peut pondérer son importance dans le calcul du seuil.", 55, y, 10)
    y -= 35

    y = nouvelle_page_si_besoin(y, limite=120)
    y = titre_section("6. Référence patient", y)
    if result["reference_ready"]:
        texte_normal("Référence patient : prête", 55, y, 10)
        y -= 16
        texte_normal(f"CoP référence : {result['cop_reference']}", 55, y, 10)
        y -= 16
        texte_normal(f"Sigma CoP : {result['sigma_cop']:.4f}", 55, y, 10)
    else:
        texte_normal("Référence patient : pas encore prête", 55, y, 10)
        y -= 16
        texte_normal("Plusieurs fenêtres stables sont encore nécessaires pour l'apprentissage.", 55, y, 10)

    c.setFont("Helvetica-Oblique", 7)
    c.drawString(40, 20,
        "Document généré automatiquement — Aide à l'interprétation de la qualité de mesure, sans valeur de diagnostic médical.")

    c.save()
    print(f"PDF généré : {nom_fichier}")


# ============================================================
# BLOC 3 — CALIBRATION INITIALE DU SEUIL PERSONNALISÉ
# Modification mineure : vérification du statut de validation
# avant de retourner le résultat (bloque les zones non fiables).
# ============================================================

@dataclass
class ThresholdConfig:
    """Paramètres réglables du Bloc 3."""
    percentile: float = 90.0
    imc_ref: float = 25.0
    gamma_imc: float = 0.05
    k_mad: float = 1.4826
    pression_activation_kpa: float = 5.0
    min_valid_windows: int = 5


@dataclass
class PatientProfile:
    """Informations morphologiques utilisées pour ajuster le seuil."""
    imc: float
    type_pied: str = "normal"


def weighted_percentile(values, weights, percentile):
    """Calcule un percentile pondéré."""
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    valid = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    values = values[valid]
    weights = weights[valid]
    if len(values) == 0:
        return np.nan
    order = np.argsort(values)
    values_sorted = values[order]
    weights_sorted = weights[order]
    cumulative_weights = np.cumsum(weights_sorted)
    cutoff = percentile / 100.0 * cumulative_weights[-1]
    return float(values_sorted[np.searchsorted(cumulative_weights, cutoff)])


def get_beta_pied(type_pied):
    """Retourne les coefficients anatomiques selon le type de pied."""
    coefficients = {
        "normal":     {"talon": 1.0,  "medio_pied": 1.0,  "avant_pied": 1.0,  "hallux": 1.0},
        "pied_plat":  {"talon": 1.0,  "medio_pied": 0.85, "avant_pied": 1.0,  "hallux": 1.0},
        "pied_creux": {"talon": 1.0,  "medio_pied": 1.0,  "avant_pied": 0.85, "hallux": 0.85},
        "charcot":    {"talon": 0.85, "medio_pied": 0.85, "avant_pied": 0.85, "hallux": 0.85},
    }
    type_pied = type_pied.lower().replace(" ", "_")
    if type_pied not in coefficients:
        raise ValueError("type_pied doit être : normal, pied_plat, pied_creux ou charcot")
    return coefficients[type_pied]


def compute_alpha_imc(imc, config: ThresholdConfig):
    """Calcule le facteur d'ajustement lié à l'IMC."""
    return 1.0 + config.gamma_imc * ((imc - config.imc_ref) / config.imc_ref)


def compute_zone_pressures(pressures_kpa, sensor_zones, config: ThresholdConfig):
    """Calcule la pression moyenne représentative de chaque zone anatomique."""
    pressures_kpa = np.asarray(pressures_kpa, dtype=float)
    sensor_zones = np.asarray(sensor_zones)
    zones = np.unique(sensor_zones)
    zone_pressures = {}
    for zone in zones:
        zone_mask = sensor_zones == zone
        zone_data = pressures_kpa[:, zone_mask]
        active_data = np.where(zone_data > config.pression_activation_kpa, zone_data, np.nan)
        zone_mean_time = np.nanmean(active_data, axis=1)
        zone_pressure = np.nanmean(zone_mean_time)
        zone_pressures[str(zone)] = float(zone_pressure)
    return zone_pressures


def compute_mad(values, config: ThresholdConfig):
    """Calcule la dispersion robuste MAD."""
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return np.nan
    median = np.median(values)
    mad = np.median(np.abs(values - median))
    return float(config.k_mad * mad)


def validate_threshold_with_literature(threshold_value, zone, alpha_imc, beta_zone):
    """Compare le seuil personnalisé aux bornes de référence littérature."""
    literature_bounds = {
        "talon":      (150.0, 350.0),
        "medio_pied": (30.0,  100.0),
        "avant_pied": (150.0, 400.0),
        "hallux":     (100.0, 300.0),
    }
    if zone not in literature_bounds:
        return {"status": "zone inconnue", "confidence": 0.0, "bounds_adjusted": None}
    lower, upper = literature_bounds[zone]
    lower_adjusted = lower * alpha_imc * beta_zone
    upper_adjusted = upper * alpha_imc * beta_zone
    if lower_adjusted <= threshold_value <= upper_adjusted:
        confidence = 1.0
        status = "S0 validé"
    else:
        distance = min(abs(threshold_value - lower_adjusted), abs(threshold_value - upper_adjusted))
        reference_range = upper_adjusted - lower_adjusted
        confidence = max(0.0, 1.0 - distance / reference_range)
        if confidence >= 0.8:
            status = "S0 validé"
        elif confidence >= 0.5:
            status = "avertissement, recalibration conseillée"
        else:
            status = "recalibration obligatoire, activation bloquée"
    return {
        "status": status,
        "confidence": float(confidence),
        "bounds_adjusted": (float(lower_adjusted), float(upper_adjusted)),
    }


def bloc3_initial_threshold_calibration(
    valid_windows,
    sensor_zones,
    patient_profile: PatientProfile,
    config: Optional[ThresholdConfig] = None
):
    """
    Fonction principale du Bloc 3.
    Modification : le champ 'recalibration_requise' est ajouté par zone
    pour signaler au Bloc 4 les zones à bloquer à l'initialisation.
    """
    if config is None:
        config = ThresholdConfig()

    if len(valid_windows) < config.min_valid_windows:
        raise ValueError(
            f"Bloc 3 impossible : au moins {config.min_valid_windows} fenêtres valides sont nécessaires."
        )

    beta_pied = get_beta_pied(patient_profile.type_pied)
    alpha_imc = compute_alpha_imc(patient_profile.imc, config)

    all_zone_values = {zone: [] for zone in beta_pied.keys()}
    all_zone_weights = {zone: [] for zone in beta_pied.keys()}

    for window in valid_windows:
        pressures_kpa = window["pressures_kpa"]
        w_post = float(window["w_post"])
        # SC_z issu du Bloc 0, transmis via Bloc 2
        sensor_confidence_by_zone = window.get("sensor_confidence_by_zone", {})

        zone_pressures = compute_zone_pressures(
            pressures_kpa=pressures_kpa,
            sensor_zones=sensor_zones,
            config=config
        )

        for zone, pressure_value in zone_pressures.items():
            if zone not in all_zone_values:
                continue
            sc_z = float(sensor_confidence_by_zone.get(zone, 1.0))
            weight = w_post * sc_z   # ← pondération conforme au doc : w_post × SC_z
            all_zone_values[zone].append(pressure_value)
            all_zone_weights[zone].append(weight)

    thresholds = {}
    for zone in beta_pied.keys():
        values = np.asarray(all_zone_values[zone], dtype=float)
        weights = np.asarray(all_zone_weights[zone], dtype=float)

        s0_raw = weighted_percentile(values=values, weights=weights, percentile=config.percentile)
        mad_z = compute_mad(values, config)
        beta_zone = beta_pied[zone]
        s0_adjusted = s0_raw * alpha_imc * beta_zone

        validation = validate_threshold_with_literature(
            threshold_value=s0_adjusted,
            zone=zone,
            alpha_imc=alpha_imc,
            beta_zone=beta_zone
        )

        # Indicateur de blocage pour le Bloc 4
        recalibration_requise = "recalibration obligatoire" in validation["status"]

        thresholds[zone] = {
            "S0_raw_kpa": float(s0_raw),
            "S0_adjusted_kpa": float(s0_adjusted),
            "MAD_kpa": float(mad_z),
            "alpha_imc": float(alpha_imc),
            "beta_pied": float(beta_zone),
            "validation": validation,
            "n_windows_used": int(np.sum(np.isfinite(values))),
            "recalibration_requise": recalibration_requise,   # ← ajout
        }

    return {
        "thresholds": thresholds,
        "patient_profile": {"imc": patient_profile.imc, "type_pied": patient_profile.type_pied},
        "config": config,
    }


def afficher_resultats_bloc3(result):
    """Affichage simplifié des résultats du Bloc 3."""
    print("\n" + "=" * 55)
    print("              RÉSULTATS DU BLOC 3")
    print("       Calibration initiale des seuils S0")
    print("=" * 55)

    print("\nProfil patient")
    print(f"   IMC : {result['patient_profile']['imc']:.2f} kg/m²")
    print(f"   Type de pied : {result['patient_profile']['type_pied']}")

    print("\nSeuils personnalisés par zone")
    for zone, data in result["thresholds"].items():
        print(f"\n   Zone : {zone}")
        print(f"      S0 brut : {data['S0_raw_kpa']:.2f} kPa")
        print(f"      S0 ajusté : {data['S0_adjusted_kpa']:.2f} kPa")
        print(f"      MAD : {data['MAD_kpa']:.2f} kPa")
        print(f"      alpha_IMC : {data['alpha_imc']:.3f}")
        print(f"      beta_pied : {data['beta_pied']:.3f}")
        print(f"      Validation : {data['validation']['status']}")
        print(f"      Confiance : {data['validation']['confidence']:.2f}")
        if data["recalibration_requise"]:
            print(f"      ⚠ RECALIBRATION OBLIGATOIRE — zone bloquée dans le Bloc 4")
    print("=" * 55 + "\n")


def generer_pdf_bloc3(result, filename=None, logo_path=None):
    """
    Génère un rapport PDF pour les résultats du Bloc 3.
    """
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image
        from reportlab.lib.styles import getSampleStyleSheet
        from reportlab.lib import colors
    except ImportError:
        raise ImportError("La bibliothèque reportlab est nécessaire. pip install reportlab")

    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"Rapport_Bloc3_{timestamp}.pdf"

    doc = SimpleDocTemplate(filename, pagesize=A4)
    styles = getSampleStyleSheet()
    story = []

    thresholds = result["thresholds"]
    patient_profile = result["patient_profile"]

    validations = [data["validation"]["status"] for data in thresholds.values()]
    confidences = [data["validation"]["confidence"] for data in thresholds.values()]
    confidence_globale = float(np.mean(confidences))

    if all(status == "S0 validé" for status in validations):
        decision = "SEUILS VALIDÉS"
        conclusion = "Les seuils personnalisés peuvent être transmis aux blocs suivants."
    elif any("recalibration obligatoire" in status for status in validations):
        decision = "RECALIBRATION OBLIGATOIRE"
        conclusion = "Les seuils ne doivent pas être utilisés sans nouvelle calibration."
    else:
        decision = "VALIDATION AVEC PRUDENCE"
        conclusion = "Les seuils peuvent être analysés, mais une vérification complémentaire est recommandée."

    if logo_path is not None:
        try:
            logo = Image(logo_path, width=160, height=60)
            story.append(logo)
            story.append(Spacer(1, 12))
        except Exception:
            story.append(Paragraph("Logo non chargé.", styles["Italic"]))
            story.append(Spacer(1, 12))

    story.append(Paragraph("Rapport Digi'Feet — Bloc 3", styles["Title"]))
    story.append(Paragraph("Calibration initiale des seuils personnalisés", styles["Heading2"]))
    story.append(Paragraph(f"Date : {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}", styles["Normal"]))
    story.append(Spacer(1, 12))

    story.append(Paragraph("1. Résumé simple", styles["Heading2"]))
    story.append(Paragraph(f"Décision : {decision}", styles["Normal"]))
    story.append(Paragraph(f"Confiance globale : {confidence_globale:.2f} / 1.00", styles["Normal"]))
    story.append(Paragraph(f"Conclusion simple : {conclusion}", styles["Normal"]))
    story.append(Spacer(1, 12))

    story.append(Paragraph("2. Explication simplifiée", styles["Heading2"]))
    story.append(Paragraph(
        "Cette étape calcule un seuil initial personnalisé S0 pour chaque zone anatomique du pied. "
        "Le calcul utilise les fenêtres validées par le Bloc 2, puis applique un percentile pondéré "
        "(w_post × SC_z), un ajustement selon l'IMC et un ajustement selon le type de pied.",
        styles["Normal"]
    ))
    story.append(Spacer(1, 12))

    story.append(Paragraph("3. Profil patient", styles["Heading2"]))
    profil_data = [
        [Paragraph("Paramètre", styles["Normal"]), Paragraph("Valeur", styles["Normal"]), Paragraph("Signification", styles["Normal"])],
        [Paragraph("IMC", styles["Normal"]), Paragraph(f"{patient_profile['imc']:.2f} kg/m²", styles["Normal"]), Paragraph("Facteur morphologique", styles["Normal"])],
        [Paragraph("Type de pied", styles["Normal"]), Paragraph(patient_profile["type_pied"], styles["Normal"]), Paragraph("Facteur anatomique zones à risque", styles["Normal"])],
    ]
    profil_table = Table(profil_data, colWidths=[105, 110, 275])
    profil_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
    ]))
    story.append(profil_table)
    story.append(Spacer(1, 12))

    story.append(Paragraph("4. Seuils calculés par zone", styles["Heading2"]))
    seuils_data = [["Zone", "S0 brut", "S0 ajusté", "MAD", "alpha IMC", "beta pied", "Confiance", "Statut"]]
    for zone, data in thresholds.items():
        statut = "BLOQUER" if data["recalibration_requise"] else "OK"
        seuils_data.append([
            zone,
            f"{data['S0_raw_kpa']:.2f} kPa",
            f"{data['S0_adjusted_kpa']:.2f} kPa",
            f"{data['MAD_kpa']:.2f} kPa",
            f"{data['alpha_imc']:.3f}",
            f"{data['beta_pied']:.3f}",
            f"{data['validation']['confidence']:.2f}",
            statut,
        ])
    seuils_table = Table(seuils_data, colWidths=[65, 65, 70, 55, 55, 55, 55, 65])
    seuils_style = [
        ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("ALIGN", (1, 1), (-1, -1), "CENTER"),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
    ]
    # FIX: colorer en rouge les lignes BLOQUER
    for row_idx, row in enumerate(seuils_data[1:], start=1):
        if row[-1] == "BLOQUER":
            seuils_style.append(("BACKGROUND", (0, row_idx), (-1, row_idx), colors.HexColor("#fadbd8")))
            seuils_style.append(("TEXTCOLOR", (-1, row_idx), (-1, row_idx), colors.HexColor("#c0392b")))
            seuils_style.append(("FONTNAME", (-1, row_idx), (-1, row_idx), "Helvetica-Bold"))
    seuils_table.setStyle(TableStyle(seuils_style))
    story.append(seuils_table)
    story.append(Spacer(1, 12))

    story.append(Paragraph("5. Critères de validation", styles["Heading2"]))
    validation_data = [["Zone", "Validation", "Bornes ajustées"]]
    for zone, data in thresholds.items():
        bounds = data["validation"]["bounds_adjusted"]
        bounds_text = f"{bounds[0]:.2f} – {bounds[1]:.2f} kPa" if bounds else "Non disponible"
        validation_data.append([zone, data["validation"]["status"], bounds_text])
    validation_table = Table(validation_data, colWidths=[100, 230, 160])
    validation_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
    ]))
    story.append(validation_table)
    story.append(Spacer(1, 12))

    story.append(Paragraph("6. Données transmises aux blocs suivants", styles["Heading2"]))
    story.append(Paragraph("Les données principales transmises au Bloc 4 sont :", styles["Normal"]))
    for zone, data in thresholds.items():
        story.append(Paragraph(
            f"- {zone} : S0 = {data['S0_adjusted_kpa']:.2f} kPa ; "
            f"MAD = {data['MAD_kpa']:.2f} kPa ; "
            f"confiance = {data['validation']['confidence']:.2f} ; "
            f"recalibration requise = {data['recalibration_requise']}",
            styles["Normal"]
        ))
    story.append(Spacer(1, 12))
    story.append(Paragraph(
        "Document généré automatiquement — Aide à l'interprétation des seuils personnalisés, "
        "sans valeur de diagnostic médical.",
        styles["Italic"]
    ))
    doc.build(story)
    return filename


# ============================================================
# BLOC 4 — MODÈLE BAYÉSIEN ADAPTATIF
# Modification mineure : initialize_bayesian_states_from_bloc3
# bloque les zones marquées recalibration_requise=True.
# Le reste est inchangé.
# ============================================================

@dataclass
class BayesianConfig:
    """Paramètres réglables du Bloc 4."""
    k_prior: float = 1.5
    lambda_forgetting: float = 0.97
    sc_min_update: float = 0.40
    confidence_min_update: float = 0.50
    variation_max_sigma: float = 3.0
    sigma_min_kpa: float = 1.0
    pression_activation_kpa: float = 5.0


@dataclass
class BayesianZoneState:
    """État bayésien courant d'une zone anatomique."""
    zone: str
    mu_kpa: float
    sigma_kpa: float
    sigma_mes_kpa: float
    confidence: float
    n_updates: int = 0
    blocked: bool = False
    last_reason: str = "initialisation"

@dataclass
class AlertDecision:
    """Décision d'alerte pour une zone à un instant donné (conforme BLOC 5)"""
    zone: str
    pressure_measured_kpa: float
    threshold_mu_kpa: float
    sigma_kpa: float
    safety_margin: float          # μ - P(t)
    exceedance_ratio: float       # P(t) / μ
    alert_level: str              # "Normal", "Avertissement", "Alarme"
    is_at_risk: bool


def initialize_bayesian_states_from_bloc3(
    result_bloc3,
    config: Optional[BayesianConfig] = None
):
    """
    Initialise l'a priori bayésien pour chaque zone anatomique.
    Modification : les zones avec recalibration_requise=True sont
    initialisées avec blocked=True dès le départ.
    """
    if config is None:
        config = BayesianConfig()

    states = {}
    for zone, data in result_bloc3["thresholds"].items():
        s0 = float(data["S0_adjusted_kpa"])
        mad = float(data["MAD_kpa"])
        confidence = float(data["validation"]["confidence"])
        recalibration_requise = data.get("recalibration_requise", False)

        sigma0 = max(config.k_prior * mad, config.sigma_min_kpa)
        sigma_mes = max(2.0 * mad, config.sigma_min_kpa)

        # Blocage immédiat si recalibration obligatoire (Bloc 3)
        if recalibration_requise:
            blocked = True
            reason = "recalibration obligatoire détectée au Bloc 3 — zone bloquée"
        else:
            blocked = False
            reason = "prior initialisé depuis le Bloc 3"

        states[zone] = BayesianZoneState(
            zone=zone,
            mu_kpa=s0,
            sigma_kpa=sigma0,
            sigma_mes_kpa=sigma_mes,
            confidence=confidence,
            n_updates=0,
            blocked=blocked,
            last_reason=reason
        )
    return states


def compute_zone_representative_pressure(
    pressures_kpa,
    sensor_zones,
    zone,
    activation_threshold_kpa=5.0
):
    """Calcule la pression représentative P_z(i) (maximum moyen de la zone)."""
    pressures_kpa = np.asarray(pressures_kpa, dtype=float)
    sensor_zones = np.asarray(sensor_zones)
    zone_mask = sensor_zones == zone
    if not np.any(zone_mask):
        return np.nan
    zone_pressures = pressures_kpa[:, zone_mask]
    zone_pressures = np.where(zone_pressures > activation_threshold_kpa, zone_pressures, np.nan)
    if np.all(np.isnan(zone_pressures)):
        return np.nan
    pressure_max_time = np.nanmax(zone_pressures, axis=1)
    return float(np.nanmean(pressure_max_time))


def check_bayesian_update_blocking(
    state: BayesianZoneState,
    p_rep_kpa,
    sensor_confidence,
    config: BayesianConfig
):
    """Vérifie si la mise à jour bayésienne doit être bloquée."""
    if state.blocked and "recalibration obligatoire" in state.last_reason:
        return True, state.last_reason
    if not np.isfinite(p_rep_kpa):
        return True, "mesure non exploitable"
    if sensor_confidence < config.sc_min_update:
        return True, "confiance capteur trop faible"
    if state.confidence < config.confidence_min_update:
        return True, "confiance initiale Bloc 3 trop faible"
    variation = abs(p_rep_kpa - state.mu_kpa)
    if variation > config.variation_max_sigma * state.sigma_kpa:
        return True, "variation brutale suspecte"
    return False, "mise à jour autorisée"


def update_bayesian_zone(
    state: BayesianZoneState,
    p_rep_kpa,
    w_post,
    sensor_confidence,
    config: BayesianConfig
):
    """Met à jour une zone par modèle normal conjugué avec forgetting factor."""
    blocked, reason = check_bayesian_update_blocking(
        state=state, p_rep_kpa=p_rep_kpa,
        sensor_confidence=sensor_confidence, config=config
    )
    if blocked:
        state.blocked = True
        state.last_reason = reason
        return state

    # La confiance pondère la PRÉCISION, PAS la valeur mesurée
    confidence_weight = max(w_post * sensor_confidence, 0.1)  # éviter division par zéro
    
    prior_precision = config.lambda_forgetting / (state.sigma_kpa ** 2)
    measurement_precision = confidence_weight / (state.sigma_mes_kpa ** 2)
    posterior_variance = 1.0 / (prior_precision + measurement_precision)
    
    # La valeur mesurée reste NON pondérée
    posterior_mean = posterior_variance * (
        prior_precision * state.mu_kpa + measurement_precision * p_rep_kpa
    )

    state.mu_kpa = float(posterior_mean)
    state.sigma_kpa = float(max(np.sqrt(posterior_variance), config.sigma_min_kpa))
    state.sigma_mes_kpa = float(max(
        config.lambda_forgetting * state.sigma_mes_kpa
        + (1.0 - config.lambda_forgetting) * min(abs(p_rep_kpa - state.mu_kpa), 50.0),  # plafonner variation
        config.sigma_min_kpa
    ))
    state.n_updates += 1
    state.blocked = False
    state.last_reason = "mise à jour effectuée"
    return state

def bloc4_bayesian_adaptive_update(
    current_window,
    sensor_zones,
    bayesian_states,
    config: Optional[BayesianConfig] = None
):
    """Fonction principale du Bloc 4 — Version améliorée avec décisions d'alerte"""
    if config is None:
        config = BayesianConfig()

    pressures_kpa = current_window["pressures_kpa"]
    w_post = float(current_window.get("w_post", 1.0))
    sensor_confidence_by_zone = current_window.get("sensor_confidence_by_zone", {})

    updated_thresholds = {}
    alert_decisions = []          # ← NOUVEAU : liste des décisions d'alerte

    for zone, state in bayesian_states.items():
        sensor_confidence = float(sensor_confidence_by_zone.get(zone, 1.0))

        p_rep = compute_zone_representative_pressure(
            pressures_kpa=pressures_kpa,
            sensor_zones=sensor_zones,
            zone=zone,
            activation_threshold_kpa=config.pression_activation_kpa
        )
        old_mu = state.mu_kpa

        state = update_bayesian_zone(
            state=state, p_rep_kpa=p_rep, w_post=w_post,
            sensor_confidence=sensor_confidence, config=config
        )
        bayesian_states[zone] = state

        # === Calcul des décisions d'alerte (conforme BLOC 5) ===
        # Variable locale pour cette zone : évite le bug alert_decisions[-1]
        # qui pointerait sur la zone précédente si p_rep n'est pas fini.
        zone_alert_level = "Normal"
        zone_is_at_risk = False
        zone_alert_decision: Optional[AlertDecision] = None

        if np.isfinite(p_rep):
            safety_margin = state.mu_kpa - p_rep
            exceedance_ratio = p_rep / state.mu_kpa if state.mu_kpa > 0 else 0.0

            # Niveaux d'alerte selon logique clinique
            if p_rep >= state.mu_kpa + 2.0 * state.sigma_kpa:
                zone_alert_level = "Alarme"
                zone_is_at_risk = True
            elif p_rep >= state.mu_kpa + state.sigma_kpa:
                zone_alert_level = "Avertissement"
                zone_is_at_risk = True
            else:
                zone_alert_level = "Normal"
                zone_is_at_risk = False

            zone_alert_decision = AlertDecision(
                zone=zone,
                pressure_measured_kpa=float(p_rep),
                threshold_mu_kpa=float(state.mu_kpa),
                sigma_kpa=float(state.sigma_kpa),
                safety_margin=float(safety_margin),
                exceedance_ratio=float(exceedance_ratio),
                alert_level=zone_alert_level,
                is_at_risk=zone_is_at_risk
            )
            alert_decisions.append(zone_alert_decision)

        updated_thresholds[zone] = {
            "P_rep_kpa": float(p_rep) if np.isfinite(p_rep) else np.nan,
            "S_previous_kpa": float(old_mu),
            "S_updated_kpa": float(state.mu_kpa),
            "sigma_kpa": float(state.sigma_kpa),
            "sigma_mes_kpa": float(state.sigma_mes_kpa),
            "sensor_confidence": sensor_confidence,
            "w_post": w_post,
            "n_updates": state.n_updates,
            "blocked": state.blocked,
            "reason": state.last_reason,
            # Utilise la variable locale de cette zone, pas alert_decisions[-1]
            "alert_level": zone_alert_level,
            "is_at_risk": zone_is_at_risk,
        }

    result = {
        "thresholds": updated_thresholds,
        "alert_decisions": alert_decisions,   # ← NOUVEAU
        "config": config
    }

    return result, bayesian_states

def afficher_resultats_bloc4(result):
    """Affichage lisible des résultats du Bloc 4."""
    print("\n" + "=" * 55)
    print("              RÉSULTATS DU BLOC 4")
    print("          Modèle bayésien adaptatif")
    print("=" * 55)

    for zone, data in result["thresholds"].items():
        print(f"\n   Zone : {zone}")
        print(f"      Pression représentative : {data['P_rep_kpa']:.2f} kPa")
        print(f"      Seuil précédent         : {data['S_previous_kpa']:.2f} kPa")
        print(f"      Seuil mis à jour        : {data['S_updated_kpa']:.2f} kPa")
        print(f"      Incertitude sigma       : {data['sigma_kpa']:.2f} kPa")
        print(f"      Sigma mesure            : {data['sigma_mes_kpa']:.2f} kPa")
        print(f"      w_post                  : {data['w_post']:.2f}")
        print(f"      Confiance capteur       : {data['sensor_confidence']:.2f}")
        print(f"      Mise à jour bloquée     : {data['blocked']}")
        print(f"      Raison                  : {data['reason']}")

    # Affichage des décisions d'alerte (une seule fois, pas dans la boucle des zones)
    print("\n" + "-" * 55)
    print("   DÉCISIONS D'ALERTE (BLOC 5)")
    print("-" * 55)

    alerts = result.get("alert_decisions", [])
    if alerts:
        for alert in alerts:
            risk_str = "(RISQUE)" if alert.is_at_risk else ""
            print(f"   {alert.zone:12s} | "
                  f"P = {alert.pressure_measured_kpa:6.1f} kPa | "
                  f"μ = {alert.threshold_mu_kpa:6.1f} kPa | "
                  f"Ratio = {alert.exceedance_ratio:.2f} | "
                  f"Marge = {alert.safety_margin:+6.1f} | "
                  f"{alert.alert_level:12s} {risk_str}")
    else:
        print("   Aucune décision d'alerte disponible.")

    print("=" * 55 + "\n")

def generer_pdf_bloc4(result, filename=None, logo_path=None):
    """Génère un rapport PDF pour les résultats du Bloc 4 avec décisions d'alerte."""
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image
        from reportlab.lib.styles import getSampleStyleSheet
        from reportlab.lib import colors
    except ImportError:
        raise ImportError("La bibliothèque reportlab est nécessaire. pip install reportlab")

    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"Rapport_Bloc4_{timestamp}.pdf"

    doc = SimpleDocTemplate(filename, pagesize=A4)
    styles = getSampleStyleSheet()
    story = []
    thresholds = result["thresholds"]

    # En-tête avec logo
    if logo_path is not None and os.path.exists(logo_path):
        story.append(Image(logo_path, width=140, height=60))
        story.append(Spacer(1, 12))

    story.append(Paragraph("Rapport Digi'Feet — Bloc 4", styles["Title"]))
    story.append(Paragraph("Modèle bayésien adaptatif des seuils personnalisés", styles["Heading2"]))
    story.append(Paragraph(f"Date : {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}", styles["Normal"]))
    story.append(Spacer(1, 12))

    # Résumé simple
    blocked_zones = [zone for zone, data in thresholds.items() if data.get("blocked", False)]
    if len(blocked_zones) == 0:
        decision = "MISE À JOUR VALIDÉE"
        conclusion = "Les seuils personnalisés ont été mis à jour correctement."
    else:
        decision = "MISE À JOUR PARTIELLE"
        conclusion = "Certaines zones n'ont pas été mises à jour car une condition de blocage a été détectée."

    story.append(Paragraph("1. Résumé simple", styles["Heading2"]))
    story.append(Paragraph(f"Décision : <b>{decision}</b>", styles["Normal"]))
    story.append(Paragraph(f"Conclusion simple : {conclusion}", styles["Normal"]))
    story.append(Spacer(1, 12))

    # Explication simplifiée
    story.append(Paragraph("2. Explication simplifiée", styles["Heading2"]))
    story.append(Paragraph(
        "Le Bloc 4 met à jour les seuils personnalisés calculés au Bloc 3 à l'aide d'un modèle "
        "bayésien adaptatif avec forgetting factor. Il produit également les décisions d'alerte "
        "conformes au BLOC 5 (marge de sécurité, ratio de dépassement, niveaux d'alerte).",
        styles["Normal"]
    ))
    story.append(Spacer(1, 12))

        # === TABLEAU DES SEUILS avec une VERSION AMÉLIORÉE AVEC WRAPPING ===
    story.append(Paragraph("3. Seuils mis à jour par zone", styles["Heading2"]))
    story.append(Paragraph(
        "<i>Note : « Seuil µ (après convergence) » est le seuil bayésien résultant après "
        "toutes les mises à jour sur les fenêtres valides.</i>",
        styles["Normal"]
    ))
    story.append(Spacer(1, 6))

    # Style pour les en-têtes (centré + gras)
    header_style = ParagraphStyle(
        'header',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=8,
        alignment=1,          # 0=left, 1=center, 2=right
        leading=10
    )

    # Création des en-têtes avec Paragraph pour permettre le retour à la ligne
    table_data = [[
        Paragraph("Zone", header_style),
        Paragraph("P rep.", header_style),
        Paragraph("Seuil µ (après<br/>convergence)", header_style),
        Paragraph("Seuil mis à jour", header_style),
        Paragraph("Sigma", header_style),
        Paragraph("Bloqué", header_style),
        Paragraph("Raison", header_style),
    ]]

    # Remplissage des données
    for zone, data in thresholds.items():
        table_data.append([
            zone,
            f"{data.get('P_rep_kpa', 0):.2f} kPa",
            f"{data.get('S_previous_kpa', 0):.2f} kPa",
            f"{data.get('S_updated_kpa', 0):.2f} kPa",
            f"{data.get('sigma_kpa', 0):.2f} kPa",
            str(data.get("blocked", False)),
            Paragraph(data.get("reason", "N/A"), styles["Normal"]),   # raison peut être longue
        ])

    # Largeurs de colonnes optimisées
    col_widths = [55, 55, 88, 80, 50, 45, 135]

    table = Table(table_data, colWidths=col_widths, repeatRows=1)

    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.lightgrey),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 8),
        ('FONTSIZE', (0, 1), (-1, -1), 8),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('LEADING', (0, 0), (-1, 0), 10),      # espacement vertical en-tête
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.lightgrey]),
    ]))

    story.append(table)
    story.append(Spacer(1, 12))

    # === AJOUT DE LA SECTION ALERTES (BLOC 5) ===
    story.append(Paragraph("4. Décisions d'alerte (BLOC 5)", styles["Heading2"]))
    
    alert_data = [["Zone", "P mesurée (kPa)", "Seuil μ (kPa)", "Ratio", "Marge", "Niveau d'alerte", "Risque"]]
    for alert in result.get("alert_decisions", []):
        alert_data.append([
            alert.zone,
            f"{alert.pressure_measured_kpa:.1f}",
            f"{alert.threshold_mu_kpa:.1f}",
            f"{alert.exceedance_ratio:.2f}",
            f"{alert.safety_margin:+.1f}",
            alert.alert_level,
            "OUI" if alert.is_at_risk else "Non"
        ])

    if len(alert_data) > 1:  # s'il y a des alertes
        alert_table = Table(alert_data, colWidths=[70, 80, 80, 60, 70, 90, 50])
        alert_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("ALIGN", (1, 1), (-1, -1), "CENTER"),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ]))
        story.append(alert_table)
    else:
        story.append(Paragraph("Aucune décision d'alerte disponible pour cette fenêtre.", styles["Normal"]))
    
    story.append(Spacer(1, 12))

    # Section qualité
    story.append(Paragraph("5. Données de qualité utilisées", styles["Heading2"]))
    quality_data = [["Zone", "w_post", "Confiance capteur", "Nombre de mises à jour", "Sigma mesure"]]
    for zone, data in thresholds.items():
        quality_data.append([
            zone,
            f"{data.get('w_post', 0):.2f}",
            f"{data.get('sensor_confidence', 0):.2f}",
            str(data.get("n_updates", 0)),
            f"{data.get('sigma_mes_kpa', 0):.2f} kPa",
        ])
    quality_table = Table(quality_data, colWidths=[90, 80, 110, 120, 100])
    quality_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
    ]))
    story.append(quality_table)
    story.append(Spacer(1, 12))

    # Pied de page
    story.append(Paragraph(
        "Document généré automatiquement — Aide à l'interprétation du modèle bayésien adaptatif, "
        "sans valeur de diagnostic médical.",
        styles["Italic"]
    ))

    doc.build(story)
    return filename


# ============================================================
# BLOC 5 — MOTEUR DE DÉCISION D'ALERTE (fonction autonome)
# Extrait les décisions d'alerte du Bloc 4, applique la règle
# de persistance multi-fenêtres, et produit une sortie structurée.
# ============================================================

@dataclass
class AlertSummary:
    """
    Résumé de décision d'alerte pour une zone, tenant compte
    de la persistance sur plusieurs fenêtres consécutives.
    """
    zone: str
    alert_level: str          # "Normal", "Avertissement", "Alarme"
    is_at_risk: bool
    consecutive_alerts: int   # nb de fenêtres consécutives en alerte
    confirmed: bool           # True si persistance ≥ seuil (voir Bloc5Config)
    latest_pressure_kpa: float
    latest_threshold_mu_kpa: float
    latest_exceedance_ratio: float
    latest_safety_margin: float
    message: str


@dataclass
class Bloc5Config:
    """Paramètres du moteur de décision d'alerte Bloc 5."""
    n_fenetres_confirmation: int = 2   # nb de fenêtres consécutives pour confirmer une alerte
    ratio_alarme: float = 1.0          # P/μ ≥ μ + 2σ → Alarme
    ratio_avertissement: float = 1.0   # P/μ ≥ μ + σ → Avertissement


class AlertEngine:
    """
    Bloc 5 : Moteur de décision d'alerte autonome.
    Maintient un historique de fenêtres et applique la règle de persistance :
    une alerte n'est « confirmée » que si elle persiste sur
    n_fenetres_confirmation fenêtres consécutives.
    """

    def __init__(self, config: Optional[Bloc5Config] = None):
        self.config = config or Bloc5Config()
        # Historique par zone : liste de niveaux d'alerte ("Normal", "Avertissement", "Alarme")
        self._historique: Dict[str, List[str]] = {}

    def run_bloc5(
        self,
        result_bloc4: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Fonction principale du Bloc 5.

        Entrée :
            result_bloc4 : résultat complet du Bloc 4, contenant
                           "alert_decisions" (liste d'AlertDecision)
                           et "thresholds" (dict par zone)

        Sortie :
            alert_summaries : dict {zone: AlertSummary}
            n_confirmees     : nb de zones avec alerte confirmée
            global_risk      : True si au moins une alarme confirmée
        """
        alert_summaries: Dict[str, AlertSummary] = {}
        alert_decisions: List[AlertDecision] = result_bloc4.get("alert_decisions", [])
        thresholds = result_bloc4.get("thresholds", {})

        # Construire un index rapide zone → AlertDecision
        alerts_by_zone: Dict[str, AlertDecision] = {a.zone: a for a in alert_decisions}

        all_zones = set(thresholds.keys())

        for zone in all_zones:
            # Récupérer la décision courante (ou "Normal" si zone non activée)
            ad = alerts_by_zone.get(zone, None)
            if ad is not None:
                current_level = ad.alert_level
                p_measured = ad.pressure_measured_kpa
                mu = ad.threshold_mu_kpa
                ratio = ad.exceedance_ratio
                margin = ad.safety_margin
            else:
                current_level = "Normal"
                zone_data = thresholds.get(zone, {})
                p_measured = float(zone_data.get("P_rep_kpa", 0.0) or 0.0)
                mu = float(zone_data.get("S_updated_kpa", 0.0))
                ratio = p_measured / mu if mu > 0 else 0.0
                margin = mu - p_measured

            # Mise à jour de l'historique
            if zone not in self._historique:
                self._historique[zone] = []
            self._historique[zone].append(current_level)

            # Ne garder que les N dernières fenêtres
            n = self.config.n_fenetres_confirmation
            self._historique[zone] = self._historique[zone][-n:]

            # Calcul de la persistance
            hist = self._historique[zone]
            # Compter les alertes consécutives depuis la fin
            consecutive = 0
            for lvl in reversed(hist):
                if lvl in ("Avertissement", "Alarme"):
                    consecutive += 1
                else:
                    break

            confirmed = consecutive >= n
            is_at_risk = current_level in ("Avertissement", "Alarme")

            # Message lisible
            if current_level == "Alarme" and confirmed:
                message = (
                    f"ALARME CONFIRMÉE : pression {p_measured:.1f} kPa dépasse "
                    f"μ+2σ ({mu:.1f} kPa) sur {consecutive} fenêtre(s) consécutive(s)."
                )
            elif current_level == "Alarme":
                message = (
                    f"Alarme détectée ({p_measured:.1f} kPa > μ+2σ={mu:.1f} kPa) "
                    f"— en attente de confirmation ({consecutive}/{n})."
                )
            elif current_level == "Avertissement" and confirmed:
                message = (
                    f"AVERTISSEMENT CONFIRMÉ : pression {p_measured:.1f} kPa dépasse "
                    f"μ+σ ({mu:.1f} kPa) sur {consecutive} fenêtre(s) consécutive(s)."
                )
            elif current_level == "Avertissement":
                message = (
                    f"Avertissement ({p_measured:.1f} kPa > μ+σ={mu:.1f} kPa) "
                    f"— en attente de confirmation ({consecutive}/{n})."
                )
            else:
                message = f"Normal — pression {p_measured:.1f} kPa dans les seuils ({mu:.1f} kPa)."

            alert_summaries[zone] = AlertSummary(
                zone=zone,
                alert_level=current_level,
                is_at_risk=is_at_risk,
                consecutive_alerts=consecutive,
                confirmed=confirmed,
                latest_pressure_kpa=p_measured,
                latest_threshold_mu_kpa=mu,
                latest_exceedance_ratio=ratio,
                latest_safety_margin=margin,
                message=message,
            )

        n_confirmees = sum(1 for s in alert_summaries.values() if s.confirmed)
        global_risk = any(
            s.confirmed and s.alert_level == "Alarme"
            for s in alert_summaries.values()
        )

        return {
            "alert_summaries": alert_summaries,
            "n_confirmees": n_confirmees,
            "global_risk": global_risk,
        }


def afficher_resultats_bloc5(result_bloc5: Dict[str, Any]) -> None:
    """Affichage console lisible des résultats du Bloc 5."""
    print("\n" + "=" * 60)
    print("              RÉSULTATS DU BLOC 5")
    print("         Moteur de décision d'alerte")
    print("=" * 60)
    summaries = result_bloc5.get("alert_summaries", {})
    for zone, s in summaries.items():
        flag = "⚠ " if s.is_at_risk else "  "
        conf_str = "[CONFIRMÉ]" if s.confirmed else f"[{s.consecutive_alerts} consec.]"
        print(
            f"  {flag}{zone:14s} | {s.alert_level:13s} {conf_str:14s} | "
            f"P={s.latest_pressure_kpa:6.1f} kPa  μ={s.latest_threshold_mu_kpa:6.1f} kPa"
        )
    print(f"\n  Zones en alerte confirmée : {result_bloc5['n_confirmees']}")
    risk_str = "OUI ⚠" if result_bloc5["global_risk"] else "Non"
    print(f"  Risque global (alarme confirmée) : {risk_str}")
    print("=" * 60 + "\n")


def generer_pdf_bloc5(result_bloc5: Dict[str, Any], filename: Optional[str] = None, logo_path: Optional[str] = None) -> str:
    """Génère un rapport PDF pour les décisions d'alerte du Bloc 5."""
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image
        from reportlab.lib.styles import getSampleStyleSheet
        from reportlab.lib import colors as rl_colors
    except ImportError:
        raise ImportError("reportlab requis. pip install reportlab")

    if filename is None:
        filename = f"Rapport_Bloc5_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"

    doc = SimpleDocTemplate(filename, pagesize=A4)
    styles = getSampleStyleSheet()
    story = []

    if logo_path and os.path.exists(logo_path):
        story.append(Image(logo_path, width=140, height=60))
        story.append(Spacer(1, 12))

    story.append(Paragraph("Rapport Digi'Feet — Bloc 5", styles["Title"]))
    story.append(Paragraph("Moteur de décision d'alerte avec persistance multi-fenêtres", styles["Heading2"]))
    story.append(Paragraph(f"Date : {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}", styles["Normal"]))
    story.append(Spacer(1, 12))

    summaries = result_bloc5.get("alert_summaries", {})
    n_confirmees = result_bloc5["n_confirmees"]
    global_risk = result_bloc5["global_risk"]

    story.append(Paragraph("1. Résumé global", styles["Heading2"]))
    risk_str = "RISQUE ÉLEVÉ DÉTECTÉ ⚠" if global_risk else "Aucun risque majeur confirmé"
    story.append(Paragraph(f"Statut global : <b>{risk_str}</b>", styles["Normal"]))
    story.append(Paragraph(f"Zones en alerte confirmée : {n_confirmees}", styles["Normal"]))
    story.append(Spacer(1, 12))

    story.append(Paragraph("2. Détail par zone anatomique", styles["Heading2"]))
    table_data = [["Zone", "Niveau", "Confirmée", "Consec.", "P mesurée", "Seuil μ", "Ratio", "Message"]]
    for zone, s in summaries.items():
        conf_str = "OUI" if s.confirmed else "Non"
        row_color = None
        table_data.append([
            zone,
            s.alert_level,
            conf_str,
            str(s.consecutive_alerts),
            f"{s.latest_pressure_kpa:.1f} kPa",
            f"{s.latest_threshold_mu_kpa:.1f} kPa",
            f"{s.latest_exceedance_ratio:.2f}",
            s.message[:60] + ("…" if len(s.message) > 60 else ""),
        ])

    tbl = Table(table_data, colWidths=[65, 75, 55, 45, 65, 60, 45, 80])
    tbl_style = [
        ("BACKGROUND", (0, 0), (-1, 0), rl_colors.lightgrey),
        ("GRID", (0, 0), (-1, -1), 0.5, rl_colors.grey),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 7.5),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
    ]
    # Colorier les lignes d'alerte confirmée
    for i, (zone, s) in enumerate(summaries.items(), start=1):
        if s.confirmed and s.alert_level == "Alarme":
            tbl_style.append(("BACKGROUND", (0, i), (-1, i), rl_colors.Color(1.0, 0.85, 0.85)))
        elif s.confirmed and s.alert_level == "Avertissement":
            tbl_style.append(("BACKGROUND", (0, i), (-1, i), rl_colors.Color(1.0, 0.95, 0.75)))
    tbl.setStyle(TableStyle(tbl_style))
    story.append(tbl)
    story.append(Spacer(1, 12))

    story.append(Paragraph("3. Messages détaillés", styles["Heading2"]))
    for zone, s in summaries.items():
        story.append(Paragraph(f"<b>{zone}</b> : {s.message}", styles["Normal"]))
    story.append(Spacer(1, 12))

    story.append(Paragraph(
        "Document généré automatiquement — Aide à la détection des zones à risque, "
        "sans valeur de diagnostic médical.",
        styles["Italic"]
    ))
    doc.build(story)
    return filename


# ============================================================
# BLOC 6 — VALIDATION STATISTIQUE ET RAPPORT DE PERFORMANCE
# Génération de données synthétiques, calcul TP/FP/TN/FN,
# métriques (sensibilité, spécificité, VPP), test de Student,
# courbe ROC + AUC, comparaison seuil fixe vs seuil personnalisé.
# ============================================================

try:
    from scipy import stats as _scipy_stats
    from sklearn.metrics import roc_curve, auc as sklearn_auc
    _SKLEARN_OK = True
except ImportError:
    _SKLEARN_OK = False


# -------------------------------------------------------
# 6.1 — Profils synthétiques
# -------------------------------------------------------

PROFILS_SYNTHETIQUES = {
    "sujet_sain": {
        "description": "Sujet sain (IMC ≈ 22) — pressions homogènes",
        "imc": 22.0,
        "base_pressures": {
            "talon": 140.0,
            "medio_pied": 70.0,
            "avant_pied": 160.0,
            "hallux": 130.0,
        },
        "sigma_bruit": 4.0,
        "risk_zone": "avant_pied",
        "risk_delta_kpa": 60.0,
        "risk_prevalence": 0.15,   # 15 % des mesures sont à risque
    },
    "surpoids": {
        "description": "Patient en surpoids (IMC ≈ 28) — pressions globalement élevées",
        "imc": 28.0,
        "base_pressures": {
            "talon": 210.0,
            "medio_pied": 110.0,
            "avant_pied": 240.0,
            "hallux": 190.0,
        },
        "sigma_bruit": 6.0,
        "risk_zone": "avant_pied",
        "risk_delta_kpa": 80.0,
        "risk_prevalence": 0.25,
    },
    "hallux_valgus": {
        "description": "Hallux valgus — surcharge localisée à l'hallux",
        "imc": 25.0,
        "base_pressures": {
            "talon": 150.0,
            "medio_pied": 80.0,
            "avant_pied": 170.0,
            "hallux": 310.0,
        },
        "sigma_bruit": 5.0,
        "risk_zone": "hallux",
        "risk_delta_kpa": 100.0,
        "risk_prevalence": 0.30,
    },
    "pied_charcot": {
        "description": "Pied de Charcot (IMC ≈ 30) — pressions élevées, forte déformation",
        "imc": 30.0,
        "base_pressures": {
            "talon": 270.0,
            "medio_pied": 160.0,
            "avant_pied": 320.0,
            "hallux": 290.0,
        },
        "sigma_bruit": 8.0,
        "risk_zone": "avant_pied",
        "risk_delta_kpa": 120.0,
        "risk_prevalence": 0.40,
    },
}


def generer_donnees_synthetiques(
    profil: Dict[str, Any],
    n_mesures: int = 100,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Génère des données synthétiques pour un profil patient.
    Retourne les pressions par zone, les labels vérité terrain,
    et le seuil personnalisé simulé (percentile 85 pondéré + ajustement IMC).

    Format de sortie :
        pressures    : dict {zone: np.ndarray(n_mesures,)}
        ground_truth : dict {zone: np.ndarray(n_mesures,) booléen}
        seuil_perso  : dict {zone: float}
    """
    rng = np.random.default_rng(seed)
    zones = list(profil["base_pressures"].keys())
    risk_zone = profil["risk_zone"]
    risk_delta = profil["risk_delta_kpa"]
    prevalence = profil["risk_prevalence"]
    sigma = profil["sigma_bruit"]
    imc = profil["imc"]

    pressures: Dict[str, np.ndarray] = {}
    ground_truth: Dict[str, np.ndarray] = {}
    seuil_perso: Dict[str, float] = {}

    # Facteur IMC (doc sect. 3) : α = 1 + 0.012 * (IMC - 22)
    alpha_imc = 1.0 + 0.012 * (imc - 22.0)

    for zone in zones:
        base = profil["base_pressures"][zone]
        # Mesures de base + bruit gaussien
        p = rng.normal(base, sigma, n_mesures)

        # Injection des événements à risque sur la zone à risque
        is_risk = np.zeros(n_mesures, dtype=bool)
        if zone == risk_zone:
            n_risk = int(n_mesures * prevalence)
            risk_indices = rng.choice(n_mesures, size=n_risk, replace=False)
            risk_values = rng.uniform(
                risk_delta * 0.7, risk_delta * 1.3, n_risk
            )
            p[risk_indices] += risk_values
            is_risk[risk_indices] = True

        p = np.clip(p, 0.0, 700.0)
        pressures[zone] = p
        ground_truth[zone] = is_risk

        # FIX: Seuil personnalisé calculé sur la baseline (sans les events à risque)
        # Le percentile 85 sur les données complètes (avec pics) monte trop haut
        # pour les profils pathologiques → sensibilité personnalisée effondrée.
        # La calibration clinique réelle se fait sur des fenêtres "normales" (Blocs 0-3),
        # pas sur des fenêtres incluant des événements à risque.
        p_baseline = rng.normal(base, sigma, n_mesures)
        p_baseline = np.clip(p_baseline, 0.0, 700.0)
        s0 = float(np.percentile(p_baseline, 90)) * alpha_imc  # cohérent avec ThresholdConfig.percentile=90
        seuil_perso[zone] = s0

    return {
        "pressures": pressures,
        "ground_truth": ground_truth,
        "seuil_perso": seuil_perso,
        "profil_nom": profil["description"],
        "imc": imc,
    }


# -------------------------------------------------------
# 6.2 — Calcul TP/FP/TN/FN et métriques
# -------------------------------------------------------

def calculer_metriques_zone(
    pressures: np.ndarray,
    ground_truth: np.ndarray,
    seuil: float,
) -> Dict[str, float]:
    """
    Calcule TP, FP, TN, FN et les métriques dérivées pour une zone et un seuil.

    Retourne un dict : tp, fp, tn, fn, sensibilite, specificite, vpp, f1
    """
    predictions = pressures >= seuil
    tp = int(np.sum(predictions & ground_truth))
    fp = int(np.sum(predictions & ~ground_truth))
    tn = int(np.sum(~predictions & ~ground_truth))
    fn = int(np.sum(~predictions & ground_truth))

    sensibilite = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificite = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    vpp = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0.0

    return {
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "sensibilite": sensibilite,
        "specificite": specificite,
        "vpp": vpp,
        "f1": f1,
    }


# -------------------------------------------------------
# 6.3 — Simulation multi-runs + test de Student + ROC
# -------------------------------------------------------

def bloc6_validation_statistique(
    seuil_fixe_kpa: float = 200.0,
    n_simulations: int = 30,
    n_mesures_par_sim: int = 100,
    profils: Optional[Dict[str, Any]] = None,
    config_zone_cible: str = "avant_pied",
) -> Dict[str, Any]:
    """
    Fonction principale du Bloc 6.

    Génère n_simulations × 4 profils de données synthétiques,
    compare seuil fixe vs seuil personnalisé, calcule les métriques,
    le test de Student et la courbe ROC.

    Paramètres :
        seuil_fixe_kpa     : seuil de référence (doc : 200 kPa)
        n_simulations      : nombre de répétitions (doc : plusieurs simulations)
        n_mesures_par_sim  : taille de chaque simulation
        profils            : dictionnaire des profils (défaut = PROFILS_SYNTHETIQUES)
        config_zone_cible  : zone utilisée pour ROC et test t (zone principale)

    Retourne un dict complet avec toutes les métriques et données pour PDF.
    """
    if profils is None:
        profils = PROFILS_SYNTHETIQUES

    # Accumulateurs par simulation
    sens_fixe_sims: List[float] = []
    sens_perso_sims: List[float] = []
    spec_fixe_sims: List[float] = []
    spec_perso_sims: List[float] = []
    vpp_fixe_sims: List[float] = []
    vpp_perso_sims: List[float] = []

    # Pour la courbe ROC (agrégation sur tous profils et simulations)
    all_pressures_roc: List[float] = []
    all_gt_roc: List[int] = []

    # Résultats détaillés par profil
    resultats_par_profil: Dict[str, Any] = {}

    for nom_profil, profil in profils.items():
        metriques_fixe_profil: List[Dict] = []
        metriques_perso_profil: List[Dict] = []
        seuils_perso_profil: List[float] = []

        risk_zone = profil["risk_zone"]

        for sim_idx in range(n_simulations):
            donnees = generer_donnees_synthetiques(
                profil=profil,
                n_mesures=n_mesures_par_sim,
                seed=sim_idx * 137 + hash(nom_profil) % 10000,
            )
            p_zone = donnees["pressures"][risk_zone]
            gt_zone = donnees["ground_truth"][risk_zone]
            s_perso = donnees["seuil_perso"][risk_zone]
            seuils_perso_profil.append(s_perso)

            m_fixe = calculer_metriques_zone(p_zone, gt_zone, seuil_fixe_kpa)
            m_perso = calculer_metriques_zone(p_zone, gt_zone, s_perso)

            metriques_fixe_profil.append(m_fixe)
            metriques_perso_profil.append(m_perso)

            # Accumulation pour ROC sur la zone cible
            if risk_zone == config_zone_cible or nom_profil == list(profils.keys())[0]:
                all_pressures_roc.extend(p_zone.tolist())
                all_gt_roc.extend(gt_zone.astype(int).tolist())

        resultats_par_profil[nom_profil] = {
            "description": profil["description"],
            "risk_zone": risk_zone,
            "seuil_perso_moyen": float(np.mean(seuils_perso_profil)),
            "seuil_perso_std": float(np.std(seuils_perso_profil)),
            "fixe": {
                "sensibilite_moy": float(np.mean([m["sensibilite"] for m in metriques_fixe_profil])),
                "sensibilite_std": float(np.std([m["sensibilite"] for m in metriques_fixe_profil])),
                "specificite_moy": float(np.mean([m["specificite"] for m in metriques_fixe_profil])),
                "vpp_moy": float(np.mean([m["vpp"] for m in metriques_fixe_profil])),
            },
            "perso": {
                "sensibilite_moy": float(np.mean([m["sensibilite"] for m in metriques_perso_profil])),
                "sensibilite_std": float(np.std([m["sensibilite"] for m in metriques_perso_profil])),
                "specificite_moy": float(np.mean([m["specificite"] for m in metriques_perso_profil])),
                "vpp_moy": float(np.mean([m["vpp"] for m in metriques_perso_profil])),
            },
        }

        # Accumulation pour test de Student global
        sens_fixe_sims.extend([m["sensibilite"] for m in metriques_fixe_profil])
        sens_perso_sims.extend([m["sensibilite"] for m in metriques_perso_profil])
        spec_fixe_sims.extend([m["specificite"] for m in metriques_fixe_profil])
        spec_perso_sims.extend([m["specificite"] for m in metriques_perso_profil])
        vpp_fixe_sims.extend([m["vpp"] for m in metriques_fixe_profil])
        vpp_perso_sims.extend([m["vpp"] for m in metriques_perso_profil])

    # --- Test de Student (deux groupes appariés) ---
    test_student: Dict[str, Any] = {}
    if _SKLEARN_OK or True:  # scipy est disponible dans tous les cas
        try:
            t_sens, p_sens = _scipy_stats.ttest_rel(sens_perso_sims, sens_fixe_sims)
            t_spec, p_spec = _scipy_stats.ttest_rel(spec_perso_sims, spec_fixe_sims)
            t_vpp, p_vpp = _scipy_stats.ttest_rel(vpp_perso_sims, vpp_fixe_sims)
            test_student = {
                "sensibilite": {"t": float(t_sens), "p_value": float(p_sens),
                                "significatif": bool(p_sens < 0.05)},
                "specificite": {"t": float(t_spec), "p_value": float(p_spec),
                                "significatif": bool(p_spec < 0.05)},
                "vpp": {"t": float(t_vpp), "p_value": float(p_vpp),
                        "significatif": bool(p_vpp < 0.05)},
            }
        except Exception as e:
            test_student = {"erreur": str(e)}

    # --- Courbe ROC ---
    roc_data: Dict[str, Any] = {}
    all_gt_arr = np.array(all_gt_roc)
    all_p_arr = np.array(all_pressures_roc)

    if _SKLEARN_OK and len(np.unique(all_gt_arr)) == 2:
        # ROC seuil personnalisé : score = pression brute (plus la pression, plus à risque)
        fpr_perso, tpr_perso, _ = roc_curve(all_gt_arr, all_p_arr)
        auc_perso = float(sklearn_auc(fpr_perso, tpr_perso))

        # ROC seuil fixe : score binaire (1 si P ≥ seuil fixe)
        scores_fixe = (all_p_arr >= seuil_fixe_kpa).astype(float)
        fpr_fixe, tpr_fixe, _ = roc_curve(all_gt_arr, scores_fixe)
        auc_fixe = float(sklearn_auc(fpr_fixe, tpr_fixe))

        roc_data = {
            "fpr_perso": fpr_perso.tolist(),
            "tpr_perso": tpr_perso.tolist(),
            "auc_perso": auc_perso,
            "fpr_fixe": fpr_fixe.tolist(),
            "tpr_fixe": tpr_fixe.tolist(),
            "auc_fixe": auc_fixe,
        }
    else:
        # Calcul ROC simplifié sans sklearn
        seuils_test = np.percentile(all_p_arr, np.linspace(0, 100, 50))
        fprs_p, tprs_p, fprs_f, tprs_f = [], [], [], []
        for seuil_t in seuils_test:
            pred = all_p_arr >= seuil_t
            tp_ = np.sum(pred & (all_gt_arr == 1))
            fp_ = np.sum(pred & (all_gt_arr == 0))
            tn_ = np.sum(~pred & (all_gt_arr == 0))
            fn_ = np.sum(~pred & (all_gt_arr == 1))
            tpr_ = tp_ / (tp_ + fn_) if (tp_ + fn_) > 0 else 0.0
            fpr_ = fp_ / (fp_ + tn_) if (fp_ + tn_) > 0 else 0.0
            tprs_p.append(tpr_)
            fprs_p.append(fpr_)
        auc_p = float(np.trapezoid(sorted(tprs_p), sorted(fprs_p))) if fprs_p else 0.5
        pred_f = all_p_arr >= seuil_fixe_kpa
        tp_f = np.sum(pred_f & (all_gt_arr == 1))
        fp_f = np.sum(pred_f & (all_gt_arr == 0))
        tn_f = np.sum(~pred_f & (all_gt_arr == 0))
        fn_f = np.sum(~pred_f & (all_gt_arr == 1))
        tpr_f = tp_f / (tp_f + fn_f) if (tp_f + fn_f) > 0 else 0.0
        fpr_f = fp_f / (fp_f + tn_f) if (fp_f + tn_f) > 0 else 0.0
        roc_data = {
            "fpr_perso": sorted(fprs_p),
            "tpr_perso": sorted(tprs_p),
            "auc_perso": auc_p,
            "fpr_fixe": [0.0, fpr_f, 1.0],
            "tpr_fixe": [0.0, tpr_f, 1.0],
            "auc_fixe": 0.5 + (tpr_f - fpr_f) / 2,
        }

    # --- Métriques globales ---
    metriques_globales = {
        "fixe": {
            "sensibilite_moy": float(np.mean(sens_fixe_sims)),
            "sensibilite_std": float(np.std(sens_fixe_sims)),
            "specificite_moy": float(np.mean(spec_fixe_sims)),
            "vpp_moy": float(np.mean(vpp_fixe_sims)),
        },
        "perso": {
            "sensibilite_moy": float(np.mean(sens_perso_sims)),
            "sensibilite_std": float(np.std(sens_perso_sims)),
            "specificite_moy": float(np.mean(spec_perso_sims)),
            "vpp_moy": float(np.mean(vpp_perso_sims)),
        },
    }

    # --- Critères de validation clinique (doc sect. 6.6) ---
    validation_clinique = _evaluer_criteres_cliniques(metriques_globales, test_student, roc_data)

    return {
        "metriques_globales": metriques_globales,
        "resultats_par_profil": resultats_par_profil,
        "test_student": test_student,
        "roc_data": roc_data,
        "validation_clinique": validation_clinique,
        "seuil_fixe_kpa": seuil_fixe_kpa,
        "n_simulations": n_simulations,
        "n_mesures_par_sim": n_mesures_par_sim,
    }


def _evaluer_criteres_cliniques(
    metriques: Dict[str, Any],
    test_student: Dict[str, Any],
    roc_data: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Évalue les critères de validation clinique décrits dans le doc sect. 6.6.

    Critères :
      1. Sensibilité personnalisée ≥ 80 %
      2. Sensibilité personnalisée > sensibilité fixe
      3. Spécificité acceptable (≥ 50 %)
      4. VPP plus élevée pour le seuil personnalisé
      5. Test de Student significatif (p < 0.05)
      6. AUC personnalisé > AUC fixe
    """
    mg_f = metriques["fixe"]
    mg_p = metriques["perso"]

    critere_sens_min = mg_p["sensibilite_moy"] >= 0.80
    critere_sens_sup = mg_p["sensibilite_moy"] > mg_f["sensibilite_moy"]
    critere_spec_ok = mg_p["specificite_moy"] >= 0.50
    critere_vpp_sup = mg_p["vpp_moy"] > mg_f["vpp_moy"]

    p_val_sens = test_student.get("sensibilite", {}).get("p_value", 1.0)
    critere_student = p_val_sens < 0.05

    auc_perso = roc_data.get("auc_perso", 0.5)
    auc_fixe = roc_data.get("auc_fixe", 0.5)
    critere_auc = auc_perso > auc_fixe

    n_criteres_ok = sum([
        critere_sens_min, critere_sens_sup, critere_spec_ok,
        critere_vpp_sup, critere_student, critere_auc
    ])

    if n_criteres_ok >= 5:
        verdict = "VALIDÉ — Le seuil personnalisé est supérieur au seuil fixe."
    elif n_criteres_ok >= 3:
        verdict = "PARTIELLEMENT VALIDÉ — Amélioration observée, validation clinique complète recommandée."
    else:
        verdict = "NON VALIDÉ — Données insuffisantes ou modèle à réévaluer."

    return {
        "critere_sens_min_80pct": critere_sens_min,
        "critere_sens_superieur": critere_sens_sup,
        "critere_spec_acceptable": critere_spec_ok,
        "critere_vpp_superieure": critere_vpp_sup,
        "critere_student_significatif": critere_student,
        "critere_auc_superieur": critere_auc,
        "n_criteres_ok": n_criteres_ok,
        "verdict": verdict,
    }


# -------------------------------------------------------
# 6.4 — Affichage console et PDF
# -------------------------------------------------------

def afficher_resultats_bloc6(result_bloc6: Dict[str, Any]) -> None:
    """Affichage console lisible des résultats du Bloc 6."""
    print("\n" + "=" * 65)
    print("              RÉSULTATS DU BLOC 6")
    print("      Validation statistique — Seuil fixe vs Personnalisé")
    print("=" * 65)

    mg = result_bloc6["metriques_globales"]
    print(f"\n  {'Métrique':20s} | {'Seuil fixe':>12s} | {'Seuil perso':>12s}")
    print(f"  {'-'*20}-+-{'-'*12}-+-{'-'*12}")
    print(f"  {'Sensibilité moy.':20s} | {mg['fixe']['sensibilite_moy']:>11.1%} | {mg['perso']['sensibilite_moy']:>11.1%}")
    print(f"  {'Spécificité moy.':20s} | {mg['fixe']['specificite_moy']:>11.1%} | {mg['perso']['specificite_moy']:>11.1%}")
    print(f"  {'VPP moy.':20s} | {mg['fixe']['vpp_moy']:>11.1%} | {mg['perso']['vpp_moy']:>11.1%}")

    ts = result_bloc6["test_student"]
    print(f"\n  Test de Student — Sensibilité :")
    if "erreur" not in ts:
        s_ts = ts.get("sensibilite", {})
        sig = "OUI ✓" if s_ts.get("significatif") else "Non"
        print(f"    t = {s_ts.get('t', 0):.3f}  |  p = {s_ts.get('p_value', 1):.4f}  |  Significatif : {sig}")

    roc = result_bloc6["roc_data"]
    print(f"\n  AUC seuil fixe       : {roc.get('auc_fixe', 0):.4f}")
    print(f"  AUC seuil personnalisé: {roc.get('auc_perso', 0):.4f}")

    vc = result_bloc6["validation_clinique"]
    print(f"\n  Critères cliniques validés : {vc['n_criteres_ok']}/6")
    print(f"  VERDICT : {vc['verdict']}")
    print("=" * 65 + "\n")


def generer_pdf_bloc6(
    result_bloc6: Dict[str, Any],
    filename: Optional[str] = None,
    logo_path: Optional[str] = None,
) -> str:
    """Génère le rapport PDF complet de validation statistique (Bloc 6)."""
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.platypus import (
            SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
            Image, HRFlowable,
        )
        from reportlab.lib.styles import getSampleStyleSheet
        from reportlab.lib import colors as rl_colors
    except ImportError:
        raise ImportError("reportlab requis. pip install reportlab")

    if filename is None:
        filename = f"Rapport_Bloc6_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"

    doc = SimpleDocTemplate(filename, pagesize=A4)
    styles = getSampleStyleSheet()
    story = []

    if logo_path and os.path.exists(logo_path):
        story.append(Image(logo_path, width=140, height=60))
        story.append(Spacer(1, 12))

    story.append(Paragraph("Rapport Digi'Feet — Bloc 6", styles["Title"]))
    story.append(Paragraph("Validation statistique : seuil fixe vs seuil personnalisé", styles["Heading2"]))
    story.append(Paragraph(f"Date : {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}", styles["Normal"]))
    story.append(Spacer(1, 10))

    # Paramètres de la simulation
    # Note méthodologique sur la simulation statique
    story.append(Paragraph(
        "<b>Note méthodologique :</b> Les données synthétiques sont générées en conditions d'appui "
        "statique. Les variations de charge entre fenêtres modélisent des <b>oscillations posturales "
        "naturelles</b> (déplacements transitoires du centre de gravité debout), et non un cycle de marche. "
        "Cette distinction est essentielle pour l'interprétation des métriques de performance.",
        styles["Italic"]
    ))
    story.append(Spacer(1, 8))

    story.append(Paragraph("1. Paramètres de la simulation", styles["Heading2"]))
    params = [
        ["Paramètre", "Valeur"],
        ["Seuil fixe de référence", f"{result_bloc6['seuil_fixe_kpa']:.0f} kPa"],
        ["Nombre de simulations par profil", str(result_bloc6['n_simulations'])],
        ["Mesures par simulation", str(result_bloc6['n_mesures_par_sim'])],
        ["Profils testés", ", ".join(result_bloc6["resultats_par_profil"].keys())],
    ]
    pt = Table(params, colWidths=[220, 270])
    pt.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), rl_colors.lightgrey),
        ("GRID", (0, 0), (-1, -1), 0.5, rl_colors.grey),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
    ]))
    story.append(pt)
    story.append(Spacer(1, 10))

    # Métriques globales
    story.append(Paragraph("2. Métriques globales (tous profils confondus)", styles["Heading2"]))
    mg = result_bloc6["metriques_globales"]
    metrics_data = [
        ["Métrique", "Seuil fixe", "Seuil personnalisé", "Δ (perso − fixe)"],
        [
            "Sensibilité moyenne",
            f"{mg['fixe']['sensibilite_moy']:.1%} ± {mg['fixe']['sensibilite_std']:.1%}",
            f"{mg['perso']['sensibilite_moy']:.1%} ± {mg['perso']['sensibilite_std']:.1%}",
            f"{mg['perso']['sensibilite_moy'] - mg['fixe']['sensibilite_moy']:+.1%}",
        ],
        [
            "Spécificité moyenne",
            f"{mg['fixe']['specificite_moy']:.1%}",
            f"{mg['perso']['specificite_moy']:.1%}",
            f"{mg['perso']['specificite_moy'] - mg['fixe']['specificite_moy']:+.1%}",
        ],
        [
            "VPP moyenne",
            f"{mg['fixe']['vpp_moy']:.1%}",
            f"{mg['perso']['vpp_moy']:.1%}",
            f"{mg['perso']['vpp_moy'] - mg['fixe']['vpp_moy']:+.1%}",
        ],
    ]
    mt = Table(metrics_data, colWidths=[130, 115, 125, 120])
    mt.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), rl_colors.lightgrey),
        ("GRID", (0, 0), (-1, -1), 0.5, rl_colors.grey),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("ALIGN", (1, 0), (-1, -1), "CENTER"),
    ]))
    story.append(mt)
    story.append(Spacer(1, 10))

    # Résultats par profil
    story.append(Paragraph("3. Résultats par profil clinique", styles["Heading2"]))
    profil_data = [["Profil", "Zone à risque", "Seuil perso moy.", "Sens. fixe", "Sens. perso", "VPP fixe", "VPP perso"]]
    for nom, rd in result_bloc6["resultats_par_profil"].items():
        profil_data.append([
            nom.replace("_", " ").capitalize(),
            rd["risk_zone"],
            f"{rd['seuil_perso_moyen']:.1f} kPa",
            f"{rd['fixe']['sensibilite_moy']:.1%}",
            f"{rd['perso']['sensibilite_moy']:.1%}",
            f"{rd['fixe']['vpp_moy']:.1%}",
            f"{rd['perso']['vpp_moy']:.1%}",
        ])
    pft = Table(profil_data, colWidths=[80, 65, 80, 65, 65, 55, 55])
    pft.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), rl_colors.lightgrey),
        ("GRID", (0, 0), (-1, -1), 0.5, rl_colors.grey),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 8),
        ("ALIGN", (2, 1), (-1, -1), "CENTER"),
    ]))
    story.append(pft)
    story.append(Spacer(1, 10))

    # Test de Student
    story.append(Paragraph("4. Test de Student (comparaison appariée)", styles["Heading2"]))
    story.append(Paragraph(
        "H0 : les deux modèles ont des performances équivalentes. "
        "H1 : le seuil personnalisé est meilleur. Seuil de significativité : p < 0,05.",
        styles["Normal"]
    ))
    story.append(Spacer(1, 6))
    ts = result_bloc6["test_student"]
    if "erreur" not in ts:
        student_data = [["Métrique", "Statistique t", "p-value", "Significatif (p<0.05)"]]
        for metrique, vals in ts.items():
            if isinstance(vals, dict):
                student_data.append([
                    metrique.capitalize(),
                    f"{vals.get('t', 0):.4f}",
                    f"{vals.get('p_value', 1):.4f}",
                    "OUI ✓" if vals.get("significatif") else "Non",
                ])
        st_tbl = Table(student_data, colWidths=[110, 110, 110, 160])
        st_tbl.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), rl_colors.lightgrey),
            ("GRID", (0, 0), (-1, -1), 0.5, rl_colors.grey),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("ALIGN", (1, 0), (-1, -1), "CENTER"),
        ]))
        story.append(st_tbl)
    else:
        story.append(Paragraph(f"Erreur lors du calcul : {ts['erreur']}", styles["Normal"]))
    story.append(Spacer(1, 10))

    # Analyse ROC
    story.append(Paragraph("5. Analyse ROC et AUC", styles["Heading2"]))
    roc = result_bloc6["roc_data"]
    story.append(Paragraph(
        f"AUC seuil fixe ({result_bloc6['seuil_fixe_kpa']:.0f} kPa) : "
        f"<b>{roc.get('auc_fixe', 0):.4f}</b>",
        styles["Normal"]
    ))
    story.append(Paragraph(
        f"AUC seuil personnalisé (bayésien adaptatif) : "
        f"<b>{roc.get('auc_perso', 0):.4f}</b>",
        styles["Normal"]
    ))
    delta_auc = roc.get("auc_perso", 0) - roc.get("auc_fixe", 0)
    story.append(Paragraph(
        f"Différence ΔAUC = {delta_auc:+.4f} "
        f"({'faveur seuil personnalisé' if delta_auc > 0 else 'faveur seuil fixe'})",
        styles["Normal"]
    ))
    story.append(Spacer(1, 10))

    # Critères de validation clinique
    story.append(Paragraph("6. Critères de validation clinique (sect. 6.6)", styles["Heading2"]))
    vc = result_bloc6["validation_clinique"]
    criteres_data = [
        ["Critère", "Statut"],
        ["Sensibilité personnalisée ≥ 80 %", "✓ Validé" if vc["critere_sens_min_80pct"] else "✗ Non validé"],
        ["Sensibilité perso > sensibilité fixe", "✓ Validé" if vc["critere_sens_superieur"] else "✗ Non validé"],
        ["Spécificité acceptable (≥ 50 %)", "✓ Validé" if vc["critere_spec_acceptable"] else "✗ Non validé"],
        ["VPP personnalisée > VPP fixe", "✓ Validé" if vc["critere_vpp_superieure"] else "✗ Non validé"],
        ["Test Student significatif (p < 0,05)", "✓ Validé" if vc["critere_student_significatif"] else "✗ Non validé"],
        ["AUC personnalisé > AUC fixe", "✓ Validé" if vc["critere_auc_superieur"] else "✗ Non validé"],
    ]
    vc_tbl = Table(criteres_data, colWidths=[350, 140])
    vc_style = [
        ("BACKGROUND", (0, 0), (-1, 0), rl_colors.lightgrey),
        ("GRID", (0, 0), (-1, -1), 0.5, rl_colors.grey),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("ALIGN", (1, 1), (-1, -1), "CENTER"),
    ]
    for i in range(1, len(criteres_data)):
        val = list(vc.values())[i - 1] if i - 1 < len(vc) else False
        if isinstance(val, bool):
            color = rl_colors.Color(0.85, 1.0, 0.85) if val else rl_colors.Color(1.0, 0.88, 0.88)
            vc_style.append(("BACKGROUND", (0, i), (-1, i), color))
    vc_tbl.setStyle(TableStyle(vc_style))
    story.append(vc_tbl)
    story.append(Spacer(1, 12))
    story.append(HRFlowable(width="100%", thickness=1, color=rl_colors.grey))
    story.append(Spacer(1, 8))
    story.append(Paragraph(
        f"<b>VERDICT ({vc['n_criteres_ok']}/6 critères validés) : {vc['verdict']}</b>",
        styles["Normal"]
    ))
    story.append(Spacer(1, 12))
    story.append(Paragraph(
        "Document généré automatiquement — Validation sur données synthétiques uniquement, "
        "sans valeur de diagnostic médical. Une validation clinique sur données réelles est requise.",
        styles["Italic"]
    ))

    doc.build(story)
    return filename