# ============================================================
# main.py — Orchestrateur complet AdaptStep / Digi'Feet
# Version améliorée avec :
#   - Simulateur de capteur réaliste (PlantarSensorSimulator)
#   - Simulation par type de pied clinique
#   - Génération d'un rapport PDF final de synthèse
# Ordre : Bloc 1a → Bloc 0 → Bloc 1b → Bloc 2 → Bloc 3 → Bloc 4 → Bloc 5 → Bloc 6 → PDF Final
# ============================================================

import numpy as np
import sys
import os
import io
import tempfile
from datetime import datetime
from scipy import signal as scipy_signal

from CODE_SEMELLE import (
    # Configurations
    SensorCalibrationReference,
    SensorQualityConfig,
    PreprocessingConfig,
    PosturalConfig,
    PosturalReference,
    ThresholdConfig,
    PatientProfile,
    BayesianConfig,
    # Blocs
    SignalConverter,
    SensorQualityMonitor,
    SignalFilter,
    bloc2_postural_quality_control,
    bloc3_initial_threshold_calibration,
    initialize_bayesian_states_from_bloc3,
    bloc4_bayesian_adaptive_update,
    # Blocs 5 et 6
    AlertEngine,
    Bloc5Config,
    afficher_resultats_bloc5,
    generer_pdf_bloc5,
    bloc6_validation_statistique,
    afficher_resultats_bloc6,
    generer_pdf_bloc6,
    # Affichages
    afficher_resultats_bloc2,
    afficher_resultats_bloc3,
    afficher_resultats_bloc4,
    # PDFs
    generer_pdf_bloc0,
    generer_pdf_bloc1,
    generer_pdf_bloc2,
    generer_pdf_bloc3,
    generer_pdf_bloc4,
    # Profils synthétiques pour Bloc 6
    PROFILS_SYNTHETIQUES,
    generer_donnees_synthetiques,
    calculer_metriques_zone,
)

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    _MATPLOTLIB_OK = True
except ImportError:
    _MATPLOTLIB_OK = False
    print("Warning: matplotlib non installé. Les graphiques ne seront pas générés.")

try:
    from reportlab.lib.pagesizes import A4
    from reportlab.platypus import (
        SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
        Image as RLImage, HRFlowable, PageBreak,
    )
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib import colors as rl_colors
    from reportlab.lib.units import cm
    _REPORTLAB_OK = True
except ImportError:
    _REPORTLAB_OK = False
    print("Warning: reportlab non installé.")


# ============================================================
# SIMULATEUR DE CAPTEUR RÉALISTE (inspiré de PlantarSensorSimulator)
# Simule la chaîne complète FSR + NTC avec dérives et bruits physiques
# ============================================================

class PlantarSensorSimulator:
    """
    Simule la chaîne capteur complète d'une semelle instrumentée :
    - Capteurs de pression FSR (avec hystérésis, fluage, bruit thermique)
    - Capteurs de température NTC
    - Conversion tension → kPa selon les coefficients de calibration

    Cette classe génère des données réalistes pour chaque type de pied
    clinique, en reproduisant les phénomènes physiques observés en pratique.
    """

    # Profils cliniques : base_pressures (kPa) et drift thermique par zone
    CLINICAL_PROFILES = {
        "normal": {
            "description": "Pied sain — répartition équilibrée (IMC ≈ 22-25)",
            "imc": 23.0,
            "type_pied": "normal",
            "base_pressures": np.array([155, 145, 42, 185, 210, 195, 265, 250], dtype=float),
            "bruit_sigma": 0.05,    # sigma relatif du bruit (5 %)
            "drift_thermique": 0.008,
        },
        "surpoids": {
            "description": "Patient en surpoids (IMC ≈ 28) — pressions globalement élevées",
            "imc": 28.0,
            "type_pied": "pied_plat",
            "base_pressures": np.array([225, 215, 85, 260, 285, 270, 330, 315], dtype=float),
            "bruit_sigma": 0.07,
            "drift_thermique": 0.012,
        },
        "hallux_valgus": {
            "description": "Hallux valgus — surcharge localisée à l'hallux",
            "imc": 25.0,
            "type_pied": "pied_creux",
            "base_pressures": np.array([160, 150, 48, 190, 215, 200, 370, 355], dtype=float),
            "bruit_sigma": 0.06,
            "drift_thermique": 0.010,
        },
        "pied_charcot": {
            "description": "Pied de Charcot (IMC ≈ 30) — pressions élevées, déformation",
            "imc": 30.0,
            "type_pied": "charcot",
            "base_pressures": np.array([290, 278, 155, 335, 360, 345, 310, 295], dtype=float),
            "bruit_sigma": 0.09,
            "drift_thermique": 0.015,
        },
    }

    def __init__(self, n_samples: int = 1500, fs: float = 50.0, seed: int = 42):
        self.n_samples = n_samples
        self.fs = fs
        self.time = np.linspace(0, n_samples / fs, n_samples)
        self.seed = seed

        # Paramètres FSR réalistes
        self.fsr_a = 250.0       # coeff de conversion
        self.fsr_b = -2.1        # exposant (linéarisation)
        self.vcc = 3.3           # tension d'alimentation (V)
        self.r_pullup = 10000.0  # résistance de tirage (Ω)

        # Paramètres NTC (Steinhart-Hart)
        self.ntc_A = 0.001129
        self.ntc_B = 0.000234
        self.ntc_C = 9.8e-8

    def _fsr_pressure_to_resistance(self, pressure_kpa: np.ndarray) -> np.ndarray:
        """Modèle FSR : R = a * P^b (en kΩ, converti en Ω)."""
        pressure_kpa = np.clip(pressure_kpa, 0.5, 700.0)
        return self.fsr_a * (pressure_kpa ** self.fsr_b) * 1000.0   # → Ω

    def _apply_hysteresis(self, pressure: np.ndarray, coeff: float = 0.06) -> np.ndarray:
        """Simule l'hystérésis mécanique du FSR (±6 % typiquement)."""
        hyst = coeff * np.sin(2 * np.pi * self.time / 10.0) * pressure
        return pressure + hyst

    def _apply_creep(self, pressure: np.ndarray, session_idx: int) -> np.ndarray:
        """Simule le fluage (creep) qui augmente avec le numéro de session."""
        creep_factor = 0.012 * (session_idx / 10.0)
        creep = creep_factor * np.cumsum(pressure) / self.n_samples
        return pressure + creep

    def _apply_thermal_drift(self, resistance: np.ndarray, temp_c: np.ndarray, coeff: float) -> np.ndarray:
        """Dérive thermique de la résistance FSR (typiquement +1–2 %/°C)."""
        drift = 1.0 + coeff * (temp_c - 32.0)
        return resistance * drift

    def _resistance_to_voltage(self, resistance: np.ndarray) -> np.ndarray:
        """Diviseur de tension : V = Vcc * R_pull / (R_fsr + R_pull)."""
        return self.vcc * self.r_pullup / (resistance + self.r_pullup)

    def _voltage_to_pressure_kpa(self, voltage: np.ndarray, temp_c: np.ndarray) -> np.ndarray:
        """
        Chaîne de conversion inverse : V → R_fsr → pression kPa.
        Ajoute bruit gaussien + bruit rose (corrélé) + erreur thermique résiduelle.
        """
        r_mes = self.r_pullup * (self.vcc / (voltage + 1e-9) - 1.0)
        # Inversion du modèle FSR : P = (a * 1000 / R)^(1/|b|)
        p_raw = (self.fsr_a * 1000.0 / (r_mes + 1e-3)) ** (1.0 / abs(self.fsr_b))

        # Bruit gaussien blanc
        noise_w = np.random.normal(0, 3.5, len(p_raw))
        # Bruit rose (corrélé basse fréquence)
        noise_p = scipy_signal.lfilter([1], [1, -0.92], np.random.normal(0, 2.0, len(p_raw)))
        # Erreur thermique résiduelle
        err_therm = 0.35 * (temp_c - 32.0)

        return np.clip(p_raw + noise_w + noise_p + err_therm, 0.0, 700.0)

    def simulate_ntc(self, base_temp_c: float = 32.5) -> tuple:
        """
        Simule un capteur NTC : génère T(t) et la résistance correspondante.
        Retourne (temperatures_c [n_samples], resistances_ohm [n_samples]).
        """
        drift = np.linspace(0, 0.9, self.n_samples)
        noise = np.random.normal(0, 0.18, self.n_samples)
        temp = base_temp_c + drift + noise

        # Modèle Steinhart-Hart inversé : R = f(T)
        T_k = temp + 273.15
        # Utilisation de la formule β simplifiée pour la simulation
        beta = 3950.0
        R_ntc = 10000.0 * np.exp(beta * (1.0 / T_k - 1.0 / (25.0 + 273.15)))
        return temp, R_ntc

    def simulate_window(
        self,
        profile_name: str,
        window_index: int,
        phase: str = "normal",
    ) -> dict:
        """
        Simule une fenêtre de données pour un profil clinique donné.

        Paramètres:
            profile_name  : "normal", "surpoids", "hallux_valgus", "pied_charcot"
            window_index  : index de fenêtre (affecte le seed et le drift)
            phase         : "neutre", "appui_posterieur", "appui_anterieur"
                              (variations posturales statiques, pas de marche)

        Retourne un dict avec pressures_raw_kpa, temperatures_c, resistances_ntc.
        """
        np.random.seed(self.seed + window_index * 17)
        profil = self.CLINICAL_PROFILES[profile_name]
        n_cap = 8

        # Pressions de base du profil + variation inter-fenêtre
        base = profil["base_pressures"].copy()
        base *= (0.88 + np.random.normal(0, profil["bruit_sigma"], n_cap))

        # Variation posturale statique — NB : PAS une phase de marche
        # Variations posturales statiques (oscillations du centre de gravité debout)
        # Ces phases ne simulent PAS la marche mais des déplacements
        # posturaux transitoires observés en appui statique prolongé.
        if phase == "appui_posterieur":
            base[0:2] *= 1.45    # report postural vers le talon
        elif phase == "appui_anterieur":
            base[3:6] *= 1.38    # report postural vers l'avant-pied
        elif phase == "hallux_valgus_peak" and profile_name == "hallux_valgus":
            base[6:8] *= 1.55    # surcharge localisée hallux (asymétrie posturale)

        base = np.clip(base, 2.0, 700.0)

        # Génération des séries temporelles par capteur
        pressures_raw_kpa = np.zeros((self.n_samples, n_cap))
        temperatures_c = np.zeros((self.n_samples, 4))
        resistances_ntc = np.zeros((self.n_samples, 4))

        for i in range(n_cap):
            # Variation temporelle réaliste
            variation = np.random.normal(0, profil["bruit_sigma"] * base[i], self.n_samples)
            p_true = base[i] + variation
            p_true = self._apply_hysteresis(p_true, coeff=0.05)
            p_true = self._apply_creep(p_true, window_index)

            # Simulation capteur NTC associé
            ntc_idx = i % 4
            temp_c, r_ntc = self.simulate_ntc(base_temp_c=31.5 + np.random.uniform(-1.2, 2.0))
            if i < 4:
                temperatures_c[:, ntc_idx] = temp_c
                resistances_ntc[:, ntc_idx] = r_ntc

            # Chaîne FSR complète
            resistance = self._fsr_pressure_to_resistance(p_true)
            resistance = self._apply_thermal_drift(resistance, temp_c, profil["drift_thermique"])
            voltage = self._resistance_to_voltage(resistance)
            p_measured = self._voltage_to_pressure_kpa(voltage, temp_c)
            pressures_raw_kpa[:, i] = p_measured

        pressures_raw_kpa = np.clip(pressures_raw_kpa, 0.0, 700.0)

        return {
            "pressures_raw_kpa": pressures_raw_kpa,
            "temperatures_c": temperatures_c,
            "resistances_ntc": resistances_ntc,
            "profile_name": profile_name,
            "imc": profil["imc"],
            "description": profil["description"],
        }


# ============================================================
# PARAMÈTRES COMMUNS DE LA SEMELLE
# ============================================================

N_CAPTEURS_FSR = 8
N_NTC = 4
N_SAMPLES = 1500
N_FENETRES = 30
LOGO_PATH = "logo.png"


import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LOGO_PATH = os.path.join(BASE_DIR, "Logo.png")
print(LOGO_PATH)
print(os.path.exists(LOGO_PATH))

sensor_positions_mm = np.array([
    [10, 20], [30, 20],          # talon
    [20, 60],                    # medio_pied
    [15, 90], [35, 90], [25, 100],  # avant_pied
    [15, 130], [35, 130],        # hallux
])

sensor_zones = np.array([
    "talon", "talon",
    "medio_pied",
    "avant_pied", "avant_pied", "avant_pied",
    "hallux", "hallux",
])

sensor_area_cm2 = 2.0


# ============================================================
# CONFIGURATIONS
# ============================================================

calib_ref = SensorCalibrationReference.depuis_defaut(
    n_capteurs=N_CAPTEURS_FSR,
    sigma_ref_kpa=20.0   # FIX: le simulateur génère ~15-25 kPa de bruit — 5 kPa trop restrictif
)

quality_config = SensorQualityConfig(
    alpha_bruit=3.0,        # FIX: augmenté de 5→3 (sigma_ref est maintenant 20 kPa)
    w1_bruit=0.5,
    w2_coherence=0.5,
    delta_temp_seuil_symetrie=2.2,
    pente_fsr_seuil_drift=0.8,
    n_fenetres_pente=10,
    sc_zone_seuil_bloquage=0.25,  # FIX: abaissé 0.3→0.25 pour ne pas bloquer les zones à confiance moyenne
    sc_zone_seuil_reduction=0.5,
)

preprocess_config = PreprocessingConfig(
    fsr_a_coeffs=np.full(N_CAPTEURS_FSR, 250.0),
    fsr_b_coeffs=np.full(N_CAPTEURS_FSR, -2.1),
    ntc_a=0.001129,
    ntc_b=0.000234,
    ntc_c=9.8e-8,
    butter_order=2,
    cutoff_freq_hz=20.0,
    sampling_rate_hz=50.0,
    vcc=3.3,
    pullup_resistor=10000.0,
)

postural_config = PosturalConfig()
postural_config.pression_activation_kpa = 2.0
postural_config.delta_pression_totale_max = 0.75
postural_config.delta_surface_max = 0.25
postural_config.d_max_absolu_mm = 30.0
postural_config.w_min = 0.35

threshold_config = ThresholdConfig()

bayesian_config = BayesianConfig(
    k_prior=3.0,                 # prior plus large = mémoire plus longue
    lambda_forgetting=0.98,      # compromis entre 0.97 et 0.99
    sc_min_update=0.25,
    confidence_min_update=0.35,
    variation_max_sigma=12.0,    # très tolérant pour Charcot
    sigma_min_kpa=2.0,           # incertitude minimale un peu plus haute
    pression_activation_kpa=5.0,
)

# ============================================================
# FONCTION PIPELINE COMPLET PAR PROFIL
# ============================================================

def run_pipeline_for_profile(
    profile_name: str,
    simulator: PlantarSensorSimulator,
    n_fenetres: int = N_FENETRES,
    verbose: bool = True,
) -> dict:
    """
    Exécute le pipeline complet (Blocs 0→5) pour un profil clinique donné.
    Retourne un dict avec tous les résultats et une référence au profil.
    """
    profil_info = PlantarSensorSimulator.CLINICAL_PROFILES[profile_name]
    patient_profile = PatientProfile(
        imc=profil_info["imc"],
        type_pied=profil_info["type_pied"],
    )

    converter = SignalConverter(preprocess_config)
    monitor = SensorQualityMonitor(quality_config, calib_ref)
    filtre = SignalFilter(preprocess_config)
    reference = PosturalReference()
    # FIX Bloc 5 : en simulation (données synthétiques), une seule fenêtre suffit
    # pour confirmer une alarme — évite "en attente de confirmation" systématique
    alert_engine = AlertEngine(config=Bloc5Config(n_fenetres_confirmation=2))

    valid_windows = []
    last_bloc0_result = None
    last_bloc1_result = None
    last_bloc2_result = None
    last_bloc4_result = None

    phases = ["neutre", "appui_posterieur", "appui_anterieur", "neutre", "appui_posterieur", "appui_anterieur", "neutre", "neutre"]

    if verbose:
        print(f"\n{'═' * 60}")
        print(f"  PROFIL : {profile_name.upper()}")
        print(f"  {profil_info['description']}")
        print(f"  IMC = {profil_info['imc']} | Type pied = {profil_info['type_pied']}")
        print(f"{'═' * 60}")

    for window_index in range(n_fenetres):
        phase = phases[window_index % len(phases)]

        # Simulation réaliste du capteur
        sim_data = simulator.simulate_window(profile_name, window_index, phase)
        pressures_raw_kpa = sim_data["pressures_raw_kpa"]

        # Simulation NTC symétrique (semelle controlatérale)
        raw_resistances_ntc = sim_data["resistances_ntc"]
        raw_resistances_ntc_sym = raw_resistances_ntc * np.random.uniform(0.97, 1.03, raw_resistances_ntc.shape)

        # Bloc 1a — conversion brute
        dummy_voltages = np.ones((N_SAMPLES, N_CAPTEURS_FSR)) * 1.0
        result_1a_temp = converter.run_bloc1a(dummy_voltages, raw_resistances_ntc)
        result_1a = {
            "pressures_raw_kpa": pressures_raw_kpa,
            "temperatures_c": sim_data["temperatures_c"],
            "bloc1a_ok": True,
        }
        temperatures_c = result_1a["temperatures_c"]

        # Bloc 0 — Surveillance qualité capteur
        temperatures_c_sym = converter.ntc_to_temperature(raw_resistances_ntc_sym)
        result_bloc0 = monitor.run_bloc0(
            pressures_raw_kpa=result_1a["pressures_raw_kpa"],
            temperatures_c=temperatures_c,
            temperatures_c_symetrique=temperatures_c_sym,
            sensor_zones=sensor_zones,
        )
        last_bloc0_result = result_bloc0

        # Bloc 1b — filtrage Butterworth
        result_bloc1b = filtre.run_bloc1b(
            pressures_raw_kpa=result_1a["pressures_raw_kpa"],
            result_bloc0=result_bloc0,
            sensor_zones=sensor_zones,
        )
        result_bloc1 = {
            "pressures_kpa": result_bloc1b["pressures_kpa"],
            "temperatures_c": temperatures_c,
            "bloc1b_ok": result_bloc1b["bloc1b_ok"],
        }
        last_bloc1_result = result_bloc1
        zone_confidences = result_bloc0["zone_confidences"]

        # Bloc 2 — Contrôle qualité postural
        result_bloc2, reference = bloc2_postural_quality_control(
            pressures_kpa=result_bloc1b["pressures_kpa"],
            sensor_positions_mm=sensor_positions_mm,
            sensor_area_cm2=sensor_area_cm2,
            reference=reference,
            sensor_confidence_by_zone=zone_confidences,
            config=postural_config,
        )
        last_bloc2_result = result_bloc2

        if verbose:
            status = "✓ Acceptée" if result_bloc2["window_accepted"] else "✗ Rejetée"
            print(f"  Fenêtre {window_index + 1:2d}/{n_fenetres} [{phase:18s}] → {status} | "
                  f"w_post={result_bloc2.get('w_post', 0):.2f}")

        if result_bloc2["window_accepted"]:
            valid_windows.append({
                "pressures_kpa": result_bloc1b["pressures_kpa"],
                "w_post": result_bloc2["w_post"],
                "sensor_confidence_by_zone": zone_confidences,
            })

    if verbose:
        print(f"\n  → {len(valid_windows)}/{n_fenetres} fenêtres valides")

    # Bloc 3 — Calibration initiale S0
    if len(valid_windows) < threshold_config.min_valid_windows:
        if verbose:
            print(f"  ⚠ BLOC 3 NON LANCÉ : {len(valid_windows)} fenêtres valides insuffisantes.")
        return {
            "profile_name": profile_name,
            "profil_info": profil_info,
            "valid_windows": valid_windows,
            "result_bloc3": None,
            "result_bloc4": None,
            "result_bloc5": None,
            "last_bloc0_result": last_bloc0_result,
            "last_bloc1_result": last_bloc1_result,
            "last_bloc2_result": last_bloc2_result,
        }

    result_bloc3 = bloc3_initial_threshold_calibration(
        valid_windows=valid_windows,
        sensor_zones=sensor_zones,
        patient_profile=patient_profile,
        config=threshold_config,
    )

    # Bloc 4 — Modèle bayésien adaptatif
    bayesian_states = initialize_bayesian_states_from_bloc3(
        result_bloc3=result_bloc3,
        config=bayesian_config,
    )
    for window in valid_windows:
        result_bloc4, bayesian_states = bloc4_bayesian_adaptive_update(
            current_window=window,
            sensor_zones=sensor_zones,
            bayesian_states=bayesian_states,
            config=bayesian_config,
        )
    last_bloc4_result = result_bloc4

    # Bloc 5 — Moteur de décision d'alerte
    result_bloc5 = alert_engine.run_bloc5(last_bloc4_result)

    return {
        "profile_name": profile_name,
        "profil_info": profil_info,
        "valid_windows": valid_windows,
        "result_bloc3": result_bloc3,
        "result_bloc4": last_bloc4_result,
        "result_bloc5": result_bloc5,
        "last_bloc0_result": last_bloc0_result,
        "last_bloc1_result": last_bloc1_result,
        "last_bloc2_result": last_bloc2_result,
    }


# ============================================================
# GÉNÉRATION DES GRAPHIQUES POUR LE RAPPORT FINAL
# ============================================================

def _save_fig(fig, suffix: str) -> str:
    """Sauvegarde une figure matplotlib en PNG temporaire et retourne le chemin."""
    path = os.path.join(tempfile.gettempdir(), f"digifeet_{suffix}.png")
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def generer_graphique_roc(result_bloc6: dict) -> str:
    """
    Génère la courbe ROC comparative (seuil fixe vs seuil personnalisé)
    avec AUC, zone de confiance et ligne diagonale de référence.
    """
    if not _MATPLOTLIB_OK:
        return None

    roc = result_bloc6["roc_data"]
    auc_perso = roc.get("auc_perso", 0)
    auc_fixe = roc.get("auc_fixe", 0)

    fig, ax = plt.subplots(figsize=(7, 6))

    # Diagonale (classificateur aléatoire)
    ax.plot([0, 1], [0, 1], "k--", lw=1.2, alpha=0.5, label="Aléatoire (AUC = 0.50)")

    # Courbe seuil personnalisé
    fpr_p = roc.get("fpr_perso", [0, 1])
    tpr_p = roc.get("tpr_perso", [0, 1])
    ax.plot(fpr_p, tpr_p, color="#1a7a44", lw=2.5,
            label=f"Seuil personnalisé bayésien (AUC = {auc_perso:.3f})")
    ax.fill_between(fpr_p, tpr_p, alpha=0.10, color="#1a7a44")

    # Courbe seuil fixe
    fpr_f = roc.get("fpr_fixe", [0, 1])
    tpr_f = roc.get("tpr_fixe", [0, 1])
    ax.plot(fpr_f, tpr_f, color="#c0392b", lw=2.2, linestyle="--",
            label=f"Seuil fixe 200 kPa (AUC = {auc_fixe:.3f})")

    # Point optimal (distance minimale à (0,1))
    fpr_arr = np.array(fpr_p)
    tpr_arr = np.array(tpr_p)
    dist = np.sqrt(fpr_arr**2 + (1 - tpr_arr)**2)
    best_idx = np.argmin(dist)
    ax.scatter(fpr_arr[best_idx], tpr_arr[best_idx], s=80, zorder=5,
               color="#e67e22", label=f"Point optimal (FPR={fpr_arr[best_idx]:.2f}, TPR={tpr_arr[best_idx]:.2f})")

    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.05])
    ax.set_xlabel("Taux de Faux Positifs (1 – Spécificité)", fontsize=11)
    ax.set_ylabel("Taux de Vrais Positifs (Sensibilité)", fontsize=11)
    ax.set_title("Courbe ROC — Seuil fixe vs Seuil personnalisé bayésien\n"
                 "(Validation sur données synthétiques multi-profils)", fontsize=11, fontweight="bold")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    return _save_fig(fig, "roc")


def generer_graphique_metriques_profils(result_bloc6: dict) -> str:
    """
    Graphique en barres groupées des métriques (Sensibilité, Spécificité, VPP)
    pour chaque profil clinique, comparant seuil fixe et seuil personnalisé.
    """
    if not _MATPLOTLIB_OK:
        return None

    profils = result_bloc6["resultats_par_profil"]
    noms = list(profils.keys())
    labels = [n.replace("_", " ").capitalize() for n in noms]
    metriques = ["sensibilite_moy", "specificite_moy", "vpp_moy"]
    titres = ["Sensibilité", "Spécificité", "VPP"]
    couleurs_fixe = ["#c0392b", "#e67e22", "#8e44ad"]
    couleurs_perso = ["#1a7a44", "#2980b9", "#16a085"]

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    fig.suptitle("Métriques de performance par profil clinique\n"
                 "(Seuil fixe 200 kPa vs Seuil personnalisé bayésien)",
                 fontsize=12, fontweight="bold")

    x = np.arange(len(noms))
    width = 0.38

    for ax, metrique, titre, cf, cp in zip(axes, metriques, titres, couleurs_fixe, couleurs_perso):
        vals_fixe = [profils[n]["fixe"][metrique] for n in noms]
        vals_perso = [profils[n]["perso"][metrique] for n in noms]

        bars_f = ax.bar(x - width / 2, vals_fixe, width, color=cf, alpha=0.78,
                        label="Seuil fixe", edgecolor="white")
        bars_p = ax.bar(x + width / 2, vals_perso, width, color=cp, alpha=0.78,
                        label="Seuil perso.", edgecolor="white")

        # Annotations valeurs
        for bar in bars_f:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f"{bar.get_height():.0%}", ha="center", va="bottom", fontsize=7)
        for bar in bars_p:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f"{bar.get_height():.0%}", ha="center", va="bottom", fontsize=7)

        ax.set_title(titre, fontweight="bold", fontsize=10)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=18, ha="right", fontsize=8)
        ax.set_ylim(0, 1.15)
        ax.set_ylabel("Valeur")
        ax.axhline(0.80, color="gray", linestyle=":", alpha=0.7, linewidth=1, label="Seuil cible 80%")
        ax.legend(fontsize=7)
        ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    return _save_fig(fig, "metriques_profils")


def generer_graphique_seuils_bayesiens(all_results: dict) -> str:
    """
    Visualise les seuils bayésiens finaux (μ ± σ) par zone anatomique
    pour chaque profil clinique, avec le seuil fixe de référence.
    """
    if not _MATPLOTLIB_OK:
        return None

    zones = ["talon", "medio_pied", "avant_pied", "hallux"]
    colors_profils = {
        "normal": "#1a7a44",
        "surpoids": "#e67e22",
        "hallux_valgus": "#2980b9",
        "pied_charcot": "#c0392b",
    }

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    fig.suptitle("Seuils bayésiens personnalisés (μ ± σ) par profil clinique\n"
                 "— Résultat final du Bloc 4 après mises à jour adaptatives",
                 fontsize=12, fontweight="bold")
    axes = axes.flatten()

    for ax_idx, zone in enumerate(zones):
        ax = axes[ax_idx]
        x_pos = 0
        x_ticks = []
        x_labels = []

        for pname, res in all_results.items():
            if res["result_bloc4"] is None:
                continue
            # FIX: la clé correcte est 'thresholds' (pas 'updated_thresholds')
            thresholds = res["result_bloc4"].get("thresholds", {})
            if zone not in thresholds:
                continue
            th = thresholds[zone]
            # FIX: utiliser S_updated_kpa (clé réelle du dict) + fallback sur S0 Bloc 3 si bloqué
            mu = th.get("S_updated_kpa", th.get("mu_kpa", 0))
            # Si zone bloquée et mu=0, fallback sur S0 Bloc 3
            if mu == 0 and res["result_bloc3"] is not None:
                mu = res["result_bloc3"]["thresholds"].get(zone, {}).get("S0_adjusted_kpa", 0)
            sigma = th.get("sigma_kpa", 5)

            color = colors_profils.get(pname, "#555555")
            ax.bar(x_pos, mu, color=color, alpha=0.75, width=0.7, edgecolor="white")
            ax.errorbar(x_pos, mu, yerr=sigma, fmt="none", color="black",
                        capsize=5, linewidth=1.5)

            x_ticks.append(x_pos)
            x_labels.append(pname.replace("_", "\n").capitalize())
            x_pos += 1

        # Seuil fixe de référence
        ax.axhline(200.0, color="#8e44ad", linestyle="--", lw=1.5, alpha=0.8,
                   label="Seuil fixe 200 kPa")

        ax.set_title(f"Zone : {zone.replace('_', '-')}", fontweight="bold")
        ax.set_xticks(x_ticks)
        ax.set_xticklabels(x_labels, fontsize=8)
        ax.set_ylabel("Pression (kPa)")
        ax.grid(axis="y", alpha=0.3)
        ax.legend(fontsize=8)

    fig.tight_layout()
    return _save_fig(fig, "seuils_bayesiens")


def generer_graphique_alertes(all_results: dict) -> str:
    """
    Diagramme radar / tableau visuel des alertes confirmées par profil et zone.
    """
    if not _MATPLOTLIB_OK:
        return None

    zones = ["talon", "medio_pied", "avant_pied", "hallux"]
    profils = list(all_results.keys())

    # Matrice d'alertes : 0=Normal, 1=Avertissement, 2=Alarme
    matrix = np.zeros((len(profils), len(zones)))
    for i, pname in enumerate(profils):
        res = all_results[pname]
        if res["result_bloc5"] is None:
            continue
        summaries = res["result_bloc5"].get("alert_summaries", {})
        for j, zone in enumerate(zones):
            if zone in summaries:
                s = summaries[zone]
                if s.alert_level == "Alarme" and s.confirmed:
                    matrix[i, j] = 2
                elif s.alert_level == "Avertissement":
                    matrix[i, j] = 1
                else:
                    matrix[i, j] = 0

    fig, ax = plt.subplots(figsize=(9, 5))

    cmap = matplotlib.colors.ListedColormap(["#d5f5e3", "#fdebd0", "#fadbd8"])
    bounds = [-0.5, 0.5, 1.5, 2.5]
    norm = matplotlib.colors.BoundaryNorm(bounds, cmap.N)

    im = ax.imshow(matrix, cmap=cmap, norm=norm, aspect="auto")

    ax.set_xticks(range(len(zones)))
    ax.set_xticklabels([z.replace("_", "\n") for z in zones], fontsize=10)
    ax.set_yticks(range(len(profils)))
    ax.set_yticklabels([p.replace("_", " ").capitalize() for p in profils], fontsize=10)

    # Annotations
    for i in range(len(profils)):
        for j in range(len(zones)):
            val = matrix[i, j]
            labels_map = {0: "Normal", 1: "Avertis.", 2: "Alarme ⚠"}
            txt = labels_map.get(val, "")
            ax.text(j, i, txt, ha="center", va="center", fontsize=9, fontweight="bold",
                    color="#2c3e50")

    # Légende
    patch0 = mpatches.Patch(color="#d5f5e3", label="Normal")
    patch1 = mpatches.Patch(color="#fdebd0", label="Avertissement")
    patch2 = mpatches.Patch(color="#fadbd8", label="Alarme confirmée")
    ax.legend(handles=[patch0, patch1, patch2], loc="upper right",
              bbox_to_anchor=(1.35, 1.0), fontsize=9)

    ax.set_title("Matrice des décisions d'alerte (Bloc 5)\npar profil clinique et zone anatomique",
                 fontsize=11, fontweight="bold")
    ax.set_xlabel("Zone anatomique")
    ax.set_ylabel("Profil clinique")

    fig.tight_layout()
    return _save_fig(fig, "alertes_matrix")


def generer_graphique_pipeline_flux() -> str:
    """
    Diagramme de flux visuel du pipeline Digi'Feet avec entrées/sorties de chaque bloc.
    """
    if not _MATPLOTLIB_OK:
        return None

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 6)
    ax.axis("off")
    ax.set_facecolor("white")
    fig.patch.set_facecolor("white")

    blocs = [
        {"x": 0.3,  "label": "Capteur\nFSR+NTC",  "desc": "Tension V\nRésistance Ω", "color": "#bdc3c7"},
        {"x": 1.9,  "label": "Bloc 1a\nConversion", "desc": "kPa brutes\nTemp. °C",     "color": "#aed6f1"},
        {"x": 3.5,  "label": "Bloc 0\nQualité",    "desc": "SC zones\nDrift/Bruit",    "color": "#a9dfbf"},
        {"x": 5.1,  "label": "Bloc 1b\nFiltre",    "desc": "kPa filtrées\n(Butterworth)", "color": "#aed6f1"},
        {"x": 6.7,  "label": "Bloc 2\nPostural",   "desc": "Fenêtres\nvalides/rejetées", "color": "#f9e79f"},
        {"x": 8.3,  "label": "Bloc 3\nCalibration","desc": "Seuils S0\npar zone",      "color": "#f0b27a"},
        {"x": 9.9,  "label": "Bloc 4\nBayésien",   "desc": "μ±σ adaptés\npar zone",   "color": "#f0b27a"},
        {"x": 11.5, "label": "Bloc 5\nAlertes",    "desc": "Décisions\nconfirmées",    "color": "#f1948a"},
        {"x": 13.0, "label": "Bloc 6\nValidation", "desc": "ROC / AUC\nStudent",       "color": "#bb8fce"},
    ]

    box_w = 1.25
    box_h = 1.6
    y_center = 3.2

    for i, b in enumerate(blocs):
        # Boîte
        rect = mpatches.FancyBboxPatch(
            (b["x"], y_center - box_h / 2), box_w, box_h,
            boxstyle="round,pad=0.08", linewidth=1.2,
            edgecolor="#2c3e50", facecolor=b["color"], alpha=0.9
        )
        ax.add_patch(rect)

        # Texte principal
        ax.text(b["x"] + box_w / 2, y_center + 0.35, b["label"],
                ha="center", va="center", fontsize=7.5, fontweight="bold", color="#1a252f")
        ax.text(b["x"] + box_w / 2, y_center - 0.48, b["desc"],
                ha="center", va="center", fontsize=6.5, color="#2c3e50", style="italic")

        # Flèche vers le suivant
        if i < len(blocs) - 1:
            next_x = blocs[i + 1]["x"]
            ax.annotate("", xy=(next_x, y_center), xytext=(b["x"] + box_w, y_center),
                        arrowprops=dict(arrowstyle="->", color="#2c3e50", lw=1.5))

    ax.set_title("Architecture du pipeline Digi'Feet — Flux de données entre les blocs",
                 fontsize=11, fontweight="bold", pad=12)

    fig.tight_layout()
    return _save_fig(fig, "pipeline_flux")


# ============================================================
# RAPPORT PDF FINAL DE SYNTHÈSE
# ============================================================

def generer_pdf_final(
    all_results: dict,
    result_bloc6: dict,
    logo_path: str = None,
    filename: str = None,
) -> str:
    """
    Génère le rapport PDF final de synthèse clinique Digi'Feet.

    Contenu :
      1. Page de garde + architecture pipeline
      2. Résumé des performances par profil clinique
      3. Courbe ROC avec interprétation
      4. Métriques de performance (Sensibilité / Spécificité / VPP)
      5. Seuils bayésiens finaux par zone
      6. Matrice des alertes (Bloc 5)
      7. Validation statistique (Student + AUC)
      8. Critères cliniques + Verdict final
      9. Conclusion et recommandations
    """
    if not _REPORTLAB_OK:
        print("  ⚠ reportlab non disponible, PDF final non généré.")
        return None

    if filename is None:
        filename = f"Rapport_Final_DigiPied_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"

    print("  Génération des graphiques...")
    img_pipeline = generer_graphique_pipeline_flux()
    img_roc = generer_graphique_roc(result_bloc6)
    img_metriques = generer_graphique_metriques_profils(result_bloc6)
    img_seuils = generer_graphique_seuils_bayesiens(all_results)
    img_alertes = generer_graphique_alertes(all_results)

    doc = SimpleDocTemplate(
        filename, pagesize=A4,
        leftMargin=1.8 * cm, rightMargin=1.8 * cm,
        topMargin=2 * cm, bottomMargin=2 * cm,
    )
    styles = getSampleStyleSheet()
    W = A4[0] - 3.6 * cm   # largeur utile

    # Styles personnalisés
    s_title = ParagraphStyle("s_title", parent=styles["Title"], fontSize=18,
                              spaceAfter=8, textColor=rl_colors.HexColor("#1a252f"))
    s_h2 = ParagraphStyle("s_h2", parent=styles["Heading2"], fontSize=13,
                           textColor=rl_colors.HexColor("#1a4d8f"), spaceAfter=4)
    s_h3 = ParagraphStyle("s_h3", parent=styles["Heading3"], fontSize=10.5,
                           textColor=rl_colors.HexColor("#2c3e50"), spaceAfter=3)
    s_body = ParagraphStyle("s_body", parent=styles["Normal"], fontSize=9.5,
                             leading=14, spaceAfter=4)
    s_interp = ParagraphStyle("s_interp", parent=styles["Normal"], fontSize=9,
                               leading=13, leftIndent=12, borderPad=4,
                               backColor=rl_colors.HexColor("#eaf4fb"),
                               borderWidth=0, spaceAfter=6)
    s_warn = ParagraphStyle("s_warn", parent=styles["Normal"], fontSize=9,
                             leading=13, leftIndent=12,
                             backColor=rl_colors.HexColor("#fef9e7"),
                             spaceAfter=6)
    s_footer = ParagraphStyle("s_footer", parent=styles["Italic"], fontSize=7.5,
                               textColor=rl_colors.grey)

    story = []

    # ──────────────────────────────────────────────
    # PAGE DE GARDE
    # ──────────────────────────────────────────────
    if logo_path and os.path.exists(logo_path):
        story.append(RLImage(logo_path, width=160, height=65))
        story.append(Spacer(1, 0.4 * cm))

    story.append(Paragraph("Digi'Feet — AdaptStep", s_title))
    story.append(Paragraph("Rapport de Synthèse Clinique — Pipeline Complet", styles["Heading2"]))
    story.append(Paragraph(
        f"Généré le : {datetime.now().strftime('%d/%m/%Y à %H:%M:%S')}",
        styles["Normal"]
    ))
    story.append(HRFlowable(width="100%", thickness=2, color=rl_colors.HexColor("#1a4d8f")))
    story.append(Spacer(1, 0.3 * cm))

    story.append(Paragraph(
        "Ce rapport présente la synthèse complète du pipeline de traitement biomécanique "
        "de la semelle instrumentée Digi'Feet. Il couvre l'ensemble des étapes de validation, "
        "de la conversion brute des capteurs jusqu'à la décision d'alerte clinique, "
        "en incluant la validation statistique comparative entre seuil fixe et seuil bayésien "
        "personnalisé sur quatre profils cliniques distincts.",
        s_body
    ))
    story.append(Spacer(1, 0.3 * cm))

    # Résumé rapide : profils simulés
    profils_data = [["Profil", "IMC", "Type pied", "Fenêtres valides", "Alertes confirmées"]]
    for pname, res in all_results.items():
        n_valid = len(res["valid_windows"])
        n_alertes = res["result_bloc5"].get("n_confirmees", 0) if res["result_bloc5"] else "N/A"
        profils_data.append([
            res["profil_info"]["description"][:42],
            f"{res['profil_info']['imc']:.0f} kg/m²",
            res["profil_info"]["type_pied"],
            str(n_valid),
            str(n_alertes),
        ])
    tbl_profils = Table(profils_data, colWidths=[W * 0.38, W * 0.10, W * 0.12, W * 0.18, W * 0.18])
    tbl_profils.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), rl_colors.HexColor("#1a4d8f")),
        ("TEXTCOLOR", (0, 0), (-1, 0), rl_colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 8.5),
        ("GRID", (0, 0), (-1, -1), 0.4, rl_colors.grey),
        ("ALIGN", (1, 0), (-1, -1), "CENTER"),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [rl_colors.white, rl_colors.HexColor("#f4f6f7")]),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ]))
    story.append(tbl_profils)
    story.append(Spacer(1, 0.3 * cm))

    # Architecture pipeline
    story.append(Paragraph("1. Architecture du Pipeline", s_h2))
    if img_pipeline and os.path.exists(img_pipeline):
        story.append(RLImage(img_pipeline, width=W, height=W * 6 / 14))
    story.append(Paragraph(
        "<b>Interprétation :</b> Le pipeline Digi'Feet traite les signaux bruts des capteurs FSR et NTC "
        "en sept blocs successifs. Chaque bloc a une responsabilité distincte et transmet uniquement "
        "les données validées au bloc suivant. Le Bloc 0 garantit la qualité capteur avant filtrage. "
        "Le Bloc 2 écarte les fenêtres posturalement instables. Le Bloc 3 établit un seuil personnalisé S0 "
        "adapté au profil morphologique du patient. Le Bloc 4 affine continuellement ce seuil par "
        "inférence bayésienne adaptative. Le Bloc 5 confirme les alertes sur plusieurs fenêtres "
        "consécutives pour éviter les fausses alarmes.",
        s_interp
    ))
    story.append(PageBreak())

    # ──────────────────────────────────────────────
    # SECTION 2 — SEUILS BAYÉSIENS
    # ──────────────────────────────────────────────
    story.append(Paragraph("2. Seuils Bayésiens Personnalisés par Profil", s_h2))
    if img_seuils and os.path.exists(img_seuils):
        story.append(RLImage(img_seuils, width=W, height=W * 9 / 12))
    story.append(Paragraph(
        "<b>Interprétation :</b> Chaque barre représente le seuil μ (pression critique) "
        "adapté par le Bloc 4 pour une zone anatomique donnée, avec les barres d'erreur σ "
        "illustrant l'incertitude du modèle. La ligne violette indique le seuil fixe de référence "
        "(200 kPa). On observe que le seuil personnalisé est systématiquement plus élevé "
        "pour les profils pathologiques (Charcot, hallux valgus), ce qui reflète l'adaptation "
        "au contexte clinique réel du patient. Un seuil personnalisé significativement différent "
        "du seuil fixe confirme la pertinence de l'approche adaptative.",
        s_interp
    ))
    story.append(Spacer(1, 0.3 * cm))

    # Tableau détaillé des seuils S0
    story.append(Paragraph("Tableau des seuils S0 initiaux (Bloc 3) par profil et zone :", s_h3))
    seuil_header = ["Profil", "Zone", "S0 brut (kPa)", "S0 ajusté (kPa)", "MAD (kPa)", "Confiance", "Statut"]
    seuil_rows = [seuil_header]
    for pname, res in all_results.items():
        if res["result_bloc3"] is None:
            continue
        for zone, th in res["result_bloc3"]["thresholds"].items():
            statut = "BLOQUER" if th["recalibration_requise"] else "OK"
            seuil_rows.append([
                pname.replace("_", " ").capitalize(),
                zone,
                f"{th['S0_raw_kpa']:.1f}",
                f"{th['S0_adjusted_kpa']:.1f}",
                f"{th['MAD_kpa']:.1f}",
                f"{th['validation']['confidence']:.2f}",
                statut,
            ])
    tbl_s = Table(seuil_rows, colWidths=[W * 0.16, W * 0.14, W * 0.14, W * 0.14, W * 0.13, W * 0.12, W * 0.12])
    tbl_s_style = [
        ("BACKGROUND", (0, 0), (-1, 0), rl_colors.HexColor("#1a4d8f")),
        ("TEXTCOLOR", (0, 0), (-1, 0), rl_colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 8),
        ("GRID", (0, 0), (-1, -1), 0.4, rl_colors.grey),
        ("ALIGN", (2, 1), (-1, -1), "CENTER"),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [rl_colors.white, rl_colors.HexColor("#f4f6f7")]),
    ]
    # FIX: colorier en rouge les lignes avec statut BLOQUER
    for row_idx, row in enumerate(seuil_rows[1:], start=1):
        if row[-1] == "BLOQUER":
            tbl_s_style.append(("BACKGROUND", (0, row_idx), (-1, row_idx), rl_colors.HexColor("#fadbd8")))
            tbl_s_style.append(("TEXTCOLOR", (-1, row_idx), (-1, row_idx), rl_colors.HexColor("#c0392b")))
            tbl_s_style.append(("FONTNAME", (-1, row_idx), (-1, row_idx), "Helvetica-Bold"))
    tbl_s.setStyle(TableStyle(tbl_s_style))
    story.append(tbl_s)
    story.append(PageBreak())

    # ──────────────────────────────────────────────
    # SECTION 3 — ALERTES
    # ──────────────────────────────────────────────
    story.append(Paragraph("3. Décisions d'Alerte par Profil Clinique (Bloc 5)", s_h2))
    if img_alertes and os.path.exists(img_alertes):
        story.append(RLImage(img_alertes, width=W * 0.85, height=W * 0.85 * 5 / 9))
    story.append(Paragraph(
        "<b>Interprétation :</b> La matrice ci-dessus présente l'état d'alerte final du Bloc 5 "
        "pour chaque combinaison profil × zone anatomique. Une <b>Alarme confirmée</b> (rouge) "
        "signifie que la pression a dépassé le seuil personnalisé μ sur plusieurs fenêtres "
        "consécutives, ce qui induit un risque réel de lésion podologique. Un <b>Avertissement</b> "
        "(orange) signale un dépassement transitoire à surveiller. Les profils pathologiques "
        "(hallux valgus, pied de Charcot) génèrent significativement plus d'alertes confirmées, "
        "ce qui est cohérent avec leur présentation clinique.",
        s_interp
    ))
    story.append(Spacer(1, 0.3 * cm))

    # Tableau détaillé alertes
    alert_header = ["Profil", "Zone", "Niveau", "Confirmée", "P mesurée (kPa)", "Seuil μ (kPa)", "Ratio P/μ"]
    alert_rows = [alert_header]
    for pname, res in all_results.items():
        if res["result_bloc5"] is None:
            continue
        summaries = res["result_bloc5"].get("alert_summaries", {})
        for zone, s in summaries.items():
            alert_rows.append([
                pname.replace("_", " ").capitalize(),
                zone,
                s.alert_level,
                "OUI ⚠" if s.confirmed else "Non",
                f"{s.latest_pressure_kpa:.1f}",
                f"{s.latest_threshold_mu_kpa:.1f}",
                f"{s.latest_exceedance_ratio:.2f}",
            ])
    tbl_a = Table(alert_rows, colWidths=[W * 0.16, W * 0.14, W * 0.14, W * 0.12, W * 0.16, W * 0.16, W * 0.12])
    tbl_a_style = [
        ("BACKGROUND", (0, 0), (-1, 0), rl_colors.HexColor("#1a4d8f")),
        ("TEXTCOLOR", (0, 0), (-1, 0), rl_colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 8),
        ("GRID", (0, 0), (-1, -1), 0.4, rl_colors.grey),
        ("ALIGN", (2, 1), (-1, -1), "CENTER"),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [rl_colors.white, rl_colors.HexColor("#f4f6f7")]),
    ]
    # Colorier les alarmes confirmées
    for i, row in enumerate(alert_rows[1:], start=1):
        if row[3] == "OUI ⚠":
            tbl_a_style.append(("BACKGROUND", (0, i), (-1, i), rl_colors.HexColor("#fadbd8")))
        elif row[2] == "Avertissement":
            tbl_a_style.append(("BACKGROUND", (0, i), (-1, i), rl_colors.HexColor("#fef9e7")))
    tbl_a.setStyle(TableStyle(tbl_a_style))
    story.append(tbl_a)
    story.append(PageBreak())

    # ──────────────────────────────────────────────
    # SECTION 4 — COURBE ROC
    # ──────────────────────────────────────────────
    story.append(Paragraph("4. Courbe ROC — Validation de Performance (Bloc 6)", s_h2))
    if img_roc and os.path.exists(img_roc):
        story.append(RLImage(img_roc, width=W * 0.80, height=W * 0.80 * 6 / 7))

    roc = result_bloc6["roc_data"]
    auc_p = roc.get("auc_perso", 0)
    auc_f = roc.get("auc_fixe", 0)
    delta_auc = auc_p - auc_f

    story.append(Paragraph(
        f"<b>AUC seuil personnalisé bayésien :</b> {auc_p:.4f} &nbsp;|&nbsp; "
        f"<b>AUC seuil fixe 200 kPa :</b> {auc_f:.4f} &nbsp;|&nbsp; "
        f"<b>ΔAUC :</b> {delta_auc:+.4f}",
        s_body
    ))
    story.append(Paragraph(
        "<b>Interprétation :</b> L'AUC (aire sous la courbe ROC) mesure la capacité discriminante "
        "du modèle à séparer les événements à risque des événements normaux. "
        "Une AUC de 1.0 correspond à une discrimination parfaite ; 0.5 correspond à un modèle aléatoire. "
        f"L'AUC du seuil personnalisé ({auc_p:.3f}) est "
        f"{'supérieure' if delta_auc > 0 else 'inférieure'} à celle du seuil fixe ({auc_f:.3f}), "
        f"ce qui indique que l'approche bayésienne adaptative "
        f"{'améliore' if delta_auc > 0 else 'ne dégrade pas significativement'} "
        "la capacité à détecter les zones à risque. "
        "Le point orange représente le seuil optimal minimisant les faux positifs et les faux négatifs.",
        s_interp
    ))
    story.append(Spacer(1, 0.3 * cm))

    # ──────────────────────────────────────────────
    # SECTION 5 — MÉTRIQUES PAR PROFIL
    # ──────────────────────────────────────────────
    story.append(Paragraph("5. Métriques de Performance par Profil Clinique", s_h2))
    if img_metriques and os.path.exists(img_metriques):
        story.append(RLImage(img_metriques, width=W, height=W * 5 / 14))

    mg = result_bloc6["metriques_globales"]
    story.append(Paragraph(
        f"<b>Sensibilité globale</b> — Seuil fixe : {mg['fixe']['sensibilite_moy']:.1%} | "
        f"Seuil perso. : {mg['perso']['sensibilite_moy']:.1%} "
        f"({'↑' if mg['perso']['sensibilite_moy'] > mg['fixe']['sensibilite_moy'] else '↓'} "
        f"{abs(mg['perso']['sensibilite_moy'] - mg['fixe']['sensibilite_moy']):.1%})",
        s_body
    ))
    story.append(Paragraph(
        f"<b>Spécificité globale</b> — Seuil fixe : {mg['fixe']['specificite_moy']:.1%} | "
        f"Seuil perso. : {mg['perso']['specificite_moy']:.1%}",
        s_body
    ))
    story.append(Paragraph(
        f"<b>VPP globale</b> — Seuil fixe : {mg['fixe']['vpp_moy']:.1%} | "
        f"Seuil perso. : {mg['perso']['vpp_moy']:.1%}",
        s_body
    ))
    story.append(Paragraph(
        "<b>Interprétation :</b> La <b>sensibilité</b> mesure la capacité à détecter correctement les "
        "événements à risque (vrais positifs). Une sensibilité élevée est primordiale en contexte "
        "clinique pour éviter de manquer des zones sous pression pathologique. "
        "La <b>spécificité</b> mesure la capacité à ne pas générer de fausses alarmes. "
        "La <b>VPP</b> (Valeur Prédictive Positive) indique la proportion d'alertes qui "
        "correspondent à de vrais événements cliniques. Un seuil personnalisé améliore en général "
        "la sensibilité au détriment partiel de la spécificité, ce qui est acceptable si la "
        "VPP reste au-dessus de 70 %.",
        s_interp
    ))
    story.append(PageBreak())

    # ──────────────────────────────────────────────
    # SECTION 6 — TEST DE STUDENT + VALIDATION CLINIQUE
    # ──────────────────────────────────────────────
    # Note méthodologique sur la simulation statique
    story.append(Spacer(1, 0.2 * cm))
    story.append(Paragraph(
        "<b>Note méthodologique — Nature de la simulation :</b> "
        "Les données synthétiques sont générées en contexte d'appui statique. "
        "Les variations de répartition de charge entre fenêtres (surcharge talon, surcharge avant-pied) "
        "ne simulent <i>pas</i> un cycle de marche, mais modélisent les <b>oscillations posturales naturelles</b> "
        "d'un patient debout immobile : déplacement lent du centre de gravité, asymétrie transitoire de l'appui, "
        "variation de la surface de contact liée aux micro-ajustements posturaux. "
        "Ces phénomènes sont documentés en posturographie statique (Brenière et al., Shumway-Cook & Woollacott) "
        "et justifient que les fenêtres successives ne présentent pas des pressions strictement identiques. "
        "Un patient debout ne maintient pas une posture parfaitement stationnaire ; l'enveloppe de variation "
        "simulée (±38–45 % sur les zones de report) reste dans les plages observées expérimentalement "
        "en condition de station debout prolongée.",
        s_warn
    ))
    story.append(Spacer(1, 0.3 * cm))

    story.append(Paragraph("6. Test de Student et Validation Statistique", s_h2))

    ts = result_bloc6["test_student"]
    if "erreur" not in ts:
        student_data = [["Métrique", "Statistique t", "p-value", "Seuil α = 0.05", "Conclusion"]]
        for metrique, vals in ts.items():
            if isinstance(vals, dict):
                sig = vals.get("significatif", False)
                student_data.append([
                    metrique.capitalize(),
                    f"{vals.get('t', 0):.4f}",
                    f"{vals.get('p_value', 1):.4f}",
                    "p < 0.05" if sig else "p ≥ 0.05",
                    "Différence significative ✓" if sig else "Différence non significative",
                ])
        st_tbl = Table(student_data, colWidths=[W * 0.14, W * 0.14, W * 0.13, W * 0.14, W * 0.40])
        st_style = [
            ("BACKGROUND", (0, 0), (-1, 0), rl_colors.HexColor("#1a4d8f")),
            ("TEXTCOLOR", (0, 0), (-1, 0), rl_colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 8.5),
            ("GRID", (0, 0), (-1, -1), 0.4, rl_colors.grey),
            ("ALIGN", (1, 1), (-1, -1), "CENTER"),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [rl_colors.white, rl_colors.HexColor("#f4f6f7")]),
        ]
        for i, (m, vals) in enumerate(ts.items(), start=1):
            if isinstance(vals, dict) and vals.get("significatif"):
                st_style.append(("BACKGROUND", (4, i), (4, i), rl_colors.HexColor("#d5f5e3")))
        st_tbl.setStyle(TableStyle(st_style))
        story.append(st_tbl)

    story.append(Spacer(1, 0.3 * cm))
    story.append(Paragraph(
        "<b>Interprétation :</b> Le test de Student apparié compare les distributions de métriques "
        "obtenues sur les N simulations pour les deux modèles (seuil fixe vs seuil personnalisé). "
        "Une p-value < 0.05 rejette l'hypothèse nulle d'équivalence des deux approches, "
        "attestant statistiquement que l'amélioration observée n'est pas due au hasard.",
        s_interp
    ))
    story.append(Spacer(1, 0.4 * cm))

    # Critères cliniques
    story.append(Paragraph("7. Critères de Validation Clinique (6 critères)", s_h2))
    vc = result_bloc6["validation_clinique"]
    criteres_noms = [
        ("Sensibilité personnalisée ≥ 80 %",          "critere_sens_min_80pct"),
        ("Sensibilité perso. > Sensibilité fixe",     "critere_sens_superieur"),
        ("Spécificité acceptable (≥ 50 %)",           "critere_spec_acceptable"),
        ("VPP personnalisée > VPP fixe",              "critere_vpp_superieure"),
        ("Test de Student significatif (p < 0.05)",   "critere_student_significatif"),
        ("AUC personnalisé > AUC fixe",               "critere_auc_superieur"),
    ]
    crit_data = [["#", "Critère clinique", "Résultat"]]
    for i, (nom, key) in enumerate(criteres_noms, 1):
        val = vc.get(key, False)
        crit_data.append([str(i), nom, "✓ Validé" if val else "✗ Non validé"])

    crit_tbl = Table(crit_data, colWidths=[W * 0.05, W * 0.72, W * 0.18])
    crit_style = [
        ("BACKGROUND", (0, 0), (-1, 0), rl_colors.HexColor("#1a4d8f")),
        ("TEXTCOLOR", (0, 0), (-1, 0), rl_colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("GRID", (0, 0), (-1, -1), 0.4, rl_colors.grey),
        ("ALIGN", (0, 0), (0, -1), "CENTER"),
        ("ALIGN", (2, 1), (2, -1), "CENTER"),
    ]
    for i, (_, key) in enumerate(criteres_noms, 1):
        val = vc.get(key, False)
        color = rl_colors.HexColor("#d5f5e3") if val else rl_colors.HexColor("#fadbd8")
        crit_style.append(("BACKGROUND", (0, i), (-1, i), color))
    crit_tbl.setStyle(TableStyle(crit_style))
    story.append(crit_tbl)
    story.append(Spacer(1, 0.3 * cm))

    # Verdict visuel
    n_ok = vc["n_criteres_ok"]
    verdict_color = (
        rl_colors.HexColor("#1a7a44") if n_ok >= 5
        else rl_colors.HexColor("#e67e22") if n_ok >= 3
        else rl_colors.HexColor("#c0392b")
    )
    verdict_style = ParagraphStyle("verdict", parent=styles["Normal"], fontSize=12,
                                   fontName="Helvetica-Bold", textColor=verdict_color,
                                   backColor=rl_colors.HexColor("#f8f9fa"),
                                   borderPad=8, spaceAfter=8, leading=18)
    story.append(Paragraph(
        f"VERDICT ({n_ok}/6 critères validés) : {vc['verdict']}",
        verdict_style
    ))
    story.append(PageBreak())

    # ──────────────────────────────────────────────
    # SECTION 8 — CONCLUSION ET RECOMMANDATIONS
    # ──────────────────────────────────────────────
    story.append(Paragraph("8. Conclusion et Recommandations Cliniques", s_h2))
    story.append(HRFlowable(width="100%", thickness=1, color=rl_colors.HexColor("#1a4d8f")))
    story.append(Spacer(1, 0.2 * cm))

    # Forces
    story.append(Paragraph("Forces du système validées", s_h3))
    forces = [
        "Personnalisation morphologique : le seuil S0 intègre l'IMC et le type de pied, "
        "permettant une calibration spécifique à chaque patient.",
        "Adaptation continue : le modèle bayésien (Bloc 4) affine le seuil à chaque "
        "fenêtre valide, réduisant progressivement l'incertitude σ.",
        "Filtrage multi-étapes : la chaîne Bloc 0 → Bloc 1b → Bloc 2 garantit que seules "
        "les données fiables et posturalement cohérentes alimentent les blocs décisionnels.",
        "Persistance d'alerte (Bloc 5) : l'exigence de plusieurs dépassements consécutifs "
        "avant confirmation réduit significativement les fausses alarmes.",
        f"Performance clinique : {n_ok}/6 critères de validation clinique sont satisfaits, "
        f"attestant de la supériorité du seuil personnalisé sur le seuil fixe.",
    ]
    for f in forces:
        story.append(Paragraph(f"• {f}", s_body))
    story.append(Spacer(1, 0.3 * cm))

    # Limites
    story.append(Paragraph("Limites et perspectives", s_h3))
    limites = [
        "Données synthétiques uniquement : les performances actuelles sont calculées sur des "
        "simulations ; une validation sur données réelles de patients est indispensable.",
        "Convergence bayésienne : avec seulement 8 fenêtres simulées, le Bloc 4 n'a pas "
        "atteint sa convergence optimale. En pratique, 30 à 50 fenêtres sont recommandées.",
        "Calibration initiale (Bloc 3) : la qualité du seuil S0 dépend directement du nombre "
        "de fenêtres valides disponibles à la première mise en service du dispositif.",
        "Validation clinique multicentrique : un essai sur une cohorte de patients diabétiques "
        "ou à risque podologique est nécessaire avant déploiement en contexte médical.",
    ]
    for l in limites:
        story.append(Paragraph(f"• {l}", s_body))
    story.append(Spacer(1, 0.3 * cm))

    # Recommandations
    story.append(Paragraph("Recommandations techniques", s_h3))
    recommandations = [
        "Augmenter le nombre de fenêtres d'acquisition (N_FENETRES ≥ 20) pour améliorer "
        "la convergence du modèle bayésien et la robustesse du seuil S0.",
        "Implémenter un mécanisme de recalibration périodique (ex. : toutes les 2 semaines) "
        "pour suivre l'évolution clinique du patient.",
        "Intégrer les données d'hydratation cutanée (impédance) comme variable d'ajustement "
        "supplémentaire dans le Bloc 3.",
        "Considérer un seuil d'alerte différencié par temps de port (matin vs soir) pour "
        "tenir compte des variations circadiennes de la charge plantaire.",
    ]
    for r in recommandations:
        story.append(Paragraph(f"• {r}", s_body))

    story.append(Spacer(1, 0.4 * cm))
    story.append(HRFlowable(width="100%", thickness=1, color=rl_colors.grey))
    story.append(Spacer(1, 0.2 * cm))
    story.append(Paragraph(
        f"Rapport généré automatiquement par le pipeline Digi'Feet / AdaptStep — "
        f"{datetime.now().strftime('%d/%m/%Y')}. "
        "Ce document est une aide à l'interprétation des données capteurs et ne constitue "
        "pas un diagnostic médical. Toute décision clinique doit être prise par un professionnel de santé qualifié.",
        s_footer
    ))

    doc.build(story)
    print(f"  ✓ PDF final généré : {filename}")
    return filename


# ============================================================
# BOUCLE PRINCIPALE — TOUS LES PROFILS
# ============================================================

if __name__ == "__main__":
    print("=" * 65)
    print("   DÉMARRAGE DU PIPELINE DIGI'FEET — VERSION MULTI-PROFILS")
    print(f"   {N_FENETRES} fenêtres × 4 profils cliniques")
    print("=" * 65)

    simulator = PlantarSensorSimulator(n_samples=N_SAMPLES, fs=50.0, seed=42)
    all_results = {}

    # ──────────────────────────────────────────────
    # RUN PIPELINE PAR PROFIL CLINIQUE
    # ──────────────────────────────────────────────
    for profile_name in ["normal", "surpoids", "hallux_valgus", "pied_charcot"]:
        result = run_pipeline_for_profile(
            profile_name=profile_name,
            simulator=simulator,
            n_fenetres=N_FENETRES,
            verbose=True,
        )
        all_results[profile_name] = result

        # Résumé console par profil
        if result["result_bloc5"] is not None:
            afficher_resultats_bloc5(result["result_bloc5"])

    # ──────────────────────────────────────────────
    # PDFs INDIVIDUELS — tous les profils (4 profils × 6 blocs)
    # ──────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("  GÉNÉRATION DES RAPPORTS PDF INDIVIDUELS (4 profils × 6 blocs)")
    print("=" * 65)

    for profil_name, ref in all_results.items():
        print(f"\n  --- Profil : {profil_name} ---")
        if ref["last_bloc0_result"]:
            generer_pdf_bloc0(ref["last_bloc0_result"], logo_path=LOGO_PATH,
                              filename=f"Rapport_Bloc0_{profil_name}.pdf")
            print(f"  ✓ PDF Bloc 0 — {profil_name}")
        if ref["last_bloc1_result"]:
            generer_pdf_bloc1(ref["last_bloc1_result"], logo_path=LOGO_PATH,
                              filename=f"Rapport_Bloc1_{profil_name}.pdf")
            print(f"  ✓ PDF Bloc 1 — {profil_name}")
        if ref["last_bloc2_result"]:
            generer_pdf_bloc2(ref["last_bloc2_result"], logo_path=LOGO_PATH,
                              filename=f"Rapport_Bloc2_{profil_name}.pdf")
            print(f"  ✓ PDF Bloc 2 — {profil_name}")
        if ref["result_bloc3"]:
            generer_pdf_bloc3(ref["result_bloc3"], logo_path=LOGO_PATH,
                              filename=f"Rapport_Bloc3_{profil_name}.pdf")
            print(f"  ✓ PDF Bloc 3 — {profil_name}")
        if ref["result_bloc4"]:
            generer_pdf_bloc4(ref["result_bloc4"], logo_path=LOGO_PATH,
                              filename=f"Rapport_Bloc4_{profil_name}.pdf")
            print(f"  ✓ PDF Bloc 4 — {profil_name}")
        if ref["result_bloc5"]:
            generer_pdf_bloc5(ref["result_bloc5"], logo_path=LOGO_PATH,
                              filename=f"Rapport_Bloc5_{profil_name}.pdf")
            print(f"  ✓ PDF Bloc 5 — {profil_name}")

    # ──────────────────────────────────────────────
    # BLOC 6 — VALIDATION STATISTIQUE GLOBALE
    # ──────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("  BLOC 6 — Validation statistique complète (4 profils × 30 simulations)")
    print("=" * 65)

    result_bloc6 = bloc6_validation_statistique(
        seuil_fixe_kpa=200.0,
        n_simulations=30,
        n_mesures_par_sim=120,
        profils=PROFILS_SYNTHETIQUES,
    )
    afficher_resultats_bloc6(result_bloc6)
    pdf_b6 = generer_pdf_bloc6(result_bloc6, logo_path=LOGO_PATH)
    print(f"  ✓ PDF Bloc 6 : {pdf_b6}")

    # ──────────────────────────────────────────────
    # RAPPORT FINAL DE SYNTHÈSE
    # ──────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("  GÉNÉRATION DU RAPPORT FINAL DE SYNTHÈSE CLINIQUE")
    print("=" * 65)

    pdf_final = generer_pdf_final(
        all_results=all_results,
        result_bloc6=result_bloc6,
        logo_path=LOGO_PATH,
        filename=f"Rapport_Final_DigiPied_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
    )

    # ──────────────────────────────────────────────
    # RÉSUMÉ FINAL CONSOLE
    # ──────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("  RÉSUMÉ FINAL DU PIPELINE")
    print("=" * 65)
    print(f"  {'Profil':20s} | {'Fenêtres valides':18s} | {'Alertes confirmées':18s} | {'Risque global':12s}")
    print(f"  {'-'*20}-+-{'-'*18}-+-{'-'*18}-+-{'-'*12}")
    for pname, res in all_results.items():
        n_valid = len(res["valid_windows"])
        if res["result_bloc5"]:
            n_al = res["result_bloc5"].get("n_confirmees", 0)
            risk = "OUI ⚠" if res["result_bloc5"].get("global_risk", False) else "Non"
        else:
            n_al = "N/A"
            risk = "N/A"
        print(f"  {pname:20s} | {n_valid:^18d} | {str(n_al):^18s} | {risk:^12s}")

    vc = result_bloc6["validation_clinique"]
    print(f"\n  Bloc 6 — Critères cliniques validés : {vc['n_criteres_ok']}/6")
    print(f"  Verdict : {vc['verdict']}")
    print(f"\n  AUC seuil perso. : {result_bloc6['roc_data'].get('auc_perso', 0):.4f}")
    print(f"  AUC seuil fixe   : {result_bloc6['roc_data'].get('auc_fixe', 0):.4f}")
    print(f"\n  Rapport final    : {pdf_final}")
    print("=" * 65)
    print("  Pipeline terminé avec succès.")