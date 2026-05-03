# Biomecanique_Pied_Diabetique

# Digi'Feet / AdaptStep

Pipeline complet de traitement biomécanique pour semelle instrumentée (FSR + NTC).  
Calibration personnalisée, détection des zones à risque, validation statistique seuil fixe vs seuil bayésien adaptatif.

---

## 📦 Bibliothèques requises

### Installation rapide

```bash
pip install numpy scipy reportlab matplotlib scikit-learn pillow

Détail des dépendances :

numpy (>= 1.21) : Calculs matriciels
scipy (>= 1.7) : Filtrage Butterworth, statistiques
reportlab (>= 3.6) : Génération des rapports PDF
matplotlib (>= 3.4) : Graphiques (ROC, barres, matrices d'alertes)
scikit-learn (>= 1.0) : Courbe ROC et AUC
Pillow (>= 8.0) : Gestion du logo (format PNG)

Vérification de l'installation :
  python
  import numpy as np
  import scipy
  import reportlab
  import matplotlib
  import sklearn
  print("Toutes les bibliothèques sont installées
  
Structure des fichiers
  text
  
  DigiFeet/
  ├── CODE_SEMELLE.py          # Modules des blocs 0 à 6
  ├── main_FINAL.py            # Orchestrateur principal
  ├── Logo.png                 # Logo pour les rapports PDF (optionnel)
  └── Rapports/                # Dossier de sortie (créé automatiquement)
      ├── Rapport_Bloc0_*.pdf
      ├── Rapport_Bloc1_*.pdf
      ├── ...
      └── Rapport_Final_*.pdf

Exécution
  bash
  python main_FINAL.py


Flux d'exécution :
  Simulateur capteur (8 FSR + 4 NTC)
  
  Bloc 1a : Conversion tensions → kPa brutes
  
  Bloc 0 : Qualité capteur (score SC_z)
  
  Bloc 1b : Filtrage Butterworth 20 Hz
  
  Bloc 2 : Contrôle postural (stabilité)
  
  Bloc 3 : Calibration initiale S0 (percentile pondéré)
  
  Bloc 4 : Mise à jour bayésienne adaptative (µ ± σ)
  
  Bloc 5 : Persistance d'alerte (2 fenêtres consécutives)
  
  Bloc 6 : Validation statistique (ROC, AUC, test de Student)
  
  Rapport final PDF

Configuration principale
  Paramètres ajustables dans main_FINAL.py :
  
  python
  N_FENETRES = 30                 # Nombre de fenêtres par profil
  N_SAMPLES = 1500                # Échantillons par fenêtre (30s à 50Hz)
  
  # Configuration bayésienne (Bloc 4)
  bayesian_config = BayesianConfig(
      k_prior=3.0,                # Largeur du prior (tolérance initiale)
      lambda_forgetting=0.98,     # Mémoire (0.97-0.99)
      variation_max_sigma=12.0,   # Tolérance aux variations
      sigma_min_kpa=2.0,          # Incertitude minimale
  )
  
  # Persistance des alertes (Bloc 5)
  alert_engine = AlertEngine(config=Bloc5Config(n_fenetres_confirmation=2))

Profils cliniques simulés
  normal : Pied sain (IMC 23)
  
  surpoids : Patient en surpoids, pied plat (IMC 28)
  
  hallux_valgus : Hallux valgus, pied creux (IMC 25)
  
  pied_charcot : Pied de Charcot (IMC 30)

Sorties générées

Rapports individuels par profil (4 profils × 6 blocs)

  Bloc 0 : Scores de confiance capteur, qualité du signal
  
  Bloc 1 : Pressions filtrées, températures NTC
  
  Bloc 2 : Stabilité posturale, w_post
  
  Bloc 3 : Seuils personnalisés S0
  
  Bloc 4 : Seuils bayésiens finaux (µ ± σ), alertes
  
  Bloc 5 : Décisions d'alerte confirmées
  
  Bloc 6 : ROC, AUC, test de Student (global)

Rapport final de synthèse
  Seuils bayésiens par zone anatomique
  
  Matrice des alertes (profil × zone)
  
  Courbe ROC comparative
  
  Métriques globales (sensibilité, spécificité, VPP)
  
  Critères de validation clinique (6/6)

Logo
  Les rapports PDF intègrent un logo en haut de page.
  Placez un fichier Logo.png (format recommandé : largeur 160-200px, fond transparent) dans le même dossier que main_FINAL.py.
  

Exemple de résultats attendus

Alertes confirmées par profil
  Normal : 0 alerte
  Surpoids : 0 alerte (avertissement non confirmé sur avant-pied)
  Hallux valgus : 0 alerte (alarme non confirmée sur avant-pied)  
  Pied de Charcot : 0 alerte (avertissements sur hallux et avant-pied)

Métriques globales (Bloc 6)
  Sensibilité : Seuil fixe 99.6% / Seuil personnalisé 100.0%
  Spécificité : Seuil fixe 25.0% / Seuil personnalisé 97.2%
  VPP : Seuil fixe 48.8% / Seuil personnalisé 90.7%
  AUC : Seuil fixe 0.6918 / Seuil personnalisé 0.8471

Notes importantes :
  Données synthétiques : Les performances sont validées sur simulations. Une validation sur données cliniques réelles est nécessaire avant    déploiement médical.

  Convergence bayésienne : Avec 30 fenêtres, la convergence est bonne. Pour une utilisation clinique, 30 à 50 fenêtres sont recommandées.
  
  Nature statique : Les phases simulées (appui postérieur, appui antérieur) modélisent des oscillations posturales debout, pas un cycle de    marche.
  
  Seuil de persistance : n_fenetres_confirmation = 2 (une alerte n'est confirmée qu'après 2 fenêtres consécutives).
  
Structure des fichiers de code :
  CODE_SEMELLE.py
  Contient toutes les classes et fonctions des blocs 0 à 6 :
  
  SensorCalibrationReference, SensorQualityConfig
  
  SignalConverter, SensorQualityMonitor, SignalFilter
  
  PosturalConfig, bloc2_postural_quality_control
  
  ThresholdConfig, bloc3_initial_threshold_calibration
  
  BayesianConfig, bloc4_bayesian_adaptive_update
  
  AlertEngine, Bloc5Config
  
  bloc6_validation_statistique
  
  Fonctions de génération PDF (generer_pdf_bloc0 à generer_pdf_bloc6)
  
  main_FINAL.py
  Orchestrateur principal :
  
  PlantarSensorSimulator : simulation réaliste des capteurs
  
  Configuration des paramètres
  
  Boucle d'exécution sur les 4 profils cliniques
  
  Génération des rapports individuels et du rapport final

Tests
  Pour exécuter une validation complète (30 simulations × 4 profils × 120 mesures) :
  
  python
  result_bloc6 = bloc6_validation_statistique(
      seuil_fixe_kpa=200.0,
      n_simulations=30,
      n_mesures_par_sim=120
  )

Licence
  Ce projet est fourni à titre de démonstration technique.
  Une validation clinique sur données réelles est obligatoire avant toute utilisation médicale.

Auteurs
  Projet AdaptStep / Digi'Feet
  
  Généré automatiquement par le pipeline — Dernière mise à jour : 2026
