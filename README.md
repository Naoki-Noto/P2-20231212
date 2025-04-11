# P2-20231212
For general machine learning (adapt.yml):
Python (3.7.16) was used as a language for the general machine-learning study, and used packages were adapt (0.4.3), lightgbm (3.3.5), matplotlib (3.5.3), mordred (1.2.0), numpy (1.21.5), pandas (1.3.5), rdkit (2023.3.2), scikit-learn (1.0.2), seaborn (0.12.2), shap (0.42.1) and xgboost (1.6.2).

For the use of Featuretools (featuretools.yml):
Python (3.10.13) was used as a language for Featuretools, and used packages were featuretools (1.27.0), numpy (1.24.4), and pandas (2.1.1).

For performing XGBoost (xgboost.yml):
Python (3.10.13) was used as a language for performing XGBoost, and used packages were pandas (2.2.2), scikit-learn (1.0.2), and xgboost (2.1.2).

# Citation Information
title={Transfer learning across different photocatalytic organic reactions}

author={Naoki Noto, Ryuga Kunisada, Tabea Rohlfs, Manami Hayashi, Ryosuke Kojima, Olga García Mancheño, Takeshi Yanai, Susumu Saito }

journal info={Nat. Commun. 2025, 16, Article number: 3388.}

# Table of Contents

Gaussian/

  • example_input: Examples of Gaussian input files for different calculations.

  • xyz: XYZ files after geometry optimization.

==============================================================================
  
ML/

  • Conventional_ML

  - Lasso: Code and results for Lasso regression.
    
  - RF: Code and results for random forest.
    
  - RF_control: Code and results for control experiments (random forest).
      
  - SVM: Code and results for support-vector machine.
    
  - XGB: Code and results for XGBoost.
  
  • DA1: Code and main results of domain adaptation.
  
  • DA2: Code and results of domain adaptation when using 10 data points as a training set.
  
  • DA3_(C4_12): Code and results of domain adaptation for alkene photoisomerization.

  - Alkene_isomerization_8OPSs: Code and results for predictions in alkene isomerization by domain adaptation using 8 OPSs as the training set.
    
  - Alkene_isomerization_DA: Code and results for predictions in alkene isomerization by domain adaptation.
      
  - Alkene_isomerization_RF: Code and results for predictions in alkene isomerization by random forest.
    
  - Alkene_isomerization_badcase: Code and results for predictions in alkene isomerization by domain adaptation using the source domain with photocatalytic activity trends less similar to the target domain.

  - Correlation_analysis: Code and results for the correlation analysis.
    
  - Learning_curve: Code and results for generating learning curves to investigate generalization performance.
  
  • DA4_(C9_10_15): Code and results of investigations into the limitation and applicability of domain adaptation.
  
  - CN: Code and results for domain adaptation in CN.
    
  - CO_a: Code and results for domain adaptation in CO_a.
      
  - CO_b: Code and results for domain adaptation in CO_b.
    
  - CO_c: Code and results for domain adaptation in CO_c.

  - CO_d: Code and results for domain adaptation in CO_d.
    
  - CO_e: Code and results for domain adaptation in CO_e.
      
  - CS: Code and results for domain adaptation in CS.
    
  - SD_data_exclusion: Code and results for investigations into the influence of excluding 30 OPSs from the source domain.
  
  • DA_SI_(C3_5_13): Code and results of domain adaptation for supporting information.
  
  - Comparison_method: Code and results for comparison of DA methods.
    
  - Increasing_training_data: Code and results for the test with larger training datasets.
      
  - Top3_and_bottom3: Code and results for domain adaptation using source domains selected based on correlation coefficients among OPSs in the training data.

  • Data_volume_(C8): Code and results of investigations into the effect of increasing the data volume (but not domain adaptation).
  
  - Lasso: Code and results for Lasso regression.
    
  - RF: Code and results for random forest.
      
  - SVM: Code and results for support-vector machine.
    
  - XGB: Code and results for XGBoost.
  
  • Make_descriptors: Code for generating descriptors.

  • Paired_t-test: Code and results of paired t-test.
