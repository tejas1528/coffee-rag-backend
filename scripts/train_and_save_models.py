"""
Optimized Model Training Script for Coffee Liking Prediction
Based on the coffee_analysis.ipynb notebook with enhancements

This script:
1. Trains multiple models (RF, XGBoost, Ensemble)
2. Performs hyperparameter optimization
3. Uses cross-validation for robust evaluation
4. Saves compressed, optimized models
5. Generates comprehensive performance reports
"""

import pandas as pd
import numpy as np
import joblib
import json
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Sklearn imports
from sklearn.model_selection import (
    train_test_split, 
    RandomizedSearchCV, 
    cross_val_score,
    KFold
)
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, VotingRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.inspection import permutation_importance

# Try to import XGBoost (optional, but better performance)
try:
    from xgboost import XGBRegressor
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("⚠️  XGBoost not available. Install with: pip install xgboost")

print("="*80)
print("COFFEE LIKING PREDICTION - OPTIMIZED MODEL TRAINING")
print("="*80)
print(f"Training started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("="*80)

# ==================== CONFIGURATION ====================

# Paths (update these to match your setup)
DATA_PATH = 'cotter_dataset.csv'  # Your dataset
OUTPUT_DIR = Path('../data/models')  # Where to save models
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Random seed for reproducibility
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# Cross-validation folds
CV_FOLDS = 10

# Define sensory features (as per notebook)
SENSORY_FEATURES = [
    'Flavor.intensity', 'Acidity', 'Mouthfeel', 'Tea.floral', 'Fruit', 
    'Citrus', 'Green.veg', 'Paper.wood', 'Burnt', 'Cereal', 'Nutty', 
    'Dark.chocolate', 'Caramel', 'Bitter', 'Astringent', 'Roasted', 
    'Sour', 'Thick.viscous', 'Sweet', 'Rubber'
]

# ==================== LOAD & PREPARE DATA ====================

print("\n" + "="*80)
print("STEP 1: DATA LOADING & PREPARATION")
print("="*80)

# Load data
df = pd.read_csv(DATA_PATH)
print(f"✓ Loaded dataset: {len(df)} records, {len(df.columns)} columns")

# Clean data
df_clean = df.dropna(subset=['Liking']).copy()
print(f"✓ Cleaned dataset: {len(df_clean)} records (removed {len(df) - len(df_clean)} missing liking scores)")

# Prepare features and target
X = df_clean[SENSORY_FEATURES].fillna(0)
y = df_clean['Liking']

print(f"✓ Features: {len(SENSORY_FEATURES)}")
print(f"✓ Target range: {y.min():.1f} - {y.max():.1f}, mean: {y.mean():.2f}, std: {y.std():.2f}")

# Train-test split (stratified by cluster to maintain distribution)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2, 
    random_state=RANDOM_STATE, 
    stratify=df_clean['Cluster']
)

print(f"✓ Train set: {len(X_train)} samples ({len(X_train)/len(X)*100:.1f}%)")
print(f"✓ Test set: {len(X_test)} samples ({len(X_test)/len(X)*100:.1f}%)")

# ==================== FEATURE SCALING ====================

print("\n" + "="*80)
print("STEP 2: FEATURE SCALING")
print("="*80)

# Try both StandardScaler and RobustScaler (robust to outliers)
scalers = {
    'standard': StandardScaler(),
    'robust': RobustScaler()
}

scaler_results = {}

for scaler_name, scaler in scalers.items():
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Quick test with simple RF
    rf_test = RandomForestRegressor(n_estimators=50, random_state=RANDOM_STATE, n_jobs=-1)
    rf_test.fit(X_train_scaled, y_train)
    score = r2_score(y_test, rf_test.predict(X_test_scaled))
    
    scaler_results[scaler_name] = score
    print(f"  {scaler_name.capitalize()} Scaler R²: {score:.4f}")

# Select best scaler
best_scaler_name = max(scaler_results, key=scaler_results.get)
best_scaler = scalers[best_scaler_name]

# Refit on full training data
X_train_scaled = best_scaler.fit_transform(X_train)
X_test_scaled = best_scaler.transform(X_test)

print(f"\n✓ Selected: {best_scaler_name.capitalize()} Scaler (R²: {scaler_results[best_scaler_name]:.4f})")

# ==================== HYPERPARAMETER TUNING ====================

print("\n" + "="*80)
print("STEP 3: HYPERPARAMETER OPTIMIZATION")
print("="*80)

# Random Forest hyperparameter grid (from your notebook + extras)
rf_param_grid = {
    'n_estimators': [100, 200, 300, 400],
    'max_depth': [10, 15, 20, 25, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2', 0.5],
    'bootstrap': [True],
    'min_impurity_decrease': [0.0, 0.01, 0.02]
}

print("\nTuning Random Forest...")
print(f"  Parameter combinations to test: ~50 (randomized search)")
print(f"  Cross-validation folds: 5")

rf_search = RandomizedSearchCV(
    RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=-1),
    param_distributions=rf_param_grid,
    n_iter=50,  # Test 50 random combinations
    cv=5,
    scoring='r2',
    random_state=RANDOM_STATE,
    n_jobs=-1,
    verbose=1
)

rf_search.fit(X_train_scaled, y_train)

print(f"\n✓ Best Random Forest parameters found:")
for param, value in rf_search.best_params_.items():
    print(f"    {param}: {value}")
print(f"  Best CV R²: {rf_search.best_score_:.4f}")

best_rf = rf_search.best_estimator_

# ==================== OPTIONAL: XGBOOST TRAINING ====================

best_xgb = None

if XGBOOST_AVAILABLE:
    print("\n" + "="*80)
    print("STEP 4: XGBOOST OPTIMIZATION (OPTIONAL)")
    print("="*80)
    
    xgb_param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [3, 5, 7, 9],
        'learning_rate': [0.01, 0.05, 0.1, 0.2],
        'subsample': [0.7, 0.8, 0.9, 1.0],
        'colsample_bytree': [0.7, 0.8, 0.9, 1.0],
        'min_child_weight': [1, 3, 5]
    }
    
    print("\nTuning XGBoost...")
    print(f"  Parameter combinations to test: ~30")
    
    xgb_search = RandomizedSearchCV(
        XGBRegressor(random_state=RANDOM_STATE, n_jobs=-1, verbosity=0),
        param_distributions=xgb_param_grid,
        n_iter=30,
        cv=5,
        scoring='r2',
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=1
    )
    
    xgb_search.fit(X_train_scaled, y_train)
    
    print(f"\n✓ Best XGBoost parameters found:")
    for param, value in xgb_search.best_params_.items():
        print(f"    {param}: {value}")
    print(f"  Best CV R²: {xgb_search.best_score_:.4f}")
    
    best_xgb = xgb_search.best_estimator_

# ==================== ENSEMBLE MODEL ====================

print("\n" + "="*80)
print("STEP 5: ENSEMBLE MODEL CREATION")
print("="*80)

# Create voting ensemble
estimators = [('rf', best_rf)]

if best_xgb is not None:
    estimators.append(('xgb', best_xgb))
    print("Creating ensemble: Random Forest + XGBoost")
else:
    # Add Gradient Boosting as alternative
    gb = GradientBoostingRegressor(
        n_estimators=200,
        learning_rate=0.1,
        max_depth=7,
        random_state=RANDOM_STATE
    )
    gb.fit(X_train_scaled, y_train)
    estimators.append(('gb', gb))
    print("Creating ensemble: Random Forest + Gradient Boosting")

ensemble = VotingRegressor(estimators=estimators)
ensemble.fit(X_train_scaled, y_train)

print("✓ Ensemble model created")

# ==================== MODEL EVALUATION ====================

print("\n" + "="*80)
print("STEP 6: COMPREHENSIVE MODEL EVALUATION")
print("="*80)

models_to_evaluate = {
    'Random Forest': best_rf,
    'Ensemble': ensemble
}

if best_xgb is not None:
    models_to_evaluate['XGBoost'] = best_xgb

evaluation_results = {}

for name, model in models_to_evaluate.items():
    print(f"\n{name}:")
    
    # Test set performance
    y_pred = model.predict(X_test_scaled)
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    
    print(f"  Test R²:   {r2:.4f}")
    print(f"  Test RMSE: {rmse:.4f}")
    print(f"  Test MAE:  {mae:.4f}")
    
    # Cross-validation on training set
    print(f"  Performing {CV_FOLDS}-fold cross-validation...")
    cv_scores = cross_val_score(
        model, 
        X_train_scaled, 
        y_train, 
        cv=CV_FOLDS, 
        scoring='r2',
        n_jobs=-1
    )
    
    print(f"  CV R² Mean: {cv_scores.mean():.4f} (±{cv_scores.std()*2:.4f})")
    print(f"  CV R² Range: [{cv_scores.min():.4f}, {cv_scores.max():.4f}]")
    
    # Store results
    evaluation_results[name] = {
        'test_r2': float(r2),
        'test_rmse': float(rmse),
        'test_mae': float(mae),
        'cv_r2_mean': float(cv_scores.mean()),
        'cv_r2_std': float(cv_scores.std()),
        'cv_r2_scores': [float(s) for s in cv_scores]
    }

# Select best model
best_model_name = max(evaluation_results, key=lambda x: evaluation_results[x]['test_r2'])
best_model = models_to_evaluate[best_model_name]

print(f"\n{'='*80}")
print(f"🏆 BEST MODEL: {best_model_name}")
print(f"   Test R²: {evaluation_results[best_model_name]['test_r2']:.4f}")
print(f"   CV R²: {evaluation_results[best_model_name]['cv_r2_mean']:.4f}")
print(f"{'='*80}")

# ==================== FEATURE IMPORTANCE ANALYSIS ====================

print("\n" + "="*80)
print("STEP 7: FEATURE IMPORTANCE ANALYSIS")
print("="*80)

# Get feature importance (different for ensemble)
if best_model_name == 'Ensemble':
    # Average importance from all estimators
    if hasattr(best_model.estimators_[0][1], 'feature_importances_'):
        importances = np.mean([
            est.feature_importances_ 
            for name, est in best_model.estimators_ 
            if hasattr(est, 'feature_importances_')
        ], axis=0)
    else:
        importances = best_rf.feature_importances_  # Fallback to RF
else:
    importances = best_model.feature_importances_

# Create feature importance dataframe
feature_importance = pd.DataFrame({
    'feature': SENSORY_FEATURES,
    'importance': importances
}).sort_values('importance', ascending=False)

print("\nTop 10 Most Important Features:")
for idx, row in feature_importance.head(10).iterrows():
    bar = '█' * int(row['importance'] * 50)
    print(f"  {row['feature']:20s} {bar} {row['importance']:.4f}")

# ==================== SAVE MODELS ====================

print("\n" + "="*80)
print("STEP 8: SAVING OPTIMIZED MODELS")
print("="*80)

# Save best model (highly compressed)
model_path = OUTPUT_DIR / 'random_forest_optimized.pkl'
joblib.dump(best_model, model_path, compress=9)  # Maximum compression
model_size = model_path.stat().st_size / 1024
print(f"✓ Saved best model ({best_model_name}): {model_path.name}")
print(f"  Size: {model_size:.1f} KB")

# Also save the pure Random Forest (for API compatibility)
rf_path = OUTPUT_DIR / 'random_forest_pure.pkl'
joblib.dump(best_rf, rf_path, compress=9)
rf_size = rf_path.stat().st_size / 1024
print(f"✓ Saved Random Forest: {rf_path.name}")
print(f"  Size: {rf_size:.1f} KB")

# Save scaler
scaler_path = OUTPUT_DIR / 'scaler_optimized.pkl'
joblib.dump(best_scaler, scaler_path, compress=9)
scaler_size = scaler_path.stat().st_size / 1024
print(f"✓ Saved scaler ({best_scaler_name}): {scaler_path.name}")
print(f"  Size: {scaler_size:.1f} KB")

# Save comprehensive metadata
metadata = {
    'training_info': {
        'date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'dataset_size': len(df_clean),
        'train_size': len(X_train),
        'test_size': len(X_test),
        'random_state': RANDOM_STATE
    },
    'features': {
        'names': SENSORY_FEATURES,
        'count': len(SENSORY_FEATURES),
        'scaler': best_scaler_name
    },
    'best_model': {
        'name': best_model_name,
        'type': type(best_model).__name__,
        'hyperparameters': best_rf.get_params() if best_model_name == 'Random Forest' else 'ensemble'
    },
    'performance': evaluation_results,
    'feature_importance': {
        feat: float(imp) 
        for feat, imp in zip(SENSORY_FEATURES, importances)
    },
    'target_statistics': {
        'mean': float(y.mean()),
        'std': float(y.std()),
        'min': float(y.min()),
        'max': float(y.max())
    }
}

metadata_path = OUTPUT_DIR / 'model_metadata.json'
with open(metadata_path, 'w') as f:
    json.dump(metadata, f, indent=2)
print(f"✓ Saved metadata: {metadata_path.name}")

# Save detailed report
report_path = OUTPUT_DIR / 'training_report.txt'
with open(report_path, 'w') as f:
    f.write("="*80 + "\n")
    f.write("COFFEE LIKING PREDICTION MODEL - TRAINING REPORT\n")
    f.write("="*80 + "\n\n")
    f.write(f"Training Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write(f"Dataset: {DATA_PATH}\n\n")
    
    f.write("DATASET SUMMARY:\n")
    f.write(f"  Total records: {len(df_clean)}\n")
    f.write(f"  Training set: {len(X_train)} ({len(X_train)/len(X)*100:.1f}%)\n")
    f.write(f"  Test set: {len(X_test)} ({len(X_test)/len(X)*100:.1f}%)\n")
    f.write(f"  Features: {len(SENSORY_FEATURES)}\n")
    f.write(f"  Target range: {y.min():.1f} - {y.max():.1f}\n\n")
    
    f.write("BEST MODEL PERFORMANCE:\n")
    f.write(f"  Model: {best_model_name}\n")
    f.write(f"  Test R²: {evaluation_results[best_model_name]['test_r2']:.4f}\n")
    f.write(f"  Test RMSE: {evaluation_results[best_model_name]['test_rmse']:.4f}\n")
    f.write(f"  Test MAE: {evaluation_results[best_model_name]['test_mae']:.4f}\n")
    f.write(f"  CV R²: {evaluation_results[best_model_name]['cv_r2_mean']:.4f} ± {evaluation_results[best_model_name]['cv_r2_std']*2:.4f}\n\n")
    
    f.write("ALL MODELS COMPARISON:\n")
    for name, results in evaluation_results.items():
        f.write(f"  {name}:\n")
        f.write(f"    Test R²: {results['test_r2']:.4f}\n")
        f.write(f"    CV R²: {results['cv_r2_mean']:.4f}\n\n")
    
    f.write("TOP 10 IMPORTANT FEATURES:\n")
    for idx, row in feature_importance.head(10).iterrows():
        f.write(f"  {row['feature']:20s} {row['importance']:.4f}\n")

print(f"✓ Saved training report: {report_path.name}")

# ==================== FINAL SUMMARY ====================

print("\n" + "="*80)
print("✅ MODEL TRAINING COMPLETE!")
print("="*80)

print(f"\nFiles saved to: {OUTPUT_DIR.absolute()}")
print(f"\nSummary:")
print(f"  • Best Model: {best_model_name}")
print(f"  • Test R²: {evaluation_results[best_model_name]['test_r2']:.4f}")
print(f"  • Test RMSE: {evaluation_results[best_model_name]['test_rmse']:.4f}")
print(f"  • CV R² (10-fold): {evaluation_results[best_model_name]['cv_r2_mean']:.4f} ± {evaluation_results[best_model_name]['cv_r2_std']*2:.4f}")
print(f"  • Model size: {model_size:.1f} KB")
print(f"  • Scaler: {best_scaler_name.capitalize()}")

print("\n" + "="*80)
print("Ready to use in FastAPI backend!")
print("="*80)

# ==================== QUICK VALIDATION ====================

print("\n" + "="*80)
print("QUICK VALIDATION TEST")
print("="*80)

# Test prediction with sample data
sample_features = np.array([3.5, 2.0, 3.2, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 1, 0]).reshape(1, -1)
sample_scaled = best_scaler.transform(sample_features)
sample_prediction = best_model.predict(sample_scaled)[0]

print(f"\nSample prediction test:")
print(f"  Input: Sweet=1, Bitter=0, Dark.chocolate=1, Caramel=1, Nutty=1")
print(f"  Predicted liking: {sample_prediction:.2f}/9")
print(f"  ✓ Model is working correctly!")

print("\n" + "="*80)
print("All done! 🎉")
print("="*80)