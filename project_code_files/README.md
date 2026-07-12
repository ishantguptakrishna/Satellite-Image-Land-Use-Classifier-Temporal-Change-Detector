1. README.md — project overview and full run instructions
2. requirements.txt — list of Python packages needed to run the project

#data/

1. datasets.py — shared dataset class (class list, file layout, transforms) used by every other script
2. download_data.py — downloads EuroSAT (main dataset) and UC Merced (holdout dataset)
3. spatial_split.py — splits data by geographic block (not by image) so train/val/test don't leak; also saves a naive random split for comparison
4. visualize_dataset.py — generates sample-image grids and class distribution charts for the report

#models/

1. baseline_cnn.py — simple 3-layer CNN trained from scratch, used as the "floor" to compare against
2. finetune_model.py — ResNet-18 wrapper implementing the two-phase fine-tuning logic, plus the embedding() function used for change detection
3. train_baseline.py — trains the baseline CNN
4. train_finetune.py — runs the two-phase fine-tuning (frozen backbone → unfrozen last layers) and saves finetuned.pt

#evaluation/

1. evaluate.py — measures accuracy on EuroSAT validation and the UC Merced holdout (zero-shot mapping + linear probe)
2. spatial_leakage.py — proves how much a naive random split inflates accuracy vs. the proper geographic split
3. error_analysis.py — shows the model's top-5 most confident wrong predictions
4. embedding_viz.py — 2D t-SNE/UMAP plot of embeddings, colored by class (Bonus C)
5. imbalance_experiment.py — tests how the model handles class imbalance and whether reweighting helps (Bonus D)

#change_detection/

1. build_embeddings.py — extracts 512-d embeddings for every tile and simulates before/after (T1/T2) pairs
2. detector.py — compares T1/T2 embeddings via cosine similarity, flags "changed" pairs, builds an ROC curve

#gradcam/

1. gradcam.py — GradCAM visualization showing which pixels drove each prediction (Bonus A)

#dashboard/

1. app.py — the Streamlit web app you've been using (upload, predict, compare, heatmap)

#configs/

1. dashboard.yaml — default checkpoint path and threshold preset for the dashboard
2. leakage.yaml — settings for the spatial leakage experiment

#report/

1. report_template.md — template/outline the final written report is meant to follow

#Other folders

1. checkpoints/ — where trained model weights (finetuned.pt, baseline.pt) are saved
2. outputs/ — where generated figures, evaluation results, and reports get written
3. notebooks/ — placeholder folder for exploratory notebooks (currently empty)
