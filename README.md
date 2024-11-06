# HEDDI-Net: Heterogeneous Network Embedding for Drug-Disease Association Prediction and Drug Repurposing

**HEDDI-Net is a heterogeneous embedding architecture designed to accurately predict drug-disease associations while preserving interpretability of biological mechanisms. By integrating graph and shallow learning techniques, HEDDI-Net extracts representative proteins and diseases, embedding them as input features in a multilayer perceptron model to predict drug-disease associations.**

---

HEDDI-Net combines protein-drug affinity profiles and disease similarity measures to generate interpretable embeddings, enhancing the model's predictive accuracy.

## Dataset information
**[chemical-disease associations]**

We utilized chemical-disease relationships from the Comparative Toxicogenomics Database (CTD) as drug-disease associations to train HEDDI-Net. Two types of associations were provided for training and evaluating the model:
- Direct evidence: Includes associations with direct evidence, such as marker/mechanism and therapeutic relationships, resulting in 71,187 chemical-disease associations between 6,074 chemicals and 2,802 diseases.
- Therapeutic: Contains only therapeutic association data, considered more reliable, with 26,789 drug-disease associations between 4,157 drugs and 2,149 diseases.

**[representative proteins]**
- We extracted 182, 228, and 311 proteins with a median z-score ≥ 3, 4, and 5, respectively, from 1,000 permutation tests to serve as representative proteins in HEDDI-Net.

**[representative diseases]**
- Based on the MeSH tree hierarchy structure, we identified 277 diseases well-connected within the network as representative diseases.

**[negative dataset]**
- A negative dataset of 8,011,328 drug-disease combinations was carefully selected to minimize misclassification risks. We excluded potential positive links by cross-referencing with the CTD database. Curated associations represent direct evidence from published literature, while inferred associations are based on shared gene interactions.

## Quick start guide	
* 1. Specify model parameters: Configure `train_type`, `pfeature`, and `pfdata` in `deepLearning_model_DNN_10fold.py`.
* 2. Run the model: Execute HEDDI-Net by running `deepLearning_model_DNN_10fold.py`.
* 3. View results: The performance metrics for drug-disease associations will be saved as `newtype2_ge4_1to1_e1600_bat11000.xlsx`.

## Code and data structure
`(./)` Root directory
* deepLearning_model_DNN_10fold.py: Script for training and evaluating the deep learning model of HEDDI-Net.
* metric_fn.py: Contains functions for calculating performance metrics.

`data/:`
* DL_CTD_wDE_data.txt: Predicted affinities with representative proteins and similarities with representative diseases in direct evidence chemical-disease association data.
* DL_CTD_thera_data.txt: Predicted affinities with representative proteins and similarities with representative diseases in therapeutic chemical-disease association data.
* neg_DL_CTD_thera_data.txt: Predicted affinities with representative proteins and similarities with representative diseases in the filtered negative dataset.

`data/npy/:`
* pos_ge3/4/5_wDE.npy: Features of positive samples for the direct evidence dataset with z-scores ≥ 3, 4, and 5, respectively.
* pos_ge3/4/5_thera.npy: Features of positive samples for the therapeutic dataset with z-scores ≥ 3, 4, and 5, respectively.
* neg_ge3/4/5.npy: Features for the negative dataset with z-scores ≥ 3, 4, and 5, respectively.

`B-dataset/:`
* DL_Bdata.txt: Predicted affinities with representative proteins and similarities with representative diseases in the smaller comparison dataset.
* neg_DL_Bdata.txt: Predicted affinities with representative proteins and similarities with representative diseases in the smaller comparison negative dataset.

`B-dataset/npy/:`
* pos_ge3/4/5.npy: Features of positive samples for the smaller comparison dataset with z-scores ≥ 3, 4, and 5, respectively.
* neg_ge3/4/5.npy: Features for the smaller comparison negative dataset with z-scores ≥ 3, 4, and 5, respectively.

Due to the size limitation, please downloaded **data** and **B-dataset** from [this link](https://140.129.68.214/share.cgi?ssid=0QiKvWs).

---
## Citation

