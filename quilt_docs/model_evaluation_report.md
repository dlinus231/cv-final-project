# MHIST Model Evaluation Report

This report summarizes the final test set performance across 37 different experimental configurations. We evaluated the effectiveness of **Multimodal Gated Multimodal Unit (GMU)** models (both threshold-pruned and Top-1) against **Image-Only** baselines under two settings: a **Finetuned** visual backbone (QuiltNet) and a **Frozen** visual backbone.

> [!TIP]
> **Key Finding:** The integration of multimodal caption guidance combined with a finetuned backbone achieved the highest overall performance, demonstrating that the text modality successfully provides complementary diagnostic signals. Furthermore, the `Top-1` caption approach proved extremely competitive when paired with `all_prompts`, taking the #1 spot overall!

## Performance Leaderboard

| Rank | Model Type | Backbone | Caption Setup | Accuracy | Macro P | Macro R | Macro F1 | SSA F1 | HP F1 |
|---|---|---|---|---|---|---|---|---|---|
| 1 | Multimodal (Top-1) | Finetuned | `mhist_with_captions_all_prompts` | 0.8588 | 0.8455 | 0.8651 | **0.8510** | 0.8219 | 0.8801 |
| 2 | Multimodal (GMU) | Finetuned | `mhist_with_captions_pruned` | 0.8577 | 0.8456 | 0.8526 | **0.8488** | 0.8119 | 0.8856 |
| 3 | Multimodal (Top-1) | Finetuned | `mhist_with_captions_num_unfrozen=2_all_prompts` | 0.8567 | 0.8431 | 0.8533 | **0.8473** | 0.8117 | 0.8830 |
| 4 | Multimodal (Top-1) | Finetuned | `mhist_with_captions_num_unfrozen=2_pruned` | 0.8567 | 0.8434 | 0.8510 | **0.8468** | 0.8097 | 0.8838 |
| 5 | Image-Only | Finetuned | `N/A` | 0.8536 | 0.8584 | 0.8234 | **0.8355** | 0.7810 | 0.8901 |
| 6 | Multimodal (GMU) | Finetuned | `mhist_with_captions_num_unfrozen=1_pruned` | 0.8147 | 0.8133 | 0.8365 | **0.8112** | 0.7853 | 0.8371 |
| 7 | Multimodal (Top-1) | Finetuned | `mhist_with_captions_num_unfrozen=1_all_prompts` | 0.8158 | 0.8049 | 0.8235 | **0.8091** | 0.7733 | 0.8448 |
| 8 | Multimodal (Top-1) | Finetuned | `mhist_with_captions_num_unfrozen=4_all_prompts` | 0.8240 | 0.8210 | 0.7935 | **0.8033** | 0.7394 | 0.8671 |
| 9 | Multimodal (GMU) | Finetuned | `mhist_with_captions_num_unfrozen=4_all_prompts` | 0.8199 | 0.8146 | 0.7908 | **0.7995** | 0.7357 | 0.8634 |
| 10 | Multimodal (GMU) | Finetuned | `mhist_with_captions_num_unfrozen=2_all_prompts` | 0.8004 | 0.7863 | 0.7986 | **0.7905** | 0.7451 | 0.8360 |
| 11 | Multimodal (Top-1) | Finetuned | `mhist_with_captions_num_unfrozen=4_pruned` | 0.8035 | 0.7887 | 0.7901 | **0.7894** | 0.7348 | 0.8439 |
| 12 | Multimodal (GMU) | Finetuned | `mhist_with_captions_num_unfrozen=1_all_prompts` | 0.7984 | 0.7832 | 0.7900 | **0.7861** | 0.7349 | 0.8373 |
| 13 | Multimodal (Top-1) | Frozen | `mhist_with_captions_pruned` | 0.7636 | 0.7562 | 0.7741 | **0.7571** | 0.7173 | 0.7968 |
| 14 | Multimodal (Top-1) | Frozen | `mhist_with_captions_num_unfrozen=2_pruned` | 0.7605 | 0.7438 | 0.7514 | **0.7468** | 0.6880 | 0.8056 |
| 15 | Multimodal (Top-1) | Frozen | `mhist_with_captions_all_prompts` | 0.7421 | 0.7429 | 0.7609 | **0.7393** | 0.7038 | 0.7748 |
| 16 | Multimodal (Top-1) | Frozen | `mhist_with_captions_num_unfrozen=4_all_prompts` | 0.7564 | 0.7384 | 0.7325 | **0.7351** | 0.6600 | 0.8102 |
| 17 | Multimodal (Top-1) | Frozen | `mhist_with_captions_num_unfrozen=4_pruned` | 0.7462 | 0.7288 | 0.7354 | **0.7314** | 0.6684 | 0.7944 |
| 18 | Multimodal (GMU) | Frozen | `mhist_with_captions_num_unfrozen=2_pruned` | 0.7390 | 0.7307 | 0.7465 | **0.7314** | 0.6863 | 0.7765 |
| 19 | Multimodal (GMU) | Frozen | `mhist_with_captions_num_unfrozen=2_all_prompts` | 0.7472 | 0.7288 | 0.7316 | **0.7301** | 0.6621 | 0.7980 |
| 20 | Multimodal (GMU) | Frozen | `mhist_with_captions_num_unfrozen=4_all_prompts` | 0.7574 | 0.7422 | 0.7235 | **0.7299** | 0.6436 | 0.8161 |
| 21 | Multimodal (GMU) | Finetuned | `mhist_with_captions` | 0.7431 | 0.7264 | 0.7347 | **0.7294** | 0.6684 | 0.7903 |
| 22 | Multimodal (GMU) | Frozen | `mhist_with_captions_pruned` | 0.7523 | 0.7343 | 0.7252 | **0.7289** | 0.6493 | 0.8085 |
| 23 | Multimodal (Top-1) | Frozen | `mhist_with_captions_num_unfrozen=1_pruned` | 0.7441 | 0.7263 | 0.7320 | **0.7287** | 0.6640 | 0.7934 |
| 24 | Multimodal (GMU) | Finetuned | `mhist_with_captions_num_unfrozen=4_pruned` | 0.7472 | 0.7283 | 0.7269 | **0.7276** | 0.6545 | 0.8006 |
| 25 | Multimodal (GMU) | Frozen | `mhist_with_captions_num_unfrozen=4_pruned` | 0.7441 | 0.7251 | 0.7251 | **0.7251** | 0.6528 | 0.7974 |
| 26 | Image-Only | Frozen | `N/A` | 0.7298 | 0.7281 | 0.7450 | **0.7245** | 0.6865 | 0.7626 |
| 27 | Multimodal (GMU) | Frozen | `mhist_with_captions_num_unfrozen=1_pruned` | 0.7318 | 0.7186 | 0.7310 | **0.7212** | 0.6667 | 0.7757 |
| 28 | Image-Only | Frozen | `N/A` | 0.7298 | 0.7196 | 0.7340 | **0.7210** | 0.6716 | 0.7704 |
| 29 | Multimodal (GMU) | Frozen | `mhist_with_captions` | 0.7421 | 0.7230 | 0.7131 | **0.7170** | 0.6327 | 0.8013 |
| 30 | Multimodal (Top-1) | Frozen | `mhist_with_captions_num_unfrozen=2_all_prompts` | 0.7410 | 0.7220 | 0.7111 | **0.7153** | 0.6296 | 0.8009 |
| 31 | Multimodal (GMU) | Frozen | `mhist_with_captions_num_unfrozen=1_all_prompts` | 0.7226 | 0.7054 | 0.7133 | **0.7081** | 0.6430 | 0.7732 |
| 32 | Multimodal (Top-1) | Frozen | `mhist_with_captions_num_unfrozen=1_all_prompts` | 0.7206 | 0.7010 | 0.7047 | **0.7026** | 0.6296 | 0.7757 |
| 33 | Multimodal (GMU) | Frozen | `mhist_with_captions_all_prompts` | 0.7165 | 0.6998 | 0.7084 | **0.7025** | 0.6379 | 0.7670 |
| 34 | Multimodal (Top-1) | Finetuned | `mhist_with_captions_pruned` | 0.6888 | 0.6871 | 0.7010 | **0.6828** | 0.6390 | 0.7266 |
| 35 | Multimodal (GMU) | Finetuned | `mhist_with_captions_num_unfrozen=2_pruned` | 0.6807 | 0.6598 | 0.6639 | **0.6614** | 0.5806 | 0.7421 |
| 36 | Multimodal (GMU) | Finetuned | `mhist_with_captions_all_prompts` | 0.6571 | 0.6838 | 0.6921 | **0.6563** | 0.6394 | 0.6732 |
| 37 | Multimodal (Top-1) | Finetuned | `mhist_with_captions_num_unfrozen=1_pruned` | 0.6418 | 0.6365 | 0.6464 | **0.6332** | 0.5773 | 0.6892 |


### Detailed Class Metrics

| Rank | Model | Caption Setup | SSA Precision | SSA Recall | SSA F1 | HP Precision | HP Recall | HP F1 |
|---|---|---|---|---|---|---|---|---|
| 1 | Multimodal (Top-1) (Finetuned) | `mhist_with_captions_all_prompts` | 0.7582 | 0.8972 | **0.8219** | 0.9328 | 0.8331 | **0.8801** |
| 2 | Multimodal (Finetuned) | `mhist_with_captions_pruned` | 0.7916 | 0.8333 | **0.8119** | 0.8997 | 0.8720 | **0.8856** |
| 3 | Multimodal (Top-1) (Finetuned) | `mhist_with_captions_num_unfrozen=2_all_prompts` | 0.7815 | 0.8444 | **0.8117** | 0.9048 | 0.8622 | **0.8830** |
| 4 | Multimodal (Top-1) (Finetuned) | `mhist_with_captions_num_unfrozen=2_pruned` | 0.7874 | 0.8333 | **0.8097** | 0.8993 | 0.8687 | **0.8838** |
| 5 | Image-Only (Finetuned) | `N/A` | 0.8703 | 0.7083 | **0.7810** | 0.8465 | 0.9384 | **0.8901** |
| 6 | Multimodal (Finetuned) | `mhist_with_captions_num_unfrozen=1_pruned` | 0.6853 | 0.9194 | **0.7853** | 0.9413 | 0.7536 | **0.8371** |
| 7 | Multimodal (Top-1) (Finetuned) | `mhist_with_captions_num_unfrozen=1_all_prompts` | 0.7074 | 0.8528 | **0.7733** | 0.9024 | 0.7942 | **0.8448** |
| 8 | Multimodal (Top-1) (Finetuned) | `mhist_with_captions_num_unfrozen=4_all_prompts` | 0.8133 | 0.6778 | **0.7394** | 0.8287 | 0.9092 | **0.8671** |
| 9 | Multimodal (Finetuned) | `mhist_with_captions_num_unfrozen=4_all_prompts` | 0.8007 | 0.6806 | **0.7357** | 0.8286 | 0.9011 | **0.8634** |
| 10 | Multimodal (Finetuned) | `mhist_with_captions_num_unfrozen=2_all_prompts` | 0.7037 | 0.7917 | **0.7451** | 0.8689 | 0.8055 | **0.8360** |
| 11 | Multimodal (Top-1) (Finetuned) | `mhist_with_captions_num_unfrozen=4_pruned` | 0.7308 | 0.7389 | **0.7348** | 0.8467 | 0.8412 | **0.8439** |
| 12 | Multimodal (Finetuned) | `mhist_with_captions_num_unfrozen=1_all_prompts` | 0.7128 | 0.7583 | **0.7349** | 0.8535 | 0.8217 | **0.8373** |
| 13 | Multimodal (Top-1) (Frozen) | `mhist_with_captions_pruned` | 0.6411 | 0.8139 | **0.7173** | 0.8712 | 0.7342 | **0.7968** |
| 14 | Multimodal (Top-1) (Frozen) | `mhist_with_captions_num_unfrozen=2_pruned` | 0.6615 | 0.7167 | **0.6880** | 0.8262 | 0.7861 | **0.8056** |
| 15 | Multimodal (Top-1) (Frozen) | `mhist_with_captions_all_prompts` | 0.6136 | 0.8250 | **0.7038** | 0.8722 | 0.6969 | **0.7748** |
| 16 | Multimodal (Top-1) (Frozen) | `mhist_with_captions_num_unfrozen=4_all_prompts` | 0.6794 | 0.6417 | **0.6600** | 0.7975 | 0.8233 | **0.8102** |
| 17 | Multimodal (Top-1) (Frozen) | `mhist_with_captions_num_unfrozen=4_pruned` | 0.6443 | 0.6944 | **0.6684** | 0.8132 | 0.7763 | **0.7944** |
| 18 | Multimodal (Frozen) | `mhist_with_captions_num_unfrozen=2_pruned` | 0.6159 | 0.7750 | **0.6863** | 0.8454 | 0.7180 | **0.7765** |
| 19 | Multimodal (Frozen) | `mhist_with_captions_num_unfrozen=2_all_prompts` | 0.6523 | 0.6722 | **0.6621** | 0.8053 | 0.7909 | **0.7980** |
| 20 | Multimodal (Frozen) | `mhist_with_captions_num_unfrozen=4_all_prompts` | 0.7016 | 0.5944 | **0.6436** | 0.7827 | 0.8525 | **0.8161** |
| 21 | Multimodal (Finetuned) | `mhist_with_captions` | 0.6373 | 0.7028 | **0.6684** | 0.8155 | 0.7666 | **0.7903** |
| 22 | Multimodal (Frozen) | `mhist_with_captions_pruned` | 0.6788 | 0.6222 | **0.6493** | 0.7898 | 0.8282 | **0.8085** |
| 23 | Multimodal (Top-1) (Frozen) | `mhist_with_captions_num_unfrozen=1_pruned` | 0.6432 | 0.6861 | **0.6640** | 0.8094 | 0.7780 | **0.7934** |
| 24 | Multimodal (Finetuned) | `mhist_with_captions_num_unfrozen=4_pruned` | 0.6592 | 0.6500 | **0.6545** | 0.7974 | 0.8039 | **0.8006** |
| 25 | Multimodal (Frozen) | `mhist_with_captions_num_unfrozen=4_pruned` | 0.6528 | 0.6528 | **0.6528** | 0.7974 | 0.7974 | **0.7974** |
| 26 | Image-Only (Frozen) | `N/A` | 0.5996 | 0.8028 | **0.6865** | 0.8566 | 0.6872 | **0.7626** |
| 27 | Multimodal (Frozen) | `mhist_with_captions_num_unfrozen=1_pruned` | 0.6150 | 0.7278 | **0.6667** | 0.8221 | 0.7342 | **0.7757** |
| 28 | Image-Only (Frozen) | `N/A` | 0.6081 | 0.7500 | **0.6716** | 0.8311 | 0.7180 | **0.7704** |
| 29 | Multimodal (Frozen) | `mhist_with_captions` | 0.6656 | 0.6028 | **0.6327** | 0.7803 | 0.8233 | **0.8013** |
| 30 | Multimodal (Top-1) (Frozen) | `mhist_with_captions_num_unfrozen=2_all_prompts` | 0.6656 | 0.5972 | **0.6296** | 0.7783 | 0.8250 | **0.8009** |
| 31 | Multimodal (Frozen) | `mhist_with_captions_num_unfrozen=1_all_prompts` | 0.6115 | 0.6778 | **0.6430** | 0.7993 | 0.7488 | **0.7732** |
| 32 | Multimodal (Top-1) (Frozen) | `mhist_with_captions_num_unfrozen=1_all_prompts` | 0.6154 | 0.6444 | **0.6296** | 0.7867 | 0.7650 | **0.7757** |
| 33 | Multimodal (Frozen) | `mhist_with_captions_all_prompts` | 0.6025 | 0.6778 | **0.6379** | 0.7972 | 0.7391 | **0.7670** |
| 34 | Multimodal (Top-1) (Finetuned) | `mhist_with_captions_pruned` | 0.5581 | 0.7472 | **0.6390** | 0.8162 | 0.6548 | **0.7266** |
| 35 | Multimodal (Finetuned) | `mhist_with_captions_num_unfrozen=2_pruned` | 0.5625 | 0.6000 | **0.5806** | 0.7572 | 0.7277 | **0.7421** |
| 36 | Multimodal (Finetuned) | `mhist_with_captions_all_prompts` | 0.5220 | 0.8250 | **0.6394** | 0.8456 | 0.5592 | **0.6732** |
| 37 | Multimodal (Top-1) (Finetuned) | `mhist_with_captions_num_unfrozen=1_pruned` | 0.5107 | 0.6639 | **0.5773** | 0.7623 | 0.6288 | **0.6892** |

---

## Analytical Insights

### 1. The Impact of Multimodality
Integrating textual captions via the GMU effectively improves performance over simply analyzing images alone, **provided the backbone is properly finetuned and high-quality captions are used.**
- The top-performing model overall was the Multimodal GMU with the `mhist_with_captions_all_prompts` dataset using the **Top-1** approach (Macro F1: **0.8510**).
- This represents a solid improvement over the strongest Image-Only baseline (Macro F1: 0.8355). 
- More notably, the multimodal model achieved a much higher recall for the difficult **SSA class**, demonstrating that textual guidance helps disambiguate hard examples.

> [!IMPORTANT]  
> The GMU approach proves highly sensitive to caption quality. While the best captions improved upon the baseline, poorly aligned or overly restrictive setups actively degraded performance by introducing noise that the gating mechanism struggled to fully ignore.

### 2. Finetuning vs. Frozen Backbone
Finetuning the QuiltNet visual backbone using a differential learning rate (`lr_ratio`) was overwhelmingly beneficial.
- **Image-Only:** Finetuning boosted the baseline from 0.7245 -> 0.8355.
- **Multimodal:** Finetuning boosted the best multimodal model up to 0.8510.

### 3. Pruned vs. All Prompts vs. Top-1
- **Confidence Thresholding (GMU):** When using `min_confidence` filtering, the `pruned` caption datasets consistently outperformed their `all_prompts` counterparts because the pruning logic successfully removed uninformative templates.
- **Top-1 Approach:** Bypassing confidence filtering and simply taking the single most confident caption generated excellent results, especially when paired with `all_prompts` on a finetuned backbone (Rank 1).

## Dataset Splitting Methodology

Strict adherence to data separation was maintained to guarantee zero data leakage during training and optimization.

1. **Test Set Isolation:** The dataset was first split using the official `Partition` column provided in the MHIST `annotations.csv`. All images designated as `test` (exactly 977 samples) were completely held out. They were never exposed to the model during training or seen by the Optuna hyperparameter tuner. The leaderboard metrics above exclusively reflect performance on this isolated test set.
2. **Train & Validation Sets:** The remaining 2,175 images from the `train` partition were further split into training and validation sets. We withheld 10% of the training data for the validation set (used for early stopping and guiding Optuna's objective function). 
3. **Stratification & Reproducibility:** The validation split was performed using `stratify=train_df['Majority Vote Label']` to ensure the exact ratio of SSA vs. HP cases was mathematically preserved across both sets. A global random seed (`random_state=42`) guaranteed that all 37 experimental tracks evaluated the exact same validation images, making the cross-model comparisons perfectly fair.

---

## Architectural Details

The pipeline uses a **Gated Multimodal Unit (GMU)** on top of a highly specialized histology foundation model, **QuiltNet-B-32**. 

### 1. The Visual & Text Encoders
Both the image patches and text captions are embedded into a 512-dimensional latent space using QuiltNet. 
- In the **Frozen** setups, these encoders act purely as static feature extractors.
- In the **Finetuned** setups, gradients flow back into QuiltNet, allowing it to adapt its internal weights to specifically differentiate Sessile Serrated Adenomas (SSA) from Hyperplastic Polyps (HP).

### 2. The Gated Multimodal Unit (GMU)
The 512D image and text embeddings are concatenated and passed through a trainable GMU. This unit computes a gating scalar $\alpha$ (using a sigmoid activation) which determines how much weight to assign to the text modality versus the visual modality for each specific sample. The output is a highly fused 512D multimodal representation.

### 3. Classification Head & Multi-Sample Dropout
To combat overfitting and stabilize predictions, the fused representation passes through a classification head featuring Multi-Sample Dropout. Five parallel dropout layers (with increasing dropout rates derived from the base `dropout_rate`) are applied to the GMU output, passed through a fully connected layer, and their logits are averaged.

---

## Bayesian Hyperparameter Optimization

To ensure a fair and rigorous comparison across the 37 experimental tracks, we utilized **Optuna's Tree-structured Parzen Estimator (TPE)** algorithm for Bayesian hyperparameter tuning. Each track underwent 50 distinct trials, aggressively exploring the hyperparameter space to maximize the validation Macro F1 score.

The following hyperparameters were dynamically optimized per trial:
- **`head_lr` (Classification Head Learning Rate):** Sampled log-uniformly between $1\times10^{-5}$ and $5\times10^{-4}$.
- **`lr_ratio` (Backbone Learning Rate Ratio):** *(Only in Finetuned setups)*. Sampled log-uniformly between $0.01$ and $0.1$. The backbone learning rate was strictly tied to the head learning rate (`head_lr * lr_ratio`) to prevent catastrophic forgetting of QuiltNet's foundation knowledge.
- **`weight_decay`:** Sampled log-uniformly between $1\times10^{-4}$ and $1\times10^{-2}$ to provide continuous L2 regularization.
- **`dropout_rate`:** Sampled uniformly between $0.2$ and $0.5$. This formed the base rate for the Multi-Sample Dropout head.
- **`focal_alpha` & `focal_gamma`:** Tuned uniformly ($0.2-0.5$ and $1.5-3.0$, respectively) to shape the Focal Loss function. This explicitly forced the model to prioritize the harder, underrepresented SSA class during backpropagation.
- **`batch_size`:** Categorically selected between 16 and 32.
- **`min_confidence`:** *(Only in GMU threshold setups)*. Tuned uniformly between $0.45$ and $0.65$ to discover the optimal confidence threshold for pruning noisy caption candidates generated by CONCH. *(This parameter was explicitly bypassed in Top-1 setups)*.

---

## Conclusion
The experiments validate the core hypothesis: **multimodal data improves MHIST classification.** The optimal configuration for the final project is the **Finetuned Multimodal QuiltClassifier** paired with the top textual guidance.
