# Gender-Conditioned Face Shape Classification with Explainable AI

A deep learning pipeline that classifies face shape separately for male and female faces, using VGG16 transfer learning, Grad-CAM explainability, and a rule-based style recommendation layer. Built as an independent research project, with an accompanying paper written in IEEE conference format.

## 🔑 Key Results

| Model | Test Accuracy |
|---|---|
| Gender Classifier | **99.5%** |
| Female Face-Shape Classifier (5 classes) | **72.5%** (after class-weighted fine-tuning; 65.0% before) |
| Male Face-Shape Classifier (4 classes) | **79.7%** |

## 🧠 Why This Project

Most published face-shape classification research is trained **exclusively on female faces**, and almost none of it explains *why* a model makes a given prediction. This project addresses both gaps:

1. **Gender-conditioned architecture** — a gender classifier routes each face to a dedicated shape classifier trained specifically for that gender (5 classes for female: heart, oblong, oval, round, square; 4 classes for male: oval, rectangle, round, square), avoiding the taxonomy mismatch that arises from forcing both genders into one label set.
2. **Explainable AI** — Grad-CAM visualizes which facial regions drive each prediction.

### 🔍 Key Finding
Comparing Grad-CAM heatmaps across genders revealed something not previously documented in the literature: the well-known "oval faces are hard to classify" problem — reported across multiple prior papers — appears to be **specific to female training data**, not an inherent property of oval faces. The male oval class performs strongly (F1 = 0.78) while the female oval class lags even after targeted fixes (F1 = 0.55). Grad-CAM suggests female misclassifications are linked to the model attending to expression/accessory regions (mouth, earrings) rather than jaw/cheekbone structure — an effect the male model doesn't show.

## 🏗️ Architecture

```
Face Image
   → Face Detection & Cropping (OpenCV Haar Cascade)
   → Gender Classifier (VGG16, binary)
   → Route to gender-specific shape classifier
        Female → {Heart, Oblong, Oval, Round, Square}
        Male   → {Oval, Rectangle, Round, Square}
   → Grad-CAM heatmap (explainability)
   → Rule-based style recommendation (spectacles + hairstyle, gender-specific)
```
## 🚧 Limitations

- Male dataset (~1,011 images) is far smaller than female (5,000), limiting direct cross-gender comparison
- Datasets were independently labeled; inter-annotator agreement is unknown
- Gender is modeled as binary due to the binary nature of the source datasets, not as a normative claim
- The gender-asymmetry Grad-CAM finding is based on illustrative examples, not a full statistical study — flagged as future work
