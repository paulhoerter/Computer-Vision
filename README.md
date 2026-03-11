# 🫁 Pneumonia Detection — Computer Vision Project

## Overview

In this project you will build a **pneumonia detection system** from chest X-ray images.  
You will implement three deep learning architectures, expose them through a **FastAPI** backend,  
and interact with them through a **Streamlit** frontend.

At the end, you generate a CSV of predictions on the test set and submit it to the class leaderboard.

---

## Project Structure

```
pneumonia_project/
│
├── models/
│   ├── unet.py              ← U-Net architecture        (to complete)
│   ├── resnet.py            ← ResNet architecture       (to complete)
│   └── inception.py        ← Inception architecture    (to complete)
│
├── api/
│   └── main.py              ← FastAPI backend            (to complete)
│
├── app/
│   └── streamlit_app.py     ← Streamlit frontend        (to complete)
│
├── data/
│   ├── train/
│   │   ├── PNEUMONIA/
│   │   └── NORMAL/
│   ├── val/
│   │   ├── PNEUMONIA/
│   │   └── NORMAL/
│   └── test_for_students/   ← unlabelled test images (for submission)
│
├── sample_submission.csv    ← format reference for leaderboard submission
├── requirements.txt
└── README.md
```

---

## Setup

```bash
git clone <repo-url>
cd pneumonia_project

pip install -r requirements.txt
```

---

## How to run

**1. Start the FastAPI backend**
```bash
cd api
uvicorn main:app --reload --port 8000
```

**2. Start the Streamlit frontend** (in a second terminal)
```bash
cd app
streamlit run streamlit_app.py
```

Then open your browser at `http://localhost:8501`.

---

## Workflow

1. **Choose a model** (U-Net, ResNet, or Inception) in the sidebar
2. **Set hyperparameters** (learning rate, epochs, batch size, …)
3. Click **Train** — the frontend calls the FastAPI `/train` endpoint
4. View **training & validation curves** and metrics live
5. Click **Generate predictions** — calls `/predict` on the test set
6. **Download** the generated `submission.csv`
7. Upload it to the class leaderboard 🏆

---

## Models to implement

| File | Architecture | Key idea |
|---|---|---|
| `models/unet.py` | U-Net | Encoder-decoder with skip connections |
| `models/resnet.py` | ResNet | Residual blocks adapted for classification |
| `models/inception.py` | Inception | Multi-scale convolutions |

Each model file contains the class skeleton and the expected interface.  
**Do not change the class names or the `forward()` signature.**

---

## Leaderboard submission format

Your `submission.csv` must have exactly two columns:

```
id,prediction
img_0001,0.91
img_0002,0.07
img_0003,0.83
...
```

- `id` — image filename without extension (e.g. `img_0001`)
- `prediction` — probability of PNEUMONIA between 0 and 1 (not a hard label)

---

## Data
- `PNEUMONIA` images → label **1**
- `NORMAL` images → label **0**
