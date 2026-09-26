# test_data

Sample photographs for demos and for testing RetiLink on new images. Rebuild them with:

```bash
python scripts/make_test_data.py
```

| Folder | Contents | Use it for |
|---|---|---|
| `mbrset/singles/` | 2 photos per category: ICDR 0–4, macular edema, unusable quality. All from the held-out test split, never used in training. | **Try an image**, or `predict.py` |
| `mbrset/patients/<scenario>/` | 6 complete patients (both eyes, 2 views each): no DR, mild NPDR, referable in both eyes, referable in one eye, unusable quality, systemic-rich history. `patient.json` holds the age, diabetes history and reported conditions to type into **New screening**. | **New screening**: drop `OD_*` on the right eye and `OS_*` on the left |
| `brset/` | Canon CR and Nikon NF5050 photos: DR grades, edema, hypertensive retinopathy, AMD, vascular occlusion, inadequate quality | Domain-shift testing (a different camera from mBRSET) |

Every folder has a `manifest.csv` with the labels, which works directly with `python scripts/predict.py test_data/brset --labels test_data/brset/manifest.csv`.

**Data use:** mBRSET and BRSET are credentialed PhysioNet datasets. The images and manifests in this folder are git-ignored and excluded from the Vercel bundle. Do not share them, and do not upload them to the public deployment.
