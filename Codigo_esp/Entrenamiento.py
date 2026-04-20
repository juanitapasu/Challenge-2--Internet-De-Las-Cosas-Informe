import json
import numpy as np
import pandas as pd

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


# -----------------------------
# Configuración
# -----------------------------
FILE = "Dataset-IOT.xlsx"
FEATURES = ["PM1", "PM25", "PM10", "NH3"]

# Percentiles para banderas (interpretables y consistentes con datos normalizados)
P_PRECAUCION = 75
P_PELIGRO = 90

# K a comparar (sanity check)
K_CANDIDATES = [2, 3, 4]


# -----------------------------
# Funciones auxiliares
# -----------------------------
def load_and_clean_excel(path: str, features: list[str]) -> np.ndarray:
    """Carga el Excel, valida columnas, fuerza numéricos, elimina NaNs y devuelve X numpy."""
    df = pd.read_excel(path)

    missing = [c for c in features if c not in df.columns]
    if missing:
        raise ValueError(
            f"Faltan columnas: {missing}\n"
            f"Columnas disponibles: {df.columns.tolist()}"
        )

    X = df[features].apply(pd.to_numeric, errors="coerce").dropna()

    if len(X) < 50:
        raise ValueError(
            f"Hay muy pocas filas válidas ({len(X)}). "
            "Revisa NaNs o nombres de columnas."
        )

    return X.to_numpy(dtype=float)


def compute_percentile_thresholds(X: np.ndarray, features: list[str]) -> dict:
    """Calcula umbrales por percentiles (por columna)."""
    p75 = np.percentile(X, P_PRECAUCION, axis=0)
    p90 = np.percentile(X, P_PELIGRO, axis=0)

    return {
        f"P{P_PRECAUCION}": {features[i]: float(p75[i]) for i in range(len(features))},
        f"P{P_PELIGRO}": {features[i]: float(p90[i]) for i in range(len(features))},
    }


def fit_kmeans(X: np.ndarray, k: int) -> KMeans:
    """Entrena KMeans con parámetros estables."""
    km = KMeans(n_clusters=k, n_init=20, random_state=42)
    km.fit(X)
    return km


def validate_k_candidates(X: np.ndarray, ks: list[int]) -> list[tuple[int, float]]:
    """Calcula silhouette para varios K."""
    results = []
    for k in ks:
        km = fit_kmeans(X, k)
        score = silhouette_score(X, km.labels_)
        results.append((k, float(score)))
    return results


def map_clusters_to_levels(centroids: np.ndarray, features: list[str]) -> dict[int, str]:
    """
    Asigna nivel a cada cluster basado en un score simple (suma de centroides).
    Con datos normalizados en 0-1, la suma es un proxy razonable de "riesgo".
    """
    scores = centroids.sum(axis=1)
    order = np.argsort(scores)  # menor→mayor

    mapping = {}
    mapping[int(order[0])] = "Normal"
    mapping[int(order[1])] = "Precaucion"
    mapping[int(order[2])] = "Peligro"
    return mapping


def cluster_distribution(labels: np.ndarray) -> dict[int, int]:
    """Cuenta cuántas muestras caen en cada cluster."""
    unique, counts = np.unique(labels, return_counts=True)
    return {int(u): int(c) for u, c in zip(unique, counts)}


# -----------------------------
# Main
# -----------------------------
def main():
    print("== Entrenamiento K-means IoT (PM1/PM25/PM10/NH3) ==")

    # 1) Cargar y limpiar
    X = load_and_clean_excel(FILE, FEATURES)
    n = X.shape[0]
    print(f"[OK] Datos cargados: {n} filas válidas, {X.shape[1]} features")

    # 2) Umbrales por percentiles (banderas)
    thresholds = compute_percentile_thresholds(X, FEATURES)
    print(f"[OK] Umbrales calculados: P{P_PRECAUCION} y P{P_PELIGRO}")

    # 3) Validación rápida de K (sanity check)
    print("\n== Validación: comparación de K por Silhouette ==")
    k_scores = validate_k_candidates(X, K_CANDIDATES)
    for k, sc in k_scores:
        print(f"K={k} -> silhouette={sc:.3f}")

    # 4) Entrenar modelo final con K=3
    print("\n== Entrenando modelo final (K=3) ==")
    kmeans = fit_kmeans(X, 3)
    centroids = kmeans.cluster_centers_
    labels = kmeans.labels_

    # 5) Verificaciones del modelo entrenado
    print("\n== Verificación del modelo (K=3) ==")
    sil = float(silhouette_score(X, labels))
    dist = cluster_distribution(labels)
    centroid_df = pd.DataFrame(centroids, columns=FEATURES)

    print(f"Silhouette score (K=3): {sil:.3f}")
    print("Distribución por cluster:", dist)
    print("\nCentroides (cada fila = un cluster):")
    print(centroid_df.to_string(index=True))

    # Recomendación textual simple (por si el silhouette sale bajito)
    if sil < 0.25:
        print("\n[AVISO] Silhouette < 0.25: clusters débiles.")
        print("Sugerencias: probar K=2, limpiar outliers o revisar si hay muchas filas repetidas.")
    elif sil < 0.5:
        print("\n[INFO] Silhouette entre 0.25 y 0.5: separación aceptable para demo IoT.")
    else:
        print("\n[OK] Silhouette > 0.5: separación fuerte.")

    # 6) Mapear cluster->nivel (Normal/Precaucion/Peligro)
    cluster_to_level = map_clusters_to_levels(centroids, FEATURES)
    print("\nMapeo cluster -> nivel:", cluster_to_level)

    # 7) Exportar todo para ESP32
    export = {
        "features": FEATURES,
        "data_assumption": "Features are normalized ~[0,1] (as in provided dataset).",
        "percentile_thresholds": thresholds,  # para banderas
        "kmeans": {
            "k": 3,
            "centroids": [[float(v) for v in row] for row in centroids],
            "cluster_to_level": {str(k): v for k, v in cluster_to_level.items()},
        },
        "validation": {
            "silhouette_k_candidates": [{"k": k, "silhouette": sc} for k, sc in k_scores],
            "silhouette_k3": sil,
            "cluster_distribution": dist,
        },
        "notes": {
            "risk_ordering": "Cluster risk assigned by centroid sum score (higher = higher risk).",
            "flags_logic": f"P{P_PRECAUCION}=Precaucion threshold, P{P_PELIGRO}=Peligro threshold per feature.",
        },
    }

    out_json = "iot_model_export.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(export, f, indent=2, ensure_ascii=False)

    print(f"\n[OK ✅] Exportado: {out_json}")
    print("Listo para copiar centroides/umbrales al firmware (ESP32).")


if __name__ == "__main__":
    main()