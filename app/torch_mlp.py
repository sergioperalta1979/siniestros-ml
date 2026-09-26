import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from torch import nn

from app.datos import leer_hechos
from app.entrenar import armar_preprocesador, dividir
from app.features import preparar

EPOCAS = 15
LOTE = 512


def armar_red(entradas):
    return nn.Sequential(
        nn.Linear(entradas, 32),
        nn.ReLU(),
        nn.Linear(32, 16),
        nn.ReLU(),
        nn.Linear(16, 1),
    )


def main():
    torch.manual_seed(0)
    X, y, anio = preparar(leer_hechos())
    X_ent, X_prueba, y_ent, y_prueba = dividir(X, y, anio)

    pre = armar_preprocesador()
    A_ent = torch.tensor(pre.fit_transform(X_ent), dtype=torch.float32)
    A_prueba = torch.tensor(pre.transform(X_prueba), dtype=torch.float32)
    b_ent = torch.tensor(y_ent.to_numpy(), dtype=torch.float32).unsqueeze(1)
    print("Forma de entrada:", tuple(A_ent.shape))

    # Los graves son pocos: la pérdida les da más peso, como class_weight en scikit-learn
    peso = (1 - y_ent.mean()) / y_ent.mean()
    perdida = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(float(peso)))
    red = armar_red(A_ent.shape[1])
    optimizador = torch.optim.Adam(red.parameters(), lr=1e-3)
    
    for epoca in range(EPOCAS):
        red.train()
        orden = torch.randperm(len(A_ent))
        total = 0.0
        for inicio in range(0, len(orden), LOTE):
            lote = orden[inicio : inicio + LOTE]
            optimizador.zero_grad()
            error = perdida(red(A_ent[lote]), b_ent[lote])
            error.backward()
            optimizador.step()
            total += error.item() * len(lote)
        print(f"Epoca {epoca + 1:2d}  perdida {total / len(orden):.4f}")
        
    red.eval()
    with torch.no_grad():
        prob = torch.sigmoid(red(A_prueba)).squeeze(1).numpy()
    print("PR-AUC :", round(average_precision_score(y_prueba, prob), 4))
    print("ROC-AUC:", round(roc_auc_score(y_prueba, prob), 4))
    print("Prevalencia (piso del PR-AUC):", round(float(np.mean(y_prueba)), 4))


if __name__ == "__main__":
    main()