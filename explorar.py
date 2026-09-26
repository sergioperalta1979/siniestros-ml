from app.datos import consultar, leer_hechos


df = leer_hechos()
print(df.shape)
print(df.isna().sum())
print(df["tipo_de_via_siniestro"].value_counts(dropna=False).head())

# 1. Cuántos siniestros hay de cada gravedad, de mayor a menor
print(consultar("""
SELECT gravedad_siniestro, COUNT(*) AS n
FROM hechos
GROUP BY gravedad_siniestro
ORDER BY n DESC
"""))

# 2. Porcentaje de graves o mortales por modo de desplazamiento
print(consultar("""
SELECT modo_desplazamiento_victima AS modo, COUNT(*) AS n,
       ROUND(100.0 * SUM(gravedad_siniestro IN ('GRAVE', 'MORTAL')) / COUNT(*), 2) AS pct_graves
FROM hechos
GROUP BY modo
ORDER BY pct_graves DESC
"""))

# 3. Las 5 horas con mayor porcentaje de graves (SUBSTR toma las 2 primeras cifras de la hora)
print(consultar("""
SELECT SUBSTR(hora_siniestro, 1, 2) AS hora, COUNT(*) AS n,
       ROUND(100.0 * SUM(gravedad_siniestro IN ('GRAVE', 'MORTAL')) / COUNT(*), 2) AS pct_graves
FROM hechos
WHERE hora_siniestro IS NOT NULL
GROUP BY hora
ORDER BY pct_graves DESC
LIMIT 5
"""))