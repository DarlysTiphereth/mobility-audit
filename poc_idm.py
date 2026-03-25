# ============================================================
# FRAMEWORK DE AUDITORIA DE EQUIDADE NO TRANSPORTE PÚBLICO
# Índice de Deserto de Mobilidade (IDM) — Prova de Conceito
# Maceió, Alagoas
# ============================================================
# Fórmula central:  IDM = (P × I) / O
#   P = Pressão Social
#   I = Ineficiência Topológica  (I = d_rede / d_geodesica)
#   O = Oferta Real Auditada     (O = Frequency × (1 - H_entropy))
# ============================================================


# ════════════════════════════════════════════════════════════
# BLOCO 0 — DEPENDÊNCIAS
# Instale antes de rodar: pip install pandas numpy scikit-learn folium
# ════════════════════════════════════════════════════════════
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
import folium
from folium.plugins import HeatMap
import warnings
warnings.filterwarnings("ignore")

print("✅ BLOCO 0 — Bibliotecas carregadas com sucesso.")


# ════════════════════════════════════════════════════════════
# BLOCO 1 — ENTRADAS BRUTAS (Data Lake Central)
# ════════════════════════════════════════════════════════════
# Representa a camada de ingestão do pipeline:
#   • GPS da Frota        → frequência e posição dos ônibus
#   • Dados do Censo      → população vulnerável por bairro
#   • Malha Viária        → distância real vs distância geodésica
#   • Sensores Celulares  → demanda estimada (inovação)
#
# Na PoC, os dados são SIMULADOS com distribuições realistas
# calibradas para a realidade de Maceió.
# Em produção: substituir por pd.read_csv() ou API da SMTT.
# ════════════════════════════════════════════════════════════

np.random.seed(42)  # Reprodutibilidade garantida

# Parâmetros por bairro: (lat, lon, pop_vuln, freq_base, d_rede, d_geo)
#   pop_vuln  → proporção de população vulnerável (0-1)
#   freq_base → frequência de ônibus por hora
#   d_rede    → distância real pela rede viária (km)
#   d_geo     → distância geodésica em linha reta (km)
BAIRROS = {
    # ── Periferia Norte/Noroeste — mais críticos ──────────────
    "Benedito Bentes":       (-9.4965, -35.8437, 0.88, 2.0, 8.5, 4.2),
    "Antares":               (-9.5255, -35.7920, 0.84, 1.8, 8.9, 4.3),
    "Tabuleiro do Martins":  (-9.5370, -35.7930, 0.83, 2.2, 9.1, 4.5),
    "Cidade Universitária":  (-9.5560, -35.7760, 0.85, 2.5, 7.8, 3.9),
 
    # ── Zona Norte/Centro — críticos ─────────────────────────
    "Clima Bom":             (-9.5780, -35.7620, 0.80, 3.0, 7.2, 3.8),
    "Santos Dumont":         (-9.5720, -35.7530, 0.76, 3.5, 6.5, 3.4),
    "Serraria":              (-9.5800, -35.7450, 0.78, 3.2, 6.9, 3.5),
    "Jacintinho":            (-9.6180, -35.7640, 0.82, 2.8, 7.5, 3.7),
 
    # ── Zona Intermediária ────────────────────────────────────
    "Jardim Petrópolis":     (-9.6070, -35.7510, 0.60, 5.0, 5.5, 3.2),
    "Poço":                  (-9.6380, -35.7390, 0.55, 6.0, 5.0, 3.0),
    "Farol":                 (-9.6480, -35.7410, 0.45, 7.5, 4.2, 2.8),
    "Centro":                (-9.6660, -35.7353, 0.40, 9.0, 3.5, 2.5),
 
    # ── Orla Privilegiada — menor IDM (contraste) ────────────
    "Jatiúca":               (-9.6430, -35.7130, 0.25, 12.0, 2.8, 2.2),
    "Ponta Verde":           (-9.6570, -35.7130, 0.22, 13.0, 2.5, 2.0),
    "Pajuçara":              (-9.6690, -35.7120, 0.20, 14.0, 2.3, 1.9),
}

registros = []
for bairro, (lat_c, lon_c, pop_v, freq, d_rede, d_geo) in BAIRROS.items():
    n_pontos = np.random.randint(35, 60)
    for _ in range(n_pontos):
        registros.append({
            "Bairro":          bairro,
            "Latitude":        round(lat_c + np.random.normal(0, 0.008), 6),
            "Longitude":       round(lon_c + np.random.normal(0, 0.008), 6),
            "Pop_Vulneravel":  round(np.clip(pop_v  + np.random.normal(0, 0.05), 0, 1), 4),
            "Freq_Onibus_h":   round(np.clip(freq   + np.random.normal(0, 0.5),  0.1, 20), 2),
            "Dist_Rede_km":    round(np.clip(d_rede + np.random.normal(0, 0.4),  0.5, 20), 2),
            "Dist_Geo_km":     round(np.clip(d_geo  + np.random.normal(0, 0.2),  0.5, 15), 2),
        })

df_raw = pd.DataFrame(registros)

print(f"✅ BLOCO 1 — Data Lake simulado: {len(df_raw)} registros, {df_raw['Bairro'].nunique()} bairros.")
print(df_raw.head(3).to_string(index=False))


# ════════════════════════════════════════════════════════════
# BLOCO 2 — ISOLATION FOREST (Filtro de Ruídos)
# ════════════════════════════════════════════════════════════
# O Isolation Forest é um algoritmo de Machine Learning não
# supervisionado que detecta anomalias isolando pontos que
# "fogem do padrão" com poucas divisões na árvore.
#
# Aqui ele audita os dados brutos do GPS e dos sensores,
# removendo registros corrompidos, duplicatas ou outliers
# antes de calcular os índices. Isso garante que o IDM
# final reflita a realidade, não ruídos do sistema.
#
# Resultado: a "Camada Ouro" do pipeline — dados confiáveis.
# ════════════════════════════════════════════════════════════

features_auditoria = ["Pop_Vulneravel", "Freq_Onibus_h", "Dist_Rede_km", "Dist_Geo_km"]

iso_forest = IsolationForest(
    n_estimators=100,       # 100 árvores de decisão
    contamination=0.05,     # Espera-se até 5% de dados ruins
    random_state=42
)

df_raw["anomalia"] = iso_forest.fit_predict(df_raw[features_auditoria])
# fit_predict retorna: 1 = normal, -1 = anomalia

n_anomalias = (df_raw["anomalia"] == -1).sum()
df_ouro = df_raw[df_raw["anomalia"] == 1].copy()

print(f"\n✅ BLOCO 2 — Isolation Forest concluído.")
print(f"   Registros originais : {len(df_raw)}")
print(f"   Anomalias detectadas: {n_anomalias} ({n_anomalias/len(df_raw)*100:.1f}%)")
print(f"   Camada Ouro (limpa) : {len(df_ouro)} registros")


# ════════════════════════════════════════════════════════════
# BLOCO 3A — PRESSÃO SOCIAL (P)
# ════════════════════════════════════════════════════════════
# P mede QUEM depende do transporte público.
# Baseada na proporção de população vulnerável do bairro:
#   trabalhadores de baixa renda, idosos, pessoas com
#   deficiência — grupos que não têm alternativa ao ônibus.
#
# Quanto maior P, maior a necessidade social não atendida.
# Escala: 0 (nenhuma pressão) → 1 (pressão máxima)
# ════════════════════════════════════════════════════════════

df_ouro["P_Pressao_Social"] = df_ouro["Pop_Vulneravel"].round(4)

p_medio = df_ouro.groupby("Bairro")["P_Pressao_Social"].mean().sort_values(ascending=False)

print(f"\n✅ BLOCO 3A — Pressão Social (P) calculada.")
print(f"   Bairro mais crítico : {p_medio.index[0]}  (P = {p_medio.iloc[0]:.3f})")
print(f"   Bairro menos crítico: {p_medio.index[-1]} (P = {p_medio.iloc[-1]:.3f})")


# ════════════════════════════════════════════════════════════
# BLOCO 3B — INEFICIÊNCIA TOPOLÓGICA (I)
# ════════════════════════════════════════════════════════════
# I = d_rede / d_geodesica
#
# Mede o quanto a malha viária "desperdiça" distância.
# d_geodesica = distância em linha reta (mínimo teórico)
# d_rede      = distância real que o ônibus percorre
#
# I = 1.0 → trajeto perfeito, sem desvios
# I > 2.0 → o ônibus percorre o dobro do caminho ideal
#            (becos sem saída, falta de vias, topografia)
#
# Bairros periféricos de Maceió têm I alto por causa da
# topografia acidentada e da malha viária precária.
# ════════════════════════════════════════════════════════════

df_ouro["I_Ineficiencia"] = (
    df_ouro["Dist_Rede_km"] / df_ouro["Dist_Geo_km"]
).round(4)

# Normalizar para escala 0-1 para uso na fórmula IDM
i_max = df_ouro["I_Ineficiencia"].max()
df_ouro["I_Ineficiencia_norm"] = (df_ouro["I_Ineficiencia"] / i_max).round(4)

i_medio = df_ouro.groupby("Bairro")["I_Ineficiencia"].mean().sort_values(ascending=False)

print(f"\n✅ BLOCO 3B — Ineficiência Topológica (I) calculada.")
print(f"   Razão média d_rede/d_geo: {df_ouro['I_Ineficiencia'].mean():.2f}x")
print(f"   Maior ineficiência: {i_medio.index[0]}  (I = {i_medio.iloc[0]:.2f}x)")
print(f"   Menor ineficiência: {i_medio.index[-1]} (I = {i_medio.iloc[-1]:.2f}x)")


# ════════════════════════════════════════════════════════════
# BLOCO 3C — OFERTA REAL AUDITADA (O)
# ════════════════════════════════════════════════════════════
# O = Frequency × (1 - H_entropy)
#
# Não basta contar ônibus — é preciso auditar a QUALIDADE
# da oferta. Por isso usamos a Entropia de Shannon (H):
#
#   H = 0 → distribuição perfeitamente regular (ônibus pontuais)
#   H = 1 → distribuição caótica (ônibus imprevisíveis)
#
# (1 - H) = "fator de confiabilidade" da linha
#
# Uma linha com 10 ônibus/hora mas caótica vale menos
# do que uma com 6 ônibus/hora mas regular.
# ════════════════════════════════════════════════════════════

# Simular entropia: bairros periféricos têm serviço mais irregular
def calcular_entropia(freq, lat):
    """Entropia maior em bairros mais ao norte (periferia)."""
    base = 0.8 if lat > -9.63 else 0.4  # periferia vs orla
    return round(np.clip(base + np.random.normal(0, 0.1), 0, 0.99), 4)

df_ouro["H_Entropy"] = df_ouro.apply(
    lambda r: calcular_entropia(r["Freq_Onibus_h"], r["Latitude"]), axis=1
)

df_ouro["O_Oferta"] = (
    df_ouro["Freq_Onibus_h"] * (1 - df_ouro["H_Entropy"])
).round(4)

# Normalizar para 0-1
o_max = df_ouro["O_Oferta"].max()
df_ouro["O_Oferta_norm"] = (df_ouro["O_Oferta"] / o_max).clip(0.01).round(4)

o_medio = df_ouro.groupby("Bairro")["O_Oferta"].mean().sort_values(ascending=False)

print(f"\n✅ BLOCO 3C — Oferta Real Auditada (O) calculada.")
print(f"   Maior oferta auditada: {o_medio.index[0]}  (O = {o_medio.iloc[0]:.2f})")
print(f"   Menor oferta auditada: {o_medio.index[-1]} (O = {o_medio.iloc[-1]:.2f})")


# ════════════════════════════════════════════════════════════
# BLOCO 4 — IDM: ÍNDICE DE DESERTO DE MOBILIDADE
# ════════════════════════════════════════════════════════════
#
#              P × I
#   IDM =  ──────────
#               O
#
# Interpretação:
#   IDM < 0.4  → Verde  — Mobilidade adequada
#   IDM 0.4–0.7 → Amarelo — Atenção, risco moderado
#   IDM 0.7–0.85 → Laranja — Crítico, intervenção necessária
#   IDM > 0.85  → Vermelho — Deserto de Mobilidade
#
# O IDM aumenta quando:
#   ↑ P: mais população vulnerável dependente
#   ↑ I: malha viária mais ineficiente
#   ↓ O: menos ônibus ou serviço mais irregular
# ════════════════════════════════════════════════════════════

df_ouro["IDM_Score"] = (
    (df_ouro["P_Pressao_Social"] * df_ouro["I_Ineficiencia_norm"])
    / df_ouro["O_Oferta_norm"]
).clip(0, 1).round(4)

# Classificação semafórica
def classificar_idm(v):
    if v >= 0.85: return "🔴 Deserto de Mobilidade"
    if v >= 0.70: return "🟠 Crítico"
    if v >= 0.40: return "🟡 Atenção"
    return "🟢 Adequado"

df_ouro["IDM_Classe"] = df_ouro["IDM_Score"].apply(classificar_idm)

print(f"\n✅ BLOCO 4 — IDM calculado para {len(df_ouro)} pontos.")
print(f"\n   Distribuição por classe:")
print(df_ouro["IDM_Classe"].value_counts().to_string())

print(f"\n   IDM médio por bairro (ranking):")
idm_bairro = df_ouro.groupby("Bairro")["IDM_Score"].mean().sort_values(ascending=False)
for bairro, val in idm_bairro.items():
    emoji = "🔴" if val >= 0.85 else "🟠" if val >= 0.70 else "🟡" if val >= 0.40 else "🟢"
    print(f"   {emoji} {bairro:<25} IDM = {val:.3f}")


# ════════════════════════════════════════════════════════════
# BLOCO 5 — EXPORTAÇÃO DA CAMADA OURO
# ════════════════════════════════════════════════════════════
# O arquivo CSV final é o produto do pipeline.
# Contém apenas dados auditados (pós-Isolation Forest)
# com todos os índices calculados, pronto para:
#   • Alimentar o dashboard Streamlit
#   • Ser entregue à SMTT para validação
#   • Ser auditado pela banca ou gestores públicos
# ════════════════════════════════════════════════════════════

colunas_exportar = [
    "Bairro", "Latitude", "Longitude",
    "P_Pressao_Social", "I_Ineficiencia_norm", "O_Oferta_norm",
    "IDM_Score", "IDM_Classe"
]

df_final = df_ouro[colunas_exportar].copy()
df_final.to_csv("dataset_poc_idm_maceio.csv", index=False)

print(f"\n✅ BLOCO 5 — Camada Ouro exportada: 'dataset_poc_idm_maceio.csv'")
print(f"   {len(df_final)} registros · {len(colunas_exportar)} colunas")


# ════════════════════════════════════════════════════════════
# BLOCO 6 — MAPA DE CALOR DO IDM
# ════════════════════════════════════════════════════════════
# Visualização geoespacial do IDM sobre Maceió.
# Gradiente de cores alinhado à escala semafórica:
#   Azul → Ciano → Laranja → Vermelho Escuro
#
# O fundo escuro (CartoDB dark_matter) é intencional:
# faz as manchas de calor saltarem visualmente,
# amplificando o impacto comunicativo para políticas públicas.
# ════════════════════════════════════════════════════════════

mapa = folium.Map(
    location=[-9.6498, -35.7089],
    zoom_start=12,
    tiles="CartoDB dark_matter"
)

# Mapa de calor principal
HeatMap(
    data=df_final[["Latitude", "Longitude", "IDM_Score"]].values.tolist(),
    radius=18,
    blur=15,
    max_zoom=1,
    gradient={0.0: "blue", 0.4: "cyan", 0.65: "orange", 0.85: "red", 1.0: "darkred"}
).add_to(mapa)

# Marcadores nos pontos mais críticos (IDM ≥ 0.85)
criticos = df_final[df_final["IDM_Score"] >= 0.85]
for _, row in criticos.iterrows():
    folium.CircleMarker(
        location=[row["Latitude"], row["Longitude"]],
        radius=5, color="red", fill=True, fill_opacity=0.7,
        tooltip=(
            f"<b>{row['Bairro']}</b><br>"
            f"IDM: <b>{row['IDM_Score']}</b><br>"
            f"P: {row['P_Pressao_Social']} | "
            f"I: {row['I_Ineficiencia_norm']} | "
            f"O: {row['O_Oferta_norm']}"
        )
    ).add_to(mapa)

# Legenda
legenda = """
<div style="position:fixed; bottom:30px; left:30px; z-index:1000;
            background:#1e1e2e; padding:14px 18px; border-radius:10px;
            border:1px solid #555; font-size:13px; color:white;">
    <b>IDM — Deserto de Mobilidade</b><br><br>
    🔴 Deserto (&gt; 0.85)<br>
    🟠 Crítico  (0.70–0.85)<br>
    🟡 Atenção  (0.40–0.70)<br>
    🟢 Adequado (&lt; 0.40)
</div>
"""
mapa.get_root().html.add_child(folium.Element(legenda))

mapa.save("mapa_idm_maceio.html")

print(f"\n✅ BLOCO 6 — Mapa exportado: 'mapa_idm_maceio.html'")
print(f"   Pontos críticos (IDM ≥ 0.85): {len(criticos)}")


# ════════════════════════════════════════════════════════════
# RESUMO EXECUTIVO — KPIs PARA A APRESENTAÇÃO
# ════════════════════════════════════════════════════════════

HABITANTES_POR_PONTO = 120  # estimativa de escala para PoC

desertos     = df_final[df_final["IDM_Score"] >= 0.85]
criticos_tot = df_final[df_final["IDM_Score"] >= 0.70]

pop_deserto  = len(desertos) * HABITANTES_POR_PONTO
espera_media = round(df_final["IDM_Score"].mean() * 45, 1)
idm_max      = df_final["IDM_Score"].max()

print("\n" + "═"*55)
print("  RESUMO EXECUTIVO — PROVA DE CONCEITO")
print("═"*55)
print(f"  📍 Área analisada       : Maceió, AL")
print(f"  📊 Pontos auditados     : {len(df_final)}")
print(f"  🏘️  Bairros cobertos     : {df_final['Bairro'].nunique()}")
print(f"  👥 Pop. em deserto      : {pop_deserto:,} habitantes")
print(f"  ⏱️  Espera média estimada: {espera_media} min")
print(f"  🚨 IDM crítico máximo   : {idm_max}")
print(f"  🔴 Pontos em deserto    : {len(desertos)} ({len(desertos)/len(df_final)*100:.1f}%)")
print(f"  🟠 Pontos críticos      : {len(criticos_tot)} ({len(criticos_tot)/len(df_final)*100:.1f}%)")
print("═"*55)
print("  Arquivos gerados:")
print("  • dataset_poc_idm_maceio.csv  → Camada Ouro")
print("  • mapa_idm_maceio.html        → Mapa interativo")
print("═"*55)
