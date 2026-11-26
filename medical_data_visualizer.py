import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# 1 : Importer les données
df = pd.read_csv('medical_examination.csv')

# 2 : Colonne overweight (IMC > 25 → 1, sinon 0)
df['overweight'] = (df['weight'] / (df['height'] / 100) ** 2 > 25).astype(int)

# 3 : Normalisation (0 = bon, 1 = mauvais)
df['cholesterol'] = (df['cholesterol'] > 1).astype(int)
df['gluc'] = (df['gluc'] > 1).astype(int)


# 4 : Diagramme catégoriel
def draw_cat_plot():
    # 5 : Format long avec pd.melt
    df_cat = pd.melt(
        df,
        id_vars=['cardio'],
        value_vars=['cholesterol', 'gluc', 'smoke', 'alco', 'active', 'overweight']
    )

    # 6 : Regrouper et compter, la colonne des comptes s'appelle 'total'
    df_cat = (
        df_cat.groupby(['cardio', 'variable', 'value'])
        .size()
        .reset_index(name='total')
    )

    # 7 : Créer le diagramme
    g = sns.catplot(
        data=df_cat,
        kind='bar',
        x='variable',
        y='total',
        hue='value',
        col='cardio'
    )

    # 8 : Récupérer la figure
    fig = g.fig

    # 9 : Ne pas modifier
    fig.savefig('catplot.png')
    return fig


# 10 : Carte de chaleur
def draw_heat_map():
    # 11 : Nettoyer les données incorrectes
    df_heat = df[
        (df['ap_lo'] <= df['ap_hi']) &
        (df['height'] >= df['height'].quantile(0.025)) &
        (df['height'] <= df['height'].quantile(0.975)) &
        (df['weight'] >= df['weight'].quantile(0.025)) &
        (df['weight'] <= df['weight'].quantile(0.975))
    ]

    # 12 : Matrice de corrélation
    corr = df_heat.corr()

    # 13 : Masque du triangle supérieur
    mask = np.triu(np.ones_like(corr, dtype=bool))

    # 14 : Figure matplotlib
    fig, ax = plt.subplots(figsize=(12, 10))

    # 15 : Carte de chaleur
    sns.heatmap(
        corr,
        mask=mask,
        annot=True,
        fmt='.1f',
        center=0,
        vmax=0.3,
        vmin=-0.1,
        square=True,
        linewidths=0.5,
        cbar_kws={'shrink': 0.5},
        ax=ax
    )

    # 16 : Ne pas modifier
    fig.savefig('heatmap.png')
    return fig