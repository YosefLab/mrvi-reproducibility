import plotnine as p9


INCH_TO_CM = 1 / 2.54
ALGO_RENAMER = {
    "scviv2": "scviv2",
    "scviv2_mlp": "scviv2(mlp)",
    "scviv2_prior": "scviv2(prior)",
    "scviv2_weighted": "scviv2(weighted)",
    "scviv2_attention": "scviv2(attention)",
    "composition_SCVI_clusterkey_subleiden1": "composition(SCVI)",
    "composition_PCA_clusterkey_subleiden1": "composition(PCA)",
}
SHARED_THEME = p9.theme(
    strip_background=p9.element_blank(),
    subplots_adjust={"wspace": 0.3},
    # panel_background=p9.element_blank(),
    axis_text=p9.element_text(family="sans-serif", size=7),
    axis_title=p9.element_text(family="sans-serif", size=8),
)

SCIPLEX_PATHWAY_CMAP = {
    "Antioxidant": "#00FFFF",  # aqua
    "Apoptotic regulation": "#DAA520",  # goldenrod
    "Cell cycle regulation": "#008080",  # teal
    "DNA damage & DNA repair": "#808000",  # olive
    "Epigenetic regulation": "#000080",  # navy
    "Focal adhesion signaling": "#A52A2A",  # brown
    "HIF signaling": "#FFC0CB",  # pink
    "JAK/STAT signaling": "#008000",  # green
    "Metabolic regulation": "#FFD700",  # gold
    "Neuronal signaling": "#FA8072",  # salmon
    "Nuclear receptor signaling": "#7FFF00",  # chartreuse
    "PKC signaling": "#DDA0DD",  # plum
    "Protein folding & Protein degradation": "#4B0082",  # indigo
    "TGF/BMP signaling": "#00FFFF",  # cyan
    "Tyrosine kinase signaling": "#ADD8E6",  # lightblue
    "Other": "#DA70D6",  # orchid
    "Vehicle": "#FF0000",  # red
}