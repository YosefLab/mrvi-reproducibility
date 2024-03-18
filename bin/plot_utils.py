import plotnine as p9


INCH_TO_CM = 1 / 2.54
ALGO_RENAMER = {
    "mrvi": "mrvi",
    "mrvi_mlp": "mrvi(mlp)",
    "mrvi_prior": "mrvi(prior)",
    "mrvi_weighted": "mrvi(weighted)",
    "mrvi_attention": "mrvi(attention)",
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
