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
