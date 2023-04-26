# %%
import scvi_v2
import anndata

import matplotlib.pyplot as plt
# %%
adata = anndata.read_h5ad("/home/justin/ghrepos/scvi-v2-reproducibility/results/sciplex_pipeline/data/sciplex_A549_simple_filtered_all_phases.preprocessed.h5ad")
model = scvi_v2.MrVI.load("/home/justin/ghrepos/scvi-v2-reproducibility/results/sciplex_pipeline/models/sciplex_A549_simple_filtered_all_phases.scviv2_attention", adata=adata)
# %%
plt.plot(model.history_["train_loss_epoch"])

# %%
plt.plot(model.history_["elbo_train"])

# %%
plt.plot(model.history_["kl_local_train"])
# %%
cell_dists = model.get_local_sample_distances(
    adata, keep_cell=False, groupby="phase"
)
# %%
