#%%
#%matplotlib inline
from brainstat.stats.terms import FixedEffect
from brainstat.datasets import fetch_mask, fetch_template_surface
from brainstat.tutorial.utils import fetch_mics_data

# Load behavioral markers
thickness, demographics = fetch_mics_data()
pial_left, pial_right = fetch_template_surface("fsaverage5", join=False)
pial_combined = fetch_template_surface("fsaverage5", join=True)
mask = fetch_mask("fsaverage5")

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from brainspace.plotting import plot_hemispheres

plot_hemispheres(pial_left, pial_right, np.mean(thickness, axis=0), color_bar=True, color_range=(1.5, 3.5),
        label_text=["Cortical Thickness"], cmap="viridis", embed_nb=True, size=(1400, 200), zoom=1.45,
        nan_color=(0.7, 0.7, 0.7, 1), cb__labelTextProperty={"fontSize": 12}, interactive=False)

from brainstat.stats.SLM import SLM
contrast_age = demographics.AGE_AT_SCAN
term_sex = FixedEffect(demographics.SEX)


from brainstat.stats.terms import MixedEffect

term_subject = MixedEffect(demographics.SUB_ID)
term_age = FixedEffect(demographics.AGE_AT_SCAN)

model_mixed = term_age + term_sex + term_age * term_sex + term_subject

slm_mixed = SLM(
    model_mixed,
    -contrast_age,
    surf=pial_combined,
    mask=mask,
    correction=["fdr", "rft"],
    cluster_threshold=0.01,
    two_tailed=False,
)
slm_mixed.fit(thickness)

cp = [np.copy(slm_mixed.P["pval"]["C"])]


# %%
