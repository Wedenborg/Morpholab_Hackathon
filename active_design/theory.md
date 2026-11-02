# Theory: the Gaussian factor model in `active_design/`

This file states the math of the Gaussian factor model (`model.py`, `designs.Session`) in the
form the code implements it. File and function names are given so each equation can be
checked against the code.

## 0. Problem and notation

The hidden embryo is one E7.5 Stereo-seq embryo. We may run experiments on it: one
experiment measures up to 4 genes in every cell of one section (one $z$ slice). After a fixed
number of experiments the method returns a full cells × genes matrix. The other E7.5
replicate is the **visible embryo**: fully observed, used for training. The hidden embryo is
never used for fitting or tuning.

| Symbol | Meaning |
|---|---|
| $g = 1..G$ | genes |
| $i$ | cells |
| $k = 1..K$, $K = 16$ | factors |
| $v$ | neighbourhoods of the visible embryo |
| $u$ | neighbourhoods of the hidden embryo |
| $\mathcal C_u$, $n_u$ | the cells of neighbourhood $u$ and their number (about 30) |
| $u(i)$, $v(i)$ | the neighbourhood of cell $i$ |
| $s$ | sections |
| $e$ | experiments; experiment $e$ measures the gene set $\mathcal G_e$ ($\lvert\mathcal G_e\rvert \le 4$) on section $s_e$ |
| $x_{ig}$ | Stereo-seq value: $\log(1 + c_{ig})$ with $c_{ig}$ the SCT-corrected count (an integer) |
| $d_g$ | detection rate of gene $g$ in the visible embryo, the fraction of cells with $x_{ig} > 0$ |
| $W \in \mathbb R^{G\times K}$, $w_g \in \mathbb R^K$ | gene weights; $w_g^\top$ is row $g$ of $W$ |
| $W_{\mathcal A}$ | the rows of $W$ for the genes in a set $\mathcal A$ ($\lvert\mathcal A\rvert\times K$) |

Neighbourhoods are k-means clusters on $(x, y)$ within one section, about 30 cells each
(`neighbourhood_bins`; the code calls them *units*). Each embryo is cut into its own
neighbourhoods. A neighbourhood never spans two sections.

The model has four parts: a factor model of expression, fitted on the visible embryo (1); a
spatial prior on the hidden embryo's factors (2); a likelihood for the measurements and the
posterior it gives (3); and the reconstruction of every cell from that posterior (4).

---

## 1. Factor model, fitted on the visible embryo (`FactorModel.fit`)

Everything in this section uses the visible embryo only.

**Neighbourhood averages and gene means.** With $V$ visible neighbourhoods,

$$\bar x_{vg} = \frac{1}{n_v}\sum_{i\in v} x_{ig}, \qquad b_g = \frac{1}{V}\sum_v \bar x_{vg}.$$

$b_g$ is gene $g$'s mean log1p value over the visible neighbourhoods. It is used as the gene
mean of the hidden embryo too: any overall difference between the embryos (sequencing depth,
a slightly different stage) has to be absorbed by the factors.

**Factors.** PCA with $K$ components on the vectors $\bar x_v$. The scores are standardised
so each factor has variance 1 across neighbourhoods; call them $Z_v \in \mathbb R^K$.

**Gene weights**, by least squares:

$$W = \arg\min_W \sum_v \lVert \bar x_v - b - W Z_v \rVert^2 .$$

**Neighbourhood-level noise** per gene, floored at $0.01\cdot\operatorname{median}_g r_g$:

$$r_g = \operatorname{Var}_v\big(\bar x_{vg} - b_g - w_g^\top Z_v\big).$$

$r_g$ holds both counting noise and whatever the $K$ factors do not explain.

**Cell level.** Each cell is projected onto the factors, and its spread around its
neighbourhood is measured:

$$z_i = (W^\top W)^{-1}W^\top(x_i - b), \qquad \delta_i = z_i - Z_{v(i)}, \qquad \Sigma_\delta = \operatorname{Cov}_i(\delta_i),$$

$$r^{c}_g = \operatorname{Var}_i\big(x_{ig} - b_g - w_g^\top z_i\big).$$

The model for one neighbourhood average is therefore $\bar x_{vg} = b_g + w_g^\top Z_v + \varepsilon_{vg}$
with $\varepsilon_{vg} \sim \mathcal N(0, r_g)$, and a cell is
$x_{ig} = b_g + w_g^\top(Z_{v(i)} + \delta_i) + e_{ig}$ with $e_{ig} \sim \mathcal N(0, r^c_g)$.

### 1.1 Factors and weights from mdCLA (`FactorModel.fit(source=...)`)

The alternative to the above takes the factor directions and the gene weights from mdCLA E7.5:
10,010 "Early embryo" single cells as $\log(1 + c\cdot \tilde L/L_i)$, with $L_i$ the cell's
library size and $\tilde L$ its median (`sources.mdcla_cells`). Genes are matched to
Stereo-seq by symbol. Everything downstream of $W$ is unchanged.

PCA with $K$ components on the mdCLA cells gives unit-norm directions with rows
$a_g \in \mathbb R^K$. The visible neighbourhoods are placed on these directions by least
squares, using only genes with $d_g \ge 0.02$ (set $\mathcal O$):

$$Z_v = \arg\min_z \sum_{g\in\mathcal O}\big(\bar x_{vg} - b_g - a_g^\top z\big)^2,$$

then standardised by their spread $\sigma_k$ per factor. mdCLA's pattern of weights is kept and
only its scale is fitted per gene. With $p_g = a_g\odot\sigma$ and $Z$ the $V\times K$ matrix of
the $Z_v$,

$$w_g = \tilde\beta_g\, p_g, \qquad \beta_g = \frac{\langle \bar x_{\cdot g}-b_g,\; Z p_g\rangle}{\lVert Z p_g\rVert^2},$$

$$\tilde\beta_g = \omega_g\beta_g + (1-\omega_g)\operatorname{median}_{g'}\beta_{g'}, \qquad \omega_g = \frac{d_g}{d_g + 0.05}.$$

Rarely detected genes, whose $\beta_g$ is unreliable, are shrunk toward the typical scale. $b$,
$r$, $\Sigma_\delta$ and $r^c$ are then fitted on the visible embryo as above, with these $W$
and $Z_v$.

---

## 2. Spatial prior on the hidden embryo (`unit_precision`)

The hidden embryo is cut into neighbourhoods from cell positions only; let $n$ be their
number. Each neighbourhood $u$ has a latent $Z_u \in \mathbb R^K$. A graph on neighbourhood
centroids:

- same section: each neighbourhood links to its 6 nearest neighbourhoods, weight 1 (symmetrised);
- adjacent sections: each neighbourhood links to the nearest neighbourhood in the previous and
  the next section, weight $\gamma = 1$.

With adjacency $A$, degrees $d_u = \sum_{u'} A_{uu'}$, Laplacian $L = D - A$ and smoothness
$\lambda = 3$:

$$Q = c\,(I + \lambda L), \qquad c = \operatorname{mean}\operatorname{diag}\big((I+\lambda L)^{-1}\big),$$

so that the average prior variance is 1. The $K$ factors are independent a priori and share
the same spatial precision:

$$\operatorname{vec}(Z) \sim \mathcal N\big(0,\; Q^{-1}\otimes I_K\big),$$

with $Z$ stacked neighbourhood by neighbourhood. Per factor $k$,

$$\log p(Z_{\cdot k}) = -\frac{c}{2}\Big(\sum_u Z_{uk}^2 + \lambda\sum_{\{u,u'\}} A_{uu'}(Z_{uk} - Z_{u'k})^2\Big) + \text{const},$$

where the second sum runs over linked pairs, each pair once. Per neighbourhood, given all the others,

$$Z_u \mid Z_{-u} \sim \mathcal N\Big(\frac{\lambda}{1+\lambda d_u}\sum_{u'} A_{uu'} Z_{u'},\ \frac{1}{c(1+\lambda d_u)} I_K\Big).$$

$\lambda$, $\gamma$, $K$ and the neighbourhood size were tuned on the visible embryo only
(`tune.py`); scores varied little across settings.

---

## 3. Likelihood and posterior (`UnitPosterior.update`, `Session.run`)

**Likelihood.** Experiment $e$ measures the genes $\mathcal G_e$ on every cell of section
$s_e$. For each neighbourhood $u$ in that section and each gene $g \in \mathcal G_e$, average
the measured values over the neighbourhood's cells:

$$\bar y_{ug} = \frac{1}{n_u}\sum_{i\in\mathcal C_u} x_{ig}.$$

The model for that average is

$$\bar y_{ug} = b_g + w_g^\top Z_u + \varepsilon_{ug}, \qquad \varepsilon_{ug}\sim\mathcal N(0,\ r_g),$$

independently across genes, neighbourhoods and experiments, with $b_g$, $w_g$ and $r_g$ from
section 1. Stacking the genes of experiment $e$,

$$y_u^{(e)} = \big(\bar y_{ug} - b_g\big)_{g\in\mathcal G_e} \sim \mathcal N\big(W_{\mathcal G_e} Z_u,\ R_e\big), \qquad R_e = \operatorname{diag}(r_g : g\in\mathcal G_e).$$

The log-likelihood of all experiments so far is

$$\log p(\text{data}\mid Z) = -\frac12\sum_e\ \sum_{u\subset s_e}\ \sum_{g\in\mathcal G_e}\frac{\big(\bar y_{ug}-b_g-w_g^\top Z_u\big)^2}{r_g} + \text{const}.$$

Each term involves only one neighbourhood's own $Z_u$; the likelihood has no links between
neighbourhoods. It assumes three things that hold only roughly:

- An average of about 30 log1p values is close to Gaussian for well-detected genes. For a gene
  detected in 2% of cells most averages are exactly 0, which a Gaussian handles poorly.
- $r_g$ does not depend on $n_u$. All neighbourhoods have about 30 cells, so this is acceptable.
- The part of $r_g$ the factors miss is spatially smooth, so the errors are not really
  independent across neighbourhoods.

**Posterior.** Prior and likelihood are both Gaussian in $Z$, so the posterior is Gaussian.
For each neighbourhood, collect what its measurements contribute:

$$F_u = \sum_{e:\,u\subset s_e}\ \sum_{g\in\mathcal G_e}\frac{w_g w_g^\top}{r_g}\quad(K\times K), \qquad h_u = \sum_{e:\,u\subset s_e}\ \sum_{g\in\mathcal G_e}\frac{w_g\,(\bar y_{ug}-b_g)}{r_g}\quad(K),$$

with $F_u = 0$, $h_u = 0$ for neighbourhoods not yet measured. Then

$$\operatorname{vec}(Z)\mid\text{data}\sim\mathcal N\big(\Lambda^{-1}h,\ \Lambda^{-1}\big),\qquad \Lambda = Q\otimes I_K + \operatorname{blockdiag}_u(F_u).$$

Per neighbourhood, given all the others,

$$Z_u\mid Z_{-u},\text{data}\ \sim\ \mathcal N\Big(P_u^{-1}\big(c\lambda\textstyle\sum_{u'}A_{uu'}Z_{u'} + h_u\big),\ P_u^{-1}\Big),\qquad P_u = c(1+\lambda d_u)\,I_K + F_u :$$

a precision-weighted combination of the neighbours and the neighbourhood's own measurements.
$F_u$ adds at most rank 4 per experiment, so in the remaining directions of factor space $Z_u$
is set by its neighbours. $F_u$ is a full $K\times K$ matrix, so the factors are no longer
independent after a measurement, and $\Lambda^{-1}$ is dense, so a measurement moves every
neighbourhood, less with distance in the graph.

**What the code does.** It keeps the posterior as a mean $m$ and a dense covariance $\Sigma$
over all $nK$ (neighbourhood, factor) pairs, starting from $m = 0$, $\Sigma = Q^{-1}\otimes I_K$,
and applies an exact Kalman update after each experiment, which gives the same result as the
formula above. For experiment $e$ with the $n_e$ neighbourhoods of section $s_e$, stack their
$y_u^{(e)}$ into $y$; then $y = H\operatorname{vec}(Z) + \varepsilon$, where $H$ places
$I_{n_e}\otimes W_{\mathcal G_e}$ on those neighbourhoods' entries, and
$R = I_{n_e}\otimes R_e$:

$$\Gamma = \Sigma H^\top(H\Sigma H^\top + R)^{-1}, \qquad m \leftarrow m + \Gamma(y - Hm), \qquad \Sigma \leftarrow \Sigma - \Gamma H\Sigma.$$

A gene already measured on a section is never measured there again (it would return the same
values and count twice).

**Note.** Neither $\Lambda$ nor the update for $\Sigma$ involves the measured values. Under this
model the uncertainty left after any sequence of experiments is known before any experiment
is run, so a design that maximises information gain is not adaptive.

---

## 4. Reconstruction (`Session.reconstruct`)

Every cell starts from its neighbourhood's posterior mean, $z_i = m_{u(i)}$. Let
$\mathcal M$ be the genes measured so far on the cell's section. If $\mathcal M$ is not empty,
the cell is corrected by its own values:

$$C = \Sigma_{uu} + \Sigma_\delta, \qquad z_i = m_u + C W_{\mathcal M}^\top\big(W_{\mathcal M} C W_{\mathcal M}^\top + \operatorname{diag}(r^c_g : g\in\mathcal M)\big)^{-1}\big(x_{i\mathcal M} - b_{\mathcal M} - W_{\mathcal M} m_u\big),$$

where $u = u(i)$ and $\Sigma_{uu}$ is its $K\times K$ posterior covariance block. The output is
$\hat x_{ig} = \max(0,\, b_g + w_g^\top z_i)$, with measured entries replaced by the measured
values.

This counts the cell's values twice: they were already part of the neighbourhood average in
section 3. With about 30 cells per neighbourhood the effect is small.
