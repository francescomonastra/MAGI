"""
Empirical latent bank: sample z from the encoded training events instead of
the learned prior p(z|cond).

Why this exists
---------------
The learned coupling prior follows the aggregated posterior q(z|cond) only
approximately. On DM1.2 (A9 in docs/CVAE_vs_AutoRegFlow.md) the gap sits at
low primary energy: for the muon crossings of muon primaries below 100 MeV the
prior gives 0.098 of the muons under 10 MeV, the posterior of the real events
0.060, the real data 0.060, and neither a weighted refit nor a larger prior
closed it. A bank of the training latents is the aggregated posterior itself:
each generated row takes the posterior q(z|x') of a training event x' with the
same crossing type and primary type and one of the K nearest primary energies,
and draws z from it. That brought the same fraction to 0.065.

The bank replays training latents, not events: z is a fresh draw from the
posterior of a real event, and every decoder head still samples its own noise,
so the generated crossing is new. What it gives up is generalisation of p(z|cond)
between the training conditionings, which a grouped generation never asks for:
it conditions on primary energies drawn from the same spectrum as training.

Conditioning layout
-------------------
cond = [crossing-type one-hot (n_types), primary-type one-hot, qt(E_prim)],
as built by build_conditioning_and_weights for primary-conditioned models.
When cond_dim == n_types (no primary conditioning) the bank is keyed on the
crossing type only and draws uniformly within it.
"""

import numpy as np
import tensorflow as tf


class LatentBank:
    """Posterior moments of training events, indexed by their conditioning.

    Parameters
    ----------
    z_mean, z_logvar : ndarray, shape (N, latent_dim)
        Encoder outputs of the training events.
    cond : ndarray, shape (N, cond_dim)
        Their conditioning rows.
    n_types : int
        Width of the crossing-type one-hot at the start of cond.
    k_neighbours : int
        A row draws from one of the k training events with the nearest qt(E_prim)
        in its (crossing type, primary type) group.
    seed : int
        Seeds the neighbour and z draws.
    """

    def __init__(self, z_mean, z_logvar, cond, n_types, k_neighbours=50, seed=0):
        self.z_mean = np.asarray(z_mean, dtype=np.float32)
        self.z_logvar = np.asarray(z_logvar, dtype=np.float32)
        cond = np.asarray(cond, dtype=np.float32)
        self.n_types = int(n_types)
        self.cond_dim = int(cond.shape[1])
        self.k_neighbours = int(k_neighbours)
        self.seed = int(seed)
        self._rng = np.random.default_rng(seed)
        self._cond = cond
        self.n_fallback = 0

        keys, e = self._keys(cond)
        # Per group: bank rows sorted by qt(E_prim), and those energies.
        self._groups = {}
        for key in np.unique(keys):
            rows = np.flatnonzero(keys == key)
            rows = rows[np.argsort(e[rows], kind="stable")]
            self._groups[int(key)] = (rows, e[rows])

    @property
    def primary_conditioned(self):
        return self.cond_dim > self.n_types

    def _keys(self, cond):
        """Group key (crossing type, primary type) and the energy coordinate."""
        t = np.argmax(cond[:, :self.n_types], axis=1)
        if not self.primary_conditioned:
            return t, np.zeros(len(cond), dtype=np.float32)
        n_prim = self.cond_dim - self.n_types - 1
        p = np.argmax(cond[:, self.n_types:self.n_types + n_prim], axis=1) if n_prim > 0 else 0
        return t * max(n_prim, 1) + p, cond[:, -1]

    def sample(self, cond):
        """One z per row of cond, and a mask of the rows with no bank entry
        (their z is left at 0 for the caller to fill from the prior)."""
        cond = np.asarray(cond, dtype=np.float32)
        if cond.shape[1] != self.cond_dim:
            raise ValueError(f"cond has {cond.shape[1]} columns, the bank {self.cond_dim}")
        keys, e = self._keys(cond)
        pick = np.full(len(cond), -1, dtype=np.int64)
        for key in np.unique(keys):
            sel = np.flatnonzero(keys == key)
            if int(key) not in self._groups:
                continue
            rows, e_bank = self._groups[int(key)]
            k = min(self.k_neighbours, len(rows))
            if self.primary_conditioned:
                pos = np.searchsorted(e_bank, e[sel])
                j = pos - k // 2 + self._rng.integers(0, k, len(sel))
                j = np.clip(j, 0, len(rows) - 1)
            else:
                j = self._rng.integers(0, len(rows), len(sel))
            pick[sel] = rows[j]
        missing = pick < 0
        self.n_fallback += int(missing.sum())
        z = np.zeros((len(cond), self.z_mean.shape[1]), dtype=np.float32)
        ok = ~missing
        eps = self._rng.standard_normal((int(ok.sum()), z.shape[1])).astype(np.float32)
        z[ok] = self.z_mean[pick[ok]] + np.exp(0.5 * self.z_logvar[pick[ok]]) * eps
        return z, missing

    def save(self, path):
        np.savez_compressed(path, z_mean=self.z_mean, z_logvar=self.z_logvar, cond=self._cond,
                            n_types=self.n_types, k_neighbours=self.k_neighbours, seed=self.seed)

    @classmethod
    def load(cls, path, k_neighbours=None, seed=None):
        d = np.load(path)
        return cls(d["z_mean"], d["z_logvar"], d["cond"], int(d["n_types"]),
                   k_neighbours=int(d["k_neighbours"]) if k_neighbours is None else k_neighbours,
                   seed=int(d["seed"]) if seed is None else seed)


def build_latent_bank(model, X_cont_s, cond, k_neighbours=50, seed=0, batch_size=65536):
    """Encode training events with `model.encoder` and return their LatentBank.

    Use the training split the model was fitted on (not the validation split),
    scaled exactly as in training: X_cont_s and cond as passed to model.fit.

    Parameters
    ----------
    model : CVAE_MixEnergy_ContPhi_TaskAdaptive
        The trained model whose encoder gives the posteriors.
    X_cont_s : array (n, d)
        Scaled continuous targets of the training split.
    cond : array (n, cond_dim)
        Conditioning rows of the same events.
    k_neighbours : int
        Passed to LatentBank: the number of nearest training events in qt(E_prim) a row draws from.
    seed : int
        Seeds the neighbour and z draws.
    batch_size : int
        Encoder batch size.

    Returns
    -------
    LatentBank
    """
    X_cont_s = np.asarray(X_cont_s, dtype=np.float32)
    cond = np.asarray(cond, dtype=np.float32)
    means, logvars = [], []
    for i in range(0, len(X_cont_s), batch_size):
        x = tf.concat([tf.constant(X_cont_s[i:i + batch_size]), tf.constant(cond[i:i + batch_size])], 1)
        m, lv = model.encoder(x, training=False)
        means.append(m.numpy())
        logvars.append(lv.numpy())
    return LatentBank(np.concatenate(means), np.concatenate(logvars), cond, int(model.n_types),
                      k_neighbours=k_neighbours, seed=seed)
