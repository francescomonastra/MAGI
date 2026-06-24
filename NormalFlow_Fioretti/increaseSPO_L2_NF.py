#!/usr/bin/env python3

from astropy.io import fits
import numpy as np
import sys, os
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


# -------------------------
# Input parameters
# -------------------------
if len(sys.argv) != 9:
    print("Usage:")
    print("python increaseSPO_L2_NF.py N_in theta_max E_start E_stop E_step N_out_factor n_layers random_state")
    sys.exit(1)

N_in = int(sys.argv[1])
theta_max = int(sys.argv[2])
E_start = int(sys.argv[3])
E_stop = int(sys.argv[4])
E_step = int(sys.argv[5])
N_out_factor = int(sys.argv[6])
n_layers = int(sys.argv[7])
random_state = int(sys.argv[8])


# -------------------------
# Flow training parameters
# -------------------------
hidden_dim = 128
epochs = 500
batch_size_train = 2048
learning_rate = 1e-3


# -------------------------
# Configuration
# -------------------------
n_dets = 2
det_type = ["_XIFU", "_WFI"]
#height_det = [567.54, 506.932]
radius_det = [90.0, 240.0]
det_name = ["X-IFU", "WFI"]

torch.manual_seed(random_state)
np.random.seed(random_state)


# -------------------------
# Plot utilities
# -------------------------
def theta_from_minus_z(ux, uy, uz):
    cos_theta = -uz
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    return np.degrees(np.arccos(cos_theta))


def make_comparison_plots(plot_dir, tag,
                          x_in, y_in, e_in, ux_in, uy_in, uz_in,
                          x_out, y_out, e_out, ux_out, uy_out, uz_out):

    os.makedirs(plot_dir, exist_ok=True)

    tx_in = ux_in / np.abs(uz_in)
    ty_in = uy_in / np.abs(uz_in)
    r_in = np.sqrt(x_in**2 + y_in**2)
    theta_in = theta_from_minus_z(ux_in, uy_in, uz_in)

    tx_out = ux_out / np.abs(uz_out)
    ty_out = uy_out / np.abs(uz_out)
    r_out = np.sqrt(x_out**2 + y_out**2)
    theta_out = theta_from_minus_z(ux_out, uy_out, uz_out)

    hist_data = [
        ("E_OUT", e_in, e_out, "Energy [keV]"),
        ("X_OUT", x_in, x_out, "X [mm]"),
        ("Y_OUT", y_in, y_out, "Y [mm]"),
        ("R_OUT", r_in, r_out, "R [mm]"),
        ("COSX_OUT", ux_in, ux_out, "COSX"),
        ("COSY_OUT", uy_in, uy_out, "COSY"),
        ("COSZ_OUT", uz_in, uz_out, "COSZ"),
        ("THETA_OUT", theta_in, theta_out, "Theta from -Z [deg]"),
        ("TX", tx_in, tx_out, "tx = COSX / |COSZ|"),
        ("TY", ty_in, ty_out, "ty = COSY / |COSZ|"),
    ]

    for name, vin, vout, xlabel in hist_data:
        plt.figure(figsize=(7, 5))

        counts, bin_edges = np.histogram(vin, bins=80)
        errors = np.sqrt(counts)

        bin_widths = np.diff(bin_edges)
        norm = np.sum(counts * bin_widths)

        density = counts / norm
        density_err = errors / norm

        bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

        plt.errorbar(
            bin_centers,
            density,
            yerr=density_err,
            fmt='o',
            markersize=3,
            label="Input",
            capsize=2
        )

        counts, bin_edges = np.histogram(vout, bins=80)
        errors = np.sqrt(counts)

        bin_widths = np.diff(bin_edges)
        norm = np.sum(counts * bin_widths)

        density = counts / norm
        density_err = errors / norm

        bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

        plt.errorbar(
            bin_centers,
            density,
            yerr=density_err,
            fmt='o',
            markersize=3,
            label="Flow output",
            capsize=2
        )

        plt.xlabel(xlabel)
        plt.ylabel("Normalized counts")
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, f"{tag}_{name}_hist.png"), dpi=150)
        plt.close()

    scatter_data = [
        ("E_vs_X", x_in, e_in, x_out, e_out, "X [mm]", "Energy [keV]"),
        ("E_vs_Y", y_in, e_in, y_out, e_out, "Y [mm]", "Energy [keV]"),
        ("E_vs_R", r_in, e_in, r_out, e_out, "R [mm]", "Energy [keV]"),
        ("E_vs_THETA", theta_in, e_in, theta_out, e_out, "Theta from -Z [deg]", "Energy [keV]"),
        ("E_vs_TX", tx_in, e_in, tx_out, e_out, "tx = COSX / |COSZ|", "Energy [keV]"),
        ("E_vs_TY", ty_in, e_in, ty_out, e_out, "ty = COSY / |COSZ|", "Energy [keV]"),
    ]

    max_points = 50000
    rng = np.random.default_rng(1)

    for name, xin, yin, xout, yout, xlabel, ylabel in scatter_data:
        if len(xin) > max_points:
            idx = rng.choice(len(xin), max_points, replace=False)
            xin = xin[idx]
            yin = yin[idx]

        if len(xout) > max_points:
            idx = rng.choice(len(xout), max_points, replace=False)
            xout = xout[idx]
            yout = yout[idx]

        plt.figure(figsize=(7, 5))
        plt.scatter(xin, yin, s=2, alpha=0.25, label="Input")
        plt.scatter(xout, yout, s=2, alpha=0.25, label="Flow output")
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.legend(markerscale=4)
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, f"{tag}_{name}.png"), dpi=150)
        plt.close()


# -------------------------
# RealNVP flow
# -------------------------
class CouplingLayer(nn.Module):
    def __init__(self, dim, hidden_dim, mask):
        super().__init__()
        self.register_buffer("mask", mask)

        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2 * dim),
        )

    def forward(self, x):
        x_masked = x * self.mask
        st = self.net(x_masked)
        s, t = st.chunk(2, dim=1)

        s = torch.tanh(s) * 2.0
        s = s * (1.0 - self.mask)
        t = t * (1.0 - self.mask)

        y = x_masked + (1.0 - self.mask) * (x * torch.exp(s) + t)
        log_det = s.sum(dim=1)

        return y, log_det

    def inverse(self, y):
        y_masked = y * self.mask
        st = self.net(y_masked)
        s, t = st.chunk(2, dim=1)

        s = torch.tanh(s) * 2.0
        s = s * (1.0 - self.mask)
        t = t * (1.0 - self.mask)

        x = y_masked + (1.0 - self.mask) * ((y - t) * torch.exp(-s))
        log_det = -s.sum(dim=1)

        return x, log_det


class RealNVP(nn.Module):
    def __init__(self, dim, hidden_dim, n_layers):
        super().__init__()
        self.dim = dim
        self.layers = nn.ModuleList()

        for i in range(n_layers):
            mask_values = [(j + i) % 2 for j in range(dim)]
            mask = torch.tensor(mask_values, dtype=torch.float32)
            self.layers.append(CouplingLayer(dim, hidden_dim, mask))

        self.base = torch.distributions.MultivariateNormal(
            torch.zeros(dim),
            torch.eye(dim)
        )

    def forward(self, x):
        z = x
        log_det_total = torch.zeros(x.shape[0], device=x.device)

        for layer in self.layers:
            z, log_det = layer(z)
            log_det_total += log_det

        return z, log_det_total

    def inverse(self, z):
        x = z

        for layer in reversed(self.layers):
            x, _ = layer.inverse(x)

        return x

    def log_prob(self, x):
        z, log_det = self.forward(x)
        return self.base.log_prob(z) + log_det

    def sample(self, n):
        z = self.base.sample((n,))
        return self.inverse(z)


def train_flow(X_scaled):
    dim = X_scaled.shape[1]

    x_tensor = torch.tensor(X_scaled, dtype=torch.float32)
    loader = DataLoader(
        TensorDataset(x_tensor),
        batch_size=min(batch_size_train, len(x_tensor)),
        shuffle=True,
        drop_last=False,
    )

    flow = RealNVP(dim=dim, hidden_dim=hidden_dim, n_layers=n_layers)
    optimizer = torch.optim.Adam(flow.parameters(), lr=learning_rate)

    flow.train()

    for epoch in range(epochs):
        losses = []

        for (xb,) in loader:
            loss = -flow.log_prob(xb).mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            losses.append(loss.item())

        if epoch % 100 == 0 or epoch == epochs - 1:
            print(f"  epoch {epoch:4d}  loss = {np.mean(losses):.5f}")

    flow.eval()
    return flow


# -------------------------
# MONO simulations
# -------------------------
for E_in in range(E_start, E_stop + 1, E_step):
    print("Energy:", E_in)

    path_input = (
        "./SPO_NF/"
        + str(E_in)
        + "keV/"
        + str(N_in)
        + "/"
        + str(theta_max)
        + "deg/NF_INPUT/"
    )

    for jdet in range(n_dets):

        input_filename_det = "G4_E" + str(E_in) + "keV" + det_type[jdet] + ".fits"
        print("reading input file...." + input_filename_det)

        hdulist_det = fits.open(path_input + input_filename_det)
        tbdata_det = hdulist_det[1].data

        vecXOut = np.asarray(tbdata_det.field("X_OUT"), dtype=float)
        vecYOut = np.asarray(tbdata_det.field("Y_OUT"), dtype=float)
        vecEOut = np.asarray(tbdata_det.field("E_OUT"), dtype=float)
        vecMDXOut = np.asarray(tbdata_det.field("COSX_OUT"), dtype=float)
        vecMDYOut = np.asarray(tbdata_det.field("COSY_OUT"), dtype=float)
        vecMDZOut = np.asarray(tbdata_det.field("COSZ_OUT"), dtype=float)
        
        vecROut = np.sqrt(vecXOut**2 + vecYOut**2)

        hdr_det = hdulist_det[1].header
        RDET = float(hdr_det["RDET"])
        NINP_G4 = int(hdr_det["NINP_G4"])
        NOUT_G4 = int(hdr_det["NOUT_G4"])

        hdulist_det.close()

        if np.any(vecEOut <= 0.0):
            raise ValueError("E_OUT contains non-positive energies")

        if np.any(vecMDZOut >= 0.0):
            raise ValueError("This script assumes COSZ_OUT < 0 for propagation toward -Z")

        N_out_L3 = NOUT_G4 * N_out_factor

        tx = vecMDXOut / np.abs(vecMDZOut)
        ty = vecMDYOut / np.abs(vecMDZOut)

        # Train the flow directly on x and y, not on radius.
        X = np.column_stack([
            np.log(vecEOut),
            vecXOut,
            vecYOut,
            tx,
            ty,
        ])

        X_mean = X.mean(axis=0)
        X_std = X.std(axis=0)
        X_std[X_std == 0.0] = 1.0
        X_scaled = (X - X_mean) / X_std

        print("Training normalizing flow...")
        flow = train_flow(X_scaled)

        output_blocks = []
        remaining = N_out_L3

        with torch.no_grad():
            while remaining > 0:

                batch_size = max(2 * remaining, 10000)

                S_scaled = flow.sample(batch_size).cpu().numpy()
                S = S_scaled * X_std + X_mean

                E_s = np.exp(S[:, 0])
                x_s = S[:, 1]
                y_s = S[:, 2]
                tx_s = S[:, 3]
                ty_s = S[:, 4]

                r_s = np.sqrt(x_s**2 + y_s**2)

                good = np.isfinite(E_s)
                good &= E_s > 0.0
                good &= E_s <= E_in
                good &= np.isfinite(x_s)
                good &= np.isfinite(y_s)
                good &= np.isfinite(tx_s)
                good &= np.isfinite(ty_s)
                good &= r_s <= RDET

                E_s = E_s[good]
                x_s = x_s[good]
                y_s = y_s[good]
                tx_s = tx_s[good]
                ty_s = ty_s[good]

                if len(E_s) == 0:
                    raise RuntimeError("Flow generated no valid samples. Increase epochs or check input data.")

                denom = np.sqrt(tx_s**2 + ty_s**2 + 1.0)

                ux_s = tx_s / denom
                uy_s = ty_s / denom
                uz_s = -1.0 / denom

                n_take = min(remaining, len(E_s))

                block = np.column_stack([
                    x_s[:n_take],
                    y_s[:n_take],
                    E_s[:n_take],
                    ux_s[:n_take],
                    uy_s[:n_take],
                    uz_s[:n_take],
                ])

                output_blocks.append(block)
                remaining -= n_take

        synthetic = np.vstack(output_blocks)

        x_out_new = synthetic[:, 0]
        y_out_new = synthetic[:, 1]
        e_out_new = synthetic[:, 2]
        ux_out_new = synthetic[:, 3]
        uy_out_new = synthetic[:, 4]
        uz_out_new = synthetic[:, 5]
        
        r_out_new = np.sqrt(x_out_new**2 + y_out_new**2)
        
        x_out_new_sel = x_out_new[r_out_new <= radius_det[jdet]]
        y_out_new_sel = y_out_new[r_out_new <= radius_det[jdet]]
        e_out_new_sel = e_out_new[r_out_new <= radius_det[jdet]]
        ux_out_new_sel = ux_out_new[r_out_new <= radius_det[jdet]]
        uy_out_new_sel = uy_out_new[r_out_new <= radius_det[jdet]]
        uz_out_new_sel = uz_out_new[r_out_new <= radius_det[jdet]]

        path_output = (
            "./SPO_L3/"
            + str(E_in)
            + "keV/"
            + str(N_out_factor)
            + "x/"
            + str(theta_max)
            + "deg/DET_INPUT/"
        )

        os.makedirs(path_output, exist_ok=True)

        print("Writing in " + path_output + input_filename_det)

        new_list_cols = fits.ColDefs([
            fits.Column(name="X_OUT", format="1D", unit="mm", array=x_out_new_sel),
            fits.Column(name="Y_OUT", format="1D", unit="mm", array=y_out_new_sel),
            fits.Column(name="E_OUT", format="1D", unit="keV", array=e_out_new_sel),
            fits.Column(name="COSX_OUT", format="1D", unit="", array=ux_out_new_sel),
            fits.Column(name="COSY_OUT", format="1D", unit="", array=uy_out_new_sel),
            fits.Column(name="COSZ_OUT", format="1D", unit="", array=uz_out_new_sel),
        ])

        new_list_header = hdr_det.copy()
        new_list_header["RDET"] = radius_det[jdet]
        new_list_header["NINP_G4"] = NINP_G4 * N_out_factor
        new_list_header["NOUT_G4"] = len(x_out_new_sel)
        new_list_header["NFLOW"] = n_layers
        new_list_header["RAND"] = random_state
        new_list_header["MODEL"] = "RealNVP_XY"
        
        # Remove all COMMENT cards
        while 'COMMENT' in new_list_header:
           del new_list_header['COMMENT']

        tbhdu = fits.BinTableHDU.from_columns(new_list_cols, header=new_list_header)
        tbhdu.name = "DET_EVENTS"
        tbhdu.writeto(path_output + input_filename_det, overwrite=True)

        plot_dir = os.path.join(path_output, "PLOTS")
        tag = "E" + str(E_in) + "keV" + det_type[jdet]

        make_comparison_plots(
            plot_dir,
            tag,
            vecXOut[vecROut <= radius_det[jdet]],
            vecYOut[vecROut <= radius_det[jdet]],
            vecEOut[vecROut <= radius_det[jdet]],
            vecMDXOut[vecROut <= radius_det[jdet]],
            vecMDYOut[vecROut <= radius_det[jdet]],
            vecMDZOut[vecROut <= radius_det[jdet]],
            x_out_new_sel,
            y_out_new_sel,
            e_out_new_sel,
            ux_out_new_sel,
            uy_out_new_sel,
            uz_out_new_sel,
        )

print(f"Normalizing-flow layers: {n_layers}")