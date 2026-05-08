
# -------------------------------------------------
# For plotting
# -------------------------------------------------
import os
import numpy as np
import matplotlib.pyplot as plt
import sys  
pf_values = [0.0, 0.03,0.06]
#pf_values = [0]
#res_folder = sys.argv[0][:-3]
res_folder = "/Users/xiao/PhD/Codes/dolfin_mech_HX2_clean/dolfin_mech/run_MicroPoroflow"

def load_qois(qois_filename):
    qois_vals = np.loadtxt(qois_filename)
    with open(qois_filename, "r") as f:
        qois_names = f.readline().split()[1:]
    return qois_vals, qois_names

def get(qois_vals, qois_names, key):
    return qois_vals[:, qois_names.index(key)]

import os
import numpy as np
import matplotlib.pyplot as plt

def plot_K_vs_pg_multi_Ex(
    res_folder,
    res_basename_prefix,
    Ex_list,
    slice_start=0,
    eps=1e-12,
    pg_key="p_f",       
    gx_key="grad_p_bar_avg_x",
    gy_key="grad_p_bar_avg_y",
    qx_key="Q_l_avg_x",
    qy_key="Q_l_avg_y",
    pg_in_kPa=True,
    savepath="plots/K_vs_pf_multi_Ex.png",
):
    os.makedirs("plots", exist_ok=True)

    fig, ax = plt.subplots(figsize=(7.6, 5.2))

    colors = [
        ("#1f77b4", "#aec7e8"),
        ("#d62728", "#ff9896"),
        ("#2ca02c", "#98df8a"),
        ("#9467bd", "#c5b0d5"),
        ("#8c564b", "#c49c94"),
    ]

    for idx, Ex in enumerate(Ex_list):
        filename = f"{res_folder}/{res_basename_prefix}-Ex={Ex}-qois.dat"
        if not os.path.exists(filename):
            print(f"[WARNING] File missing: {filename}")
            continue

        qois_vals, names = load_qois(filename)

        pf = get(qois_vals, names, pg_key)[slice_start:].astype(float)
        if not pg_in_kPa:
            pf = pf / 1000.0

        gx = get(qois_vals, names, gx_key)[slice_start:].astype(float)
        gy = get(qois_vals, names, gy_key)[slice_start:].astype(float)
        qx = get(qois_vals, names, qx_key)[slice_start:].astype(float)
        qy = get(qois_vals, names, qy_key)[slice_start:].astype(float)

        Kxx = -qx / (gx + eps)
        Kyy = -qy / (gy + eps)

        order = np.argsort(pf)
        pf, Kxx, Kyy = pf[order], Kxx[order], Kyy[order]

        c_dark, c_light = colors[idx % len(colors)]
        ax.plot(pf, Kxx, color=c_dark,  lw=2.6, label=rf"$\tilde{{K}}_{{xx}}$, $E_x={Ex}$")
        ax.plot(pf, Kyy, color=c_light, lw=2.6, label=rf"$\tilde{{K}}_{{yy}}$, $E_x={Ex}$")

        print(f"Read: {os.path.basename(filename)}  points={len(pf)}")

    ax.set_xlabel(r"$p_f\,(kPa)$", fontsize=16)  
    ax.set_ylabel(r"$\tilde{K}_{xx},\,\tilde{K}_{yy}\,(m^2/(Pa\cdot s))$", fontsize=16)
    #ax.grid(ls="--", alpha=0.4)
    ax.legend(fontsize=11, framealpha=0.9, loc="upper left")

    plt.tight_layout()
    plt.savefig(savepath, bbox_inches="tight",dpi=300)
    plt.close()
    print(f"Saved: {savepath}")


def plot_q_vs_gradp_multi_pf(res_folder, pf_list, res_basename_prefix, k_hom=None):
    import numpy as np
    import matplotlib.pyplot as plt
    import os

    os.makedirs("plots", exist_ok=True)

    colors = [
        ("#1f77b4", "#aec7e8"),
        ("#d62728", "#ff9896"),
        ("#2ca02c", "#98df8a"),
        ("#9467bd", "#c5b0d5"),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    axx, axy = axes

    gx_all, qx_all, gy_all, qy_all = [], [], [], []

    for idx, pf in enumerate(pf_list):
        filename = f"{res_folder}/{res_basename_prefix}-pf={pf}-qois.dat"
        if not os.path.exists(filename):
            print(f"[WARNING] File missing: {filename}")
            continue

        qois_vals, names = load_qois(filename)

        qx = get(qois_vals, names, "q_avg_x")[2:]
        qy = get(qois_vals, names, "q_avg_y")[2:]
        gx = get(qois_vals, names, "grad_p_bar_x")[2:]
        gy = get(qois_vals, names, "grad_p_bar_y")[2:]

        # store for reference line range
        gx_all.append(gx); qx_all.append(qx)
        gy_all.append(gy); qy_all.append(qy)

        c_dark, c_light = colors[idx % len(colors)]

        axx.plot(gx, qx, color=c_dark, linewidth=2.5, label=rf"$p_f={pf}$")
        axy.plot(gy, qy, color=c_light, linewidth=2.5, label=rf"$p_f={pf}$")

    # ---- add theoretical reference lines ----
    if k_hom is not None:
        k_hom = np.asarray(k_hom, dtype=float)
        kxx = k_hom[0, 0]
        kyy = k_hom[1, 1]

        # pick a reasonable x-range from all datasets
        if len(gx_all) > 0:
            gx_min = min([np.min(g) for g in gx_all])
            gx_max = max([np.max(g) for g in gx_all])
            gx_ref = np.linspace(gx_min, gx_max, 200)
            axx.plot(gx_ref, -kxx * gx_ref, "k--", linewidth=2.0,
                     label=rf"linear model: $q_x=-k_{{xx}}\nabla\bar p_x$ ($k_{{xx}}={kxx:.3g}$)")

        if len(gy_all) > 0:
            gy_min = min([np.min(g) for g in gy_all])
            gy_max = max([np.max(g) for g in gy_all])
            gy_ref = np.linspace(gy_min, gy_max, 200)
            axy.plot(gy_ref, -kyy * gy_ref, "k--", linewidth=2.0,
                     label=rf"linear model: $q_y=-k_{{yy}}\nabla\bar p_y$ ($k_{{yy}}={kyy:.3g}$)")

    axx.set_xlabel(r"$\nabla \bar{p}_x$", fontsize=16)
    axx.set_ylabel(r"$q_x$", fontsize=16)
    axx.grid(ls="--", alpha=0.4)
    axx.legend(fontsize=11, framealpha=0.9)

    axy.set_xlabel(r"$\nabla \bar{p}_y$", fontsize=16)
    axy.set_ylabel(r"$q_y$", fontsize=16)
    axy.grid(ls="--", alpha=0.4)
    axy.legend(fontsize=11, framealpha=0.9)

    plt.tight_layout()
    plt.savefig("plots/q_vs_gradp_multi_pf.png", bbox_inches="tight")
    plt.close()
    print("Saved: plots/q_vs_gradp_multi_pf.png")

def plot_Kxx_Kyy_vs_Uxx_multi_pf(
    res_folder,
    pf_list,
    res_basename_prefix,
    K0_ref=None,           # 2x2 initial reference permeability tensor
    slice_start=5,
    eps=1e-12,
    normalize=True,        # plot K/K0 if True
    add_prediction=True,   # add purely kinematic prediction
    save_name="plots/Kxx_Kyy_vs_Uxx_multi_pf.png",
):
    import os
    import numpy as np
    import matplotlib.pyplot as plt

    os.makedirs("plots", exist_ok=True)

    colors = [
        ("#1f77b4", "#aec7e8"),
        ("#d62728", "#ff9896"),
        ("#2ca02c", "#98df8a"),
        ("#9467bd", "#c5b0d5"),
        ("#ff7f0e", "#ffbb78"),
    ]

    fig, ax = plt.subplots(figsize=(7.6, 5.2))

    # sanitize optional global K0
    if K0_ref is not None:
        K0_global = np.asarray(K0_ref, dtype=float)
        if K0_global.shape != (2, 2):
            raise ValueError(f"K0_ref must be shape (2,2), got {K0_global.shape}")
    else:
        K0_global = None

    for idx, pf in enumerate(pf_list):
        filename = f"{res_folder}/{res_basename_prefix}-pf={pf}-qois.dat"
        if not os.path.exists(filename):
            print(f"[WARNING] File missing: {filename}")
            continue

        qois_vals, names = load_qois(filename)

        # --- macro displacement-gradient components ---
        Uxx = get(qois_vals, names, "U_bar_XX")[slice_start:]
        Uyy = get(qois_vals, names, "U_bar_YY")[slice_start:]
        Uxy = get(qois_vals, names, "U_bar_XY")[slice_start:]  # assume Uyx = Uxy

        # --- Darcy outputs ---
        Qx = get(qois_vals, names, "Q_l_avg_x")[slice_start:]
        Qy = get(qois_vals, names, "Q_l_avg_y")[slice_start:]
        gx = get(qois_vals, names, "grad_p_bar_avg_x")[slice_start:]
        gy = get(qois_vals, names, "grad_p_bar_avg_y")[slice_start:]

        Uxx = np.asarray(Uxx, dtype=float)
        Uyy = np.asarray(Uyy, dtype=float)
        Uxy = np.asarray(Uxy, dtype=float)
        Qx  = np.asarray(Qx,  dtype=float)
        Qy  = np.asarray(Qy,  dtype=float)
        gx  = np.asarray(gx,  dtype=float)
        gy  = np.asarray(gy,  dtype=float)

        npts = len(Uxx)
        if not (len(Uyy) == len(Uxy) == len(Qx) == len(Qy) == len(gx) == len(gy) == npts):
            raise ValueError(f"Inconsistent array lengths in file: {filename}")

        # --- measured reference permeability ---
        # Q = - K_ref * grad_X(p)
        # This is exact only if each probing case isolates one gradient direction.
        Kxx_ref = -Qx / (gx + eps)
        Kyy_ref = -Qy / (gy + eps)

        # choose K0
        if K0_global is not None:
            K0 = K0_global.copy()
        else:
            # fallback: diagonal tensor from first available measured point
            K0 = np.array([
                [float(Kxx_ref[0]), 0.0],
                [0.0, float(Kyy_ref[0])]
            ], dtype=float)

        # --- purely kinematic prediction ---
        Kxx_pred = np.full_like(Kxx_ref, np.nan)
        Kyy_pred = np.full_like(Kyy_ref, np.nan)

        F0 = np.array([
        [1.0 + Uxx[0], Uxy[0]],
        [Uxy[0], 1.0 + Uyy[0]]
    ], dtype=float)
        print("F0 =\n", F0)
        print("det(F0) =", np.linalg.det(F0))

        if add_prediction:
            for n in range(npts):
                F = np.array([
                    [1.0 + Uxx[n], Uxy[n]],
                    [Uxy[n], 1.0 + Uyy[n]]
                ], dtype=float)

                J = float(np.linalg.det(F))
                if abs(J) < 1e-14:
                    continue

                try:
                    Finv = np.linalg.inv(F)
                except np.linalg.LinAlgError:
                    continue

                K_pred = J * (Finv @ K0 @ Finv.T)

                Kxx_pred[n] = K_pred[0, 0]
                Kyy_pred[n] = K_pred[1, 1]

        # --- normalization ---
        if normalize:
            Kxx0 = K0[0, 0]
            Kyy0 = K0[1, 1]

            if abs(Kxx0) < eps or abs(Kyy0) < eps:
                raise ValueError(
                    f"K0 diagonal too small for normalization: "
                    f"K0_xx={Kxx0}, K0_yy={Kyy0}"
                )

            yKxx = Kxx_ref / Kxx0
            yKyy = Kyy_ref / Kyy0
            yKxx_pred = Kxx_pred / Kxx0
            yKyy_pred = Kyy_pred / Kyy0

            ylabel = r"$K_{xx}^{ref}/K_{xx,0}^{ref},\;K_{yy}^{ref}/K_{yy,0}^{ref}$"
        else:
            yKxx = Kxx_ref
            yKyy = Kyy_ref
            yKxx_pred = Kxx_pred
            yKyy_pred = Kyy_pred

            ylabel = r"$K_{xx}^{ref},\;K_{yy}^{ref}\;(\mathrm{m}^2\,\mathrm{Pa}^{-1}\,\mathrm{s}^{-1})$"

        c_dark, c_light = colors[idx % len(colors)]

        # --- measured curves ---
        ax.plot(
            Uxx, yKxx,
            color=c_dark, linewidth=2.4,
            label=rf"$K_{{xx}}^{{ref}}$, $p_f={pf}$"
        )
        ax.plot(
            Uxx, yKyy,
            color=c_light, linewidth=2.4,
            label=rf"$K_{{yy}}^{{ref}}$, $p_f={pf}$"
        )

        # --- predicted curves ---
        if add_prediction:
            ax.plot(
                Uxx, yKxx_pred,
                "--", color=c_dark, linewidth=1.8,
                label=rf"$J F^{{-1}} K_0 F^{{-T}}$ ($xx$), $p_f={pf}$"
            )
            ax.plot(
                Uxx, yKyy_pred,
                "--", color=c_light, linewidth=1.8,
                label=rf"$J F^{{-1}} K_0 F^{{-T}}$ ($yy$), $p_f={pf}$"
            )

        print(f"pf = {pf}")
        print(f"K0 used for prediction:\n{K0}")

    ax.set_xlabel(r"$U_{XX}$", fontsize=14)
    ax.set_ylabel(ylabel, fontsize=14)
    ax.tick_params(axis="both", labelsize=12)
    ax.legend(fontsize=8.8, framealpha=0.95, ncol=1)
    plt.tight_layout()
    plt.savefig(save_name, bbox_inches="tight", dpi=300)
    plt.close()

    print(f"Saved: {save_name}")



def plot_fig7_summary(
    cases,
    r0_list,
    r0_to_phi=None,
    slice_start=5,
    final_index=-1,
    eps=1e-12,
    save_name="plots/Figure7_summary.png",
    show_plot=False,
):
    import os
    import json
    import numpy as np
    import matplotlib.pyplot as plt

    os.makedirs(os.path.dirname(save_name) or ".", exist_ok=True)

    colors = {
        "stretch-x": "#0072B2",
        "stretch-y": "#D55E00",
        "pure shear": "#009E73",
        "gas pressure": "#CC79A7",
    }

    markers = {
        "stretch-x": "o",
        "stretch-y": "s",
        "pure shear": "^",
        "gas pressure": "D",
    }

    def build_basename(case, r0, pf, probe):
        return f"{case['res_folder']}/{case['prefix']}-r0={r0}-pf={pf}-{probe}"

    def read_phi(basename, r0):
        metadata_file = basename + "-metadata.json"

        if os.path.exists(metadata_file):
            with open(metadata_file, "r") as f:
                metadata = json.load(f)

            for key in ["mesh_porosity", "porosity", "phi"]:
                if key in metadata:
                    return float(metadata[key])

        if r0_to_phi is not None:
            return float(r0_to_phi[r0])

        return float(r0)

    def read_probe(filename):
        qois_vals, names = load_qois(filename)

        data = {
            "Uxx": np.asarray(get(qois_vals, names, "U_bar_XX")[slice_start:], dtype=float),
            "Uyy": np.asarray(get(qois_vals, names, "U_bar_YY")[slice_start:], dtype=float),
            "Uxy": np.asarray(get(qois_vals, names, "U_bar_XY")[slice_start:], dtype=float),
            "Uyx": np.asarray(get(qois_vals, names, "U_bar_YX")[slice_start:], dtype=float),
            "Qx": np.asarray(get(qois_vals, names, "Q_l_avg_x")[slice_start:], dtype=float),
            "Qy": np.asarray(get(qois_vals, names, "Q_l_avg_y")[slice_start:], dtype=float),
            "gx": np.asarray(get(qois_vals, names, "grad_p_bar_avg_x")[slice_start:], dtype=float),
            "gy": np.asarray(get(qois_vals, names, "grad_p_bar_avg_y")[slice_start:], dtype=float),
        }

        npts = len(data["Uxx"])

        for key, val in data.items():
            if len(val) != npts:
                raise ValueError(f"Inconsistent length for {key} in {filename}")

        return data

    def read_K_and_F(case, r0, pf):
        basename_gx = build_basename(case, r0, pf, "gx")
        basename_gy = build_basename(case, r0, pf, "gy")

        file_gx = basename_gx + "-qois.dat"
        file_gy = basename_gy + "-qois.dat"

        if not os.path.exists(file_gx):
            raise FileNotFoundError(file_gx)

        if not os.path.exists(file_gy):
            raise FileNotFoundError(file_gy)

        data_gx = read_probe(file_gx)
        data_gy = read_probe(file_gy)

        gx = data_gx["gx"]
        gy = data_gy["gy"]

        Kxx = -data_gx["Qx"] / (gx + eps)
        Kyx = -data_gx["Qy"] / (gx + eps)
        Kxy = -data_gy["Qx"] / (gy + eps)
        Kyy = -data_gy["Qy"] / (gy + eps)

        K_list = []
        F_list = []

        for n in range(len(Kxx)):
            K_list.append(
                np.array(
                    [
                        [Kxx[n], Kxy[n]],
                        [Kyx[n], Kyy[n]],
                    ],
                    dtype=float,
                )
            )

            F_list.append(
                np.array(
                    [
                        [1.0 + data_gx["Uxx"][n], data_gx["Uxy"][n]],
                        [data_gx["Uyx"][n], 1.0 + data_gx["Uyy"][n]],
                    ],
                    dtype=float,
                )
            )

        phi = read_phi(basename_gx, r0)

        return K_list, F_list, phi

    def compute_indicators(K0, Kf, Ff, use_prediction):
        C_K = np.trace(Kf) / (np.trace(K0) + eps)

        Ksym = 0.5 * (Kf + Kf.T)
        eigvals = np.linalg.eigvalsh(Ksym)
        eig_min = np.min(eigvals)
        eig_max = np.max(eigvals)

        if eig_min <= eps:
            A_K = np.nan
        else:
            A_K = eig_max / eig_min

        if use_prediction:
            J = float(np.linalg.det(Ff))
            Finv = np.linalg.inv(Ff)
            K_pred = J * (Finv @ K0 @ Finv.T)
            E_K = np.linalg.norm(Kf - K_pred, ord="fro") / (
                np.linalg.norm(Kf, ord="fro") + eps
            )
        else:
            E_K = np.nan

        return C_K, A_K, E_K

    rows = []

    for case in cases:
        label = case["label"]
        kind = case.get("kind", "deformation")

        for r0 in r0_list:
            try:
                if kind == "deformation":
                    pf = case.get("pf", 0.0)
                    K_list, F_list, phi = read_K_and_F(case, r0, pf)

                    K0 = K_list[0]
                    Kf = K_list[final_index]
                    Ff = F_list[final_index]

                    C_K, A_K, E_K = compute_indicators(
                        K0=K0,
                        Kf=Kf,
                        Ff=Ff,
                        use_prediction=True,
                    )

                elif kind == "gas_pressure":
                    pf_ref = case.get("pf_ref", 0.0)
                    pf_target = case["pf_target"]

                    K_ref_list, F_ref_list, phi = read_K_and_F(case, r0, pf_ref)
                    K_tar_list, F_tar_list, _ = read_K_and_F(case, r0, pf_target)

                    K0 = K_ref_list[final_index]
                    Kf = K_tar_list[final_index]
                    Ff = F_tar_list[final_index]

                    C_K, A_K, E_K = compute_indicators(
                        K0=K0,
                        Kf=Kf,
                        Ff=Ff,
                        use_prediction=False,
                    )

                else:
                    raise ValueError(f"Unknown case kind: {kind}")

                rows.append(
                    {
                        "label": label,
                        "r0": r0,
                        "phi": phi,
                        "C_K": C_K,
                        "A_K": A_K,
                        "E_K": E_K,
                    }
                )

                print(
                    f"{label}, r0={r0}, phi={phi:.6f}, "
                    f"C_K={C_K:.6f}, A_K={A_K:.6f}, E_K={E_K:.6f}"
                )

            except FileNotFoundError as e:
                print(f"[WARNING] Missing file: {e}")
                continue

    fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.6))

    plot_settings = [
        ("C_K", r"$C_K=\mathrm{tr}(\mathbf{K})/\mathrm{tr}(\mathbf{K}_0)$"),
        ("A_K", r"$A_K=k_{\max}/k_{\min}$"),
        ("E_K", r"$E_K=\|\mathbf{K}-\mathbf{K}^{pred}\|_F/\|\mathbf{K}\|_F$"),
    ]

    for ax, (key, ylabel) in zip(axes, plot_settings):
        for case in cases:
            label = case["label"]
            case_rows = [row for row in rows if row["label"] == label]

            if not case_rows:
                continue

            case_rows = sorted(case_rows, key=lambda row: row["phi"])

            x = np.array([row["phi"] for row in case_rows], dtype=float)
            y = np.array([row[key] for row in case_rows], dtype=float)

            valid = np.isfinite(y)

            if not np.any(valid):
                continue

            ax.plot(
                x[valid],
                y[valid],
                color=colors.get(label, "black"),
                marker=markers.get(label, "o"),
                markersize=5.5,
                linewidth=2.0,
                label=label,
            )

        ax.set_xlabel(r"Porosity $\phi$", fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.tick_params(axis="both", labelsize=10)
        ax.grid(True, linewidth=0.5, alpha=0.3)

    axes[0].axhline(1.0, color="0.5", linewidth=1.0, linestyle="--")
    axes[1].axhline(1.0, color="0.5", linewidth=1.0, linestyle="--")
    axes[2].axhline(0.0, color="0.5", linewidth=1.0, linestyle="--")

    handles, labels = axes[0].get_legend_handles_labels()

    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=max(1, len(labels)),
        frameon=False,
        fontsize=11,
    )

    plt.tight_layout(rect=[0.0, 0.0, 1.0, 0.88])
    plt.savefig(save_name, bbox_inches="tight", dpi=300)

    if show_plot:
        plt.show()

    plt.close()

    print(f"Saved: {save_name}")

    return rows

