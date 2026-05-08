
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
    phi=None,
    include_pf_in_filename=True,
    K0_ref=None,
    slice_start=5,
    eps=1e-12,
    normalize=True,
    add_prediction=True,
    save_name="plots/Kxx_Kyy_vs_Uxx_multi_pf.png",
    show_plot=False,
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
        filename = f"{res_folder}/{res_basename_prefix}"

        if include_pf_in_filename:
            filename += f"-pf={pf}"

        if phi is not None:
            phi_str = f"{phi:.3f}".replace(".", "p") if isinstance(phi, float) else str(phi)
            filename += f"-phi={phi_str}"

        filename += "-qois.dat"

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
    if show_plot:
        plt.show()
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


def plot_K_vs_U(
    res_folder,
    res_basename_prefix,
    r0_list,
    pf_list=(0.0,),
    K_components=("xx", "xy", "yx", "yy"),
    x_component="xx",
    phi=None,
    slice_start=5,
    eps=1e-12,
    normalize=True,
    add_prediction=True,
    save_name="plots/K_vs_U.png",
    show_plot=False,
    xdmf_folder=None,
    xdmf_basename_prefix=None,
    stream_pf=None,
    stream_probe="gx",
    stream_density=1.0,
    stream_scale=1.0,
    stream_grid_n=400,
    add_stream_colorbar=True,
):
    import os
    import json
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.tri as mtri
    import pyvista as pv
    from matplotlib.lines import Line2D

    os.makedirs(os.path.dirname(save_name) or ".", exist_ok=True)

    comp_to_idx = {
        "xx": (0, 0),
        "xy": (0, 1),
        "yx": (1, 0),
        "yy": (1, 1),
    }

    if x_component not in comp_to_idx:
        raise ValueError(f"x_component must be one of {list(comp_to_idx.keys())}")

    for comp in K_components:
        if comp not in comp_to_idx:
            raise ValueError(f"K component must be one of {list(comp_to_idx.keys())}, got {comp}")

    if xdmf_folder is None:
        xdmf_folder = res_folder

    if xdmf_basename_prefix is None:
        xdmf_basename_prefix = res_basename_prefix

    if stream_pf is None:
        stream_pf = pf_list[0]

    def build_basename(folder, prefix, r0, pf, probe):
        filename = f"{folder}/{prefix}-r0={r0}-pf={pf}-{probe}"
        if phi is not None:
            phi_str = f"{phi:.3f}".replace(".", "p") if isinstance(phi, float) else str(phi)
            filename += f"-phi={phi_str}"
        return filename

    def build_filename(r0, pf, probe):
        return build_basename(res_folder, res_basename_prefix, r0, pf, probe) + "-qois.dat"

    def build_xdmf_filename(r0, pf, probe):
        return build_basename(xdmf_folder, xdmf_basename_prefix, r0, pf, probe) + ".xdmf"

    def read_phi_from_metadata(r0, pf, probe="gx"):
        basename = build_basename(res_folder, res_basename_prefix, r0, pf, probe)
        metadata_file = basename + "-metadata.json"

        if os.path.exists(metadata_file):
            with open(metadata_file, "r") as f:
                metadata = json.load(f)

            for key in ["mesh_porosity", "porosity", "phi"]:
                if key in metadata:
                    return float(metadata[key])

        return None

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

    def plot_stream_subplot(ax, xdmf_file):
        if not os.path.exists(xdmf_file):
            ax.text(0.5, 0.5, "missing xdmf", ha="center", va="center", transform=ax.transAxes)
            ax.set_xticks([])
            ax.set_yticks([])
            return None

        reader = pv.get_reader(xdmf_file)

        if hasattr(reader, "number_time_points") and reader.number_time_points > 0:
            reader.set_active_time_point(reader.number_time_points - 1)

        mesh = reader.read()
        mesh = mesh.cell_data_to_point_data()

        warped = mesh.warp_by_vector("U_tot", factor=stream_scale)
        surf = warped.extract_surface().triangulate()

        pts = surf.points[:, :2]
        faces = surf.faces.reshape(-1, 4)[:, 1:4]

        p = np.asarray(surf.point_data["pl_tot"], dtype=float)
        q = np.asarray(surf.point_data["q_l"][:, :2], dtype=float)

        x = pts[:, 0]
        y = pts[:, 1]
        qx = q[:, 0]
        qy = q[:, 1]

        triang = mtri.Triangulation(x, y, triangles=faces)

        interp_p = mtri.LinearTriInterpolator(triang, p)
        interp_qx = mtri.LinearTriInterpolator(triang, qx)
        interp_qy = mtri.LinearTriInterpolator(triang, qy)

        xi = np.linspace(x.min(), x.max(), stream_grid_n)
        yi = np.linspace(y.min(), y.max(), stream_grid_n)
        X, Y = np.meshgrid(xi, yi)

        P = interp_p(X, Y)
        QX = interp_qx(X, Y)
        QY = interp_qy(X, Y)

        finder = triang.get_trifinder()
        inside = finder(X, Y) != -1

        P_mask = np.ma.getmaskarray(P) | (~inside)
        QX_mask = np.ma.getmaskarray(QX) | (~inside)
        QY_mask = np.ma.getmaskarray(QY) | (~inside)

        P = np.ma.array(P, mask=P_mask)
        QX = np.ma.array(QX, mask=QX_mask)
        QY = np.ma.array(QY, mask=QY_mask)

        boundary = surf.extract_feature_edges(
            boundary_edges=True,
            feature_edges=False,
            manifold_edges=False,
            non_manifold_edges=False,
        )

        cf = ax.contourf(X, Y, P, levels=60, cmap="inferno")

        stream_kwargs = dict(
            density=stream_density,
            color="#66F7FF",
            linewidth=1.2,
            arrowsize=1.2,
            arrowstyle="->",
            minlength=0.02,
            maxlength=10.0,
            integration_direction="both",
        )

        try:
            ax.streamplot(
                xi,
                yi,
                QX,
                QY,
                broken_streamlines=False,
                **stream_kwargs,
            )
        except TypeError:
            ax.streamplot(
                xi,
                yi,
                QX,
                QY,
                **stream_kwargs,
            )

        if boundary.n_cells > 0:
            lines = boundary.lines.reshape(-1, 3)
            bpts = boundary.points[:, :2]

            for line in lines:
                i0, i1 = line[1], line[2]
                ax.plot(
                    [bpts[i0, 0], bpts[i1, 0]],
                    [bpts[i0, 1], bpts[i1, 1]],
                    color="white",
                    linewidth=1.1,
                )

        pad_x = 0.005 * (x.max() - x.min())
        pad_y = 0.005 * (y.max() - y.min())

        ax.set_xlim(x.min() - pad_x, x.max() + pad_x)
        ax.set_ylim(y.min() - pad_y, y.max() + pad_y)
        ax.set_aspect("equal", adjustable="box")
        ax.set_anchor("C")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.margins(0)

        return cf

    comp_colors = {
        "xx": "#0072B2",
        "xy": "#E69F00",
        "yx": "#009E73",
        "yy": "#CC79A7",
    }

    comp_markers = {
        "xx": "o",
        "xy": "^",
        "yx": "D",
        "yy": "s",
    }

    n_cols = len(r0_list)
    fig_width = max(4.3 * n_cols, 7.0)

    fig, axes = plt.subplots(
        2,
        n_cols,
        figsize=(fig_width, 8.4),
        sharey=False,
        squeeze=False,
        gridspec_kw={"height_ratios": [1.0, 1.25], "hspace": 0.18},
    )

    curve_axes = axes[0]
    stream_axes = axes[1]

    cf_last = None

    for i_r0, r0 in enumerate(r0_list):
        ax = curve_axes[i_r0]
        sax = stream_axes[i_r0]

        phi_val = None

        for pf in pf_list:
            filename_gx = build_filename(r0, pf, "gx")
            filename_gy = build_filename(r0, pf, "gy")

            if not os.path.exists(filename_gx):
                print(f"[WARNING] File missing: {filename_gx}")
                continue

            if not os.path.exists(filename_gy):
                print(f"[WARNING] File missing: {filename_gy}")
                continue

            if phi_val is None:
                phi_val = read_phi_from_metadata(r0, pf, "gx")

            data_gx = read_probe(filename_gx)
            data_gy = read_probe(filename_gy)

            Uxx = data_gx["Uxx"]
            Uyy = data_gx["Uyy"]
            Uxy = data_gx["Uxy"]
            Uyx = data_gx["Uyx"]

            for key in ["Uxx", "Uyy", "Uxy", "Uyx"]:
                if not np.allclose(data_gx[key], data_gy[key], rtol=1e-6, atol=1e-10):
                    print(f"[WARNING] {key} differs between gx and gy probes for r0={r0}, pf={pf}")

            gx = data_gx["gx"]
            gy = data_gy["gy"]

            Kxx = -data_gx["Qx"] / (gx + eps)
            Kyx = -data_gx["Qy"] / (gx + eps)
            Kxy = -data_gy["Qx"] / (gy + eps)
            Kyy = -data_gy["Qy"] / (gy + eps)

            K = {
                "xx": Kxx,
                "xy": Kxy,
                "yx": Kyx,
                "yy": Kyy,
            }

            xvals = {
                "xx": Uxx,
                "yy": Uyy,
                "xy": Uxy,
                "yx": Uyx,
            }[x_component]

            npts = len(xvals)
            markevery = max(1, npts // 8)

            K0 = np.array(
                [
                    [Kxx[0], Kxy[0]],
                    [Kyx[0], Kyy[0]],
                ],
                dtype=float,
            )

            K_pred = {
                "xx": np.full(npts, np.nan),
                "xy": np.full(npts, np.nan),
                "yx": np.full(npts, np.nan),
                "yy": np.full(npts, np.nan),
            }

            if add_prediction:
                for n in range(npts):
                    F = np.array(
                        [
                            [1.0 + Uxx[n], Uxy[n]],
                            [Uyx[n], 1.0 + Uyy[n]],
                        ],
                        dtype=float,
                    )

                    J = float(np.linalg.det(F))

                    if abs(J) < 1e-14:
                        continue

                    try:
                        Finv = np.linalg.inv(F)
                    except np.linalg.LinAlgError:
                        continue

                    Kp = J * (Finv @ K0 @ Finv.T)

                    K_pred["xx"][n] = Kp[0, 0]
                    K_pred["xy"][n] = Kp[0, 1]
                    K_pred["yx"][n] = Kp[1, 0]
                    K_pred["yy"][n] = Kp[1, 1]

            for comp in K_components:
                i, j = comp_to_idx[comp]

                if normalize:
                    if i == j:
                        scaleK = K0[i, j]
                    else:
                        scaleK = np.sqrt(abs(K0[0, 0] * K0[1, 1]))

                    if abs(scaleK) < eps:
                        raise ValueError(f"Normalization scale too small for K{comp}")

                    y_meas = K[comp] / scaleK
                    y_pred = K_pred[comp] / scaleK
                else:
                    y_meas = K[comp]
                    y_pred = K_pred[comp]

                color = comp_colors[comp]
                marker = comp_markers[comp]

                ax.plot(
                    xvals,
                    y_meas,
                    color=color,
                    linestyle="--",
                    linewidth=2.0,
                    marker=marker,
                    markersize=4.8,
                    markerfacecolor="white",
                    markeredgecolor=color,
                    markeredgewidth=1.0,
                    markevery=markevery,
                )

                if add_prediction:
                    ax.plot(
                        xvals,
                        y_pred,
                        color=color,
                        linestyle="-",
                        linewidth=2.0,
                    )

            print(f"r0 = {r0}, pf = {pf}")
            print(f"K0 =\n{K0}")

        if phi_val is not None:
            ax.set_title(rf"porosity = {phi_val:.3f}", fontsize=13)
        else:
            ax.set_title(rf"$r_0={r0}$", fontsize=13)

        sax.set_title("")
        ax.set_xlabel(rf"$U_{{{x_component.upper()}}}$", fontsize=13)
        ax.tick_params(axis="x", labelsize=11)
        ax.tick_params(axis="y", labelsize=11)
        ax.grid(False)

        if i_r0 > 0:
            ax.tick_params(axis="y", left=False, labelleft=False)

        xdmf_file = build_xdmf_filename(r0, stream_pf, stream_probe)
        cf_last = plot_stream_subplot(sax, xdmf_file)

    ymins = []
    ymaxs = []

    for ax in curve_axes:
        ymin, ymax = ax.get_ylim()
        ymins.append(ymin)
        ymaxs.append(ymax)

    if ymins and ymaxs:
        ymin = min(ymins)
        ymax = max(ymaxs)
        dy = ymax - ymin

        if dy > 0:
            ymin -= 0.05 * dy
            ymax += 0.05 * dy

        for ax in curve_axes:
            ax.set_ylim(ymin, ymax)

    if normalize:
        fig.text(
            0.012,
            0.70,
            r"$K_{ij}^{ref}/K_{ij,0}^{ref}$",
            rotation="vertical",
            va="center",
            ha="center",
            fontsize=13,
        )
    else:
        fig.text(
            0.012,
            0.70,
            r"$K_{ij}^{ref}$",
            rotation="vertical",
            va="center",
            ha="center",
            fontsize=13,
        )

    line_legend = []

    for comp in K_components:
        color = comp_colors[comp]
        marker = comp_markers[comp]

        line_legend.append(
            Line2D(
                [0],
                [0],
                color=color,
                linestyle="--",
                linewidth=2.0,
                marker=marker,
                markerfacecolor="white",
                markeredgecolor=color,
                label=rf"$K_{{{comp}}}^{{ref}}$",
            )
        )

        if add_prediction:
            line_legend.append(
                Line2D(
                    [0],
                    [0],
                    color=color,
                    linestyle="-",
                    linewidth=2.0,
                    label=rf"$K_{{{comp}}}^{{pred}}$",
                )
            )

    fig.legend(
        handles=line_legend,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.995),
        ncol=4,
        fontsize=10,
        frameon=False,
    )

    plt.tight_layout(rect=(0.04, 0.11, 1.0, 0.90))

    if add_stream_colorbar and cf_last is not None:
        fig.canvas.draw()

        stream_positions = [ax.get_position() for ax in stream_axes]
        left = min(pos.x0 for pos in stream_positions)
        right = max(pos.x1 for pos in stream_positions)
        bottom = min(pos.y0 for pos in stream_positions)

        cbar_height = 0.018
        cbar_pad = 0.055

        cax = fig.add_axes(
            [
                left,
                bottom - cbar_pad,
                right - left,
                cbar_height,
            ]
        )

        cbar = fig.colorbar(
            cf_last,
            cax=cax,
            orientation="horizontal",
        )
        cbar.set_label(r"$p_\ell$")

    plt.savefig(save_name, bbox_inches="tight", dpi=300)

    if show_plot:
        plt.show()

    plt.close()

    print(f"Saved: {save_name}")


def plot_principal_K_vs_U(
    res_folder,
    res_basename_prefix,
    r0_list,
    pf_list=(0.0,),
    x_component="xx",
    phi=None,
    slice_start=5,
    eps=1e-12,
    add_prediction=True,
    save_name="plots/principal_K_vs_U.png",
    show_plot=False,
    xdmf_folder=None,
    xdmf_basename_prefix=None,
    stream_pf=None,
    stream_probe="gx",
    stream_density=1.0,
    stream_scale=1.0,
    stream_grid_n=400,
    add_stream_colorbar=True,
    theta_in_degrees=True,
    theta_lift_threshold_deg=-85.0,
):
    import os
    import json
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.tri as mtri
    import pyvista as pv
    from matplotlib.lines import Line2D

    os.makedirs(os.path.dirname(save_name) or ".", exist_ok=True)

    x_components = {
        "xx": "Uxx",
        "yy": "Uyy",
        "xy": "Uxy",
        "yx": "Uyx",
    }

    if x_component not in x_components:
        raise ValueError(f"x_component must be one of {list(x_components.keys())}")

    if xdmf_folder is None:
        xdmf_folder = res_folder

    if xdmf_basename_prefix is None:
        xdmf_basename_prefix = res_basename_prefix

    if stream_pf is None:
        stream_pf = pf_list[0]

    def build_basename(folder, prefix, r0, pf, probe):
        filename = f"{folder}/{prefix}-r0={r0}-pf={pf}-{probe}"
        if phi is not None:
            phi_str = f"{phi:.3f}".replace(".", "p") if isinstance(phi, float) else str(phi)
            filename += f"-phi={phi_str}"
        return filename

    def build_filename(r0, pf, probe):
        return build_basename(res_folder, res_basename_prefix, r0, pf, probe) + "-qois.dat"

    def build_xdmf_filename(r0, pf, probe):
        return build_basename(xdmf_folder, xdmf_basename_prefix, r0, pf, probe) + ".xdmf"

    def read_phi_from_metadata(r0, pf, probe="gx"):
        basename = build_basename(res_folder, res_basename_prefix, r0, pf, probe)
        metadata_file = basename + "-metadata.json"

        if os.path.exists(metadata_file):
            with open(metadata_file, "r") as f:
                metadata = json.load(f)

            for key in ["mesh_porosity", "porosity", "phi"]:
                if key in metadata:
                    return float(metadata[key])

        return None

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

    def continuous_axis_angle_deg(theta_raw):
        theta_raw = np.asarray(theta_raw, dtype=float)
        theta_cont = np.empty_like(theta_raw)

        if len(theta_raw) == 0:
            return theta_cont

        theta_cont[0] = theta_raw[0]

        for i in range(1, len(theta_raw)):
            delta = (theta_raw[i] - theta_cont[i - 1] + 90.0) % 180.0 - 90.0
            theta_cont[i] = theta_cont[i - 1] + delta

        return theta_cont

    def continuous_axis_angle_rad(theta_raw):
        theta_raw = np.asarray(theta_raw, dtype=float)
        theta_cont = np.empty_like(theta_raw)

        if len(theta_raw) == 0:
            return theta_cont

        theta_cont[0] = theta_raw[0]

        for i in range(1, len(theta_raw)):
            delta = (theta_raw[i] - theta_cont[i - 1] + 0.5 * np.pi) % np.pi - 0.5 * np.pi
            theta_cont[i] = theta_cont[i - 1] + delta

        return theta_cont

    def lift_negative_vertical_deg(theta):
        theta = np.asarray(theta, dtype=float).copy()
        theta[theta <= theta_lift_threshold_deg] += 180.0
        return theta

    def lift_negative_vertical_rad(theta):
        theta = np.asarray(theta, dtype=float).copy()
        threshold = np.deg2rad(theta_lift_threshold_deg)
        theta[theta <= threshold] += np.pi
        return theta

    def principal_quantities(K_list):
        k1 = []
        k2 = []
        theta = []

        for K in K_list:
            Ksym = 0.5 * (K + K.T)

            a = Ksym[0, 0]
            b = Ksym[0, 1]
            c = Ksym[1, 1]

            tr = a + c
            delta = np.sqrt((a - c) ** 2 + 4.0 * b ** 2)

            lam1 = 0.5 * (tr + delta)
            lam2 = 0.5 * (tr - delta)

            angle = 0.5 * np.arctan2(2.0 * b, a - c)

            k1.append(lam1)
            k2.append(lam2)
            theta.append(angle)

        k1 = np.asarray(k1, dtype=float)
        k2 = np.asarray(k2, dtype=float)
        theta = np.asarray(theta, dtype=float)

        if theta_in_degrees:
            theta = np.rad2deg(theta)
            theta = continuous_axis_angle_deg(theta)
            theta = lift_negative_vertical_deg(theta)
        else:
            theta = continuous_axis_angle_rad(theta)
            theta = lift_negative_vertical_rad(theta)

        return k1, k2, theta

    def plot_stream_subplot(ax, xdmf_file):
        if not os.path.exists(xdmf_file):
            ax.text(0.5, 0.5, "missing xdmf", ha="center", va="center", transform=ax.transAxes)
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
            return None

        reader0 = pv.get_reader(xdmf_file)
        if hasattr(reader0, "number_time_points") and reader0.number_time_points > 0:
            reader0.set_active_time_point(0)
        mesh0 = reader0.read().cell_data_to_point_data()
        surf0 = mesh0.extract_surface().triangulate()

        reader = pv.get_reader(xdmf_file)
        if hasattr(reader, "number_time_points") and reader.number_time_points > 0:
            reader.set_active_time_point(reader.number_time_points - 1)
        mesh = reader.read().cell_data_to_point_data()

        warped = mesh.warp_by_vector("U_tot", factor=stream_scale)
        surf = warped.extract_surface().triangulate()

        pts = surf.points[:, :2]
        faces = surf.faces.reshape(-1, 4)[:, 1:4]

        pts0 = surf0.points[:, :2]
        faces0 = surf0.faces.reshape(-1, 4)[:, 1:4]

        c0 = pts0.mean(axis=0)
        c = pts.mean(axis=0)
        pts0_shift = pts0 + (c - c0)

        p = np.asarray(surf.point_data["pl_tot"], dtype=float)
        q = np.asarray(surf.point_data["q_l"][:, :2], dtype=float)

        x = pts[:, 0]
        y = pts[:, 1]
        qx = q[:, 0]
        qy = q[:, 1]

        triang = mtri.Triangulation(x, y, triangles=faces)
        triang0 = mtri.Triangulation(pts0_shift[:, 0], pts0_shift[:, 1], triangles=faces0)

        interp_p = mtri.LinearTriInterpolator(triang, p)
        interp_qx = mtri.LinearTriInterpolator(triang, qx)
        interp_qy = mtri.LinearTriInterpolator(triang, qy)

        xi = np.linspace(x.min(), x.max(), stream_grid_n)
        yi = np.linspace(y.min(), y.max(), stream_grid_n)
        X, Y = np.meshgrid(xi, yi)

        P = interp_p(X, Y)
        QX = interp_qx(X, Y)
        QY = interp_qy(X, Y)

        finder = triang.get_trifinder()
        inside = finder(X, Y) != -1

        P_mask = np.ma.getmaskarray(P) | (~inside)
        QX_mask = np.ma.getmaskarray(QX) | (~inside)
        QY_mask = np.ma.getmaskarray(QY) | (~inside)

        P = np.ma.array(P, mask=P_mask)
        QX = np.ma.array(QX, mask=QX_mask)
        QY = np.ma.array(QY, mask=QY_mask)

        ax.tripcolor(
            triang0,
            np.ones(len(pts0_shift)),
            shading="gouraud",
            cmap="Greys",
            vmin=0.0,
            vmax=2.0,
            alpha=0.7,
            edgecolors="none",
            zorder=0,
        )

        ax.triplot(
            triang0,
            color="0.75",
            linewidth=0.35,
            alpha=0.45,
            zorder=1,
        )

        cf = ax.contourf(
            X, Y, P,
            levels=60,
            cmap="inferno",
            alpha=0.65,
            zorder=2,
        )

        stream_kwargs = dict(
            density=stream_density,
            color="#66F7FF",
            linewidth=1.2,
            arrowsize=1.2,
            arrowstyle="->",
            minlength=0.02,
            maxlength=10.0,
            integration_direction="both",
        )

        try:
            ax.streamplot(
                xi,
                yi,
                QX,
                QY,
                broken_streamlines=False,
                **stream_kwargs,
            )
        except TypeError:
            ax.streamplot(
                xi,
                yi,
                QX,
                QY,
                **stream_kwargs,
            )

        pad_x = 0.01 * (x.max() - x.min())
        pad_y = 0.01 * (y.max() - y.min())

        xmin = min(x.min(), pts0_shift[:, 0].min()) - pad_x
        xmax = max(x.max(), pts0_shift[:, 0].max()) + pad_x
        ymin = min(y.min(), pts0_shift[:, 1].min()) - pad_y
        ymax = max(y.max(), pts0_shift[:, 1].max()) + pad_y

        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        ax.set_aspect("equal", adjustable="box")
        ax.set_anchor("C")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.margins(0)

        for spine in ax.spines.values():
            spine.set_visible(False)

        return cf

    colors = {
        "k1": "#0072B2",
        "k2": "#D55E00",
        "theta": "#009E73",
    }

    markers = {
        "k1": "o",
        "k2": "s",
        "theta": "^",
    }

    n_cols = len(r0_list)
    fig_width = max(4.3 * n_cols, 7.0)

    fig, axes = plt.subplots(
        3,
        n_cols,
        figsize=(fig_width, 10.5),
        sharex=False,
        sharey=False,
        squeeze=False,
        gridspec_kw={
            "height_ratios": [1.0, 0.85, 1.25],
            "hspace": 0.22,
        },
    )

    eig_axes = axes[0]
    theta_axes = axes[1]
    stream_axes = axes[2]

    cf_last = None
    all_theta_values = []

    for i_r0, r0 in enumerate(r0_list):
        eig_ax = eig_axes[i_r0]
        theta_ax = theta_axes[i_r0]
        stream_ax = stream_axes[i_r0]

        phi_val = None

        for pf in pf_list:
            filename_gx = build_filename(r0, pf, "gx")
            filename_gy = build_filename(r0, pf, "gy")

            if not os.path.exists(filename_gx):
                print(f"[WARNING] File missing: {filename_gx}")
                continue

            if not os.path.exists(filename_gy):
                print(f"[WARNING] File missing: {filename_gy}")
                continue

            if phi_val is None:
                phi_val = read_phi_from_metadata(r0, pf, "gx")

            data_gx = read_probe(filename_gx)
            data_gy = read_probe(filename_gy)

            Uxx = data_gx["Uxx"]
            Uyy = data_gx["Uyy"]
            Uxy = data_gx["Uxy"]
            Uyx = data_gx["Uyx"]

            for key in ["Uxx", "Uyy", "Uxy", "Uyx"]:
                if not np.allclose(data_gx[key], data_gy[key], rtol=1e-6, atol=1e-10):
                    print(f"[WARNING] {key} differs between gx and gy probes for r0={r0}, pf={pf}")

            gx = data_gx["gx"]
            gy = data_gy["gy"]

            Kxx = -data_gx["Qx"] / (gx + eps)
            Kyx = -data_gx["Qy"] / (gx + eps)
            Kxy = -data_gy["Qx"] / (gy + eps)
            Kyy = -data_gy["Qy"] / (gy + eps)

            xvals = data_gx[x_components[x_component]]
            npts = len(xvals)
            markevery = max(1, npts // 8)

            K_list = []
            F_list = []

            for n in range(npts):
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
                            [1.0 + Uxx[n], Uxy[n]],
                            [Uyx[n], 1.0 + Uyy[n]],
                        ],
                        dtype=float,
                    )
                )

            k1, k2, theta = principal_quantities(K_list)

            k1_0 = k1[0]
            k2_0 = k2[0]

            if abs(k1_0) < eps:
                raise ValueError(f"k1_0 is too small for r0={r0}, pf={pf}")

            if abs(k2_0) < eps:
                raise ValueError(f"k2_0 is too small for r0={r0}, pf={pf}")

            eig_ax.plot(
                xvals,
                k1 / k1_0,
                color=colors["k1"],
                linestyle="--",
                linewidth=2.0,
                marker=markers["k1"],
                markersize=4.8,
                markerfacecolor="white",
                markeredgecolor=colors["k1"],
                markeredgewidth=1.0,
                markevery=markevery,
            )

            eig_ax.plot(
                xvals,
                k2 / k2_0,
                color=colors["k2"],
                linestyle="--",
                linewidth=2.0,
                marker=markers["k2"],
                markersize=4.8,
                markerfacecolor="white",
                markeredgecolor=colors["k2"],
                markeredgewidth=1.0,
                markevery=markevery,
            )

            theta_ax.plot(
                xvals,
                theta,
                color=colors["theta"],
                linestyle="--",
                linewidth=2.0,
                marker=markers["theta"],
                markersize=4.8,
                markerfacecolor="white",
                markeredgecolor=colors["theta"],
                markeredgewidth=1.0,
                markevery=markevery,
            )

            all_theta_values.extend(theta[np.isfinite(theta)].tolist())

            if add_prediction:
                K0 = K_list[0]
                K_pred_list = []

                for n in range(npts):
                    F = F_list[n]
                    J = float(np.linalg.det(F))

                    if abs(J) < 1e-14:
                        K_pred_list.append(np.full((2, 2), np.nan))
                        continue

                    try:
                        Finv = np.linalg.inv(F)
                    except np.linalg.LinAlgError:
                        K_pred_list.append(np.full((2, 2), np.nan))
                        continue

                    K_pred_list.append(J * (Finv @ K0 @ Finv.T))

                k1_pred, k2_pred, theta_pred = principal_quantities(K_pred_list)

                eig_ax.plot(
                    xvals,
                    k1_pred / k1_0,
                    color=colors["k1"],
                    linestyle="-",
                    linewidth=2.0,
                )

                eig_ax.plot(
                    xvals,
                    k2_pred / k2_0,
                    color=colors["k2"],
                    linestyle="-",
                    linewidth=2.0,
                )

                theta_ax.plot(
                    xvals,
                    theta_pred,
                    color=colors["theta"],
                    linestyle="-",
                    linewidth=2.0,
                )

                all_theta_values.extend(theta_pred[np.isfinite(theta_pred)].tolist())

            asym = []
            for K in K_list:
                denom = np.linalg.norm(K, ord="fro") + eps
                asym.append(np.linalg.norm(K - K.T, ord="fro") / denom)

            print(f"r0 = {r0}, pf = {pf}")
            print(f"k1_0 = {k1_0}")
            print(f"k2_0 = {k2_0}")
            print(f"max asymmetry = {np.nanmax(asym)}")

        if phi_val is not None:
            eig_ax.set_title(rf"$\tilde{{\Phi}}_{{g0}} = {phi_val:.2f}$", fontsize=13)
        else:
            eig_ax.set_title(rf"$r_0={r0}$", fontsize=13)

        eig_ax.grid(False)
        theta_ax.grid(False)

        eig_ax.tick_params(axis="x", labelbottom=False)
        eig_ax.tick_params(axis="y", labelsize=11)

        theta_ax.set_xlabel(rf"$U_{{{x_component.upper()}}}$", fontsize=13)
        theta_ax.tick_params(axis="x", labelsize=11)
        theta_ax.tick_params(axis="y", labelsize=11)

        if i_r0 > 0:
            eig_ax.tick_params(axis="y", left=False, labelleft=False)
            theta_ax.tick_params(axis="y", left=False, labelleft=False)

        xdmf_file = build_xdmf_filename(r0, stream_pf, stream_probe)
        cf_last = plot_stream_subplot(stream_ax, xdmf_file)

    eig_ymins = []
    eig_ymaxs = []

    for ax in eig_axes:
        ymin, ymax = ax.get_ylim()
        eig_ymins.append(ymin)
        eig_ymaxs.append(ymax)

    if eig_ymins and eig_ymaxs:
        ymin = min(eig_ymins)
        ymax = max(eig_ymaxs)
        dy = ymax - ymin

        if dy > 0:
            ymin -= 0.05 * dy
            ymax += 0.05 * dy

        for ax in eig_axes:
            ax.set_ylim(ymin, ymax)

    if theta_in_degrees:
        if all_theta_values:
            theta_min = min(v for v in all_theta_values if np.isfinite(v))
            theta_max = max(v for v in all_theta_values if np.isfinite(v))
            theta_low = min(-100.0, theta_min - 5.0)
            theta_high = max(100.0, theta_max + 5.0)
        else:
            theta_low = -100.0
            theta_high = 100.0

        for ax in theta_axes:
            ax.set_ylim(theta_low, theta_high)
            ticks = [-90, 0, 90]
            ax.set_yticks(ticks)
    else:
        if all_theta_values:
            theta_min = min(v for v in all_theta_values if np.isfinite(v))
            theta_max = max(v for v in all_theta_values if np.isfinite(v))
            theta_low = min(-0.5 * np.pi - np.deg2rad(5.0), theta_min - np.deg2rad(5.0))
            theta_high = max(0.5 * np.pi + np.deg2rad(5.0), theta_max + np.deg2rad(5.0))
        else:
            theta_low = -0.5 * np.pi - np.deg2rad(5.0)
            theta_high = 0.5 * np.pi + np.deg2rad(5.0)

        for ax in theta_axes:
            ax.set_ylim(theta_low, theta_high)
            ax.set_yticks([-0.5 * np.pi, 0.0, 0.5 * np.pi])
            ax.set_yticklabels([r"$-\pi/2$", r"$0$", r"$\pi/2$"])

    eig_axes[0].set_ylabel(r"$k_i/k_{i,0}$", fontsize=13, labelpad=4)
    theta_axes[0].set_ylabel(r"$\theta$ (deg)" if theta_in_degrees else r"$\theta$ (rad)", fontsize=13, labelpad=4)

    eig_axes[0].yaxis.set_label_coords(-0.13, 0.5)
    theta_axes[0].yaxis.set_label_coords(-0.13, 0.5)

    legend_handles = [
        Line2D(
            [0],
            [0],
            color=colors["k1"],
            linestyle="--",
            linewidth=2.0,
            marker=markers["k1"],
            markerfacecolor="white",
            markeredgecolor=colors["k1"],
            label=r"$k_1/k_{1,0}$",
        ),
    ]

    if add_prediction:
        legend_handles.append(
            Line2D(
                [0],
                [0],
                color=colors["k1"],
                linestyle="-",
                linewidth=2.0,
                label=r"$k_1^{pred}/k_{1,0}$",
            )
        )

    legend_handles.append(
        Line2D(
            [0],
            [0],
            color=colors["k2"],
            linestyle="--",
            linewidth=2.0,
            marker=markers["k2"],
            markerfacecolor="white",
            markeredgecolor=colors["k2"],
            label=r"$k_2/k_{2,0}$",
        )
    )

    if add_prediction:
        legend_handles.append(
            Line2D(
                [0],
                [0],
                color=colors["k2"],
                linestyle="-",
                linewidth=2.0,
                label=r"$k_2^{pred}/k_{2,0}$",
            )
        )

    legend_handles.append(
        Line2D(
            [0],
            [0],
            color=colors["theta"],
            linestyle="--",
            linewidth=2.0,
            marker=markers["theta"],
            markerfacecolor="white",
            markeredgecolor=colors["theta"],
            label=r"$\theta$",
        )
    )

    if add_prediction:
        legend_handles.append(
            Line2D(
                [0],
                [0],
                color=colors["theta"],
                linestyle="-",
                linewidth=2.0,
                label=r"$\theta^{pred}$",
            )
        )

    fig.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.965),
        ncol=len(legend_handles),
        fontsize=10,
        frameon=False,
        handlelength=2.2,
        columnspacing=1.2,
    )

    plt.tight_layout(rect=(0.055, 0.10, 1.0, 0.93))

    if add_stream_colorbar and cf_last is not None:
        from matplotlib.ticker import FuncFormatter

        fig.canvas.draw()

        stream_positions = [ax.get_position() for ax in stream_axes]
        left = min(pos.x0 for pos in stream_positions)
        bottom = min(pos.y0 for pos in stream_positions)
        top = max(pos.y1 for pos in stream_positions)

        cbar_width = 0.012
        cbar_pad = 0.050

        cax = fig.add_axes(
            [
                left - cbar_pad - cbar_width,
                bottom,
                cbar_width,
                top - bottom,
            ]
        )

        cbar = fig.colorbar(
            cf_last,
            cax=cax,
            orientation="vertical",
        )

        def sci_fmt(x, pos):
            if abs(x) < 1e-14:
                return "0"
            s = f"{x:.1e}"
            s = s.replace("e-0", "e-")
            s = s.replace("e+0", "e")
            s = s.replace("e+", "e")
            return s

        cbar.ax.yaxis.set_major_formatter(FuncFormatter(sci_fmt))
        cbar.ax.tick_params(labelsize=9)
        cbar.set_label(r"$p_\ell$ (kPa)", fontsize=11)

    plt.savefig(save_name, bbox_inches="tight", dpi=300)

    if show_plot:
        plt.show()

    plt.close()

    print(f"Saved: {save_name}")


def plot_gas_pressure_loading_summary(
    res_folder,
    res_basename_prefix,
    r0,
    mode_list=("stretch-x", "volumic", "shear"),
    pf_list=(0.0, 0.2),
    probe_list=("gx", "gy"),
    phi=None,
    slice_start=5,
    eps=1e-12,
    save_name="plots/Figure6_gas_loading.png",
    show_plot=False,
    stream_probe="gx",
    stream_density=0.8,
    stream_scale=1.0,
    stream_grid_n=500,
    add_stream_colorbar=True,
    theta_lift_threshold_deg=-85.0,
):
    import os
    import json
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.tri as mtri
    import pyvista as pv
    from matplotlib.lines import Line2D

    os.makedirs(os.path.dirname(save_name) or ".", exist_ok=True)

    if len(pf_list) != 2:
        raise ValueError("pf_list should contain exactly two values, e.g. (0.0, 0.2).")

    if set(probe_list) != {"gx", "gy"}:
        raise ValueError("probe_list must contain gx and gy.")

    mode_to_xkey = {
        "stretch-x": "Uxx",
        "volumic": "Uxx",
        "shear": "Uxy",
    }

    mode_to_xlabel = {
        "stretch-x": r"$U_{XX}$",
        "volumic": r"$U_{XX}=U_{YY}$",
        "shear": r"$U_{XY}$",
    }

    mode_to_label = {
        "stretch-x": r"Stretch $x$",
        "volumic": r"Volumetric stretch",
        "shear": r"Simple shear",
    }

    colors = {
        "pg0": "#204a9a",
        "pg1": "#b22222",
    }

    markers = {
        "k1": "o",
        "k2": "s",
        "theta": "^",
    }

    linestyles = {
        "k1": "-",
        "k2": "--",
        "theta": "-.",
    }

    def mode_filename_candidates(mode):
        return [mode]

    def build_basename(mode, pf, probe, suffix=None):
        first_filename = None

        for mode_file in mode_filename_candidates(mode):
            filename = f"{res_folder}/{res_basename_prefix}-{mode_file}-r0={r0}-pf={pf}-{probe}"

            if phi is not None:
                phi_str = f"{phi:.3f}".replace(".", "p") if isinstance(phi, float) else str(phi)
                filename += f"-phi={phi_str}"

            if first_filename is None:
                first_filename = filename

            if suffix is not None and os.path.exists(filename + suffix):
                return filename

        return first_filename

    def build_qois_filename(mode, pf, probe):
        return build_basename(mode, pf, probe, suffix="-qois.dat") + "-qois.dat"

    def build_xdmf_filename(mode, pf, probe):
        return build_basename(mode, pf, probe, suffix=".xdmf") + ".xdmf"

    def read_phi_from_metadata(mode, pf, probe="gx"):
        metadata_file = build_basename(mode, pf, probe, suffix="-metadata.json") + "-metadata.json"

        if os.path.exists(metadata_file):
            with open(metadata_file, "r") as f:
                metadata = json.load(f)

            for key in ["mesh_porosity", "porosity", "phi"]:
                if key in metadata:
                    return float(metadata[key])

        return None

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

    def read_K_and_U(mode, pf):
        file_gx = build_qois_filename(mode, pf, "gx")
        file_gy = build_qois_filename(mode, pf, "gy")

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

        U = {
            "Uxx": data_gx["Uxx"],
            "Uyy": data_gx["Uyy"],
            "Uxy": data_gx["Uxy"],
            "Uyx": data_gx["Uyx"],
        }

        return K_list, U

    def continuous_axis_angle_deg(theta_raw):
        theta_raw = np.asarray(theta_raw, dtype=float)
        theta_cont = np.empty_like(theta_raw)

        if len(theta_raw) == 0:
            return theta_cont

        theta_cont[0] = theta_raw[0]

        for i in range(1, len(theta_raw)):
            delta = (theta_raw[i] - theta_cont[i - 1] + 90.0) % 180.0 - 90.0
            theta_cont[i] = theta_cont[i - 1] + delta

        return theta_cont

    def lift_negative_vertical_deg(theta):
        theta = np.asarray(theta, dtype=float).copy()
        theta[theta <= theta_lift_threshold_deg] += 180.0
        return theta

    def principal_quantities(K_list):
        k1 = []
        k2 = []
        theta = []

        for K in K_list:
            Ksym = 0.5 * (K + K.T)

            a = Ksym[0, 0]
            b = Ksym[0, 1]
            c = Ksym[1, 1]

            tr = a + c
            delta = np.sqrt((a - c) ** 2 + 4.0 * b ** 2)

            lam1 = 0.5 * (tr + delta)
            lam2 = 0.5 * (tr - delta)

            angle = 0.5 * np.arctan2(2.0 * b, a - c)

            k1.append(lam1)
            k2.append(lam2)
            theta.append(angle)

        k1 = np.asarray(k1, dtype=float)
        k2 = np.asarray(k2, dtype=float)
        theta = np.rad2deg(np.asarray(theta, dtype=float))
        theta = continuous_axis_angle_deg(theta)
        theta = lift_negative_vertical_deg(theta)

        return k1, k2, theta

    def make_xvals(mode, U):
        return U[mode_to_xkey[mode]]

    def weighted_kde(values, weights, xgrid):
        values = np.asarray(values, dtype=float)
        weights = np.asarray(weights, dtype=float)

        valid = np.isfinite(values) & np.isfinite(weights) & (weights > 0.0)
        values = values[valid]
        weights = weights[valid]

        if len(values) == 0:
            return np.zeros_like(xgrid)

        weights = weights / (np.sum(weights) + eps)

        mu = np.sum(weights * values)
        var = np.sum(weights * (values - mu) ** 2)
        sigma = np.sqrt(max(var, eps))

        n_eff = 1.0 / (np.sum(weights ** 2) + eps)
        bw = 1.06 * sigma * max(n_eff, 2.0) ** (-1.0 / 5.0)

        span = max(np.max(values) - np.min(values), eps)
        bw = max(bw, 0.04 * span)

        z = (xgrid[:, None] - values[None, :]) / bw
        density = np.sum(
            weights[None, :] * np.exp(-0.5 * z ** 2),
            axis=1,
        ) / (np.sqrt(2.0 * np.pi) * bw)

        return density

    def read_q_distribution(mode, pf, probe):
        xdmf_file = build_xdmf_filename(mode, pf, probe)

        if not os.path.exists(xdmf_file):
            raise FileNotFoundError(xdmf_file)

        reader = pv.get_reader(xdmf_file)

        if hasattr(reader, "number_time_points") and reader.number_time_points > 0:
            reader.set_active_time_point(reader.number_time_points - 1)

        mesh = reader.read()
        mesh = mesh.cell_data_to_point_data()

        warped = mesh.warp_by_vector("U_tot", factor=stream_scale)
        surf = warped.extract_surface().triangulate()
        surf = surf.point_data_to_cell_data()

        if "q_l" not in surf.cell_data:
            raise ValueError(f"'q_l' not found in cell data of {xdmf_file}")

        q = np.asarray(surf.cell_data["q_l"], dtype=float)[:, :2]
        qmag = np.linalg.norm(q, axis=1)

        valid = np.isfinite(qmag)
        qmag = qmag[valid]

        if len(qmag) == 0:
            return {
                "values": np.array([]),
                "mean": np.nan,
                "std": np.nan,
                "n_cells": 0,
            }

        return {
            "values": qmag,
            "mean": float(np.mean(qmag)),
            "std": float(np.std(qmag)),
            "n_cells": int(len(qmag)),
        }

    def plot_distribution_subplot(ax, data0, data1, xlim, ymax):
        vals0 = data0["values"]
        vals1 = data1["values"]

        bins = np.linspace(xlim[0], xlim[1], 26)
        xgrid = np.linspace(xlim[0], xlim[1], 400)
        bin_width = bins[1] - bins[0]

        if len(vals0) > 0:
            weights0 = np.ones_like(vals0, dtype=float) * 100.0 / len(vals0)

            ax.hist(
                vals0,
                bins=bins,
                weights=weights0,
                density=False,
                color=colors["pg0"],
                alpha=0.32,
                edgecolor="white",
                linewidth=0.8,
            )

        if len(vals1) > 0:
            weights1 = np.ones_like(vals1, dtype=float) * 100.0 / len(vals1)

            ax.hist(
                vals1,
                bins=bins,
                weights=weights1,
                density=False,
                color=colors["pg1"],
                alpha=0.25,
                edgecolor="white",
                linewidth=0.8,
            )

        if len(vals0) > 1:
            w0 = np.ones_like(vals0, dtype=float) / len(vals0)
            kde0 = weighted_kde(vals0, w0, xgrid)

            ax.plot(
                xgrid,
                kde0 * bin_width * 100.0,
                color=colors["pg0"],
                linewidth=2.0,
            )

        if len(vals1) > 1:
            w1 = np.ones_like(vals1, dtype=float) / len(vals1)
            kde1 = weighted_kde(vals1, w1, xgrid)

            ax.plot(
                xgrid,
                kde1 * bin_width * 100.0,
                color=colors["pg1"],
                linewidth=2.0,
            )

        if np.isfinite(data0["mean"]):
            ax.axvline(
                data0["mean"],
                color=colors["pg0"],
                linewidth=1.8,
            )

        if np.isfinite(data1["mean"]):
            ax.axvline(
                data1["mean"],
                color=colors["pg1"],
                linewidth=1.8,
            )

        ax.set_xlim(*xlim)
        ax.set_ylim(0.0, ymax)
        ax.grid(False)
        ax.tick_params(axis="both", labelsize=10)
        ax.set_xlabel(r"$|\mathbf{q}_{\ell}|$", fontsize=11)

    def plot_stream_subplot(ax, xdmf_file):
        if not os.path.exists(xdmf_file):
            ax.text(0.5, 0.5, "missing xdmf", ha="center", va="center", transform=ax.transAxes)
            ax.set_xticks([])
            ax.set_yticks([])

            for spine in ax.spines.values():
                spine.set_visible(False)

            return None

        reader = pv.get_reader(xdmf_file)

        if hasattr(reader, "number_time_points") and reader.number_time_points > 0:
            reader.set_active_time_point(0)
            mesh0 = reader.read()
            reader.set_active_time_point(reader.number_time_points - 1)
            meshf = reader.read()
        else:
            mesh0 = reader.read()
            meshf = mesh0.copy()

        mesh0 = mesh0.cell_data_to_point_data()
        meshf = meshf.cell_data_to_point_data()

        surf0 = mesh0.extract_surface().triangulate()
        warped = meshf.warp_by_vector("U_tot", factor=stream_scale)
        surf = warped.extract_surface().triangulate()

        pts0 = surf0.points[:, :2]
        faces0 = surf0.faces.reshape(-1, 4)[:, 1:4]

        pts = surf.points[:, :2]
        faces = surf.faces.reshape(-1, 4)[:, 1:4]

        center0 = np.array(
            [
                0.5 * (pts0[:, 0].min() + pts0[:, 0].max()),
                0.5 * (pts0[:, 1].min() + pts0[:, 1].max()),
            ]
        )
        center = np.array(
            [
                0.5 * (pts[:, 0].min() + pts[:, 0].max()),
                0.5 * (pts[:, 1].min() + pts[:, 1].max()),
            ]
        )
        pts0_shift = pts0 + (center - center0)

        triang0 = mtri.Triangulation(
            pts0_shift[:, 0],
            pts0_shift[:, 1],
            triangles=faces0,
        )

        p = np.asarray(surf.point_data["pl_tot"], dtype=float)
        q = np.asarray(surf.point_data["q_l"][:, :2], dtype=float)

        x = pts[:, 0]
        y = pts[:, 1]
        qx = q[:, 0]
        qy = q[:, 1]

        triang = mtri.Triangulation(x, y, triangles=faces)

        interp_p = mtri.LinearTriInterpolator(triang, p)
        interp_qx = mtri.LinearTriInterpolator(triang, qx)
        interp_qy = mtri.LinearTriInterpolator(triang, qy)

        xi = np.linspace(x.min(), x.max(), stream_grid_n)
        yi = np.linspace(y.min(), y.max(), stream_grid_n)
        X, Y = np.meshgrid(xi, yi)

        P = interp_p(X, Y)
        QX = interp_qx(X, Y)
        QY = interp_qy(X, Y)

        finder = triang.get_trifinder()
        inside = finder(X, Y) != -1

        P_mask = np.ma.getmaskarray(P) | (~inside)
        QX_mask = np.ma.getmaskarray(QX) | (~inside)
        QY_mask = np.ma.getmaskarray(QY) | (~inside)

        P = np.ma.array(P, mask=P_mask)
        QX = np.ma.array(QX, mask=QX_mask)
        QY = np.ma.array(QY, mask=QY_mask)

        ax.tripcolor(
            triang0,
            np.ones(len(pts0_shift)),
            shading="gouraud",
            cmap="Greys",
            vmin=0.0,
            vmax=2.0,
            alpha=0.7,
            edgecolors="none",
            zorder=0,
        )

        ax.triplot(
            triang0,
            color="0.75",
            linewidth=0.35,
            alpha=0.45,
            zorder=1,
        )

        cf = ax.contourf(
            X,
            Y,
            P,
            levels=60,
            cmap="inferno",
            alpha=0.65,
            zorder=2,
        )

        stream_kwargs = dict(
            density=stream_density,
            color="#66F7FF",
            linewidth=1.1,
            arrowsize=1.1,
            arrowstyle="->",
            minlength=0.02,
            maxlength=10.0,
            integration_direction="both",
        )

        try:
            sp = ax.streamplot(
                xi,
                yi,
                QX,
                QY,
                broken_streamlines=False,
                **stream_kwargs,
            )
        except TypeError:
            sp = ax.streamplot(
                xi,
                yi,
                QX,
                QY,
                **stream_kwargs,
            )

        sp.lines.set_zorder(4)
        sp.arrows.set_zorder(5)

        xmin = min(pts0_shift[:, 0].min(), x.min())
        xmax = max(pts0_shift[:, 0].max(), x.max())
        ymin = min(pts0_shift[:, 1].min(), y.min())
        ymax = max(pts0_shift[:, 1].max(), y.max())

        pad_x = 0.02 * (xmax - xmin)
        pad_y = 0.02 * (ymax - ymin)

        ax.set_xlim(xmin - pad_x, xmax + pad_x)
        ax.set_ylim(ymin - pad_y, ymax + pad_y)
        ax.set_aspect("equal", adjustable="box")
        ax.set_anchor("C")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.margins(0)

        for spine in ax.spines.values():
            spine.set_visible(False)

        return cf

    dist_cache = {}
    all_q = []

    for mode in mode_list:
        for pf in pf_list:
            for probe in probe_list:
                try:
                    data = read_q_distribution(mode, pf, probe)
                    dist_cache[(mode, pf, probe)] = data

                    if len(data["values"]) > 0:
                        all_q.extend(data["values"].tolist())

                except FileNotFoundError:
                    dist_cache[(mode, pf, probe)] = {
                        "values": np.array([]),
                        "mean": np.nan,
                        "std": np.nan,
                        "n_cells": 0,
                    }

    if len(all_q) > 0:
        all_q = np.asarray(all_q, dtype=float)
        qmin = 0.0
        qmax = np.nanpercentile(all_q, 99.2)

        if not np.isfinite(qmax) or qmax <= qmin:
            qmax = np.nanmax(all_q)

        qpad = 0.06 * max(qmax - qmin, eps)
        dist_xlim = (qmin, qmax + qpad)

    else:
        dist_xlim = (0.0, 1.0)

    dist_ymax = 1.0
    bins_tmp = np.linspace(dist_xlim[0], dist_xlim[1], 26)
    xgrid_tmp = np.linspace(dist_xlim[0], dist_xlim[1], 400)
    bin_width_tmp = bins_tmp[1] - bins_tmp[0]

    for key, data in dist_cache.items():
        vals = data["values"]

        if len(vals) == 0:
            continue

        weights = np.ones_like(vals, dtype=float) * 100.0 / len(vals)
        hist, _ = np.histogram(vals, bins=bins_tmp, weights=weights, density=False)
        local_max = np.nanmax(hist)

        if len(vals) > 1:
            ww = np.ones_like(vals, dtype=float) / len(vals)
            kde = weighted_kde(vals, ww, xgrid_tmp)
            kde_percent = kde * bin_width_tmp * 100.0
            local_max = max(local_max, np.nanmax(kde_percent))

        if np.isfinite(local_max):
            dist_ymax = max(dist_ymax, 1.12 * local_max)

    n_rows = len(mode_list)

    fig, axes = plt.subplots(
        n_rows,
        6,
        figsize=(22.0, 3.25 * n_rows),
        squeeze=False,
        gridspec_kw={
            "width_ratios": [1.15, 0.95, 1.00, 1.00, 1.00, 1.00],
            "wspace": 0.30,
            "hspace": 0.36,
        },
    )

    cf_last = None
    all_k_values = []
    all_theta_values = []
    phi_val = None

    for i_mode, mode in enumerate(mode_list):
        if mode not in mode_to_xkey:
            raise ValueError(f"Unknown mode: {mode}")

        ax_k = axes[i_mode, 0]
        ax_theta = axes[i_mode, 1]
        ax_dist_gx = axes[i_mode, 2]
        ax_dist_gy = axes[i_mode, 3]
        ax_pf0 = axes[i_mode, 4]
        ax_pf1 = axes[i_mode, 5]

        ref_K_list, ref_U = read_K_and_U(mode, pf_list[0])
        ref_k1, ref_k2, ref_theta = principal_quantities(ref_K_list)

        k1_ref = ref_k1[0]
        k2_ref = ref_k2[0]

        if abs(k1_ref) < eps:
            raise ValueError(f"k1_ref is too small for mode={mode}, r0={r0}.")

        if abs(k2_ref) < eps:
            raise ValueError(f"k2_ref is too small for mode={mode}, r0={r0}.")

        for i_pf, pf in enumerate(pf_list):
            if phi_val is None:
                phi_val = read_phi_from_metadata(mode, pf, "gx")

            K_list, U = read_K_and_U(mode, pf)
            xvals = make_xvals(mode, U)
            k1, k2, theta = principal_quantities(K_list)

            markevery = max(1, len(xvals) // 7)
            curve_color = colors["pg0"] if i_pf == 0 else colors["pg1"]

            ax_k.plot(
                xvals,
                k1 / k1_ref,
                color=curve_color,
                linestyle=linestyles["k1"],
                linewidth=2.0,
                marker=markers["k1"],
                markersize=4.5,
                markerfacecolor="white",
                markeredgecolor=curve_color,
                markeredgewidth=1.0,
                markevery=markevery,
            )

            ax_k.plot(
                xvals,
                k2 / k2_ref,
                color=curve_color,
                linestyle=linestyles["k2"],
                linewidth=2.0,
                marker=markers["k2"],
                markersize=4.5,
                markerfacecolor="white",
                markeredgecolor=curve_color,
                markeredgewidth=1.0,
                markevery=markevery,
            )

            ax_theta.plot(
                xvals,
                theta,
                color=curve_color,
                linestyle=linestyles["theta"],
                linewidth=2.0,
                marker=markers["theta"],
                markersize=4.5,
                markerfacecolor="white",
                markeredgecolor=curve_color,
                markeredgewidth=1.0,
                markevery=markevery,
            )

            all_k_values.extend((k1 / k1_ref)[np.isfinite(k1 / k1_ref)].tolist())
            all_k_values.extend((k2 / k2_ref)[np.isfinite(k2 / k2_ref)].tolist())
            all_theta_values.extend(theta[np.isfinite(theta)].tolist())

        data_gx_0 = dist_cache[(mode, pf_list[0], "gx")]
        data_gx_1 = dist_cache[(mode, pf_list[1], "gx")]
        data_gy_0 = dist_cache[(mode, pf_list[0], "gy")]
        data_gy_1 = dist_cache[(mode, pf_list[1], "gy")]

        plot_distribution_subplot(
            ax_dist_gx,
            data_gx_0,
            data_gx_1,
            dist_xlim,
            dist_ymax,
        )

        plot_distribution_subplot(
            ax_dist_gy,
            data_gy_0,
            data_gy_1,
            dist_xlim,
            dist_ymax,
        )

        xdmf_pg0 = build_xdmf_filename(mode, pf_list[0], stream_probe)
        xdmf_pg1 = build_xdmf_filename(mode, pf_list[1], stream_probe)

        cf_last = plot_stream_subplot(ax_pf0, xdmf_pg0)
        cf_last = plot_stream_subplot(ax_pf1, xdmf_pg1)

        #ax_k.set_ylabel(mode_to_label[mode], fontsize=13, labelpad=10)
        ax_k.tick_params(axis="both", labelsize=10)
        ax_theta.tick_params(axis="both", labelsize=10)

        ax_k.grid(False)
        ax_theta.grid(False)

        ax_k.set_xlabel(mode_to_xlabel[mode], fontsize=12)
        ax_theta.set_xlabel(mode_to_xlabel[mode], fontsize=12)

        if i_mode == 0:
            ax_k.set_title(r"Principal permeabilities", fontsize=13)
            ax_theta.set_title(r"Principal direction", fontsize=13)
            ax_dist_gx.set_title(r"$|\mathbf{q}_{\ell}|$ distribution ($g_x$ probe)", fontsize=12)
            ax_dist_gy.set_title(r"$|\mathbf{q}_{\ell}|$ distribution ($g_y$ probe)", fontsize=12)
            ax_pf0.set_title(rf"$p_g={pf_list[0]}$", fontsize=13)
            ax_pf1.set_title(rf"$p_g={pf_list[1]}$", fontsize=13)

        if i_mode == 0:
            ax_dist_gx.legend(
                handles=[
                    Line2D(
                        [0],
                        [0],
                        color=colors["pg0"],
                        linewidth=2.0,
                        label=rf"$p_g={pf_list[0]}$",
                    ),
                    Line2D(
                        [0],
                        [0],
                        color=colors["pg1"],
                        linewidth=2.0,
                        label=rf"$p_g={pf_list[1]}$",
                    ),
                ],
                loc="upper left",
                fontsize=8.8,
                frameon=True,
            )

        if i_mode == 0:
            ax_dist_gx.set_ylabel("Frequency (%)", fontsize=11)
        else:
            ax_dist_gx.set_ylabel("")

        ax_dist_gy.set_ylabel("")

    if all_k_values:
        k_min = min(all_k_values)
        k_max = max(all_k_values)
        dk = k_max - k_min

        if dk > 0:
            k_min -= 0.06 * dk
            k_max += 0.06 * dk

        for ax in axes[:, 0]:
            ax.set_ylim(k_min, k_max)

    if all_theta_values:
        theta_min = min(all_theta_values)
        theta_max = max(all_theta_values)
        theta_low = min(-95.0, theta_min - 5.0)
        theta_high = max(95.0, theta_max + 5.0)
    else:
        theta_low = -95.0
        theta_high = 95.0

    for ax in axes[:, 1]:
        ax.set_ylim(theta_low, theta_high)
        ax.set_yticks([-90, 0, 90])

    axes[0, 0].set_ylabel(r"$k_i/k_{i,\mathrm{ref}}$", fontsize=13)

    axes[0, 1].set_ylabel(r"$\theta$ (deg)", fontsize=13)

    for i in range(1, n_rows):
        axes[i, 1].set_ylabel("")

    legend_handles = [
        Line2D(
            [0],
            [0],
            color=colors["pg0"],
            linestyle=linestyles["k1"],
            linewidth=2.0,
            marker=markers["k1"],
            markerfacecolor="white",
            markeredgecolor=colors["pg0"],
            label=rf"$k_1/k_{{1,\mathrm{{ref}}}},\ p_g={pf_list[0]}$",
        ),
        Line2D(
            [0],
            [0],
            color=colors["pg1"],
            linestyle=linestyles["k1"],
            linewidth=2.0,
            marker=markers["k1"],
            markerfacecolor="white",
            markeredgecolor=colors["pg1"],
            label=rf"$k_1/k_{{1,\mathrm{{ref}}}},\ p_g={pf_list[1]}$",
        ),
        Line2D(
            [0],
            [0],
            color=colors["pg0"],
            linestyle=linestyles["k2"],
            linewidth=2.0,
            marker=markers["k2"],
            markerfacecolor="white",
            markeredgecolor=colors["pg0"],
            label=rf"$k_2/k_{{2,\mathrm{{ref}}}},\ p_g={pf_list[0]}$",
        ),
        Line2D(
            [0],
            [0],
            color=colors["pg1"],
            linestyle=linestyles["k2"],
            linewidth=2.0,
            marker=markers["k2"],
            markerfacecolor="white",
            markeredgecolor=colors["pg1"],
            label=rf"$k_2/k_{{2,\mathrm{{ref}}}},\ p_g={pf_list[1]}$",
        ),
        Line2D(
            [0],
            [0],
            color=colors["pg0"],
            linestyle=linestyles["theta"],
            linewidth=2.0,
            marker=markers["theta"],
            markerfacecolor="white",
            markeredgecolor=colors["pg0"],
            label=rf"$\theta,\ p_g={pf_list[0]}$",
        ),
        Line2D(
            [0],
            [0],
            color=colors["pg1"],
            linestyle=linestyles["theta"],
            linewidth=2.0,
            marker=markers["theta"],
            markerfacecolor="white",
            markeredgecolor=colors["pg1"],
            label=rf"$\theta,\ p_g={pf_list[1]}$",
        ),
    ]


    fig.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.968),
        ncol=3,
        fontsize=9.3,
        frameon=False,
        handlelength=2.2,
        columnspacing=1.1,
    )

    plt.tight_layout(rect=(0.075, 0.08, 1.0, 0.91))

    if add_stream_colorbar and cf_last is not None:
        from matplotlib.ticker import FuncFormatter

        fig.canvas.draw()

        mesh_ax_pos = axes[0, 4].get_position()

        cbar_width = 0.012
        cbar_pad = 0.025

        cax = fig.add_axes(
            [
                mesh_ax_pos.x0 - cbar_pad - cbar_width,
                mesh_ax_pos.y0,
                cbar_width,
                mesh_ax_pos.height,
            ]
        )

        cbar = fig.colorbar(
            cf_last,
            cax=cax,
            orientation="vertical",
        )

        def sci_fmt(x, pos):
            if abs(x) < 1e-14:
                return "0"
            s = f"{x:.1e}"
            s = s.replace("e-0", "e-")
            s = s.replace("e+0", "e")
            s = s.replace("e+", "e")
            return s

        cbar.ax.yaxis.set_major_formatter(FuncFormatter(sci_fmt))
        cbar.ax.tick_params(labelsize=8)
        cbar.set_label(r"$p_\ell$ (kPa)", fontsize=10)

    plt.savefig(save_name, bbox_inches="tight", dpi=300)

    if show_plot:
        plt.show()

    plt.close()

    print(f"Saved: {save_name}")

