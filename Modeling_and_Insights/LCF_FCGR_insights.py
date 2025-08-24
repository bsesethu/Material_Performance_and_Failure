import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# -------------------------------
# Load datasets
# -------------------------------
# Load the Excel file
xls = pd.ExcelFile('Fatigue database of High-entropy alloys.xlsx')

# Parse a specific sheet
df1 = xls.parse('LCF summary') 
df2 = xls.parse('LCF individual dataset')

# Merge datasets on ID
df_lcf = pd.merge(df1, df2, on="ID", how="left")

# Clean up column names: strip spaces and replace multiple spaces with single space NOTE Really important function
df_lcf.columns = df_lcf.columns.str.strip().str.replace(r'\s+', ' ', regex=True)

# --- Map your column names here if needed ---
colmap = {
    "Number of cycles" : "N",
    "Total strain amplitude" : "eps_total",
    "Plastic strain amplitude" : "eps_plastic",
    "Elastic strain amplitude" : "eps_elastic",
    # optional if present
    "Elastic_modulus" "E": "E",
    "Grain size (um)" : "grain"
}
# Create a normalized view with safe names
df_fit = df_lcf.rename(columns=colmap).copy()
print(df_fit.info())

# Keep only positive, finite rows needed for log fits
df_fit = df_fit.replace([np.inf, -np.inf], np.nan)
df_fit = df_fit.dropna(subset=["N", "eps_total", "eps_plastic", "eps_elastic"])
df_fit = df_fit[(df_fit["N"] > 0) & (df_fit["eps_plastic"] > 0) & (df_fit["eps_elastic"] > 0)]

# =========================
# 4) (OPTIONAL) CRACK GROWTH RATE (Paris Law) IF FCGR SHEET EXISTS
#    da/dN = C * (ΔK)^m  -> log(da/dN) = log C + m log(ΔK)
# =========================
try:
    xls = pd.ExcelFile('Fatigue database of High-entropy alloys.xlsx')  # adjust path if needed
    if "FCGR individual dataset" in xls.sheet_names:
        df_fcgr = xls.parse("FCGR individual dataset")
        # Try to auto-detect standard columns
        # Common headers might be like: 'Delta K', 'da/dN', etc. Clean headers first
        df_fcgr.columns = df_fcgr.columns.astype(str).str.strip().str.replace(r"\s+", " ", regex=True)
        # Heuristic guesses for column names
        dk_col = next((c for c in df_fcgr.columns if "Delta" in c or "ΔK" in c or "Delta K" in c), None)
        dadn_col = next((c for c in df_fcgr.columns if "da/dN" in c or "dadn" in c.lower() or "crack" in c.lower()), None)

        if dk_col and dadn_col:
            dfp = df_fcgr[[dk_col, dadn_col]].rename(columns={dk_col:"DeltaK", dadn_col:"dadn"}).copy()
            dfp = dfp.replace([np.inf, -np.inf], np.nan).dropna()
            dfp = dfp[(dfp["DeltaK"]>0) & (dfp["dadn"]>0)]

            m_slope, c_inter, r2_paris = (lambda x,y: (
                np.polyfit(np.log10(x), np.log10(y), 1)[0],
                np.polyfit(np.log10(x), np.log10(y), 1)[1],
                1 - np.sum((np.log10(y)-(np.poly1d(np.polyfit(np.log10(x), np.log10(y), 1))(np.log10(x))))**2) /
                    np.sum((np.log10(y)-np.log10(y).mean())**2)
            ))(dfp["DeltaK"].values, dfp["dadn"].values)

            C_paris = 10**c_inter
            print("\n=== Paris Law Fit (FCGR) ===")
            print(f"da/dN = {C_paris:.3e} * (ΔK)^{m_slope:.3f}   | R^2_log = {r2_paris:.3f}")

            # Plot
            K_grid = np.logspace(np.log10(dfp["DeltaK"].min()), np.log10(dfp["DeltaK"].max()), 200)
            dadn_pred = C_paris * (K_grid**m_slope)

            plt.figure(figsize=(9,6))
            plt.scatter(dfp["DeltaK"], dfp["dadn"], s=25, alpha=0.7, label="FCGR data", edgecolor='none')
            plt.plot(K_grid, dadn_pred, linewidth=2, label="Paris fit")
            plt.xscale('log'); plt.yscale('log')
            plt.xlabel("ΔK"); plt.ylabel("da/dN")
            plt.title("Paris Law Fit (FCGR)")
            plt.legend(); plt.tight_layout(); plt.show()
        else:
            print("\n[Info] FCGR sheet found but could not auto-detect ΔK / da/dN columns. Inspect headers and set them explicitly.")
    else:
        print("\n[Info] No 'FCGR individual dataset' sheet present. Skipping Paris law step.")
except Exception as e:
    print(f"\n[Info] Skipping FCGR step due to: {e}")

# =========================
# 5) MICROSTRUCTURAL INSIGHTS
#    (A) Grain size vs fatigue life
#    (B) Composition PCA (if elemental columns exist)
# =========================

# (A) Grain size analysis (if column available)
if "grain" in df_fit.columns:
    if "N" in df_fit.columns:
        df_fit["logN"] = np.log10(df_fit["N"].clip(lower=1))
        # Scatter: Grain size vs log N
        plt.figure(figsize=(8,6))
        plt.scatter(df_fit["grain"], df_fit["logN"], alpha=0.7)
        plt.xlabel("grain"); plt.ylabel("log10(Cycles to Failure)")
        plt.title("Grain Size vs Fatigue Life (LCF)")
        plt.tight_layout(); plt.show()

        # Spearman correlation (robust to nonlinearity)
        from scipy.stats import spearmanr
        rho, p = spearmanr(df_fit["grain"], df_fit["logN"], nan_policy='omit')
        print(f"\nSpearman(Grain Size, log10 N): rho={rho:.3f}, p={p:.3e}")

# (B) Composition PCA (auto-detect element columns like 'Fe', 'Ni', 'Co', etc.)
# Detect plausible element columns by regex (capital letter followed by optional lower-case, typical symbols)
import re
element_cols = [c for c in df_lcf.columns
                if re.fullmatch(r"[A-Z][a-z]?$", str(c).strip())]

if len(element_cols) >= 3: # Didn't trigger this 'if' statement
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA

    comp = df_lcf[element_cols].replace([np.inf, -np.inf], np.nan).dropna()
    comp_scaled = StandardScaler().fit_transform(comp.values)
    pca = PCA(n_components=2, random_state=42)
    pcs = pca.fit_transform(comp_scaled)

    # If cycles exist for coloring:
    color_vals = None
    if "Number of cycles" in df_lcf.columns:
        # Align lengths (dropna above cut rows)
        idx = comp.index
        color_vals = np.log10(df_lcf.loc[idx, "Number of cycles"].clip(lower=1).values)

    plt.figure(figsize=(8,6))
    if color_vals is not None:
        sc = plt.scatter(pcs[:,0], pcs[:,1], c=color_vals, alpha=0.8)
        cbar = plt.colorbar(sc); cbar.set_label("log10(Cycles)")
    else:
        plt.scatter(pcs[:,0], pcs[:,1], alpha=0.8)

    plt.xlabel("PC1"); plt.ylabel("PC2")
    plt.title("Composition PCA (colored by fatigue life)")
    plt.tight_layout(); plt.show()

    print(f"PCA explained variance: PC1={pca.explained_variance_ratio_[0]:.2%}, "
          f"PC2={pca.explained_variance_ratio_[1]:.2%}")
else:
    print("\n[Info] Composition columns not detected (like Fe, Ni, Co...). Skipping PCA.")
