import pandas as pd
import numpy as np
from pathlib import Path
import shutil, json, textwrap
import matplotlib.pyplot as plt

SRC = Path('/mnt/data/results3_extract/results')
OUT = Path('/mnt/data/heinzel_table1_extension_final')
if OUT.exists():
    shutil.rmtree(OUT)
(OUT/'data').mkdir(parents=True)
(OUT/'output').mkdir()
(OUT/'plots').mkdir()

raw = pd.read_csv(SRC/'table1_full_cone_prd.csv')
shutil.copy2(SRC/'table1_full_cone_prd.csv', OUT/'data'/'table1_full_cone_prd_raw.csv')
shutil.copy2(SRC/'table1_full_cone_prd.metadata.json', OUT/'data'/'table1_full_cone_prd.metadata.json')
shutil.copy2(SRC/'table1_full_cone_prd_report.txt', OUT/'data'/'table1_full_cone_prd_report.txt')

temps=[6000,8000,10000,12000,14000]
pressures=[0.01,0.02,0.05,0.10,0.20]
i_ref={
10:[[0.74,0.62,0.44,0.31,0.20],[0.83,0.72,0.55,0.44,0.35],[0.87,0.79,0.70,0.69,0.73],[0.91,0.85,0.82,0.85,0.89],[0.93,0.89,0.89,0.92,0.94]],
20:[[0.73,0.60,0.41,0.29,0.18],[0.81,0.70,0.52,0.41,0.33],[0.86,0.78,0.68,0.68,0.72],[0.90,0.84,0.81,0.84,0.88],[0.92,0.88,0.89,0.91,0.94]],
30:[[0.71,0.58,0.39,0.27,0.17],[0.80,0.68,0.50,0.39,0.32],[0.85,0.76,0.66,0.67,0.71],[0.89,0.83,0.81,0.84,0.88],[0.92,0.88,0.88,0.91,0.93]],
}
f_ref={
10:[[5.0,4.6,4.2,4.0,4.0],[6.7,5.8,5.0,4.8,4.7],[8.1,6.8,5.3,5.1,5.3],[9.1,7.0,5.0,4.9,5.2],[9.8,7.1,4.9,4.8,5.0]],
20:[[4.7,4.2,3.8,3.7,3.6],[6.3,5.4,4.6,4.4,4.3],[7.6,6.3,4.8,4.7,4.9],[8.6,6.5,4.6,4.5,4.8],[9.2,6.4,4.5,4.4,4.6]],
30:[[4.5,4.0,3.5,3.4,3.4],[6.1,5.1,4.3,4.0,4.0],[7.4,6.0,4.5,4.3,4.5],[8.3,6.1,4.2,4.2,4.5],[8.9,6.0,4.2,4.1,4.3]],
}
refs=[]
for H in (10,20,30):
    for ti,T in enumerate(temps):
        for pi,P in enumerate(pressures):
            refs.append(dict(height_Mm=H,temperature_K=T,pressure_dyn_cm2=P,
                             i_heinzel2015=i_ref[H][ti][pi],
                             f_heinzel2015_1e16_cm3=f_ref[H][ti][pi]))
ref=pd.DataFrame(refs)
ref.to_csv(OUT/'data'/'heinzel2015_table1_reference.csv', index=False)

# Primary Promweaver quantities chosen to match the hydrogen-only definition implicit in Heinzel+2015.
pw=raw[['height_Mm','temperature_K','pressure_dyn_cm2','proton_fraction_np_over_nH','f_proton_1e16_cm3',
        'i_ne_over_nH','f_1e16_cm3','iterations','duration_s']].copy()
pw=pw.rename(columns={'proton_fraction_np_over_nH':'i_pw_H','f_proton_1e16_cm3':'f_pw_H_1e16_cm3',
                      'i_ne_over_nH':'i_pw_total_e','f_1e16_cm3':'f_pw_total_e_1e16_cm3'})
anchor=pw.merge(ref,on=['height_Mm','temperature_K','pressure_dyn_cm2'])
anchor['C_i']=anchor['i_heinzel2015']/anchor['i_pw_H']
anchor['C_f']=anchor['f_heinzel2015_1e16_cm3']/anchor['f_pw_H_1e16_cm3']
anchor.to_csv(OUT/'output'/'calibration_factors_at_10_20_30Mm.csv',index=False)

# Geometric interpolation is used because the calibration is multiplicative and all quantities are positive.
def interp_log_correction(H, arr_h, arr_c):
    if H <= arr_h[0]:
        return arr_c[0]
    if H >= arr_h[-1]:
        return arr_c[-1]
    j=np.searchsorted(arr_h,H)-1
    h0,h1=arr_h[j],arr_h[j+1]
    c0,c1=arr_c[j],arr_c[j+1]
    lam=(H-h0)/(h1-h0)
    return float(np.exp((1-lam)*np.log(c0)+lam*np.log(c1)))

corr_lookup={}
for (T,P),g in anchor.groupby(['temperature_K','pressure_dyn_cm2']):
    g=g.sort_values('height_Mm')
    hs=g.height_Mm.to_numpy(dtype=float)
    corr_lookup[(T,P)] = (hs,g.C_i.to_numpy(dtype=float),g.C_f.to_numpy(dtype=float))

final=pw.copy()
ci=[]; cf=[]
for r in final.itertuples(index=False):
    hs,cis,cfs=corr_lookup[(r.temperature_K,r.pressure_dyn_cm2)]
    ci.append(interp_log_correction(r.height_Mm,hs,cis))
    cf.append(interp_log_correction(r.height_Mm,hs,cfs))
final['C_i_applied']=ci
final['C_f_applied']=cf
final['i_extension']=final['i_pw_H']*final['C_i_applied']
final['f_extension_1e16_cm3']=final['f_pw_H_1e16_cm3']*final['C_f_applied']
# Put the scientifically useful columns first.
cols=['height_Mm','temperature_K','pressure_dyn_cm2','i_extension','f_extension_1e16_cm3',
      'i_pw_H','f_pw_H_1e16_cm3','C_i_applied','C_f_applied','i_pw_total_e','f_pw_total_e_1e16_cm3','iterations','duration_s']
final=final[cols].sort_values(['height_Mm','temperature_K','pressure_dyn_cm2']).reset_index(drop=True)
final.to_csv(OUT/'output'/'heinzel_calibrated_extension_1_to_100Mm.csv',index=False,float_format='%.10g')

# Wide products: rows = height; columns = T/p combinations.
def make_wide(value_col):
    x=final.pivot_table(index='height_Mm',columns=['temperature_K','pressure_dyn_cm2'],values=value_col)
    x.columns=[f'T{int(T)}K_p{P:g}' for T,P in x.columns]
    return x.reset_index()
make_wide('i_extension').to_csv(OUT/'output'/'heinzel_extension_i_by_height.csv',index=False,float_format='%.8g')
make_wide('f_extension_1e16_cm3').to_csv(OUT/'output'/'heinzel_extension_f_by_height_1e16_cm3.csv',index=False,float_format='%.8g')

# Exact-anchor check.
check=final.merge(ref,on=['height_Mm','temperature_K','pressure_dyn_cm2'])
check['delta_i']=check.i_extension-check.i_heinzel2015
check['delta_f_1e16_cm3']=check.f_extension_1e16_cm3-check.f_heinzel2015_1e16_cm3
check.to_csv(OUT/'output'/'published_anchor_exact_match_check.csv',index=False)

# Cross-validation: calibrate at a single published height and predict the other published heights.
cv=[]
for H0 in (10,20,30):
    b=anchor[anchor.height_Mm==H0][['temperature_K','pressure_dyn_cm2','i_pw_H','f_pw_H_1e16_cm3','i_heinzel2015','f_heinzel2015_1e16_cm3']].rename(
        columns={'i_pw_H':'i_pw_anchor','f_pw_H_1e16_cm3':'f_pw_anchor','i_heinzel2015':'i_ref_anchor','f_heinzel2015_1e16_cm3':'f_ref_anchor'})
    for Ht in (10,20,30):
        if Ht==H0: continue
        t=anchor[anchor.height_Mm==Ht].merge(b,on=['temperature_K','pressure_dyn_cm2'])
        t['anchor_height_Mm']=H0
        t['target_height_Mm']=Ht
        t['i_pred_from_anchor']=t['i_ref_anchor']*t['i_pw_H']/t['i_pw_anchor']
        t['f_pred_from_anchor_1e16_cm3']=t['f_ref_anchor']*t['f_pw_H_1e16_cm3']/t['f_pw_anchor']
        t['i_abs_rel_error']=abs((t['i_pred_from_anchor']-t['i_heinzel2015'])/t['i_heinzel2015'])
        t['f_abs_rel_error']=abs((t['f_pred_from_anchor_1e16_cm3']-t['f_heinzel2015_1e16_cm3'])/t['f_heinzel2015_1e16_cm3'])
        cv.append(t[['anchor_height_Mm','target_height_Mm','temperature_K','pressure_dyn_cm2',
                     'i_heinzel2015','i_pred_from_anchor','i_abs_rel_error',
                     'f_heinzel2015_1e16_cm3','f_pred_from_anchor_1e16_cm3','f_abs_rel_error']])
cv=pd.concat(cv,ignore_index=True)
cv.to_csv(OUT/'output'/'single_anchor_cross_validation.csv',index=False)
summary=[]
for (H0,Ht),g in cv.groupby(['anchor_height_Mm','target_height_Mm']):
    summary.append(dict(anchor_height_Mm=H0,target_height_Mm=Ht,n=len(g),
                        i_median_abs_rel_error_pct=100*g.i_abs_rel_error.median(),
                        i_max_abs_rel_error_pct=100*g.i_abs_rel_error.max(),
                        f_median_abs_rel_error_pct=100*g.f_abs_rel_error.median(),
                        f_max_abs_rel_error_pct=100*g.f_abs_rel_error.max()))
cvsum=pd.DataFrame(summary).sort_values(['anchor_height_Mm','target_height_Mm'])
cvsum.to_csv(OUT/'output'/'single_anchor_cross_validation_summary.csv',index=False,float_format='%.5f')

# Representative compact table at p=0.05.
rep=final[(final.pressure_dyn_cm2==0.05)&(final.height_Mm.isin([1,5,10,20,30,50,100]))][
    ['height_Mm','temperature_K','pressure_dyn_cm2','i_extension','f_extension_1e16_cm3']]
rep.to_csv(OUT/'output'/'representative_values_p0p05.csv',index=False,float_format='%.6g')

# Plots: default matplotlib color cycle, one figure per plot.
for P in [0.01,0.05,0.20]:
    g=final[final.pressure_dyn_cm2==P]
    fig,ax=plt.subplots(figsize=(8,5.2))
    for T in temps:
        q=g[g.temperature_K==T]
        ax.plot(q.height_Mm,q.i_extension,label=f'{T} K')
    ax.set_xlabel('Height [Mm]'); ax.set_ylabel('Ionization degree i')
    ax.set_title(f'Heinzel-calibrated extension: i(H), p={P:g} dyn cm$^{{-2}}$')
    ax.set_xlim(1,100); ax.grid(alpha=0.25); ax.legend(title='Temperature')
    fig.tight_layout(); fig.savefig(OUT/'plots'/f'i_vs_height_p{str(P).replace(".","p")}.png',dpi=220); plt.close(fig)

    fig,ax=plt.subplots(figsize=(8,5.2))
    for T in temps:
        q=g[g.temperature_K==T]
        ax.plot(q.height_Mm,q.f_extension_1e16_cm3,label=f'{T} K')
    ax.set_xlabel('Height [Mm]'); ax.set_ylabel(r'$f$ [$10^{16}$ cm$^{-3}$]')
    ax.set_title(f'Heinzel-calibrated extension: f(H), p={P:g} dyn cm$^{{-2}}$')
    ax.set_xlim(1,100); ax.grid(alpha=0.25); ax.legend(title='Temperature')
    fig.tight_layout(); fig.savefig(OUT/'plots'/f'f_vs_height_p{str(P).replace(".","p")}.png',dpi=220); plt.close(fig)

# Summary text.
raw_match=anchor.copy()
raw_match['i_abs_rel']=abs((raw_match.i_pw_H-raw_match.i_heinzel2015)/raw_match.i_heinzel2015)
raw_match['f_abs_rel']=abs((raw_match.f_pw_H_1e16_cm3-raw_match.f_heinzel2015_1e16_cm3)/raw_match.f_heinzel2015_1e16_cm3)
cv10=cv[(cv.anchor_height_Mm==10)]
summary_txt=f'''FINAL 1-100 Mm HEINZEL TABLE-1 EXTENSION\n\nInput Promweaver models: {len(raw)}\nSuccessful models: {(raw.status=="ok").sum()} / {len(raw)}\nHeight grid: 1-100 Mm in 1-Mm steps\nTemperature grid: {temps}\nPressure grid [dyn cm^-2]: {pressures}\nBoundary condition: ConePromBc / Promweaver FAL-C tabulated boundary\nHydrogen solution: PRD\n\nDIRECT PROMWEAVER vs HEINZEL 2015 (75 anchor cells)\nMedian |relative error| i_H = {100*raw_match.i_abs_rel.median():.2f}%\nMedian |relative error| f_H = {100*raw_match.f_abs_rel.median():.2f}%\n\nHEIGHT-RESPONSE VALIDATION: CALIBRATE ONLY AT 10 Mm\nPredict published 20 Mm: median |error| i = {100*cv10[cv10.target_height_Mm==20].i_abs_rel_error.median():.2f}%, f = {100*cv10[cv10.target_height_Mm==20].f_abs_rel_error.median():.2f}%\nPredict published 30 Mm: median |error| i = {100*cv10[cv10.target_height_Mm==30].i_abs_rel_error.median():.2f}%, f = {100*cv10[cv10.target_height_Mm==30].f_abs_rel_error.median():.2f}%\nMaximum over these 50 predictions: i = {100*cv10.i_abs_rel_error.max():.2f}%, f = {100*cv10.f_abs_rel_error.max():.2f}%\n\nFINAL CALIBRATION\nAt 10, 20, 30 Mm, multiplicative correction factors force exact agreement with the published Table 1.\nBetween 10-20 and 20-30 Mm, correction factors are interpolated geometrically (linearly in log correction).\nBelow 10 Mm, the 10-Mm correction factor is held fixed, preserving Promweaver's relative height dependence.\nAbove 30 Mm, the 30-Mm correction factor is held fixed, preserving Promweaver's relative height dependence.\n\nCAUTION\nThe 1-5 Mm regime is supplied because it was requested, but the isolated prominence-slab assumptions become less secure very low in the atmosphere. Treat that regime with more caution than the coronal-prominence heights.\n'''
(OUT/'SUMMARY.txt').write_text(summary_txt)

readme=f'''# Heinzel et al. (2015) Table 1 — calibrated 1–100 Mm extension\n\nThis package contains the 1–100 Mm extension constructed from the completed **Promweaver PRD + cone-boundary** grid and calibrated to the published Table 1 at 10, 20, and 30 Mm.\n\n## Recommended final products\n\n- `output/heinzel_calibrated_extension_1_to_100Mm.csv` — tidy master table (2500 rows).\n- `output/heinzel_extension_i_by_height.csv` — wide ionization-degree table.\n- `output/heinzel_extension_f_by_height_1e16_cm3.csv` — wide f table.\n- `output/single_anchor_cross_validation_summary.csv` — independent height-response validation.\n- `plots/` — representative height curves.\n\n## Definitions\n\nThe primary Promweaver quantities are\n\n`i_pw_H = n_p / n_H`\n\nand\n\n`f_pw_H = n_p^2 / n_2`,\n\nwith f reported in units of `1e16 cm^-3`. These are used instead of total-electron quantities because the original Heinzel Table-1 treatment neglects helium ionization in the pressure relation. The raw total-electron quantities are retained for diagnostics.\n\n## Why calibrate?\n\nThe modern Promweaver/FAL-C calculation reproduces the published ionization degree closely, but its absolute `f` normalization differs substantially from the 2015 table. Crucially, the *relative height response* agrees very well.\n\nUsing only the published 10-Mm values to normalize Promweaver, the model predicts the published values at:\n\n- **20 Mm:** median error {100*cv10[cv10.target_height_Mm==20].i_abs_rel_error.median():.2f}% for i and {100*cv10[cv10.target_height_Mm==20].f_abs_rel_error.median():.2f}% for f.\n- **30 Mm:** median error {100*cv10[cv10.target_height_Mm==30].i_abs_rel_error.median():.2f}% for i and {100*cv10[cv10.target_height_Mm==30].f_abs_rel_error.median():.2f}% for f.\n\nThis is the empirical justification for using Promweaver to supply the height dependence.\n\n## Final calibration formula\n\nFor quantity `X` (i or f), at each `(T,p)` and published anchor height `Ha`:\n\n`C_X(Ha,T,p) = X_Heinzel(Ha,T,p) / X_PW(Ha,T,p)`\n\nand\n\n`X_extension(H,T,p) = C_X(H,T,p) * X_PW(H,T,p)`.\n\n`C_X` is held fixed below 10 Mm and above 30 Mm. Between the published anchors it is interpolated geometrically so that the final table is continuous and exactly reproduces the published 10/20/30-Mm values.\n\n## Rebuild\n\n`build_extension.py` reconstructs all outputs from the raw Promweaver CSV and the included published-reference CSV.\n\n## Scientific caution\n\nThis is a **model-based calibrated extension**, not a new published non-LTE reference table. It is strongest where the prominence-slab approximation is appropriate. Values at roughly 1–5 Mm should be treated with additional caution.\n'''
(OUT/'README.md').write_text(readme)

# Put standalone reproducibility script into package by copying this script itself later outside this process.
print(summary_txt)
print('created',OUT)
