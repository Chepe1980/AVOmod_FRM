import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.colors as colors
from pyavo.seismodel import tuning_wedge as tw
from pyavo.seismodel import wavelet

# Set page config
st.set_page_config(layout="wide", page_title="Rock Physics & AVO Modeling")

# Title and description
st.title("Rock Physics & AVO Modeling Tool")
st.markdown("""
This app performs rock physics modeling and AVO analysis for brine, oil, and gas scenarios.
""")

# Sidebar for input parameters
with st.sidebar:
    st.header("Mineral and Fluid Properties")
    
    # Mineral properties
    st.subheader("Mineral Properties")
    col1, col2 = st.columns(2)
    with col1:
        rho_qz = st.number_input("Quartz Density (g/cc)", value=2.65, step=0.01)
        k_qz = st.number_input("Quartz Bulk Modulus (GPa)", value=37.0, step=0.1)
        mu_qz = st.number_input("Quartz Shear Modulus (GPa)", value=44.0, step=0.1)
    with col2:
        rho_sh = st.number_input("Shale Density (g/cc)", value=2.81, step=0.01)
        k_sh = st.number_input("Shale Bulk Modulus (GPa)", value=15.0, step=0.1)
        mu_sh = st.number_input("Shale Shear Modulus (GPa)", value=5.0, step=0.1)
    
    # Fluid properties
    st.subheader("Fluid Properties")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("**Brine**")
        rho_b = st.number_input("Brine Density (g/cc)", value=1.09, step=0.01)
        k_b = st.number_input("Brine Bulk Modulus (GPa)", value=2.8, step=0.1)
    with col2:
        st.markdown("**Oil**")
        rho_o = st.number_input("Oil Density (g/cc)", value=0.78, step=0.01)
        k_o = st.number_input("Oil Bulk Modulus (GPa)", value=0.94, step=0.1)
    with col3:
        st.markdown("**Gas**")
        rho_g = st.number_input("Gas Density (g/cc)", value=0.25, step=0.01)
        k_g = st.number_input("Gas Bulk Modulus (GPa)", value=0.06, step=0.01)
    
    # AVO modeling parameters
    st.subheader("AVO Modeling Parameters")
    min_angle = st.slider("Minimum Angle (deg)", 0, 10, 0)
    max_angle = st.slider("Maximum Angle (deg)", 30, 50, 45)
    wavelet_freq = st.slider("Wavelet Frequency (Hz)", 20, 80, 50)
    sand_cutoff = st.slider("Sand Cutoff (VSH)", 0.0, 0.3, 0.12, step=0.01)
    
    # File upload
    st.subheader("Well Log Data")
    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])
    
    # Execute button
    run_analysis = st.button("Run Analysis")

# Main content area
if run_analysis and uploaded_file is not None:
    # Read data
    logs = pd.read_csv(uploaded_file)
    
    # VRH function
    def vrh(volumes, k, mu):
        f = np.array(volumes).T
        k = np.resize(np.array(k), np.shape(f))
        mu = np.resize(np.array(mu), np.shape(f))

        k_u = np.sum(f*k, axis=1)
        k_l = 1. / np.sum(f/k, axis=1)
        mu_u = np.sum(f*mu, axis=1)
        mu_l = 1. / np.sum(f/mu, axis=1)
        k0 = (k_u+k_l) / 2.
        mu0 = (mu_u+mu_l) / 2.
        return k_u, k_l, mu_u, mu_l, k0, mu0

    # Fluid substitution function
    def frm(vp1, vs1, rho1, rho_f1, k_f1, rho_f2, k_f2, k0, phi):
        vp1 = vp1 / 1000.
        vs1 = vs1 / 1000.
        mu1 = rho1 * vs1**2.
        k_s1 = rho1 * vp1**2 - (4./3.)*mu1

        # The dry rock bulk modulus
        kdry = (k_s1 * ((phi*k0)/k_f1+1-phi)-k0) / ((phi*k0)/k_f1+(k_s1/k0)-1-phi)

        # Now we can apply Gassmann to get the new values
        k_s2 = kdry + (1- (kdry/k0))**2 / ( (phi/k_f2) + ((1-phi)/k0) - (kdry/k0**2) )
        rho2 = rho1-phi * rho_f1+phi * rho_f2
        mu2 = mu1
        vp2 = np.sqrt(((k_s2+(4./3)*mu2))/rho2)
        vs2 = np.sqrt((mu2/rho2))

        return vp2*1000, vs2*1000, rho2, k_s2

    # Process data
    shale = logs.VSH.values
    sand = 1 - shale - logs.PHI.values
    shaleN = shale / (shale+sand)
    sandN = sand / (shale+sand)
    k_u, k_l, mu_u, mu_l, k0, mu0 = vrh([shaleN, sandN], [k_sh, k_qz], [mu_sh, mu_qz])

    # Fluid mixtures
    water = logs.SW.values
    hc = 1 - logs.SW.values
    tmp, k_fl, tmp, tmp, tmp, tmp = vrh([water, hc], [k_b, k_o], [0, 0])
    rho_fl = water*rho_b + hc*rho_o

    # Fluid substitution
    vpb, vsb, rhob, kb = frm(logs.VP, logs.VS, logs.RHO, rho_fl, k_fl, rho_b, k_b, k0, logs.PHI)
    vpo, vso, rhoo, ko = frm(logs.VP, logs.VS, logs.RHO, rho_fl, k_fl, rho_o, k_o, k0, logs.PHI)
    vpg, vsg, rhog, kg = frm(logs.VP, logs.VS, logs.RHO, rho_fl, k_fl, rho_g, k_g, k0, logs.PHI)

    # Litho-fluid classification
    brine_sand = ((logs.VSH <= sand_cutoff) & (logs.SW >= 0.65))
    oil_sand = ((logs.VSH <= sand_cutoff) & (logs.SW < 0.65))
    shale_flag = (logs.VSH > sand_cutoff)

    # Add results to logs
    logs['VP_FRMB'] = logs.VP
    logs['VS_FRMB'] = logs.VS
    logs['RHO_FRMB'] = logs.RHO
    logs['VP_FRMB'][brine_sand|oil_sand] = vpb[brine_sand|oil_sand]
    logs['VS_FRMB'][brine_sand|oil_sand] = vsb[brine_sand|oil_sand]
    logs['RHO_FRMB'][brine_sand|oil_sand] = rhob[brine_sand|oil_sand]
    logs['IP_FRMB'] = logs.VP_FRMB*logs.RHO_FRMB
    logs['IS_FRMB'] = logs.VS_FRMB*logs.RHO_FRMB
    logs['VPVS_FRMB'] = logs.VP_FRMB/logs.VS_FRMB

    logs['VP_FRMO'] = logs.VP
    logs['VS_FRMO'] = logs.VS
    logs['RHO_FRMO'] = logs.RHO
    logs['VP_FRMO'][brine_sand|oil_sand] = vpo[brine_sand|oil_sand]
    logs['VS_FRMO'][brine_sand|oil_sand] = vso[brine_sand|oil_sand]
    logs['RHO_FRMO'][brine_sand|oil_sand] = rhoo[brine_sand|oil_sand]
    logs['IP_FRMO'] = logs.VP_FRMO*logs.RHO_FRMO
    logs['IS_FRMO'] = logs.VS_FRMO*logs.RHO_FRMO
    logs['VPVS_FRMO'] = logs.VP_FRMO/logs.VS_FRMO

    logs['VP_FRMG'] = logs.VP
    logs['VS_FRMG'] = logs.VS
    logs['RHO_FRMG'] = logs.RHO
    logs['VP_FRMG'][brine_sand|oil_sand] = vpg[brine_sand|oil_sand]
    logs['VS_FRMG'][brine_sand|oil_sand] = vsg[brine_sand|oil_sand]
    logs['RHO_FRMG'][brine_sand|oil_sand] = rhog[brine_sand|oil_sand]
    logs['IP_FRMG'] = logs.VP_FRMG*logs.RHO_FRMG
    logs['IS_FRMG'] = logs.VS_FRMG*logs.RHO_FRMG
    logs['VPVS_FRMG'] = logs.VP_FRMG/logs.VS_FRMG

    # LFC flags
    temp_lfc_b = np.zeros(np.shape(logs.VSH))
    temp_lfc_b[brine_sand.values | oil_sand.values] = 1
    temp_lfc_b[shale_flag.values] = 4
    logs['LFC_B'] = temp_lfc_b

    temp_lfc_o = np.zeros(np.shape(logs.VSH))
    temp_lfc_o[brine_sand.values | oil_sand.values] = 2
    temp_lfc_o[shale_flag.values] = 4
    logs['LFC_O'] = temp_lfc_o

    temp_lfc_g = np.zeros(np.shape(logs.VSH))
    temp_lfc_g[brine_sand.values | oil_sand.values] = 3
    temp_lfc_g[shale_flag.values] = 4
    logs['LFC_G'] = temp_lfc_g

    # Visualization
    ccc = ['#B3B3B3','blue','green','red','#996633']
    cmap_facies = colors.ListedColormap(ccc[0:len(ccc)], 'indexed')

    # Display logs
    st.header("Well Log Visualization")
    ztop = st.slider("Top Depth", float(logs.DEPTH.min()), float(logs.DEPTH.max()), float(logs.DEPTH.min()))
    zbot = st.slider("Bottom Depth", float(logs.DEPTH.min()), float(logs.DEPTH.max()), float(logs.DEPTH.max()))
    
    ll = logs.loc[(logs.DEPTH>=ztop) & (logs.DEPTH<=zbot)]
    cluster = np.repeat(np.expand_dims(ll['LFC_B'].values,1), 100, 1)

    fig, ax = plt.subplots(nrows=1, ncols=4, figsize=(12, 8))
    ax[0].plot(ll.VSH, ll.DEPTH, '-g', label='Vsh')
    ax[0].plot(ll.SW, ll.DEPTH, '-b', label='Sw')
    ax[0].plot(ll.PHI, ll.DEPTH, '-k', label='phi')
    ax[1].plot(ll.IP_FRMG, ll.DEPTH, '-r')
    ax[1].plot(ll.IP_FRMB, ll.DEPTH, '-b')
    ax[1].plot(ll.IP, ll.DEPTH, '-', color='0.5')
    ax[2].plot(ll.VPVS_FRMG, ll.DEPTH, '-r')
    ax[2].plot(ll.VPVS_FRMB, ll.DEPTH, '-b')
    ax[2].plot(ll.VPVS, ll.DEPTH, '-', color='0.5')
    im = ax[3].imshow(cluster, interpolation='none', aspect='auto', cmap=cmap_facies, vmin=0, vmax=4)

    cbar = plt.colorbar(im, ax=ax[3])
    cbar.set_label((12*' ').join(['undef', 'brine', 'oil', 'gas', 'shale']))
    cbar.set_ticks(range(0,1)); cbar.set_ticklabels('')

    for i in ax[:-1]:
        i.set_ylim(ztop,zbot)
        i.invert_yaxis()
        i.grid()
        i.locator_params(axis='x', nbins=4)
    ax[0].legend(fontsize='small', loc='lower right')
    ax[0].set_xlabel("Vcl/phi/Sw"); ax[0].set_xlim(-.1,1.1)
    ax[1].set_xlabel("Ip [m/s*g/cc]"); ax[1].set_xlim(6000,15000)
    ax[2].set_xlabel("Vp/Vs"); ax[2].set_xlim(1.5,2)
    ax[3].set_xlabel('LFC')
    ax[1].set_yticklabels([]); ax[2].set_yticklabels([]); ax[3].set_yticklabels([]); ax[3].set_xticklabels([])
    
    st.pyplot(fig)

    # Crossplots
    st.header("Crossplots")
    fig2, ax2 = plt.subplots(nrows=1, ncols=4, figsize=(16, 4))
    ax2[0].scatter(logs.IP, logs.VPVS, 20, logs.LFC_B, marker='o', edgecolors='none', alpha=0.5, cmap=cmap_facies, vmin=0, vmax=4)
    ax2[1].scatter(logs.IP_FRMB, logs.VPVS_FRMB, 20, logs.LFC_B, marker='o', edgecolors='none', alpha=0.5, cmap=cmap_facies, vmin=0, vmax=4)
    ax2[2].scatter(logs.IP_FRMO, logs.VPVS_FRMO, 20, logs.LFC_O, marker='o', edgecolors='none', alpha=0.5, cmap=cmap_facies, vmin=0, vmax=4)
    ax2[3].scatter(logs.IP_FRMG, logs.VPVS_FRMG, 20, logs.LFC_G, marker='o', edgecolors='none', alpha=0.5, cmap=cmap_facies, vmin=0, vmax=4)
    ax2[0].set_xlim(3000,16000); ax2[0].set_ylim(1.5,3)
    ax2[0].set_title('Original Data')
    ax2[1].set_title('FRM to Brine')
    ax2[2].set_title('FRM to Oil')
    ax2[3].set_title('FRM to Gas')
    st.pyplot(fig2)

    # AVO Modeling
    st.header("AVO Modeling")
    
    # Zone selection
    st.subheader("Zone Selection for AVO Modeling")
    col1, col2, col3 = st.columns(3)
    with col1:
        ztopu = st.number_input("Upper Zone Top", value=float(logs.DEPTH.min()))
        zbotu = st.number_input("Upper Zone Bottom", value=float(logs.DEPTH.min() + (logs.DEPTH.max()-logs.DEPTH.min())/3))
    with col2:
        ztopm = st.number_input("Middle Zone Top", value=float(logs.DEPTH.min() + (logs.DEPTH.max()-logs.DEPTH.min())/3))
        zbotm = st.number_input("Middle Zone Bottom", value=float(logs.DEPTH.min() + 2*(logs.DEPTH.max()-logs.DEPTH.min())/3))
    with col3:
        ztopl = st.number_input("Lower Zone Top", value=float(logs.DEPTH.min() + 2*(logs.DEPTH.max()-logs.DEPTH.min())/3))
        zbotl = st.number_input("Lower Zone Bottom", value=float(logs.DEPTH.max()))
    
    # AVO for each case
    cases = ['Brine', 'Oil', 'Gas']
    case_data = {
        'Brine': {'vp': 'VP_FRMB', 'vs': 'VS_FRMB', 'rho': 'RHO_FRMB'},
        'Oil': {'vp': 'VP_FRMO', 'vs': 'VS_FRMO', 'rho': 'RHO_FRMO'},
        'Gas': {'vp': 'VP_FRMG', 'vs': 'VS_FRMG', 'rho': 'RHO_FRMG'}
    }
    
    for case in cases:
        st.subheader(f"{case} Case AVO Modeling")
        
        # Get average properties for each zone
vp_u = logs.loc[((logs.DEPTH >= ztopu) & (logs.DEPTH <= zbotu)), case_data[case]['vp']].values
vs_u = logs.loc[((logs.DEPTH >= ztopu) & (logs.DEPTH <= zbotu)), case_data[case]['vs']].values
rho_u = logs.loc[((logs.DEPTH >= ztopu) & (logs.DEPTH <= zbotu)), case_data[case]['rho']].values

vp_m = logs.loc[((logs.DEPTH >= ztopm) & (logs.DEPTH <= zbotm)), case_data[case]['vp']].values
vs_m = logs.loc[((logs.DEPTH >= ztopm) & (logs.DEPTH <= zbotm)), case_data[case]['vs']].values
rho_m = logs.loc[((logs.DEPTH >= ztopm) & (logs.DEPTH <= zbotm)), case_data[case]['rho']].values

vp_l = logs.loc[((logs.DEPTH >= ztopl) & (logs.DEPTH <= zbotl)), case_data[case]['vp']].values
vs_l = logs.loc[((logs.DEPTH >= ztopl) & (logs.DEPTH <= zbotl)), case_data[case]['vs']].values
rho_l = logs.loc[((logs.DEPTH >= ztopl) & (logs.DEPTH <= zbotl)), case_data[case]['rho']].values

# Round averages to 2 decimal places (adjust as needed)
vp_data = [round(vp_u.mean(), 2), round(vp_m.mean(), 2), round(vp_l.mean(), 2)]
vs_data = [round(vs_u.mean(), 2), round(vs_m.mean(), 2), round(vs_l.mean(), 2)]
rho_data = [round(rho_u.mean(), 2), round(rho_m.mean(), 2), round(rho_l.mean(), 2)]

        # Create model
        nangles = tw.n_angles(min_angle, max_angle)
        rc_zoep = []
        theta1 = []

        for angle in range(0, nangles):
            theta1_samp, rc_1, rc_2 = tw.calc_theta_rc(theta1_min=0, theta1_step=1, 
                                                      vp=vp_data, vs=vs_data, rho=rho_data, ang=angle)
            theta1.append(theta1_samp)
            rc_zoep.append([rc_1[0, 0], rc_2[0, 0]])

        # Generate wavelet
        wlt_time, wlt_amp = wavelet.ricker(sample_rate=0.0001, length=0.128, c_freq=wavelet_freq)
        t_samp = tw.time_samples(t_min=0, t_max=0.5)

        # Generate synthetic
        syn_zoep = []
        lyr_times = []

        for angle in range(0, nangles):
            z_int = tw.int_depth(h_int=[500.0], thickness=10)
            t_int = tw.calc_times(z_int, vp_data)
            lyr_times.append(t_int)
            rc = tw.mod_digitize(rc_zoep[angle], t_int, t_samp)
            s = tw.syn_seis(ref_coef=rc, wav_amp=wlt_amp)
            syn_zoep.append(s)

        syn_zoep = np.array(syn_zoep)
        rc_zoep = np.array(rc_zoep)
        t = np.array(t_samp)
        lyr_times = np.array(lyr_times)

        # Plot results
        fig3, (ax3a, ax3b) = plt.subplots(1, 2, figsize=(12, 6))
        
        # Wavelet plot
        ax3a.plot(wlt_time, wlt_amp)
        ax3a.set_title(f"Ricker Wavelet ({wavelet_freq} Hz)")
        ax3a.set_xlabel("Time (s)")
        ax3a.set_ylabel("Amplitude")
        ax3a.grid(True)
        
        # AVO gather plot
        excursion = 2
        thickness = 80
        min_plot_time = 0.10
        max_plot_time = 0.25
        
        lyr1_indx, lyr2_indx = tw.layer_index(lyr_times)
        vp_dig, vs_dig, rho_dig = tw.t_domain(t=t, vp=vp_data, vs=vs_data, rho=rho_data, 
                                             lyr1_index=lyr1_indx, lyr2_index=lyr2_indx)
        
        tw.syn_angle_gather(min_plot_time, max_plot_time, lyr_times, thickness, 
                           syn_zoep[:, lyr1_indx], syn_zoep[:, lyr2_indx],
                           vp_dig, vs_dig, rho_dig, syn_zoep, rc_zoep, t, excursion, ax=ax3b)
        ax3b.set_title(f"{case} Case Angle Gather")
        
        st.pyplot(fig3)

elif uploaded_file is None and run_analysis:
    st.warning("Please upload a CSV file first!")
elif not run_analysis:
    st.info("Configure parameters and click 'Run Analysis' to start")
