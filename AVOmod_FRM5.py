import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.colors as colors
from pyavo.seismodel import wavelet

# Set page config
st.set_page_config(layout="wide", page_title="Rock Physics & AVO Modeling")

# Title and description
st.title("Rock Physics & AVO Modeling Tool")
st.markdown("""
This app performs rock physics modeling and AVO analysis for brine, oil, and gas scenarios.
""")

# Available colormaps for seismic displays
seismic_colormaps = ['seismic', 'RdBu', 'bwr', 'coolwarm', 'viridis', 'plasma']

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
    
    # Seismic display options
    st.subheader("Seismic Display Options")
    selected_cmap = st.selectbox("Color Map", seismic_colormaps, index=0)
    
    # File upload
    st.subheader("Well Log Data")
    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

# Main content area
if uploaded_file is not None:
    # Read data
    logs = pd.read_csv(uploaded_file)
    
    # Depth range selection for three layers
    st.header("Three-Layer Model Configuration")
    depth_min = float(logs.DEPTH.min())
    depth_max = float(logs.DEPTH.max())
    
    col1, col2, col3 = st.columns(3)
    with col1:
        upper_top = st.number_input("Upper Zone Top (m)", value=depth_min, step=1.0)
        upper_bot = st.number_input("Upper Zone Base (m)", value=depth_min + (depth_max - depth_min) * 0.3, step=1.0)
    with col2:
        middle_top = st.number_input("Middle Zone Top (m)", value=upper_bot, step=1.0)
        middle_bot = st.number_input("Middle Zone Base (m)", value=depth_min + (depth_max - depth_min) * 0.6, step=1.0)
    with col3:
        lower_top = st.number_input("Lower Zone Top (m)", value=middle_bot, step=1.0)
        lower_bot = st.number_input("Lower Zone Base (m)", value=depth_max, step=1.0)
    
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

    # Process data (same as before)
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

    # Add results to logs (same as before)
    for case, vp, vs, rho in [('B', vpb, vsb, rhob), ('O', vpo, vso, rhoo), ('G', vpg, vsg, rhog)]:
        logs[f'VP_FRM{case}'] = logs.VP
        logs[f'VS_FRM{case}'] = logs.VS
        logs[f'RHO_FRM{case}'] = logs.RHO
        logs[f'VP_FRM{case}'][brine_sand|oil_sand] = vp[brine_sand|oil_sand]
        logs[f'VS_FRM{case}'][brine_sand|oil_sand] = vs[brine_sand|oil_sand]
        logs[f'RHO_FRM{case}'][brine_sand|oil_sand] = rho[brine_sand|oil_sand]
        logs[f'IP_FRM{case}'] = logs[f'VP_FRM{case}']*logs[f'RHO_FRM{case}']
        logs[f'IS_FRM{case}'] = logs[f'VS_FRM{case}']*logs[f'RHO_FRM{case}']
        logs[f'VPVS_FRM{case}'] = logs[f'VP_FRM{case}']/logs[f'VS_FRM{case}']

    # LFC flags (same as before)
    for case, val in [('B', 1), ('O', 2), ('G', 3)]:
        temp_lfc = np.zeros(np.shape(logs.VSH))
        temp_lfc[brine_sand.values | oil_sand.values] = val
        temp_lfc[shale_flag.values] = 4
        logs[f'LFC_{case}'] = temp_lfc

    # Generate wavelet
    wlt_time, wlt_amp = wavelet.ricker(sample_rate=0.0001, length=0.128, c_freq=wavelet_freq)
    t_samp = np.arange(0, 0.5, 0.0001)  # Time samples
    t_middle = 0.2  # Fixed time for middle interface
    
    # AVO cases
    cases = ['Brine', 'Oil', 'Gas']
    case_data = {
        'Brine': {'vp': 'VP_FRMB', 'vs': 'VS_FRMB', 'rho': 'RHO_FRMB', 'color': 'b'},
        'Oil': {'vp': 'VP_FRMO', 'vs': 'VS_FRMO', 'rho': 'RHO_FRMO', 'color': 'g'},
        'Gas': {'vp': 'VP_FRMG', 'vs': 'VS_FRMG', 'rho': 'RHO_FRMG', 'color': 'r'}
    }
    
    def calculate_reflection_coefficients(vp1, vp2, vs1, vs2, rho1, rho2, angle):
        """Calculate PP reflection coefficients using Aki-Richards approximation"""
        theta = np.radians(angle)
        vp_avg = (vp1 + vp2)/2
        vs_avg = (vs1 + vs2)/2
        rho_avg = (rho1 + rho2)/2
        
        dvp = vp2 - vp1
        dvs = vs2 - vs1
        drho = rho2 - rho1
        
        a = 0.5 * (1 + np.tan(theta)**2)
        b = -4 * (vs_avg**2/vp_avg**2) * np.sin(theta)**2
        c = 0.5 * (1 - 4 * (vs_avg**2/vp_avg**2) * np.sin(theta)**2)
        
        rc = a*(dvp/vp_avg) + b*(dvs/vs_avg) + c*(drho/rho_avg)
        return rc
    
    # Create synthetic gathers for middle interface only
    st.header("Synthetic Seismic Gathers (Middle Interface)")
    
    fig, ax = plt.subplots(1, 3, figsize=(18, 5))
    
    angles = np.arange(min_angle, max_angle + 1, 1)
    
    for idx, case in enumerate(cases):
        # Upper layer properties
        vp_upper = logs.loc[(logs.DEPTH >= upper_top) & (logs.DEPTH <= upper_bot), 
                           case_data[case]['vp']].values.mean()
        vs_upper = logs.loc[(logs.DEPTH >= upper_top) & (logs.DEPTH <= upper_bot), 
                           case_data[case]['vs']].values.mean()
        rho_upper = logs.loc[(logs.DEPTH >= upper_top) & (logs.DEPTH <= upper_bot), 
                             case_data[case]['rho']].values.mean()
        
        # Middle layer properties
        vp_middle = logs.loc[(logs.DEPTH >= middle_top) & (logs.DEPTH <= middle_bot), 
                            case_data[case]['vp']].values.mean()
        vs_middle = logs.loc[(logs.DEPTH >= middle_top) & (logs.DEPTH <= middle_bot), 
                            case_data[case]['vs']].values.mean()
        rho_middle = logs.loc[(logs.DEPTH >= middle_top) & (logs.DEPTH <= middle_bot), 
                              case_data[case]['rho']].values.mean()
        
        # Generate synthetic gather for middle interface
        syn_gather = []
        for angle in angles:
            rc = calculate_reflection_coefficients(
                vp_upper, vp_middle, vs_upper, vs_middle, rho_upper, rho_middle, angle
            )
            
            # Create reflectivity series
            rc_series = np.zeros(len(t_samp))
            idx_middle = np.argmin(np.abs(t_samp - t_middle))
            rc_series[idx_middle] = rc
            
            # Convolve with wavelet
            syn_trace = np.convolve(rc_series, wlt_amp, mode='same')
            syn_gather.append(syn_trace)
        
        syn_gather = np.array(syn_gather)
        
        # Plot the gather
        extent = [angles[0], angles[-1], t_samp[-1], t_samp[0]]
        im = ax[idx].imshow(syn_gather.T, aspect='auto', extent=extent,
                          cmap=selected_cmap, vmin=-np.max(np.abs(syn_gather)), 
                          vmax=np.max(np.abs(syn_gather)))
        
        # Add title and labels
        ax[idx].set_title(f"{case} Case", fontweight='bold')
        ax[idx].set_xlabel("Angle (degrees)")
        ax[idx].set_ylabel("Time (s)")
        ax[idx].set_ylim(0.25, 0.15)  # Focus on the middle interface
        
        # Add colorbar
        plt.colorbar(im, ax=ax[idx], label='Amplitude')
    
    plt.tight_layout()
    st.pyplot(fig)

else:
    st.info("Please upload a CSV file to begin analysis")
