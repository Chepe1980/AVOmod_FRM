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
    
    # Seismic display parameters
    st.subheader("Seismic Display Options")
    selected_cmap = st.selectbox("Color Map for Synthetic Gathers", seismic_colormaps, index=0)
    clip_percent = st.slider("Clip Percentile for Color Scaling", 1, 100, 99)
    
    # File upload
    st.subheader("Well Log Data")
    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

# Main content area
if uploaded_file is not None:
    # Read data
    logs = pd.read_csv(uploaded_file)
    
    # Depth range selection
    st.header("Well Log Visualization")
    ztop, zbot = st.slider(
        "Select Depth Range", 
        float(logs.DEPTH.min()), 
        float(logs.DEPTH.max()), 
        (float(logs.DEPTH.min()), float(logs.DEPTH.max()))
    )
    
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
    # ... [keep all the existing processing code unchanged until AVO Modeling section]

    # AVO Modeling
    st.header("AVO Modeling")
    
    # Fixed zone depths (focusing on lower interface)
    ztopm = ztop + (zbot - ztop) * 0.4
    ztopl = ztop + (zbot - ztop) * 0.7
    
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
    
    # Generate wavelet
    wlt_time, wlt_amp = wavelet.ricker(sample_rate=0.0001, length=0.128, c_freq=wavelet_freq)
    t_samp = np.arange(0, 0.5, 0.0001)
    t_lower = 0.2
    
    # Create figure for AVO results
    fig3, ax3 = plt.subplots(1, 2, figsize=(12, 6))
    
    # Wavelet plot
    ax3[0].plot(wlt_time, wlt_amp)
    ax3[0].set_title(f"Ricker Wavelet ({wavelet_freq} Hz)")
    ax3[0].set_xlabel("Time (s)")
    ax3[0].set_ylabel("Amplitude")
    ax3[0].grid(True)
    
    # Process each case for AVO curves
    angles = np.arange(min_angle, max_angle + 1, 1)
    
    for case in cases:
        # Get properties for lower zone and shale below
        vp_m = logs.loc[((logs.DEPTH >= ztopl) & (logs.DEPTH <= zbot)), case_data[case]['vp']].values.mean()
        vs_m = logs.loc[((logs.DEPTH >= ztopl) & (logs.DEPTH <= zbot)), case_data[case]['vs']].values.mean()
        rho_m = logs.loc[((logs.DEPTH >= ztopl) & (logs.DEPTH <= zbot)), case_data[case]['rho']].values.mean()
        
        # Shale properties
        vp_sh = logs.loc[logs.VSH > 0.5, 'VP'].values.mean()
        vs_sh = logs.loc[logs.VSH > 0.5, 'VS'].values.mean()
        rho_sh = logs.loc[logs.VSH > 0.5, 'RHO'].values.mean()
        
        # Calculate reflection coefficients
        rc_lower = []
        for angle in angles:
            rc_lower.append(calculate_reflection_coefficients(
                vp_m, vp_sh, vs_m, vs_sh, rho_m, rho_sh, angle
            ))
        
        # Plot AVO curve
        ax3[1].plot(angles, rc_lower, f"{case_data[case]['color']}-", label=f"{case}")
    
    # Finalize AVO plot
    ax3[1].set_title("AVO Response (Lower Interface)")
    ax3[1].set_xlabel("Angle (degrees)")
    ax3[1].set_ylabel("Reflection Coefficient")
    ax3[1].grid(True)
    ax3[1].legend()
    
    st.pyplot(fig3)

    # Create synthetic gathers for each case with interactive colormap
    st.header("Synthetic Seismic Gathers")
    
    # Create figure with subplots for each case
    fig4, ax4 = plt.subplots(1, 3, figsize=(18, 6))
    
    # Calculate global min/max for consistent color scaling
    all_syn_data = []
    for case in cases:
        vp_m = logs.loc[((logs.DEPTH >= ztopl) & (logs.DEPTH <= zbot)), case_data[case]['vp']].values.mean()
        vs_m = logs.loc[((logs.DEPTH >= ztopl) & (logs.DEPTH <= zbot)), case_data[case]['vs']].values.mean()
        rho_m = logs.loc[((logs.DEPTH >= ztopl) & (logs.DEPTH <= zbot)), case_data[case]['rho']].values.mean()
        vp_sh = logs.loc[logs.VSH > 0.5, 'VP'].values.mean()
        vs_sh = logs.loc[logs.VSH > 0.5, 'VS'].values.mean()
        rho_sh = logs.loc[logs.VSH > 0.5, 'RHO'].values.mean()
        
        syn_gather = []
        for angle in angles:
            rc = calculate_reflection_coefficients(vp_m, vp_sh, vs_m, vs_sh, rho_m, rho_sh, angle)
            rc_series = np.zeros(len(t_samp))
            idx_lower = np.argmin(np.abs(t_samp - t_lower))
            rc_series[idx_lower] = rc
            syn_trace = np.convolve(rc_series, wlt_amp, mode='same')
            syn_gather.append(syn_trace)
        
        all_syn_data.extend(np.array(syn_gather).flatten())
    
    # Calculate clip values based on percentile
    vmin = np.percentile(np.abs(all_syn_data), 100-clip_percent)
    vmax = np.percentile(np.abs(all_syn_data), clip_percent)
    
    for idx, case in enumerate(cases):
        # Get properties
        vp_m = logs.loc[((logs.DEPTH >= ztopl) & (logs.DEPTH <= zbot)), case_data[case]['vp']].values.mean()
        vs_m = logs.loc[((logs.DEPTH >= ztopl) & (logs.DEPTH <= zbot)), case_data[case]['vs']].values.mean()
        rho_m = logs.loc[((logs.DEPTH >= ztopl) & (logs.DEPTH <= zbot)), case_data[case]['rho']].values.mean()
        vp_sh = logs.loc[logs.VSH > 0.5, 'VP'].values.mean()
        vs_sh = logs.loc[logs.VSH > 0.5, 'VS'].values.mean()
        rho_sh = logs.loc[logs.VSH > 0.5, 'RHO'].values.mean()
        
        # Generate synthetic gather
        syn_gather = []
        for angle in angles:
            rc = calculate_reflection_coefficients(vp_m, vp_sh, vs_m, vs_sh, rho_m, rho_sh, angle)
            rc_series = np.zeros(len(t_samp))
            idx_lower = np.argmin(np.abs(t_samp - t_lower))
            rc_series[idx_lower] = rc
            syn_trace = np.convolve(rc_series, wlt_amp, mode='same')
            syn_gather.append(syn_trace)
        
        syn_gather = np.array(syn_gather)
        
        # Plot the gather with selected colormap
        extent = [angles[0], angles[-1], t_samp[-1], t_samp[0]]
        im = ax4[idx].imshow(syn_gather.T, aspect='auto', extent=extent,
                           cmap=selected_cmap, vmin=-vmax, vmax=vmax)
        ax4[idx].set_title(f"{case} Case")
        ax4[idx].set_xlabel("Angle (degrees)")
        ax4[idx].set_ylabel("Time (s)")
        ax4[idx].set_ylim(0.3, 0.1)
        
        # Add colorbar to each gather
        plt.colorbar(im, ax=ax4[idx], label='Amplitude')
    
    st.pyplot(fig4)

elif uploaded_file is None:
    st.info("Please upload a CSV file to begin analysis")
