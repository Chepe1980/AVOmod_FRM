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
    
    # Add colormap selection
    st.subheader("Synthetic Gather Options")
    selected_cmap = st.selectbox("Color Map", seismic_colormaps, index=0)
    
    # File upload
    st.subheader("Well Log Data")
    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

# [Keep all the existing code exactly the same until the synthetic gathers section]

    # Create synthetic gathers for each case
    st.header("Synthetic Seismic Gathers")
    
    # Create figure with subplots for each case
    fig4, ax4 = plt.subplots(1, 3, figsize=(18, 6))
    
    for idx, case in enumerate(cases):
        # Get properties for lower zone and shale below
        vp_m = logs.loc[((logs.DEPTH >= ztopl) & (logs.DEPTH <= zbot)), case_data[case]['vp']].values.mean()
        vs_m = logs.loc[((logs.DEPTH >= ztopl) & (logs.DEPTH <= zbot)), case_data[case]['vs']].values.mean()
        rho_m = logs.loc[((logs.DEPTH >= ztopl) & (logs.DEPTH <= zbot)), case_data[case]['rho']].values.mean()
        
        # Shale properties
        vp_sh = logs.loc[logs.VSH > 0.5, 'VP'].values.mean()
        vs_sh = logs.loc[logs.VSH > 0.5, 'VS'].values.mean()
        rho_sh = logs.loc[logs.VSH > 0.5, 'RHO'].values.mean()
        
        # Generate synthetic gather
        syn_gather = []
        for angle in angles:
            # Calculate reflection coefficient
            rc = calculate_reflection_coefficients(vp_m, vp_sh, vs_m, vs_sh, rho_m, rho_sh, angle)
            
            # Create reflectivity series
            rc_series = np.zeros(len(t_samp))
            idx_lower = np.argmin(np.abs(t_samp - t_lower))
            rc_series[idx_lower] = rc
            
            # Convolve with wavelet
            syn_trace = np.convolve(rc_series, wlt_amp, mode='same')
            syn_gather.append(syn_trace)
        
        syn_gather = np.array(syn_gather)
        
        # Plot the gather with selected colormap
        extent = [angles[0], angles[-1], t_samp[-1], t_samp[0]]
        im = ax4[idx].imshow(syn_gather.T, aspect='auto', extent=extent,
                           cmap=selected_cmap, vmin=-np.max(np.abs(syn_gather)), vmax=np.max(np.abs(syn_gather)))
        ax4[idx].set_title(f"{case} Case")
        ax4[idx].set_xlabel("Angle (degrees)")
        ax4[idx].set_ylabel("Time (s)")
        ax4[idx].set_ylim(0.3, 0.1)  # Zoom in on the reflection
        
        # Add colorbar to each gather
        plt.colorbar(im, ax=ax4[idx], label='Amplitude')
    
    st.pyplot(fig4)

elif uploaded_file is None:
    st.info("Please upload a CSV file to begin analysis")
