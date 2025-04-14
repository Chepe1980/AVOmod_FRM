import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from mpl_toolkits.mplot3d import Axes3D
from io import BytesIO
import base64
from streamlit_bokeh_events import streamlit_bokeh_events
from bokeh.plotting import figure
from bokeh.models import ColumnDataSource, HoverTool, CustomJS
from bokeh.palettes import Category10
from bokeh.transform import factor_cmap
import scipy.stats as stats
from scipy.optimize import curve_fit
from scipy import signal
import pywt

# Import rockphypy with error handling
try:
    from rockphypy import QI, GM, Fluid
    rockphypy_available = True
except ImportError:
    rockphypy_available = False

# Set page config
st.set_page_config(layout="wide", page_title="Enhanced Rock Physics & AVO Modeling")

# Title and description
st.title("Enhanced Rock Physics & AVO Modeling Tool")
st.markdown("""
This app performs advanced rock physics modeling and AVO analysis with multiple models, visualization options, 
and uncertainty analysis.
""")

# Available colormaps for seismic displays
seismic_colormaps = ['seismic', 'RdBu', 'bwr', 'coolwarm', 'viridis', 'plasma']

# Sidebar for input parameters
with st.sidebar:
    st.header("Model Configuration")
    
    # Rock physics model selection
    model_options = [
        "Gassmann's Fluid Substitution", 
        "Critical Porosity Model (Nur)", 
        "Contact Theory (Hertz-Mindlin)",
        "Dvorkin-Nur Soft Sand Model",
        "Raymer-Hunt-Gardner Model"
    ]
    
    if rockphypy_available:
        model_options.extend([
            "Soft Sand RPT (rockphypy)",
            "Stiff Sand RPT (rockphypy)"
        ])
    
    model_choice = st.selectbox("Rock Physics Model", model_options, index=0)
    
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
    
    # Additional parameters for selected models
    if model_choice == "Critical Porosity Model (Nur)":
        critical_porosity = st.slider("Critical Porosity (φc)", 0.3, 0.5, 0.4, 0.01)
    elif model_choice in ["Contact Theory (Hertz-Mindlin)", "Dvorkin-Nur Soft Sand Model"]:
        coordination_number = st.slider("Coordination Number", 6, 12, 9)
        effective_pressure = st.slider("Effective Pressure (MPa)", 1, 50, 10)
        if model_choice == "Dvorkin-Nur Soft Sand Model":
            critical_porosity = st.slider("Critical Porosity (φc)", 0.3, 0.5, 0.4, 0.01)
    
    # Rockphypy specific parameters
    if model_choice in ["Soft Sand RPT (rockphypy)", "Stiff Sand RPT (rockphypy)"]:
        st.subheader("RPT Model Parameters")
        rpt_phi_c = st.slider("RPT Critical Porosity", 0.3, 0.5, 0.4, 0.01)
        rpt_Cn = st.slider("RPT Coordination Number", 6.0, 12.0, 8.6, 0.1)
        rpt_sigma = st.slider("RPT Effective Stress (MPa)", 1, 50, 20)
    
    # Fluid properties with uncertainty ranges
    st.subheader("Fluid Properties")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("**Brine**")
        rho_b = st.number_input("Brine Density (g/cc)", value=1.09, step=0.01)
        k_b = st.number_input("Brine Bulk Modulus (GPa)", value=2.8, step=0.1)
        rho_b_std = st.number_input("Brine Density Std Dev", value=0.05, step=0.01, min_value=0.0)
        k_b_std = st.number_input("Brine Bulk Modulus Std Dev", value=0.1, step=0.01, min_value=0.0)
    with col2:
        st.markdown("**Oil**")
        rho_o = st.number_input("Oil Density (g/cc)", value=0.78, step=0.01)
        k_o = st.number_input("Oil Bulk Modulus (GPa)", value=0.94, step=0.1)
        rho_o_std = st.number_input("Oil Density Std Dev", value=0.05, step=0.01, min_value=0.0)
        k_o_std = st.number_input("Oil Bulk Modulus Std Dev", value=0.05, step=0.01, min_value=0.0)
    with col3:
        st.markdown("**Gas**")
        rho_g = st.number_input("Gas Density (g/cc)", value=0.25, step=0.01)
        k_g = st.number_input("Gas Bulk Modulus (GPa)", value=0.06, step=0.01)
        rho_g_std = st.number_input("Gas Density Std Dev", value=0.02, step=0.01, min_value=0.0)
        k_g_std = st.number_input("Gas Bulk Modulus Std Dev", value=0.01, step=0.01, min_value=0.0)
    
    # AVO modeling parameters
    st.subheader("AVO Modeling Parameters")
    min_angle = st.slider("Minimum Angle (deg)", 0, 10, 0)
    max_angle = st.slider("Maximum Angle (deg)", 30, 50, 45)
    angle_step = st.slider("Angle Step (deg)", 1, 5, 1)
    wavelet_freq = st.slider("Wavelet Frequency (Hz)", 20, 80, 50)
    sand_cutoff = st.slider("Sand Cutoff (VSH)", 0.0, 0.3, 0.12, step=0.01)
    
    # Time-Frequency Analysis Parameters
    st.subheader("Time-Frequency Analysis")
    cwt_scales = st.slider("CWT Scales Range", 1, 100, (1, 50))
    cwt_wavelet = st.selectbox("CWT Wavelet", ['morl', 'cmor', 'gaus', 'mexh'], index=0)
    
    # Monte Carlo parameters
    st.subheader("Uncertainty Analysis")
    mc_iterations = st.slider("Monte Carlo Iterations", 10, 1000, 100)
    include_uncertainty = st.checkbox("Include Uncertainty Analysis", value=False)
    
    # Visualization options
    st.subheader("Visualization Options")
    selected_cmap = st.selectbox("Color Map", seismic_colormaps, index=0)
    show_3d_crossplot = st.checkbox("Show 3D Crossplot", value=False)
    show_histograms = st.checkbox("Show Histograms", value=True)
    show_smith_gidlow = st.checkbox("Show Smith-Gidlow AVO Attributes", value=True)
    
    # File upload
    st.subheader("Well Log Data")
    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

# Error handling decorator
def handle_errors(func):
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            st.error(f"Error in {func.__name__}: {str(e)}")
            st.stop()
    return wrapper

# [Previous rock physics model functions remain the same...]

# Wavelet function
def ricker_wavelet(frequency, length=0.128, dt=0.001):
    """Generate a Ricker wavelet"""
    t = np.linspace(-length/2, length/2, int(length/dt))
    y = (1 - 2*(np.pi**2)*(frequency**2)*(t**2)) * np.exp(-(np.pi**2)*(frequency**2)*(t**2))
    return t, y

# Smith-Gidlow AVO approximation
def smith_gidlow(vp1, vp2, vs1, vs2, rho1, rho2):
    """Calculate Smith-Gidlow AVO attributes (intercept, gradient)"""
    # Calculate reflectivities
    rp = 0.5 * (vp2 - vp1) / (vp2 + vp1) + 0.5 * (rho2 - rho1) / (rho2 + rho1)
    rs = 0.5 * (vs2 - vs1) / (vs2 + vs1) + 0.5 * (rho2 - rho1) / (rho2 + rho1)
    
    # Smith-Gidlow coefficients
    intercept = rp
    gradient = rp - 2 * rs
    fluid_factor = rp + 1.16 * (vp1/vs1) * rs
    
    return intercept, gradient, fluid_factor

# [Previous processing functions remain the same until the AVO modeling section...]

# Main content area
if uploaded_file is not None:
    try:
        # [Previous processing code remains the same until after the synthetic gathers section...]
        
        # After the synthetic gathers section, add the time-frequency analysis
        
        # Time-Frequency Analysis Section
        st.header("Time-Frequency Analysis of Synthetic Gathers")
        
        # Create synthetic gathers for all cases first
        cases = ['Brine', 'Oil', 'Gas']
        case_data = {
            'Brine': {'vp': 'VP_FRMB', 'vs': 'VS_FRMB', 'rho': 'RHO_FRMB', 'color': 'b'},
            'Oil': {'vp': 'VP_FRMO', 'vs': 'VS_FRMO', 'rho': 'RHO_FRMO', 'color': 'g'},
            'Gas': {'vp': 'VP_FRMG', 'vs': 'VS_FRMG', 'rho': 'RHO_FRMG', 'color': 'r'}
        }
        
        wlt_time, wlt_amp = ricker_wavelet(wavelet_freq)
        t_samp = np.arange(0, 0.5, 0.001)  # Higher resolution for better CWT
        t_middle = 0.2
        
        # Store all gathers for time-frequency analysis
        all_gathers = {}
        
        for case in cases:
            vp_upper = logs.loc[(logs.DEPTH >= middle_top - (middle_bot-middle_top)), 'VP'].values.mean()
            vs_upper = logs.loc[(logs.DEPTH >= middle_top - (middle_bot-middle_top)), 'VS'].values.mean()
            rho_upper = logs.loc[(logs.DEPTH >= middle_top - (middle_bot-middle_top)), 'RHO'].values.mean()
            
            vp_middle = logs.loc[(logs.DEPTH >= middle_top) & (logs.DEPTH <= middle_bot), case_data[case]['vp']].values.mean()
            vs_middle = logs.loc[(logs.DEPTH >= middle_top) & (logs.DEPTH <= middle_bot), case_data[case]['vs']].values.mean()
            rho_middle = logs.loc[(logs.DEPTH >= middle_top) & (logs.DEPTH <= middle_bot), case_data[case]['rho']].values.mean()
            
            syn_gather = []
            for angle in angles:
                rc = calculate_reflection_coefficients(
                    vp_upper, vp_middle, vs_upper, vs_middle, rho_upper, rho_middle, angle
                )
                
                rc_series = np.zeros(len(t_samp))
                idx_middle = np.argmin(np.abs(t_samp - t_middle))
                rc_series[idx_middle] = rc
                
                syn_trace = np.convolve(rc_series, wlt_amp, mode='same')
                syn_gather.append(syn_trace)
            
            all_gathers[case] = np.array(syn_gather)
        
        # Frequency Domain Analysis (FFT)
        st.subheader("Frequency Domain Analysis (FFT)")
        
        fig_freq, ax_freq = plt.subplots(1, 3, figsize=(18, 5))
        
        for idx, case in enumerate(cases):
            syn_gather = all_gathers[case]
            
            # Compute FFT for each trace in the gather
            n = len(t_samp)
            dt = t_samp[1] - t_samp[0]
            freqs = np.fft.rfftfreq(n, dt)
            
            # Create array to store frequency spectra
            freq_spectra = np.zeros((len(angles), len(freqs)))
            
            for i, trace in enumerate(syn_gather):
                spectrum = np.abs(np.fft.rfft(trace))
                freq_spectra[i,:] = spectrum
            
            # Normalize spectra for display
            freq_spectra = freq_spectra / np.max(freq_spectra)
            
            # Plot frequency spectra
            extent = [angles[0], angles[-1], freqs[-1], freqs[0]]
            im = ax_freq[idx].imshow(freq_spectra, aspect='auto', extent=extent,
                                   cmap='jet', vmin=0, vmax=1)
            
            ax_freq[idx].set_title(f"{case} Case Frequency Spectrum")
            ax_freq[idx].set_xlabel("Angle (degrees)")
            ax_freq[idx].set_ylabel("Frequency (Hz)")
            ax_freq[idx].set_ylim(wavelet_freq*3, 0)  # Focus on relevant frequencies
            
            plt.colorbar(im, ax=ax_freq[idx], label='Normalized Amplitude')
        
        plt.tight_layout()
        st.pyplot(fig_freq)
        
        # Continuous Wavelet Transform (CWT) Analysis
        st.subheader("Time-Frequency Analysis (CWT)")
        
        # Calculate scales for CWT
        scales = np.arange(cwt_scales[0], cwt_scales[1]+1)
        
        fig_cwt, ax_cwt = plt.subplots(3, len(cases), figsize=(18, 12))
        
        for col_idx, case in enumerate(cases):
            syn_gather = all_gathers[case]
            
            # Select a representative trace (middle angle)
            rep_trace_idx = len(angles) // 2
            trace = syn_gather[rep_trace_idx]
            
            # Perform CWT
            coefficients, frequencies = pywt.cwt(trace, scales, cwt_wavelet, sampling_period=t_samp[1]-t_samp[0])
            
            # Plot time series
            ax_cwt[0, col_idx].plot(t_samp, trace)
            ax_cwt[0, col_idx].set_title(f"{case} - Time Series (@ {angles[rep_trace_idx]}°)")
            ax_cwt[0, col_idx].set_xlabel("Time (s)")
            ax_cwt[0, col_idx].set_ylabel("Amplitude")
            ax_cwt[0, col_idx].grid(True)
            
            # Plot scalogram
            extent = [t_samp[0], t_samp[-1], scales[-1], scales[0]]
            im = ax_cwt[1, col_idx].imshow(np.abs(coefficients), extent=extent,
                                          cmap='jet', aspect='auto',
                                          vmax=np.abs(coefficients).max(),
                                          vmin=-np.abs(coefficients).max())
            
            ax_cwt[1, col_idx].set_title(f"{case} - Scalogram")
            ax_cwt[1, col_idx].set_xlabel("Time (s)")
            ax_cwt[1, col_idx].set_ylabel("Scale")
            plt.colorbar(im, ax=ax_cwt[1, col_idx], label='Magnitude')
            
            # Plot dominant frequencies
            dominant_freqs = frequencies[np.argmax(np.abs(coefficients), axis=0)]
            ax_cwt[2, col_idx].plot(t_samp, dominant_freqs)
            ax_cwt[2, col_idx].set_title(f"{case} - Dominant Frequency")
            ax_cwt[2, col_idx].set_xlabel("Time (s)")
            ax_cwt[2, col_idx].set_ylabel("Frequency (Hz)")
            ax_cwt[2, col_idx].grid(True)
            ax_cwt[2, col_idx].set_ylim(0, wavelet_freq*2)
        
        plt.tight_layout()
        st.pyplot(fig_cwt)
        
        # Spectral Comparison at Selected Angles
        st.subheader("Spectral Comparison at Selected Angles")
        
        selected_angles = st.multiselect(
            "Select angles to compare spectra",
            angles.tolist(),
            default=[angles[0], angles[len(angles)//2], angles[-1]]
        )
        
        if selected_angles:
            fig_compare, ax_compare = plt.subplots(figsize=(10, 6))
            
            for case in cases:
                syn_gather = all_gathers[case]
                
                for angle in selected_angles:
                    angle_idx = np.where(angles == angle)[0][0]
                    trace = syn_gather[angle_idx]
                    
                    # FFT
                    spectrum = np.abs(np.fft.rfft(trace))
                    freqs = np.fft.rfftfreq(len(trace), t_samp[1]-t_samp[0])
                    
                    # Normalize spectrum
                    spectrum = spectrum / np.max(spectrum)
                    
                    ax_compare.plot(freqs, spectrum, 
                                   label=f"{case} @ {angle}°",
                                   linestyle='-' if case == 'Brine' else 
                                           '--' if case == 'Oil' else ':',
                                   color='blue' if case == 'Brine' else
                                        'green' if case == 'Oil' else 'red')
            
            ax_compare.set_title("Normalized Frequency Spectra Comparison")
            ax_compare.set_xlabel("Frequency (Hz)")
            ax_compare.set_ylabel("Normalized Amplitude")
            ax_compare.set_xlim(0, wavelet_freq*3)
            ax_compare.grid(True)
            ax_compare.legend()
            
            st.pyplot(fig_compare)
        
        # [Rest of your existing code continues...]
        
    except Exception as e:
        st.error(f"An error occurred during processing: {str(e)}")
