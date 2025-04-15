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
from scipy.interpolate import interp2d

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

@handle_errors
def gassmann_fluid_substitution(vp, vs, rho, phi, k_min, mu_min, rho_min, k_fl, rho_fl):
    """Gassmann's fluid substitution"""
    # Dry rock moduli from input velocities
    k_dry = rho*(vp**2 - (4/3)*vs**2)
    mu_dry = rho*vs**2
    
    # Fluid substitution
    k_sat = k_dry + (1 - k_dry/k_min)**2 / (phi/k_fl + (1-phi)/k_min - k_dry/k_min**2)
    mu_sat = mu_dry
    rho_sat = rho + phi*(rho_fl - rho)
    
    # Calculate new velocities
    vp_sat = np.sqrt((k_sat + (4/3)*mu_sat)/rho_sat)
    vs_sat = np.sqrt(mu_sat/rho_sat)
    
    return vp_sat, vs_sat, rho_sat

@handle_errors
def critical_porosity_model(vp, vs, rho, phi, phi_c, k_min, mu_min, rho_min):
    """Critical porosity model (Nur)"""
    # Calculate dry rock moduli
    k_dry = k_min*(1 - phi/phi_c)
    mu_dry = mu_min*(1 - phi/phi_c)
    
    # Calculate saturated velocities
    vp_sat = np.sqrt((k_dry + (4/3)*mu_dry)/rho)
    vs_sat = np.sqrt(mu_dry/rho)
    
    return vp_sat, vs_sat, rho

@handle_errors
def hertz_mindlin(k_min, mu_min, phi, Cn, sigma, rho_min):
    """Hertz-Mindlin contact theory"""
    pr = (3*k_min - 2*mu_min)/(6*k_min + 2*mu_min)  # Poisson's ratio
    
    # Effective moduli
    k_hm = (sigma*(Cn**2*(1-phi)**2*mu_min**2)/(18*np.pi**2*(1-pr)**2))**(1/3)
    mu_hm = ((2+3*pr)/(5*(2-pr)))*((3*sigma*Cn**2*(1-phi)**2*mu_min**2)/(2*np.pi**2*(1-pr)**2))**(1/3)
    
    # Calculate velocities
    vp_hm = np.sqrt((k_hm + (4/3)*mu_hm)/rho_min)
    vs_hm = np.sqrt(mu_hm/rho_min)
    
    return vp_hm, vs_hm, rho_min

@handle_errors
def soft_sand_model(vp, vs, rho, phi, phi_c, k_min, mu_min, rho_min, Cn, sigma):
    """Dvorkin-Nur soft sand model"""
    # Hertz-Mindlin end member
    vp_hm, vs_hm, _ = hertz_mindlin(k_min, mu_min, phi_c, Cn, sigma, rho_min)
    
    # Modified Hashin-Shtrikman lower bound
    k_dry = ((phi/phi_c)/(k_hm + (4/3)*mu_hm) + (1 - phi/phi_c)/(k_min + (4/3)*mu_hm))**-1 - (4/3)*mu_hm
    mu_dry = ((phi/phi_c)/(mu_hm + z) + (1 - phi/phi_c)/(mu_min + z))**-1 - z
    z = (mu_hm/6)*((9*k_hm + 8*mu_hm)/(k_hm + 2*mu_hm))
    
    # Calculate velocities
    vp_ss = np.sqrt((k_dry + (4/3)*mu_dry)/rho)
    vs_ss = np.sqrt(mu_dry/rho)
    
    return vp_ss, vs_ss, rho

@handle_errors
def calculate_reflection_coefficients(vp1, vp2, vs1, vs2, rho1, rho2, angle):
    """Calculate P-wave reflection coefficients using Zoeppritz equations approximation"""
    theta = np.radians(angle)
    vp_avg = (vp1 + vp2)/2
    vs_avg = (vs1 + vs2)/2
    rho_avg = (rho1 + rho2)/2
    
    dvp = vp2 - vp1
    dvs = vs2 - vs1
    drho = rho2 - rho1
    
    # Aki-Richards approximation
    rc = (0.5*(dvp/vp_avg + drho/rho_avg) + 
          0.5*(dvp/vp_avg - 4*(vs_avg**2/vp_avg**2)*(dvs/vs_avg + drho/rho_avg))*np.sin(theta)**2 + 
          0.5*(dvp/vp_avg)*np.tan(theta)**2*np.sin(theta)**2)
    
    return rc

@handle_errors
def ricker_wavelet(frequency, length=0.128, dt=0.001):
    """Generate a Ricker wavelet"""
    t = np.linspace(-length/2, length/2, int(length/dt))
    y = (1 - 2*(np.pi**2)*(frequency**2)*(t**2)) * np.exp(-(np.pi**2)*(frequency**2)*(t**2))
    return t, y

@handle_errors
def smith_gidlow(vp1, vp2, vs1, vs2, rho1, rho2):
    """Calculate Smith-Gidlow AVO attributes (intercept, gradient)"""
    rp = 0.5 * (vp2 - vp1)/(vp2 + vp1) + 0.5 * (rho2 - rho1)/(rho2 + rho1)
    rs = 0.5 * (vs2 - vs1)/(vs2 + vs1) + 0.5 * (rho2 - rho1)/(rho2 + rho1)
    
    intercept = rp
    gradient = rp - 2 * rs
    fluid_factor = rp + 1.16 * (vp1/vs1) * rs
    
    return intercept, gradient, fluid_factor

@handle_errors
def perform_time_frequency_analysis(logs, angles, wavelet_freq, cwt_scales, cwt_wavelet, middle_top, middle_bot):
    """Perform time-frequency analysis on synthetic gathers"""
    cases = ['Brine', 'Oil', 'Gas']
    case_data = {
        'Brine': {'vp': 'VP_FRMB', 'vs': 'VS_FRMB', 'rho': 'RHO_FRMB', 'color': 'b'},
        'Oil': {'vp': 'VP_FRMO', 'vs': 'VS_FRMO', 'rho': 'RHO_FRMO', 'color': 'g'},
        'Gas': {'vp': 'VP_FRMG', 'vs': 'VS_FRMG', 'rho': 'RHO_FRMG', 'color': 'r'}
    }

    wlt_time, wlt_amp = ricker_wavelet(wavelet_freq)
    t_samp = np.arange(0, 0.5, 0.001)
    t_middle = 0.2
    
    all_gathers = {}
    
    for case in cases:
        vp_upper = logs.loc[(logs.DEPTH >= middle_top - (middle_bot-middle_top)), 'VP'].mean()
        vs_upper = logs.loc[(logs.DEPTH >= middle_top - (middle_bot-middle_top)), 'VS'].mean()
        rho_upper = logs.loc[(logs.DEPTH >= middle_top - (middle_bot-middle_top)), 'RHO'].mean()
        
        vp_middle = logs.loc[(logs.DEPTH >= middle_top) & (logs.DEPTH <= middle_bot), case_data[case]['vp']].mean()
        vs_middle = logs.loc[(logs.DEPTH >= middle_top) & (logs.DEPTH <= middle_bot), case_data[case]['vs']].mean()
        rho_middle = logs.loc[(logs.DEPTH >= middle_top) & (logs.DEPTH <= middle_bot), case_data[case]['rho']].mean()
        
        syn_gather = []
        for angle in angles:
            rc = calculate_reflection_coefficients(vp_upper, vp_middle, vs_upper, vs_middle, rho_upper, rho_middle, angle)
            rc_series = np.zeros(len(t_samp))
            rc_series[np.argmin(np.abs(t_samp - t_middle))] = rc
            syn_trace = np.convolve(rc_series, wlt_amp, mode='same')
            syn_gather.append(syn_trace)
        
        all_gathers[case] = np.array(syn_gather)
    
    return all_gathers, t_samp

@handle_errors
def monte_carlo_analysis(logs, params, n_iterations=100):
    """Perform Monte Carlo uncertainty analysis"""
    results = []
    for _ in range(n_iterations):
        # Perturb input parameters
        perturbed = {}
        for param, (mean, std) in params.items():
            perturbed[param] = np.random.normal(mean, std)
        
        # Run rock physics model with perturbed parameters
        try:
            result = run_rock_physics_model(logs, perturbed)
            results.append(result)
        except:
            continue
    
    return pd.DataFrame(results)

# Main processing function
@handle_errors
def process_data(logs, params):
    """Main processing pipeline"""
    # 1. Run rock physics model
    if model_choice == "Gassmann's Fluid Substitution":
        vp_b, vs_b, rho_b = gassmann_fluid_substitution(
            logs['VP'], logs['VS'], logs['RHO'], logs['PHI'],
            params['k_min'], params['mu_min'], params['rho_min'],
            params['k_b'], params['rho_b']
        )
        # Similarly for oil and gas cases...
    
    # 2. AVO modeling
    angles = np.arange(params['min_angle'], params['max_angle']+1, params['angle_step'])
    avo_results = []
    for angle in angles:
        rc = calculate_reflection_coefficients(vp1, vp2, vs1, vs2, rho1, rho2, angle)
        avo_results.append(rc)
    
    # 3. Time-frequency analysis
    all_gathers, t_samp = perform_time_frequency_analysis(
        logs, angles, params['wavelet_freq'], 
        params['cwt_scales'], params['cwt_wavelet'],
        params['middle_top'], params['middle_bot']
    )
    
    return {
        'vp_models': {'Brine': vp_b, 'Oil': vp_o, 'Gas': vp_g},
        'avo_results': avo_results,
        'time_freq': all_gathers
    }

# Main execution
if uploaded_file is not None:
    try:
        logs = pd.read_csv(uploaded_file)
        
        # Validate required columns
        required_cols = ['DEPTH', 'VP', 'VS', 'RHO', 'PHI']
        if not all(col in logs.columns for col in required_cols):
            missing = [col for col in required_cols if col not in logs.columns]
            st.error(f"Missing required columns: {missing}")
            st.stop()
        
        # Prepare parameters
        params = {
            'k_min': k_qz,
            'mu_min': mu_qz,
            'rho_min': rho_qz,
            'k_b': k_b, 'rho_b': rho_b,
            'k_o': k_o, 'rho_o': rho_o,
            'k_g': k_g, 'rho_g': rho_g,
            'min_angle': min_angle,
            'max_angle': max_angle,
            'angle_step': angle_step,
            'wavelet_freq': wavelet_freq,
            'cwt_scales': cwt_scales,
            'cwt_wavelet': cwt_wavelet,
            'middle_top': logs.DEPTH.quantile(0.4),
            'middle_bot': logs.DEPTH.quantile(0.6)
        }
        
        # Run processing
        results = process_data(logs, params)
        
        # Display results
        st.header("Rock Physics Modeling Results")
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("VP Models")
            fig_vp, ax = plt.subplots()
            for case, vp in results['vp_models'].items():
                ax.plot(vp, logs['DEPTH'], label=case)
            ax.invert_yaxis()
            st.pyplot(fig_vp)
        
        with col2:
            st.subheader("AVO Response")
            fig_avo, ax = plt.subplots()
            ax.plot(angles, results['avo_results'])
            st.pyplot(fig_avo)
        
        # Time-frequency analysis
        st.header("Time-Frequency Analysis")
        plot_frequency_analysis(results['time_freq'], t_samp, angles, wavelet_freq)
        plot_cwt_analysis(results['time_freq'], t_samp, angles, cwt_scales, cwt_wavelet, wavelet_freq)
        
        # Uncertainty analysis
        if include_uncertainty:
            st.header("Uncertainty Analysis")
            mc_params = {
                'k_b': (k_b, k_b_std),
                'rho_b': (rho_b, rho_b_std),
                'k_o': (k_o, k_o_std),
                'rho_o': (rho_o, rho_o_std),
                'k_g': (k_g, k_g_std),
                'rho_g': (rho_g, rho_g_std)
            }
            mc_results = monte_carlo_analysis(logs, mc_params, mc_iterations)
            st.write(mc_results.describe())
            
    except Exception as e:
        st.error(f"Processing error: {str(e)}")
