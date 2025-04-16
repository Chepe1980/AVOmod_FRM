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
and uncertainty analysis. Adjust fluid saturations interactively to see their impact on rock properties.
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
    
    # Fluid properties with interactive saturation sliders
    st.subheader("Fluid Properties")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("**Brine**")
        rho_b = st.number_input("Brine Density (g/cc)", value=1.09, step=0.01)
        k_b = st.number_input("Brine Bulk Modulus (GPa)", value=2.8, step=0.1)
        sw = st.slider("Water Saturation (Sw)", 0.0, 1.0, 0.8, 0.01)
    with col2:
        st.markdown("**Oil**")
        rho_o = st.number_input("Oil Density (g/cc)", value=0.78, step=0.01)
        k_o = st.number_input("Oil Bulk Modulus (GPa)", value=0.94, step=0.1)
        so = st.slider("Oil Saturation (So)", 0.0, 1.0, 0.15, 0.01)
    with col3:
        st.markdown("**Gas**")
        rho_g = st.number_input("Gas Density (g/cc)", value=0.25, step=0.01)
        k_g = st.number_input("Gas Bulk Modulus (GPa)", value=0.06, step=0.01)
        sg = st.slider("Gas Saturation (Sg)", 0.0, 1.0, 0.05, 0.01)
    
    # Normalize saturations to sum to 1
    total_sat = sw + so + sg
    if total_sat > 0:
        sw_norm = sw / total_sat
        so_norm = so / total_sat
        sg_norm = sg / total_sat
    else:
        sw_norm, so_norm, sg_norm = 0, 0, 0
    
    # Calculate mixed fluid properties using Wood's equation
    rho_fl = sw_norm * rho_b + so_norm * rho_o + sg_norm * rho_g
    k_fl = 1 / (sw_norm/k_b + so_norm/k_o + sg_norm/k_g) if (sw_norm + so_norm + sg_norm) > 0 else 0
    
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

# Rock Physics Models
def frm(vp1, vs1, rho1, rho_f1, k_f1, rho_f2, k_f2, k0, mu0, phi):
    """Gassmann's Fluid Substitution"""
    vp1 = vp1/1000.  # Convert m/s to km/s
    vs1 = vs1/1000.
    mu1 = rho1 * vs1**2
    k_s1 = rho1 * vp1**2 - (4./3.) * mu1

    # Dry rock bulk modulus (Gassmann's equation)
    kdry = (k_s1*((phi*k0)/k_f1 + 1 - phi) - k0) / \
           ((phi*k0)/k_f1 + (k_s1/k0) - 1 - phi)

    # Apply Gassmann to get new fluid properties
    k_s2 = kdry + (1 - (kdry/k0))**2 / \
           ((phi/k_f2) + ((1 - phi)/k0) - (kdry/k0**2))
    rho2 = rho1 - phi*rho_f1 + phi*rho_f2
    mu2 = mu1  # Shear modulus unaffected by fluid change
    vp2 = np.sqrt((k_s2 + (4./3)*mu2) / rho2)
    vs2 = np.sqrt(mu2 / rho2)

    return vp2*1000, vs2*1000, rho2, k_s2

def critical_porosity_model(vp1, vs1, rho1, rho_f1, k_f1, rho_f2, k_f2, k0, mu0, phi, phi_c):
    """Critical Porosity Model (Nur et al.)"""
    vp1 = vp1/1000.
    vs1 = vs1/1000.
    mu1 = rho1*vs1**2.
    k_s1 = rho1*vp1**2 - (4./3.)*mu1
    
    # Modified dry rock modulus for critical porosity
    kdry = k0 * (1 - phi/phi_c)
    mudry = mu0 * (1 - phi/phi_c)
    
    # Gassmann substitution
    k_s2 = kdry + (1-(kdry/k0))**2/((phi/k_f2)+((1-phi)/k0)-(kdry/k0**2))
    rho2 = rho1-phi*rho_f1+phi*rho_f2
    mu2 = mudry  # Shear modulus not affected by fluid in Gassmann
    vp2 = np.sqrt((k_s2+(4./3)*mu2)/rho2)
    vs2 = np.sqrt(mu2/rho2)
    
    return vp2*1000, vs2*1000, rho2, k_s2

def hertz_mindlin_model(vp1, vs1, rho1, rho_f1, k_f1, rho_f2, k_f2, k0, mu0, phi, Cn, P):
    """Hertz-Mindlin contact theory model"""
    vp1 = vp1/1000.
    vs1 = vs1/1000.
    mu1 = rho1*vs1**2.
    k_s1 = rho1*vp1**2 - (4./3.)*mu1
    
    # Hertz-Mindlin dry rock moduli
    PR0 = (3*k0 - 2*mu0)/(6*k0 + 2*mu0)  # Poisson's ratio
    kdry = (Cn**2 * (1 - phi)**2 * P * mu0**2 / (18 * np.pi**2 * (1 - PR0)**2))**(1/3)
    mudry = ((2 + 3*PR0 - PR0**2)/(5*(2 - PR0))) * (
        (3*Cn**2 * (1 - phi)**2 * P * mu0**2)/(2 * np.pi**2 * (1 - PR0)**2)
    )**(1/3)
    
    # Gassmann substitution
    k_s2 = kdry + (1-(kdry/k0))**2/((phi/k_f2)+((1-phi)/k0)-(kdry/k0**2))
    rho2 = rho1-phi*rho_f1+phi*rho_f2
    mu2 = mudry
    vp2 = np.sqrt((k_s2+(4./3)*mu2)/rho2)
    vs2 = np.sqrt(mu2/rho2)
    
    return vp2*1000, vs2*1000, rho2, k_s2

def dvorkin_nur_model(vp1, vs1, rho1, rho_f1, k_f1, rho_f2, k_f2, k0, mu0, phi, Cn=9, P=10, phi_c=0.4):
    """Dvorkin-Nur Soft Sand model for unconsolidated sands"""
    vp1 = vp1/1000.  # Convert to km/s
    vs1 = vs1/1000.
    
    # Hertz-Mindlin for dry rock moduli at critical porosity
    PR0 = (3*k0 - 2*mu0)/(6*k0 + 2*mu0)  # Poisson's ratio
    
    # Dry rock moduli at critical porosity
    k_hm = (Cn**2 * (1-phi_c)**2 * P * mu0**2 / (18 * np.pi**2 * (1-PR0)**2))**(1/3)
    mu_hm = ((2 + 3*PR0 - PR0**2)/(5*(2-PR0))) * (
        (3*Cn**2 * (1-phi_c)**2 * P * mu0**2)/(2*np.pi**2*(1-PR0)**2)
    )**(1/3)
    
    # Modified Hashin-Shtrikman lower bound for dry rock
    k_dry = (phi/phi_c)/(k_hm + (4/3)*mu_hm) + (1 - phi/phi_c)/(k0 + (4/3)*mu_hm)
    k_dry = 1/k_dry - (4/3)*mu_hm
    k_dry = np.maximum(k_dry, 0)  # Ensure positive values
    
    mu_dry = (phi/phi_c)/(mu_hm + (mu_hm/6)*((9*k_hm + 8*mu_hm)/(k_hm + 2*mu_hm))) + \
             (1 - phi/phi_c)/(mu0 + (mu_hm/6)*((9*k_hm + 8*mu_hm)/(k_hm + 2*mu_hm)))
    mu_dry = 1/mu_dry - (mu_hm/6)*((9*k_hm + 8*mu_hm)/(k_hm + 2*mu_hm))
    mu_dry = np.maximum(mu_dry, 0)
    
    # Gassmann fluid substitution
    k_sat = k_dry + (1 - (k_dry/k0))**2 / ((phi/k_f2) + ((1-phi)/k0) - (k_dry/k0**2))
    rho2 = rho1 - phi*rho_f1 + phi*rho_f2
    vp2 = np.sqrt((k_sat + (4/3)*mu_dry)/rho2) * 1000  # Convert back to m/s
    vs2 = np.sqrt(mu_dry/rho2) * 1000
    
    return vp2, vs2, rho2, k_sat

def raymer_hunt_model(vp1, vs1, rho1, rho_f1, k_f1, rho_f2, k_f2, k0, mu0, phi):
    """Raymer-Hunt-Gardner empirical model"""
    # Empirical relationships for dry rock
    vp_dry = (1 - phi)**2 * np.sqrt(k0/rho1) + phi * np.sqrt(k_f1/rho_f1)
    vp_dry = vp_dry * 1000  # Convert to m/s
    
    # For saturated rock
    vp_sat = (1 - phi)**2 * np.sqrt(k0/rho1) + phi * np.sqrt(k_f2/rho_f2)
    vp_sat = vp_sat * 1000
    
    # VS is less affected by fluids (use empirical relationship)
    vs_sat = vs1 * (1 - 1.5*phi)  # Simple porosity correction
    
    # Density calculation
    rho2 = rho1 - phi*rho_f1 + phi*rho_f2
    
    return vp_sat, vs_sat, rho2, None  # No bulk modulus returned in this model

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

# Calculate reflection coefficients
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

# Fit AVO curve to get intercept and gradient
def fit_avo_curve(angles, rc_values):
    """Fit a line to AVO response to get intercept and gradient"""
    def linear_func(x, intercept, gradient):
        return intercept + gradient * np.sin(np.radians(x))**2
    
    try:
        popt, pcov = curve_fit(linear_func, angles, rc_values)
        intercept, gradient = popt
        return intercept, gradient, np.sqrt(np.diag(pcov))
    except:
        return np.nan, np.nan, (np.nan, np.nan)

# Monte Carlo simulation for uncertainty analysis
def monte_carlo_simulation(logs, model_func, params, iterations=100):
    """Perform Monte Carlo simulation for uncertainty analysis"""
    results = {
        'VP': [], 'VS': [], 'RHO': [], 
        'IP': [], 'VPVS': [], 'Intercept': [], 
        'Gradient': [], 'Fluid_Factor': []
    }
    
    for _ in range(iterations):
        # Perturb input parameters with normal distribution
        perturbed_params = {}
        for param, (mean, std) in params.items():
            perturbed_params[param] = np.random.normal(mean, std) if std > 0 else mean
        
        # Apply model with perturbed parameters
        vp, vs, rho, _ = model_func(**perturbed_params)
        
        # Calculate derived properties
        ip = vp * rho
        vpvs = vp / vs
        
        # Calculate Smith-Gidlow attributes (using mean values for simplicity)
        vp_upper = logs.VP.mean()
        vs_upper = logs.VS.mean()
        rho_upper = logs.RHO.mean()
        intercept, gradient, fluid_factor = smith_gidlow(vp_upper, vp, vs_upper, vs, rho_upper, rho)
        
        # Store results
        results['VP'].append(vp)
        results['VS'].append(vs)
        results['RHO'].append(rho)
        results['IP'].append(ip)
        results['VPVS'].append(vpvs)
        results['Intercept'].append(intercept)
        results['Gradient'].append(gradient)
        results['Fluid_Factor'].append(fluid_factor)
    
    return results

# Create interactive crossplot with improved error handling
def create_interactive_crossplot(logs):
    """Create interactive Bokeh crossplot with proper error handling"""
    try:
        # Define litho-fluid class labels
        lfc_labels = ['Undefined', 'Brine', 'Oil', 'Gas', 'Shale']
        
        # Handle NaN values and ensure integers
        if 'LFC_B' not in logs.columns:
            logs['LFC_B'] = 0
        logs['LFC_B'] = logs['LFC_B'].fillna(0).clip(0, 4).astype(int)
        
        # Create labels - handle any unexpected values gracefully
        logs['LFC_Label'] = logs['LFC_B'].apply(
            lambda x: lfc_labels[x] if x in range(len(lfc_labels)) else 'Undefined'
        )
        
        # Filter out NaN values and ensure numeric data
        plot_data = logs[['IP', 'VPVS', 'LFC_Label', 'DEPTH']].dropna()
        plot_data = plot_data[plot_data.apply(lambda x: pd.to_numeric(x, errors='coerce')).notnull().all(axis=1)]
        
        if len(plot_data) == 0:
            st.warning("No valid data available for crossplot - check your input data")
            return None
            
        # Create ColumnDataSource
        source = ColumnDataSource(plot_data)
        
        # Get unique labels present in data
        unique_labels = sorted(plot_data['LFC_Label'].unique())
        
        # Create figure
        p = figure(width=800, height=500, 
                  tools="box_select,lasso_select,pan,wheel_zoom,box_zoom,reset",
                  title="IP vs Vp/Vs Crossplot")
        
        # Create color map based on actual labels present
        if len(unique_labels) > 0:
            color_map = factor_cmap('LFC_Label', 
                                  palette=Category10[len(unique_labels)], 
                                  factors=unique_labels)
            
            # Create scatter plot
            scatter = p.scatter('IP', 'VPVS', source=source, size=5,
                              color=color_map, legend_field='LFC_Label',
                              alpha=0.6)
            
            # Configure axes and legend
            p.xaxis.axis_label = 'IP (m/s*g/cc)'
            p.yaxis.axis_label = 'Vp/Vs'
            p.legend.title = 'Litho-Fluid Class'
            p.legend.location = "top_right"
            p.legend.click_policy = "hide"
            
            # Add hover tool
            hover = HoverTool(tooltips=[
                ("Depth", "@DEPTH{0.2f}"),
                ("IP", "@IP{0.2f}"),
                ("Vp/Vs", "@VPVS{0.2f}"),
                ("Class", "@LFC_Label")
            ])
            p.add_tools(hover)
            
            return p
        else:
            st.warning("No valid class labels found for crossplot")
            return None
            
    except Exception as e:
        st.error(f"Error creating interactive crossplot: {str(e)}")
        return None

# Main processing function with error handling
@handle_errors
def process_data(uploaded_file, model_choice, include_uncertainty=False, mc_iterations=100, **kwargs):
    # Read and validate data
    if isinstance(uploaded_file, str):
        logs = pd.read_csv(uploaded_file)
    else:
        # Reset file pointer if it's a file upload object
        uploaded_file.seek(0)
        logs = pd.read_csv(uploaded_file)
    
    required_columns = {'DEPTH', 'VP', 'VS', 'RHO', 'VSH', 'SW', 'PHI'}
    if not required_columns.issubset(logs.columns):
        missing = required_columns - set(logs.columns)
        raise ValueError(f"Missing required columns: {missing}")
    
    # Skip fluid substitution for RPT models (they're visualization-only)
    if model_choice in ["Soft Sand RPT (rockphypy)", "Stiff Sand RPT (rockphypy)"]:
        # Just add placeholder columns for consistency
        for case in ['B', 'O', 'G']:
            logs[f'VP_FRM{case}'] = logs.VP
            logs[f'VS_FRM{case}'] = logs.VS
            logs[f'RHO_FRM{case}'] = logs.RHO
            logs[f'IP_FRM{case}'] = logs.VP * logs.RHO
            logs[f'VPVS_FRM{case}'] = logs.VP/logs.VS
            logs[f'LFC_{case}'] = 0  # Default to undefined
        
        return logs, None  # No MC results for RPT models
    
    # Extract parameters from kwargs with defaults
    rho_qz = kwargs.get('rho_qz', 2.65)
    k_qz = kwargs.get('k_qz', 37.0)
    mu_qz = kwargs.get('mu_qz', 44.0)
    rho_sh = kwargs.get('rho_sh', 2.81)
    k_sh = kwargs.get('k_sh', 15.0)
    mu_sh = kwargs.get('mu_sh', 5.0)
    rho_b = kwargs.get('rho_b', 1.09)
    k_b = kwargs.get('k_b', 2.8)
    rho_o = kwargs.get('rho_o', 0.78)
    k_o = kwargs.get('k_o', 0.94)
    rho_g = kwargs.get('rho_g', 0.25)
    k_g = kwargs.get('k_g', 0.06)
    sand_cutoff = kwargs.get('sand_cutoff', 0.12)
    sw_norm = kwargs.get('sw_norm', 0.8)
    so_norm = kwargs.get('so_norm', 0.15)
    sg_norm = kwargs.get('sg_norm', 0.05)
    
    # VRH function
    def vrh(volumes, k, mu):
        f = np.array(volumes).T
        k = np.resize(np.array(k), np.shape(f))
        mu = np.resize(np.array(mu), np.shape(f))

        k_u = np.sum(f*k, axis=1)
        k_l = 1. / np.sum(f/k, axis=1)
        mu_u = np.sum(f*mu, axis=1)
        mu_l = 1. / np.sum(f/mu, axis=1)
        k0 = (k_u+k_l)/2.
        mu0 = (mu_u+mu_l)/2.
        return k_u, k_l, mu_u, mu_l, k0, mu0

    # Process data
    shale = logs.VSH.values
    sand = 1 - shale - logs.PHI.values
    shaleN = shale/(shale+sand)
    sandN = sand/(shale+sand)
    k_u, k_l, mu_u, mu_l, k0, mu0 = vrh([shaleN, sandN], [k_sh, k_qz], [mu_sh, mu_qz])

    # Calculate mixed fluid properties using interactive saturations
    rho_fl = sw_norm * rho_b + so_norm * rho_o + sg_norm * rho_g
    k_fl = 1 / (sw_norm/k_b + so_norm/k_o + sg_norm/k_g) if (sw_norm + so_norm + sg_norm) > 0 else 0

    # Select model function and prepare arguments
    if model_choice == "Gassmann's Fluid Substitution":
        def model_func(rho_f1, k_f1, rho_f2, k_f2, k0, mu0, phi):
            return frm(logs.VP, logs.VS, logs.RHO, rho_f1, k_f1, rho_f2, k_f2, k0, mu0, phi)
    elif model_choice == "Critical Porosity Model (Nur)":
        def model_func(rho_f1, k_f1, rho_f2, k_f2, k0, mu0, phi, phi_c):
            return critical_porosity_model(logs.VP, logs.VS, logs.RHO, rho_f1, k_f1, rho_f2, k_f2, k0, mu0, phi, phi_c)
    elif model_choice == "Contact Theory (Hertz-Mindlin)":
        def model_func(rho_f1, k_f1, rho_f2, k_f2, k0, mu0, phi, Cn, P):
            return hertz_mindlin_model(logs.VP, logs.VS, logs.RHO, rho_f1, k_f1, rho_f2, k_f2, k0, mu0, phi, Cn, P)
    elif model_choice == "Dvorkin-Nur Soft Sand Model":
        def model_func(rho_f1, k_f1, rho_f2, k_f2, k0, mu0, phi, Cn, P, phi_c):
            return dvorkin_nur_model(logs.VP, logs.VS, logs.RHO, rho_f1, k_f1, rho_f2, k_f2, k0, mu0, phi, Cn, P, phi_c)
    elif model_choice == "Raymer-Hunt-Gardner Model":
        def model_func(rho_f1, k_f1, rho_f2, k_f2, k0, mu0, phi):
            return raymer_hunt_model(logs.VP, logs.VS, logs.RHO, rho_f1, k_f1, rho_f2, k_f2, k0, mu0, phi)

    # Apply selected model with all required parameters
    if model_choice == "Gassmann's Fluid Substitution":
        vp2, vs2, rho2, k2 = model_func(rho_b, k_b, rho_fl, k_fl, k0, mu0, logs.PHI)
    elif model_choice == "Critical Porosity Model (Nur)":
        critical_porosity = kwargs.get('critical_porosity', 0.4)
        vp2, vs2, rho2, k2 = model_func(rho_b, k_b, rho_fl, k_fl, k0, mu0, logs.PHI, critical_porosity)
    elif model_choice == "Contact Theory (Hertz-Mindlin)":
        coordination_number = kwargs.get('coordination_number', 9)
        effective_pressure = kwargs.get('effective_pressure', 10)
        vp2, vs2, rho2, k2 = model_func(rho_b, k_b, rho_fl, k_fl, k0, mu0, logs.PHI, coordination_number, effective_pressure)
    elif model_choice == "Dvorkin-Nur Soft Sand Model":
        coordination_number = kwargs.get('coordination_number', 9)
        effective_pressure = kwargs.get('effective_pressure', 10)
        critical_porosity = kwargs.get('critical_porosity', 0.4)
        vp2, vs2, rho2, k2 = model_func(rho_b, k_b, rho_fl, k_fl, k0, mu0, logs.PHI, coordination_number, effective_pressure, critical_porosity)
    elif model_choice == "Raymer-Hunt-Gardner Model":
        vp2, vs2, rho2, k2 = model_func(rho_b, k_b, rho_fl, k_fl, k0, mu0, logs.PHI)

    # Litho-fluid classification
    brine_sand = ((logs.VSH <= sand_cutoff) & (logs.SW >= 0.65))
    oil_sand = ((logs.VSH <= sand_cutoff) & (logs.SW < 0.65))
    shale_flag = (logs.VSH > sand_cutoff)

    # Add results to logs
    logs['VP_FRM'] = logs.VP
    logs['VS_FRM'] = logs.VS
    logs['RHO_FRM'] = logs.RHO
    logs['VP_FRM'][brine_sand|oil_sand] = vp2[brine_sand|oil_sand]
    logs['VS_FRM'][brine_sand|oil_sand] = vs2[brine_sand|oil_sand]
    logs['RHO_FRM'][brine_sand|oil_sand] = rho2[brine_sand|oil_sand]
    logs['IP_FRM'] = logs['VP_FRM']*logs['RHO_FRM']
    logs['IS_FRM'] = logs['VS_FRM']*logs['RHO_FRM']
    logs['VPVS_FRM'] = logs['VP_FRM']/logs['VS_FRM']

    # LFC flags
    temp_lfc = np.zeros(np.shape(logs.VSH))
    temp_lfc[brine_sand.values | oil_sand.values] = 1  # Brine
    temp_lfc[shale_flag.values] = 4  # Shale
    logs['LFC'] = temp_lfc

    # Uncertainty analysis if enabled
    mc_results = None
    if include_uncertainty:
        # Define parameter distributions
        params = {
            'rho_f1': (rho_b, 0.05),
            'k_f1': (k_b, 0.1),
            'rho_f2': (rho_fl, 0.05),
            'k_f2': (k_fl, 0.1),
            'k0': (k0.mean(), 0.1 * k0.mean()),  # 10% uncertainty in mineral moduli
            'mu0': (mu0.mean(), 0.1 * mu0.mean()),
            'phi': (logs.PHI.mean(), 0.05)  # 5% porosity uncertainty
        }
        
        # Add model-specific parameters
        if model_choice == "Critical Porosity Model (Nur)":
            params['phi_c'] = (critical_porosity, 0.01)
        elif model_choice in ["Contact Theory (Hertz-Mindlin)", "Dvorkin-Nur Soft Sand Model"]:
            params['Cn'] = (coordination_number, 1)
            params['P'] = (effective_pressure, 5)
            if model_choice == "Dvorkin-Nur Soft Sand Model":
                params['phi_c'] = (critical_porosity, 0.01)
        
        # Run Monte Carlo simulation
        mc_results = monte_carlo_simulation(logs, model_func, params, mc_iterations)

    return logs, mc_results

# Download link generator
def get_table_download_link(df, filename="results.csv"):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">Download CSV File</a>'
    return href

# Enhanced Time-Frequency Analysis Functions
def perform_time_frequency_analysis(logs, angles, wavelet_freq, cwt_scales, cwt_wavelet, middle_top, middle_bot):
    """Perform comprehensive time-frequency analysis on synthetic gathers"""
    # Generate synthetic gathers for all cases
    wlt_time, wlt_amp = ricker_wavelet(wavelet_freq)
    t_samp = np.arange(0, 0.5, 0.001)  # Higher resolution for better CWT
    t_middle = 0.2
    
    # Get average properties for upper layer (shale)
    vp_upper = logs.loc[(logs.DEPTH >= middle_top - (middle_bot-middle_top)), 'VP'].values.mean()
    vs_upper = logs.loc[(logs.DEPTH >= middle_top - (middle_bot-middle_top)), 'VS'].values.mean()
    rho_upper = logs.loc[(logs.DEPTH >= middle_top - (middle_bot-middle_top)), 'RHO'].values.mean()
    
    # Get average properties for middle layer (sand with fluid substitution)
    vp_middle = logs.loc[(logs.DEPTH >= middle_top) & (logs.DEPTH <= middle_bot), 'VP_FRM'].values.mean()
    vs_middle = logs.loc[(logs.DEPTH >= middle_top) & (logs.DEPTH <= middle_bot), 'VS_FRM'].values.mean()
    rho_middle = logs.loc[(logs.DEPTH >= middle_top) & (logs.DEPTH <= middle_bot), 'RHO_FRM'].values.mean()
    
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
    
    return np.array(syn_gather), t_samp

def plot_frequency_analysis(syn_gather, t_samp, angles, wavelet_freq, time_range, freq_range):
    """Plot frequency domain analysis (FFT) with frequency-based color coding"""
    st.subheader("Frequency Domain Analysis (FFT)")
    
    fig_freq, ax_freq = plt.subplots(figsize=(10, 5))
    
    # Time range filtering
    time_mask = (t_samp >= time_range[0]) & (t_samp <= time_range[1])
    t_samp_filtered = t_samp[time_mask]
    
    # Compute FFT parameters
    n = syn_gather.shape[1]  # Number of time samples
    dt = t_samp[1] - t_samp[0]
    freqs = np.fft.rfftfreq(n, dt)  # Frequency bins
    
    # Initialize array to store frequency spectra (time vs frequency)
    freq_spectra = np.zeros((len(t_samp_filtered), len(freqs)))
    
    # Calculate FFT for each time sample across all angles
    for i, t in enumerate(t_samp_filtered):
        time_idx = np.where(t_samp == t)[0][0]
        time_slice = syn_gather[:, time_idx]
        
        # Apply Hanning window to reduce spectral leakage
        window = np.hanning(len(time_slice))
        windowed_signal = time_slice * window
        
        # Compute FFT and take magnitude
        spectrum = np.abs(np.fft.rfft(windowed_signal))
        
        # Handle case where spectrum length doesn't match frequency bins
        min_len = min(len(spectrum), len(freqs))
        freq_spectra[i, :min_len] = spectrum[:min_len]
    
    # Normalize for better visualization
    if np.max(freq_spectra) > 0:
        freq_spectra = freq_spectra / np.max(freq_spectra)
    
    # Create frequency-based color coding
    X, Y = np.meshgrid(freqs, t_samp_filtered)
    
    # Plot with frequency-based color
    im = ax_freq.pcolormesh(
        X, Y, freq_spectra,
        cmap='jet',
        shading='auto',
        vmin=0,
        vmax=1
    )
    
    ax_freq.set_title("Frequency Spectrum")
    ax_freq.set_xlabel("Frequency (Hz)")
    ax_freq.set_ylabel("Time (s)")
    ax_freq.set_ylim(time_range[1], time_range[0])  # Inverted for seismic display
    ax_freq.set_xlim(freq_range[0], freq_range[1])  # Focus on relevant frequencies
    
    plt.colorbar(im, ax=ax_freq, label='Normalized Amplitude')
    plt.tight_layout()
    st.pyplot(fig_freq)

def plot_cwt_analysis(syn_gather, t_samp, angles, cwt_scales, cwt_wavelet, wavelet_freq, time_range, freq_range):
    """Plot CWT analysis with frequency-based color coding and extract 0.20s data"""
    st.subheader("Time-Frequency Analysis (CWT)")
    
    try:
        scales = np.arange(cwt_scales[0], cwt_scales[1]+1)
        # Convert scales to approximate frequencies
        freqs = pywt.scale2frequency(cwt_wavelet, scales) / (t_samp[1]-t_samp[0])
        
        fig_cwt, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 12))
        
        # Time range filtering
        time_mask = (t_samp >= time_range[0]) & (t_samp <= time_range[1])
        t_samp_filtered = t_samp[time_mask]
        
        # Initialize array to store CWT magnitudes (time vs frequency)
        cwt_magnitudes = np.zeros((len(t_samp_filtered), len(freqs)))
        
        for i, t in enumerate(t_samp_filtered):
            time_idx = np.where(t_samp == t)[0][0]
            trace = syn_gather[:, time_idx]  # All angles at this time sample
            if len(trace) == 0:
                continue
                
            coefficients, _ = pywt.cwt(trace, scales, cwt_wavelet, 
                                     sampling_period=t_samp[1]-t_samp[0])
            
            if coefficients.size > 0:
                # Sum across angles for each scale (store raw magnitudes)
                cwt_magnitudes[i, :] = np.sum(np.abs(coefficients), axis=1)
        
        if cwt_magnitudes.size == 0:
            st.warning("No valid CWT data available")
            return None
            
        # Find global max for consistent normalization in display (not used in exported data)
        global_max = np.max(cwt_magnitudes) if np.max(cwt_magnitudes) > 0 else 1
        
        # Create frequency-based color coding
        X, Y = np.meshgrid(freqs, t_samp_filtered)
        
        # Plot CWT magnitude with frequency-based color (inverted y-axis)
        im = ax1.pcolormesh(
            X, Y, cwt_magnitudes/global_max,  # Normalized only for display
            shading='auto',
            cmap='jet', 
            vmin=0, 
            vmax=1
        )
        ax1.set_title("CWT Magnitude")
        ax1.set_xlabel("Frequency (Hz)")
        ax1.set_ylabel("Time (s)")
        ax1.set_ylim(time_range[1], time_range[0])  # Inverted y-axis
        ax1.set_xlim(freq_range[0], freq_range[1])
        plt.colorbar(im, ax=ax1, label='Normalized Magnitude')
        
        # Plot time series at middle angle (inverted y-axis)
        mid_angle_idx = len(angles) // 2
        time_series = syn_gather[mid_angle_idx, time_mask]
        ax2.plot(t_samp_filtered, time_series, 'k-')
        ax2.set_title(f"Time Series (@ {angles[mid_angle_idx]}°)")
        ax2.set_xlabel("Time (s)")
        ax2.set_ylabel("Amplitude")
        ax2.grid(True)
        ax2.set_xlim(time_range[0], time_range[1])
        ax2.set_ylim(time_range[1], time_range[0])  # Inverted y-axis
        
        # Plot dominant frequency (inverted y-axis)
        if cwt_magnitudes.size > 0:
            max_freq_indices = np.argmax(cwt_magnitudes, axis=1)
            dominant_freqs = freqs[max_freq_indices]
            
            ax3.plot(t_samp_filtered, dominant_freqs, 'r-')
            ax3.set_title("Dominant Frequency")
            ax3.set_xlabel("Time (s)")
            ax3.set_ylabel("Frequency (Hz)")
            ax3.grid(True)
            ax3.set_ylim(freq_range[1], freq_range[0])  # Inverted y-axis
        
        plt.tight_layout()
        st.pyplot(fig_cwt)
        
        # Extract CWT magnitudes at t=0.20s (using raw magnitudes)
        time_target = 0.20
        time_idx = np.argmin(np.abs(t_samp_filtered - time_target))
        
        if time_idx < len(t_samp_filtered):
            # Plot Frequency vs. Magnitude at t=0.20s (using raw magnitudes)
            st.subheader("CWT Frequency vs. Magnitude at t=0.20s")
            fig_freq_mag, ax_freq_mag = plt.subplots(figsize=(10, 5))
            
            ax_freq_mag.plot(freqs, cwt_magnitudes[time_idx, :], 'b-')
            ax_freq_mag.set_xlabel("Frequency (Hz)")
            ax_freq_mag.set_ylabel("Magnitude (Raw Sum)")
            ax_freq_mag.set_title("CWT Magnitude Spectrum at t=0.20s")
            ax_freq_mag.grid(True)
            ax_freq_mag.set_xlim(freq_range[0], freq_range[1])
            
            st.pyplot(fig_freq_mag)
        
    except Exception as e:
        st.error(f"Error in CWT analysis: {str(e)}")

def plot_spectral_comparison(syn_gather, t_samp, angles, wavelet_freq, time_range, freq_range):
    """Plot spectral comparison at selected angles with frequency-based color coding"""
    st.subheader("Spectral Comparison at Selected Angles")
    
    selected_angles = st.multiselect(
        "Select angles to compare spectra",
        angles.tolist(),
        default=[angles[0], angles[len(angles)//2], angles[-1]],
        key='spectral_angles'
    )
    
    if selected_angles:
        # Time range filtering
        time_mask = (t_samp >= time_range[0]) & (t_samp <= time_range[1])
        t_samp_filtered = t_samp[time_mask]
        
        fig_compare = plt.figure(figsize=(12, 8))
        ax_compare = fig_compare.add_subplot(111, projection='3d')
        
        # Create colormap based on frequency
        norm = plt.Normalize(freq_range[0], freq_range[1])
        cmap = plt.get_cmap('jet')
        
        for angle in selected_angles:
            angle_idx = np.where(angles == angle)[0][0]
            trace = syn_gather[angle_idx, time_mask]
            
            # FFT
            spectrum = np.abs(np.fft.rfft(trace))
            freqs = np.fft.rfftfreq(len(trace), t_samp[1]-t_samp[0])
            
            # Filter frequencies
            freq_mask = (freqs >= freq_range[0]) & (freqs <= freq_range[1])
            freqs_filtered = freqs[freq_mask]
            spectrum_filtered = spectrum[freq_mask]
            
            # Normalize spectrum
            if np.max(spectrum_filtered) > 0:
                spectrum_filtered = spectrum_filtered / np.max(spectrum_filtered)
            
            # Create color array based on frequency
            colors = cmap(norm(freqs_filtered))
            
            # Plot each frequency component separately with its own color
            for i in range(len(freqs_filtered)-1):
                ax_compare.plot(
                    [angle, angle],  # X (angle)
                    [freqs_filtered[i], freqs_filtered[i+1]],  # Y (frequency)
                    [spectrum_filtered[i], spectrum_filtered[i+1]],  # Z (amplitude)
                    color=colors[i],
                    linewidth=2
                )
        
        ax_compare.set_title("3D Spectral Comparison (Color by Frequency)")
        ax_compare.set_xlabel("Angle (degrees)")
        ax_compare.set_ylabel("Frequency (Hz)")
        ax_compare.set_zlabel("Normalized Amplitude")
        ax_compare.set_xlim(min(selected_angles), max(selected_angles))
        ax_compare.set_ylim(freq_range[0], freq_range[1])
        ax_compare.set_zlim(0, 1)
        
        # Add colorbar
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        fig_compare.colorbar(sm, ax=ax_compare, label='Frequency (Hz)', shrink=0.6)
        
        st.pyplot(fig_compare)

# Main content area
if uploaded_file is not None:
    try:
        # Store the original file object
        original_file = uploaded_file
        
        # Process data with selected model
        logs, mc_results = process_data(
            original_file, 
            model_choice,
            include_uncertainty=include_uncertainty,
            mc_iterations=mc_iterations,
            rho_qz=rho_qz, k_qz=k_qz, mu_qz=mu_qz,
            rho_sh=rho_sh, k_sh=k_sh, mu_sh=mu_sh,
            rho_b=rho_b, k_b=k_b,
            rho_o=rho_o, k_o=k_o,
            rho_g=rho_g, k_g=k_g,
            sw_norm=sw_norm,
            so_norm=so_norm,
            sg_norm=sg_norm,
            sand_cutoff=sand_cutoff,
            critical_porosity=critical_porosity if 'critical_porosity' in locals() else None,
            coordination_number=coordination_number if 'coordination_number' in locals() else None,
            effective_pressure=effective_pressure if 'effective_pressure' in locals() else None
        )
        
        # Depth range selection
        st.header("Well Log Visualization")
        ztop, zbot = st.slider(
            "Select Depth Range", 
            float(logs.DEPTH.min()), 
            float(logs.DEPTH.max()), 
            (float(logs.DEPTH.min()), float(logs.DEPTH.max()))
        )
        
        # Visualization
        ccc = ['#B3B3B3','blue','green','red','#996633']
        cmap_facies = colors.ListedColormap(ccc[0:len(ccc)], 'indexed')

        # Create a filtered dataframe for the selected depth range
        ll = logs.loc[(logs.DEPTH>=ztop) & (logs.DEPTH<=zbot)]
        cluster = np.repeat(np.expand_dims(ll['LFC'].values,1), 100, 1)

        # Only show well log visualization for non-RPT models
        if model_choice not in ["Soft Sand RPT (rockphypy)", "Stiff Sand RPT (rockphypy)"]:
            # Create the well log figure
            fig, ax = plt.subplots(nrows=1, ncols=4, figsize=(12, 8))
            ax[0].plot(ll.VSH, ll.DEPTH, '-g', label='Vsh')
            ax[0].plot(ll.SW, ll.DEPTH, '-b', label='Sw')
            ax[0].plot(ll.PHI, ll.DEPTH, '-k', label='phi')
            ax[1].plot(ll.IP_FRM, ll.DEPTH, '-b', label='FRM Result')
            ax[1].plot(ll.IP, ll.DEPTH, '-', color='0.5', label='Original')
            ax[2].plot(ll.VPVS_FRM, ll.DEPTH, '-b', label='FRM Result')
            ax[2].plot(ll.VPVS, ll.DEPTH, '-', color='0.5', label='Original')
            im = ax[3].imshow(cluster, interpolation='none', aspect='auto', cmap=cmap_facies, vmin=0, vmax=4)

            cbar = plt.colorbar(im, ax=ax[3])
            cbar.set_label((12*' ').join(['undef', 'brine', 'oil', 'gas', 'shale']))
            cbar.set_ticks(range(0,5))
            cbar.set_ticklabels(['']*5)

            for i in ax[:-1]:
                i.set_ylim(ztop,zbot)
                i.invert_yaxis()
                i.grid()
                i.locator_params(axis='x', nbins=4)
            ax[0].legend(fontsize='small', loc='lower right')
            ax[1].legend(fontsize='small', loc='lower right')
            ax[2].legend(fontsize='small', loc='lower right')
            ax[0].set_xlabel("Vcl/phi/Sw"); ax[0].set_xlim(-.1,1.1)
            ax[1].set_xlabel("Ip [m/s*g/cc]"); ax[1].set_xlim(6000,15000)
            ax[2].set_xlabel("Vp/Vs"); ax[2].set_xlim(1.5,2)
            ax[3].set_xlabel('LFC')
            ax[1].set_yticklabels([]); ax[2].set_yticklabels([]); ax[3].set_yticklabels([]); ax[3].set_xticklabels([])
            
            # Display the well log plot
            st.pyplot(fig)

        # Interactive Crossplot
        st.header("Interactive Crossplots with Selection")
        crossplot = create_interactive_crossplot(logs)
        if crossplot:
            event_result = streamlit_bokeh_events(
                bokeh_plot=crossplot,
                events="SELECTION_CHANGED",
                key="crossplot",
                refresh_on_update=False,
                debounce_time=0,
                override_height=500
            )
        else:
            st.warning("Could not generate interactive crossplot due to data issues")
        
        # Original 2D crossplots
        st.header("2D Crossplots")
        fig2, ax2 = plt.subplots(figsize=(10, 5))
        ax2.scatter(logs.IP, logs.VPVS, 20, logs.LFC, marker='o', edgecolors='none', alpha=0.5, cmap=cmap_facies, vmin=0, vmax=4)
        ax2.set_xlim(3000,16000); ax2.set_ylim(1.5,3)
        ax2.set_title('Rock Physics Crossplot')
        ax2.set_xlabel('IP (m/s*g/cc)')
        ax2.set_ylabel('Vp/Vs')
        st.pyplot(fig2)

        # 3D Crossplot if enabled
        if show_3d_crossplot and model_choice not in ["Soft Sand RPT (rockphypy)", "Stiff Sand RPT (rockphypy)"]:
            st.header("3D Crossplot")
            fig3d = plt.figure(figsize=(10, 8))
            ax3d = fig3d.add_subplot(111, projection='3d')
            
            # Color by litho-fluid class
            colors = ['#B3B3B3','blue','green','red','#996633']
            for lfc in range(5):
                mask = logs['LFC'] == lfc
                ax3d.scatter(
                    logs.loc[mask, 'IP_FRM'],
                    logs.loc[mask, 'VPVS_FRM'],
                    logs.loc[mask, 'RHO_FRM'],
                    c=colors[lfc], label=f'Class {lfc}', alpha=0.5
                )
            
            ax3d.set_xlabel('IP (m/s*g/cc)')
            ax3d.set_ylabel('Vp/Vs')
            ax3d.set_zlabel('Density (g/cc)')
            ax3d.set_title('3D Rock Physics Crossplot')
            ax3d.legend()
            st.pyplot(fig3d)

        # Histograms if enabled
        if show_histograms and model_choice not in ["Soft Sand RPT (rockphypy)", "Stiff Sand RPT (rockphypy)"]:
            st.header("Property Distributions")
            fig_hist, ax_hist = plt.subplots(2, 2, figsize=(12, 8))
            
            ax_hist[0,0].hist(logs.IP_FRM, bins=30, alpha=0.5, label='FRM Result', color='blue')
            ax_hist[0,0].hist(logs.IP, bins=30, alpha=0.5, label='Original', color='0.5')
            ax_hist[0,0].set_xlabel('IP (m/s*g/cc)')
            ax_hist[0,0].set_ylabel('Frequency')
            ax_hist[0,0].legend()
            
            ax_hist[0,1].hist(logs.VPVS_FRM, bins=30, alpha=0.5, label='FRM Result', color='blue')
            ax_hist[0,1].hist(logs.VPVS, bins=30, alpha=0.5, label='Original', color='0.5')
            ax_hist[0,1].set_xlabel('Vp/Vs')
            ax_hist[0,1].legend()
            
            ax_hist[1,0].hist(logs.RHO_FRM, bins=30, color='blue', alpha=0.7)
            ax_hist[1,0].hist(logs.RHO, bins=30, color='0.5', alpha=0.7)
            ax_hist[1,0].set_xlabel('Density (g/cc)')
            ax_hist[1,0].set_ylabel('Frequency')
            ax_hist[1,0].legend(['FRM Result', 'Original'])
            
            ax_hist[1,1].hist(logs.LFC, bins=[0,1,2,3,4,5], alpha=0.5, rwidth=0.8, align='left')
            ax_hist[1,1].set_xlabel('Litho-Fluid Class')
            ax_hist[1,1].set_xticks([0.5,1.5,2.5,3.5,4.5])
            ax_hist[1,1].set_xticklabels(['Undef','Brine','Oil','Gas','Shale'])
            
            plt.tight_layout()
            st.pyplot(fig_hist)

        # AVO Modeling
        if model_choice not in ["Soft Sand RPT (rockphypy)", "Stiff Sand RPT (rockphypy)"]:
            st.header("AVO Modeling")
            middle_top = ztop + (zbot - ztop) * 0.4
            middle_bot = ztop + (zbot - ztop) * 0.6
            
            wlt_time, wlt_amp = ricker_wavelet(wavelet_freq)
            t_samp = np.arange(0, 0.5, 0.0001)
            t_middle = 0.2
            
            fig3, (ax_wavelet, ax_avo) = plt.subplots(1, 2, figsize=(12, 5), gridspec_kw={'width_ratios': [1, 2]})
            
            ax_wavelet.plot(wlt_time, wlt_amp, color='purple', linewidth=2)
            ax_wavelet.fill_between(wlt_time, wlt_amp, color='purple', alpha=0.3)
            ax_wavelet.set_title(f"Wavelet ({wavelet_freq} Hz)")
            ax_wavelet.set_xlabel("Time (s)")
            ax_wavelet.set_ylabel("Amplitude")
            ax_wavelet.grid(True)
            
            rc_min, rc_max = st.slider(
                "Reflection Coefficient Range",
                -0.5, 0.5, (-0.2, 0.2),
                step=0.01,
                key='rc_range'
            )
            
            angles = np.arange(min_angle, max_angle + 1, angle_step)
            
            # Get average properties for upper layer (shale)
            vp_upper = logs.loc[(logs.DEPTH >= middle_top - (middle_bot-middle_top)), 'VP'].values.mean()
            vs_upper = logs.loc[(logs.DEPTH >= middle_top - (middle_bot-middle_top)), 'VS'].values.mean()
            rho_upper = logs.loc[(logs.DEPTH >= middle_top - (middle_bot-middle_top)), 'RHO'].values.mean()
            
            # Get average properties for middle layer (sand with fluid substitution)
            vp_middle = logs.loc[(logs.DEPTH >= middle_top) & (logs.DEPTH <= middle_bot), 'VP_FRM'].values.mean()
            vs_middle = logs.loc[(logs.DEPTH >= middle_top) & (logs.DEPTH <= middle_bot), 'VS_FRM'].values.mean()
            rho_middle = logs.loc[(logs.DEPTH >= middle_top) & (logs.DEPTH <= middle_bot), 'RHO_FRM'].values.mean()
            
            # Calculate reflection coefficients
            rc = []
            for angle in angles:
                rc.append(calculate_reflection_coefficients(
                    vp_upper, vp_middle, vs_upper, vs_middle, rho_upper, rho_middle, angle
                ))
            
            # Fit AVO curve to get intercept and gradient
            intercept, gradient, _ = fit_avo_curve(angles, rc)
            fluid_factor = intercept + 1.16 * (vp_upper/vs_upper) * (intercept - gradient)
            
            # Plot AVO curve
            ax_avo.plot(angles, rc, "b-", label="AVO Response")
            ax_avo.set_title("AVO Reflection Coefficients (Middle Interface)")
            ax_avo.set_xlabel("Angle (degrees)")
            ax_avo.set_ylabel("Reflection Coefficient")
            ax_avo.set_ylim(rc_min, rc_max)
            ax_avo.grid(True)
            ax_avo.legend()
            
            st.pyplot(fig3)

            # Smith-Gidlow AVO Analysis
            if show_smith_gidlow:
                st.header("Smith-Gidlow AVO Attributes")
                
                # Display attributes
                st.markdown(f"""
                - **Intercept (A):** {intercept:.4f}
                - **Gradient (B):** {gradient:.4f}
                - **Fluid Factor:** {fluid_factor:.4f}
                """)
                
                # Plot intercept vs gradient
                fig_sg, ax_sg = plt.subplots(figsize=(8, 6))
                ax_sg.scatter(intercept, gradient, color='blue', s=100)
                ax_sg.text(intercept, gradient, f"Sw={sw_norm:.2f}\nSo={so_norm:.2f}\nSg={sg_norm:.2f}", 
                          fontsize=9, ha='right', va='bottom')
                
                # Add background classification
                x = np.linspace(-0.5, 0.5, 100)
                ax_sg.plot(x, -x, 'k--', alpha=0.3)  # Typical brine line
                ax_sg.plot(x, -4*x, 'k--', alpha=0.3)  # Typical gas line
                
                ax_sg.set_xlabel('Intercept (A)')
                ax_sg.set_ylabel('Gradient (B)')
                ax_sg.set_title('Smith-Gidlow AVO Crossplot')
                ax_sg.grid(True)
                ax_sg.axhline(0, color='k', alpha=0.3)
                ax_sg.axvline(0, color='k', alpha=0.3)
                ax_sg.set_xlim(-0.3, 0.3)
                ax_sg.set_ylim(-0.3, 0.3)
                
                st.pyplot(fig_sg)
                
                # Fluid Factor analysis
                st.subheader("Fluid Factor Analysis")
                st.markdown(f"""
                The fluid factor value of **{fluid_factor:.4f}** suggests:
                - Positive values may indicate hydrocarbon presence
                - Negative values typically indicate brine-filled rocks
                - The magnitude relates to fluid contrast
                """)

            # Time-Frequency Analysis of Synthetic Gathers
            st.header("Time-Frequency Analysis of Synthetic Gathers")
            
            # Time-Frequency Analysis Controls
            st.sidebar.header("Time-Frequency Analysis Controls")
            
            # Time range slider
            default_time_min = 0.15
            default_time_max = 0.25
            time_range = st.sidebar.slider(
                "Time Range (s)",
                float(t_samp[0]), float(t_samp[-1]),
                (default_time_min, default_time_max),
                step=0.01
            )
            
            # Frequency range slider
            max_freq = wavelet_freq * 3
            freq_range = st.sidebar.slider(
                "Frequency Range (Hz)",
                0, int(max_freq * 1.5),
                (0, int(max_freq)),
                step=5
            )
            
            # Generate synthetic gathers for time-frequency analysis
            syn_gather, t_samp = perform_time_frequency_analysis(
                logs, angles, wavelet_freq, cwt_scales, cwt_wavelet, middle_top, middle_bot
            )
            
            # Plot frequency domain analysis
            plot_frequency_analysis(syn_gather, t_samp, angles, wavelet_freq, time_range, freq_range)
            
            # Plot CWT analysis
            plot_cwt_analysis(syn_gather, t_samp, angles, cwt_scales, cwt_wavelet, wavelet_freq, time_range, freq_range)
            
            # Plot spectral comparison
            plot_spectral_comparison(syn_gather, t_samp, angles, wavelet_freq, time_range, freq_range)

            # Synthetic gathers
            st.header("Synthetic Seismic Gathers (Middle Interface)")
            time_min, time_max = st.slider(
                "Time Range for Synthetic Gathers (s)",
                0.0, 0.5, (0.15, 0.25),
                step=0.01,
                key='time_range'
            )
            
            fig4, ax4 = plt.subplots(figsize=(10, 5))
            
            extent = [angles[0], angles[-1], t_samp[-1], t_samp[0]]
            im = ax4.imshow(syn_gather.T, aspect='auto', extent=extent,
                          cmap=selected_cmap, vmin=-np.max(np.abs(syn_gather)), 
                          vmax=np.max(np.abs(syn_gather)))
            
            props_text = f"Vp: {vp_middle:.0f} m/s\n" \
                        f"Vs: {vs_middle:.0f} m/s\n" \
                        f"Rho: {rho_middle:.2f} g/cc\n" \
                        f"Sw: {sw_norm:.2f}, So: {so_norm:.2f}, Sg: {sg_norm:.2f}"
            ax4.text(0.05, 0.95, props_text, transform=ax4.transAxes,
                     fontsize=9, verticalalignment='top', bbox=dict(facecolor='white', alpha=0.7))
            
            ax4.set_title("Synthetic Gather", fontweight='bold')
            ax4.set_xlabel("Angle (degrees)")
            ax4.set_ylabel("Time (s)")
            ax4.set_ylim(time_max, time_min)
            
            plt.colorbar(im, ax=ax4, label='Amplitude')
            plt.tight_layout()
            st.pyplot(fig4)

        # Rock Physics Templates (RPT)
        if model_choice in ["Soft Sand RPT (rockphypy)", "Stiff Sand RPT (rockphypy)"] and rockphypy_available:
            st.header("Rock Physics Templates (RPT) with Gassmann Fluid Substitution")
            
            # Model parameters
            D0, K0, G0 = 2.65, 36.6, 45  # Mineral properties (quartz)
            Db, Kb = rho_b, k_b            # Brine from user inputs
            Do, Ko = rho_o, k_o            # Oil from user inputs
            Dg, Kg = rho_g, k_g            # Gas from user inputs
            
            phi = np.linspace(0.1, rpt_phi_c, 10)  # Porosity range
            sw = np.linspace(0, 1, 5)              # Water saturation
            
            # Function to generate and display RPT plot with Gassmann points
            def plot_rpt_with_gassmann(title, fluid='gas'):
                plt.figure(figsize=(8, 6))
                
                # Generate RPT background
                if model_choice == "Soft Sand RPT (rockphypy)":
                    Kdry, Gdry = GM.softsand(K0, G0, phi, rpt_phi_c, rpt_Cn, rpt_sigma, f=0.5)
                else:
                    Kdry, Gdry = GM.stiffsand(K0, G0, phi, rpt_phi_c, rpt_Cn, rpt_sigma, f=0.5)
                
                if fluid == 'gas':
                    QI.plot_rpt(Kdry, Gdry, K0, D0, Kb, Db, Kg, Dg, phi, sw)
                else:
                    QI.plot_rpt(Kdry, Gdry, K0, D0, Kb, Db, Ko, Do, phi, sw)
                
                plt.title(f"{model_choice.split(' ')[0]} RPT - {fluid.capitalize()} Case")
                
                # Add Gassmann fluid substitution points from logs
                if original_file is not None:
                    try:
                        # Reset file pointer to beginning
                        original_file.seek(0)
                        
                        # Process data with Gassmann model
                        logs_gassmann, _ = process_data(
                            original_file, 
                            "Gassmann's Fluid Substitution",
                            include_uncertainty=False,
                            mc_iterations=1,
                            rho_qz=rho_qz, k_qz=k_qz, mu_qz=mu_qz,
                            rho_sh=rho_sh, k_sh=k_sh, mu_sh=mu_sh,
                            rho_b=rho_b, k_b=k_b,
                            rho_o=rho_o, k_o=k_o,
                            rho_g=rho_g, k_g=k_g,
                            sw_norm=sw_norm,
                            so_norm=so_norm,
                            sg_norm=sg_norm,
                            sand_cutoff=sand_cutoff
                        )
                        
                        # Filter sand intervals (VSH < sand_cutoff)
                        sand_mask = logs_gassmann['VSH'] <= sand_cutoff
                        
                        # Select appropriate columns based on fluid case
                        ip = logs_gassmann.loc[sand_mask, 'IP_FRM']
                        vpvs = logs_gassmann.loc[sand_mask, 'VPVS_FRM']
                        
                        # Plot the points if we have valid data
                        if len(ip) > 0 and len(vpvs) > 0:
                            plt.scatter(ip, vpvs, c='k', s=20, alpha=0.7, label=f'Current (Sw={sw_norm:.2f})', edgecolors='k', linewidths=0.5)
                            plt.legend()
                        else:
                            st.warning("No valid sand points found for plotting")
                        
                    except Exception as e:
                        st.warning(f"Could not plot Gassmann points: {str(e)}")
                
                # Save plot to a buffer and display in Streamlit
                buf = BytesIO()
                plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
                buf.seek(0)
                st.image(buf, use_column_width=True)
                plt.close()
            
            # Display Gas Case RPT with Gassmann points
            st.subheader("Gas Case RPT with Gassmann Fluid Substitution")
            plot_rpt_with_gassmann("Gas Case RPT", fluid='gas')
            
            # Display Oil Case RPT with Gassmann points
            st.subheader("Oil Case RPT with Gassmann Fluid Substitution")
            plot_rpt_with_gassmann("Oil Case RPT", fluid='oil')

        # Uncertainty Analysis Results
        if include_uncertainty and mc_results and model_choice not in ["Soft Sand RPT (rockphypy)", "Stiff Sand RPT (rockphypy)"]:
            st.header("Uncertainty Analysis Results")
            
            # Create summary statistics
            mc_df = pd.DataFrame(mc_results)
            summary_stats = mc_df.describe().T
            
            st.subheader("Monte Carlo Simulation Statistics")
            if not summary_stats.empty:
                numeric_cols = summary_stats.select_dtypes(include=[np.number]).columns
                st.dataframe(summary_stats.style.format("{:.2f}", subset=numeric_cols))
            else:
                st.warning("No statistics available - check your Monte Carlo simulation parameters")
            
            # Plot uncertainty distributions
            st.subheader("Property Uncertainty Distributions")
            fig_unc, ax_unc = plt.subplots(2, 2, figsize=(12, 8))
            
            # VP distribution
            ax_unc[0,0].hist(mc_results['VP'], bins=30, color='blue', alpha=0.7)
            ax_unc[0,0].set_xlabel('VP (m/s)')
            ax_unc[0,0].set_ylabel('Frequency')
            ax_unc[0,0].set_title('P-wave Velocity Distribution')
            
            # VS distribution
            ax_unc[0,1].hist(mc_results['VS'], bins=30, color='green', alpha=0.7)
            ax_unc[0,1].set_xlabel('VS (m/s)')
            ax_unc[0,1].set_title('S-wave Velocity Distribution')
            
            # IP distribution
            ax_unc[1,0].hist(mc_results['IP'], bins=30, color='red', alpha=0.7)
            ax_unc[1,0].set_xlabel('IP (m/s*g/cc)')
            ax_unc[1,0].set_ylabel('Frequency')
            ax_unc[1,0].set_title('Acoustic Impedance Distribution')
            
            # Vp/Vs distribution
            ax_unc[1,1].hist(mc_results['VPVS'], bins=30, color='purple', alpha=0.7)
            ax_unc[1,1].set_xlabel('Vp/Vs')
            ax_unc[1,1].set_title('Vp/Vs Ratio Distribution')
            
            plt.tight_layout()
            st.pyplot(fig_unc)
            
            # AVO attribute uncertainty
            st.subheader("AVO Attribute Uncertainty")
            fig_avo_unc, ax_avo_unc = plt.subplots(1, 3, figsize=(15, 4))
            
            # Intercept distribution
            ax_avo_unc[0].hist(mc_results['Intercept'], bins=30, color='blue', alpha=0.7)
            ax_avo_unc[0].set_xlabel('Intercept')
            ax_avo_unc[0].set_ylabel('Frequency')
            ax_avo_unc[0].set_title('Intercept Distribution')
            
            # Gradient distribution
            ax_avo_unc[1].hist(mc_results['Gradient'], bins=30, color='green', alpha=0.7)
            ax_avo_unc[1].set_xlabel('Gradient')
            ax_avo_unc[1].set_title('Gradient Distribution')
            
            # Fluid Factor distribution
            ax_avo_unc[2].hist(mc_results['Fluid_Factor'], bins=30, color='red', alpha=0.7)
            ax_avo_unc[2].set_xlabel('Fluid Factor')
            ax_avo_unc[2].set_title('Fluid Factor Distribution')
            
            plt.tight_layout()
            st.pyplot(fig_avo_unc)
            
            # Crossplot of AVO attributes with uncertainty
            st.subheader("AVO Attribute Crossplot with Uncertainty")
            fig_avo_cross, ax_avo_cross = plt.subplots(figsize=(8, 6))
            
            # Plot all Monte Carlo samples
            ax_avo_cross.scatter(mc_results['Intercept'], mc_results['Gradient'], 
                                c=mc_results['Fluid_Factor'], cmap='coolwarm', 
                                alpha=0.3, s=10)
            
            # Add current result
            ax_avo_cross.scatter(intercept, gradient, c='k', s=100, label='Current')
            
            # Add colorbar
            sc = ax_avo_cross.scatter([], [], c=[], cmap='coolwarm')
            plt.colorbar(sc, label='Fluid Factor', ax=ax_avo_cross)
            
            # Add background classification
            x = np.linspace(-0.5, 0.5, 100)
            ax_avo_cross.plot(x, -x, 'k--', alpha=0.3)  # Typical brine line
            ax_avo_cross.plot(x, -4*x, 'k--', alpha=0.3)  # Typical gas line
            
            ax_avo_cross.set_xlabel('Intercept (A)')
            ax_avo_cross.set_ylabel('Gradient (B)')
            ax_avo_cross.set_title('AVO Attribute Uncertainty')
            ax_avo_cross.grid(True)
            ax_avo_cross.axhline(0, color='k', alpha=0.3)
            ax_avo_cross.axvline(0, color='k', alpha=0.3)
            ax_avo_cross.set_xlim(-0.3, 0.3)
            ax_avo_cross.set_ylim(-0.3, 0.3)
            ax_avo_cross.legend()
            
            st.pyplot(fig_avo_cross)

        # Export functionality
        st.header("Export Results")
        st.markdown(get_table_download_link(logs), unsafe_allow_html=True)
        
        plot_export_options = st.multiselect(
            "Select plots to export as PNG",
            ["Well Log Visualization", "2D Crossplots", "3D Crossplot", "Histograms", 
             "AVO Analysis", "Smith-Gidlow Analysis", "Uncertainty Analysis", "RPT Crossplots",
             "Frequency Analysis", "Time-Frequency Analysis"],
            default=["Well Log Visualization", "AVO Analysis"]
        )
        
        if st.button("Export Selected Plots"):
            if not plot_export_options:
                st.warning("No plots selected for export")
            else:
                def export_plot(figure, plot_name, file_name):
                    buf = BytesIO()
                    try:
                        figure.savefig(buf, format="png", dpi=300)
                        st.download_button(
                            label=f"Download {plot_name}",
                            data=buf.getvalue(),
                            file_name=file_name,
                            mime="image/png"
                        )
                        return True, ""
                    except Exception as e:
                        return False, str(e)
                
                results = []
                for plot_name in plot_export_options:
                    if plot_name == "Well Log Visualization" and model_choice not in ["Soft Sand RPT (rockphypy)", "Stiff Sand RPT (rockphypy)"]:
                        success, error = export_plot(fig, plot_name, "well_log_visualization.png")
                    elif plot_name == "2D Crossplots":
                        success, error = export_plot(fig2, plot_name, "2d_crossplots.png")
                    elif plot_name == "3D Crossplot" and show_3d_crossplot and model_choice not in ["Soft Sand RPT (rockphypy)", "Stiff Sand RPT (rockphypy)"]:
                        success, error = export_plot(fig3d, plot_name, "3d_crossplot.png")
                    elif plot_name == "Histograms" and show_histograms and model_choice not in ["Soft Sand RPT (rockphypy)", "Stiff Sand RPT (rockphypy)"]:
                        success, error = export_plot(fig_hist, plot_name, "histograms.png")
                    elif plot_name == "AVO Analysis" and model_choice not in ["Soft Sand RPT (rockphypy)", "Stiff Sand RPT (rockphypy)"]:
                        success, error = export_plot(fig3, plot_name, "avo_analysis.png")
                    elif plot_name == "Smith-Gidlow Analysis" and show_smith_gidlow and model_choice not in ["Soft Sand RPT (rockphypy)", "Stiff Sand RPT (rockphypy)"]:
                        success, error = export_plot(fig_sg, plot_name, "smith_gidlow_analysis.png")
                    elif plot_name == "Uncertainty Analysis" and include_uncertainty and model_choice not in ["Soft Sand RPT (rockphypy)", "Stiff Sand RPT (rockphypy)"]:
                        # Need to handle multiple figures for uncertainty
                        success1, error1 = export_plot(fig_unc, "Uncertainty_Distributions", "uncertainty_distributions.png")
                        success2, error2 = export_plot(fig_avo_unc, "AVO_Uncertainty", "avo_uncertainty.png")
                        success3, error3 = export_plot(fig_avo_cross, "AVO_Crossplot_Uncertainty", "avo_crossplot_uncertainty.png")
                        
                        if all([success1, success2, success3]):
                            results.append(f"✓ Successfully exported {plot_name} plots")
                        else:
                            errors = [e for e in [error1, error2, error3] if e]
                            results.append(f"✗ Partially exported {plot_name}: {', '.join(errors)}")
                        continue
                    elif plot_name == "RPT Crossplots" and model_choice in ["Soft Sand RPT (rockphypy)", "Stiff Sand RPT (rockphypy)"]:
                        # Create temporary figures for RPT export
                        fig_rpt_gas = plt.figure()
                        if model_choice == "Soft Sand RPT (rockphypy)":
                            Kdry, Gdry = GM.softsand(K0, G0, phi, rpt_phi_c, rpt_Cn, rpt_sigma, f=0.5)
                        else:
                            Kdry, Gdry = GM.stiffsand(K0, G0, phi, rpt_phi_c, rpt_Cn, rpt_sigma, f=0.5)
                        QI.plot_rpt(Kdry, Gdry, K0, D0, Kb, Db, Kg, Dg, phi, sw)
                        plt.title(f"{model_choice.split(' ')[0]} RPT - Gas Case")
                        buf_gas = BytesIO()
                        plt.savefig(buf_gas, format='png', dpi=150, bbox_inches='tight')
                        buf_gas.seek(0)
                        plt.close()
                        
                        fig_rpt_oil = plt.figure()
                        QI.plot_rpt(Kdry, Gdry, K0, D0, Kb, Db, Ko, Do, phi, sw)
                        plt.title(f"{model_choice.split(' ')[0]} RPT - Oil Case")
                        buf_oil = BytesIO()
                        plt.savefig(buf_oil, format='png', dpi=150, bbox_inches='tight')
                        buf_oil.seek(0)
                        plt.close()
                        
                        # Create download buttons
                        st.download_button(
                            label="Download Gas RPT",
                            data=buf_gas.getvalue(),
                            file_name="rpt_gas.png",
                            mime="image/png"
                        )
                        st.download_button(
                            label="Download Oil RPT",
                            data=buf_oil.getvalue(),
                            file_name="rpt_oil.png",
                            mime="image/png"
                        )
                        results.append("✓ Successfully exported RPT plots")
                        continue
                    elif plot_name == "Frequency Analysis" and model_choice not in ["Soft Sand RPT (rockphypy)", "Stiff Sand RPT (rockphypy)"]:
                        # Need to recreate the frequency analysis plot
                        syn_gather, t_samp = perform_time_frequency_analysis(
                            logs, angles, wavelet_freq, cwt_scales, cwt_wavelet, middle_top, middle_bot
                        )
                        fig_freq, _ = plt.subplots(figsize=(10, 5))
                        plot_frequency_analysis(syn_gather, t_samp, angles, wavelet_freq, time_range, freq_range)
                        success, error = export_plot(fig_freq, plot_name, "frequency_analysis.png")
                        plt.close()
                    elif plot_name == "Time-Frequency Analysis" and model_choice not in ["Soft Sand RPT (rockphypy)", "Stiff Sand RPT (rockphypy)"]:
                        # Need to recreate the CWT analysis plot
                        syn_gather, t_samp = perform_time_frequency_analysis(
                            logs, angles, wavelet_freq, cwt_scales, cwt_wavelet, middle_top, middle_bot
                        )
                        fig_cwt, _ = plt.subplots(3, 1, figsize=(10, 12))
                        plot_cwt_analysis(syn_gather, t_samp, angles, cwt_scales, cwt_wavelet, wavelet_freq, time_range, freq_range)
                        success, error = export_plot(fig_cwt, plot_name, "time_frequency_analysis.png")
                        plt.close()
                    else:
                        continue
                    
                    if success:
                        results.append(f"✓ Successfully exported {plot_name}")
                    else:
                        results.append(f"✗ Failed to export {plot_name}: {error}")
                
                st.write("\n".join(results))
                
                if all("✓" in result for result in results):
                    st.success("All exports completed successfully!")
                elif any("✓" in result for result in results):
                    st.warning("Some exports completed with errors")
                else:
                    st.error("All exports failed")
    
    except Exception as e:
        st.error(f"An error occurred during processing: {str(e)}")
