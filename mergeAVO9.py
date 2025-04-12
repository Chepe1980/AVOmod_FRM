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
from scipy.interpolate import interp1d
from scipy.spatial import ConvexHull

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
    model_choice = st.selectbox(
        "Rock Physics Model",
        ["Gassmann's Fluid Substitution", 
         "Critical Porosity Model (Nur)", 
         "Contact Theory (Hertz-Mindlin)",
         "Dvorkin-Nur Soft Sand Model",
         "Raymer-Hunt-Gardner Model"],
        index=0
    )
    
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
    show_templates = st.checkbox("Show Rock Physics Templates", value=True)
    
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

    return vp2*1000, vs2*1000, rho2, k_s2  # Convert back to m/s

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

def odegaard_avseth_templates(vsh, phi, fluid_case, k_qz, mu_qz, rho_qz, k_sh, mu_sh, rho_sh, rho_b, k_b, rho_o, k_o, rho_g, k_g):
    """
    Generate Ødegaard & Avseth rock physics templates for given Vsh and Phi
    Fluid cases: 'brine', 'oil', 'gas'
    Returns: Vp, Vs, Rho
    """
    # Use the passed mineral properties
    k_qz, mu_qz = k_qz, mu_qz  # GPa
    k_sh, mu_sh = k_sh, mu_sh    # GPa
    rho_qz, rho_sh = rho_qz, rho_sh  # g/cc
    
    # Use the passed fluid properties
    if fluid_case == 'brine':
        rho_fl, k_fl = rho_b, k_b
    elif fluid_case == 'oil':
        rho_fl, k_fl = rho_o, k_o
    elif fluid_case == 'gas':
        rho_fl, k_fl = rho_g, k_g
    
    # Mineral mixing (VRH)
    shale_frac = vsh / (vsh + (1 - vsh - phi))
    sand_frac = (1 - vsh - phi) / (vsh + (1 - vsh - phi))
    
    k0 = (shale_frac * k_sh + sand_frac * k_qz) / 2 + \
         (shale_frac / k_sh + sand_frac / k_qz)**-1 / 2
    mu0 = (shale_frac * mu_sh + sand_frac * mu_qz) / 2 + \
          (shale_frac / mu_sh + sand_frac / mu_qz)**-1 / 2
    
    # Dry rock moduli (Critical porosity model)
    phi_c = 0.4  # Critical porosity
    k_dry = k0 * (1 - phi/phi_c)
    mu_dry = mu0 * (1 - phi/phi_c)
    
    # Gassmann fluid substitution
    k_sat = k_dry + (1 - k_dry/k0)**2 / (phi/k_fl + (1-phi)/k0 - k_dry/k0**2)
    rho = (1 - phi) * (vsh*rho_sh + (1-vsh)*rho_qz) + phi * rho_fl
    
    # Calculate velocities
    vp = np.sqrt((k_sat + 4/3 * mu_dry) / rho) * 1000  # m/s
    vs = np.sqrt(mu_dry / rho) * 1000  # m/s
    
    return vp, vs, rho

def plot_rock_physics_templates(k_qz, mu_qz, rho_qz, k_sh, mu_sh, rho_sh, rho_b, k_b, rho_o, k_o, rho_g, k_g):
    """Create Ødegaard & Avseth template crossplot"""
    st.header("Ødegaard & Avseth Rock Physics Templates")
    
    # Create grid of Vsh and Phi values
    vsh_values = np.linspace(0, 0.5, 6)  # Vsh from 0 to 50%
    phi_values = np.linspace(0.05, 0.35, 7)  # Porosity from 5% to 35%
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Plot templates for each fluid case
    for fluid, color in [('brine', 'blue'), ('oil', 'green'), ('gas', 'red')]:
        vp_values = []
        vs_values = []
        ip_values = []
        vpvs_values = []
        
        for vsh in vsh_values:
            for phi in phi_values:
                vp, vs, rho = odegaard_avseth_templates(vsh, phi, fluid, 
                                                      k_qz, mu_qz, rho_qz,
                                                      k_sh, mu_sh, rho_sh,
                                                      rho_b, k_b, rho_o, k_o, rho_g, k_g)
                vp_values.append(vp)
                vs_values.append(vs)
                ip_values.append(vp * rho)
                vpvs_values.append(vp/vs)
        
        # Create convex hull for the template
        points = np.column_stack((np.array(ip_values), np.array(vpvs_values)))
        hull = ConvexHull(points)
        
        # Plot the template
        ax.plot(points[hull.vertices,0], points[hull.vertices,1], 
                '--', color=color, alpha=0.7, label=f'{fluid.capitalize()} Sand')
        ax.fill(points[hull.vertices,0], points[hull.vertices,1], 
               color=color, alpha=0.1)
    
    # Add shale trend
    shale_ip = np.linspace(7000, 9000, 10)
    shale_vpvs = np.linspace(1.8, 2.4, 10)
    ax.plot(shale_ip, shale_vpvs, '--', color='black', label='Shale Trend')
    
    # Format plot
    ax.set_xlabel('Acoustic Impedance (m/s*g/cc)')
    ax.set_ylabel('Vp/Vs')
    ax.set_title('Ødegaard & Avseth Rock Physics Templates')
    ax.legend()
    ax.grid(True)
    
    return fig

# [Rest of the code remains the same until the main processing section]

# Main content area
if uploaded_file is not None:
    try:
        # Process data with selected model
        logs, mc_results = process_data(
            uploaded_file, 
            model_choice,
            include_uncertainty=include_uncertainty,
            mc_iterations=mc_iterations,
            critical_porosity=critical_porosity if 'critical_porosity' in locals() else None,
            coordination_number=coordination_number if 'coordination_number' in locals() else None,
            effective_pressure=effective_pressure if 'effective_pressure' in locals() else None
        )
        
        # [Previous visualization code remains the same until the templates section]
        
        # Rock Physics Templates
        if show_templates:
            fig_templates = plot_rock_physics_templates(
                k_qz, mu_qz, rho_qz,
                k_sh, mu_sh, rho_sh,
                rho_b, k_b,
                rho_o, k_o,
                rho_g, k_g
            )
            st.pyplot(fig_templates)
            
            # Combined plot with well data overlay
            st.subheader("Well Data Overlay on Templates")
            fig_combined = plt.figure(figsize=(10, 8))
            ax_combined = fig_combined.add_subplot(111)
            
            # Plot templates again
            for fluid, color in [('brine', 'blue'), ('oil', 'green'), ('gas', 'red')]:
                ip_values = []
                vpvs_values = []
                
                for vsh in vsh_values:
                    for phi in phi_values:
                        vp, vs, rho = odegaard_avseth_templates(
                            vsh, phi, fluid, 
                            k_qz, mu_qz, rho_qz,
                            k_sh, mu_sh, rho_sh,
                            rho_b, k_b, rho_o, k_o, rho_g, k_g
                        )
                        ip_values.append(vp * rho)
                        vpvs_values.append(vp/vs)
                
                points = np.column_stack((np.array(ip_values), np.array(vpvs_values)))
                hull = ConvexHull(points)
                ax_combined.plot(points[hull.vertices,0], points[hull.vertices,1], 
                               '--', color=color, alpha=0.7, label=f'{fluid.capitalize()} Sand')
                ax_combined.fill(points[hull.vertices,0], points[hull.vertices,1], 
                               color=color, alpha=0.1)
            
            # Plot shale trend
            ax_combined.plot(shale_ip, shale_vpvs, '--', color='black', label='Shale Trend')
            
            # Overlay well data points colored by LFC
            for case, color in [('B', 'blue'), ('O', 'green'), ('G', 'red')]:
                mask = logs[f'LFC_{case}'] == int(case == 'B')*1 + int(case == 'O')*2 + int(case == 'G')*3
                ax_combined.scatter(logs.loc[mask, f'IP_FRM{case}'], 
                                  logs.loc[mask, f'VPVS_FRM{case}'],
                                  c=color, s=20, alpha=0.7, 
                                  label=f'{case} - {["", "Brine", "Oil", "Gas"][int(case == "B")*1 + int(case == "O")*2 + int(case == "G")*3]}')
            
            # Format combined plot
            ax_combined.set_xlabel('Acoustic Impedance (m/s*g/cc)')
            ax_combined.set_ylabel('Vp/Vs')
            ax_combined.set_title('Well Data Overlay on Rock Physics Templates')
            ax_combined.legend()
            ax_combined.grid(True)
            
            st.pyplot(fig_combined)

        # [Rest of the code remains the same]

    except Exception as e:
        st.error(f"An error occurred during processing: {str(e)}")
