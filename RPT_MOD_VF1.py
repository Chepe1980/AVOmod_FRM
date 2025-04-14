import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import Ellipse
from matplotlib.lines import Line2D
from io import BytesIO
import base64
from streamlit_bokeh_events import streamlit_bokeh_events
from bokeh.plotting import figure
from bokeh.models import ColumnDataSource, HoverTool, CustomJS
from bokeh.palettes import Category10
from bokeh.transform import factor_cmap
import scipy.stats as stats
from scipy.optimize import curve_fit

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
    show_templates = st.checkbox("Show Rock Physics Templates (Ødegaard & Avseth)", value=False)
    
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

# Rock Physics Models (keep all existing model functions exactly as they were)
# [Previous model functions: frm(), critical_porosity_model(), hertz_mindlin_model(), 
#  dvorkin_nur_model(), raymer_hunt_model() remain unchanged]

# Wavelet function (unchanged)
def ricker_wavelet(frequency, length=0.128, dt=0.001):
    """Generate a Ricker wavelet"""
    t = np.linspace(-length/2, length/2, int(length/dt))
    y = (1 - 2*(np.pi**2)*(frequency**2)*(t**2)) * np.exp(-(np.pi**2)*(frequency**2)*(t**2))
    return t, y

# Smith-Gidlow AVO approximation (unchanged)
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

# Monte Carlo simulation for uncertainty analysis (unchanged)
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

# Create interactive crossplot with improved error handling (unchanged)
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

# Main processing function with error handling (unchanged except for template addition)
@handle_errors
def process_data(uploaded_file, model_choice, include_uncertainty=False, mc_iterations=100, **kwargs):
    # [Previous processing code remains exactly the same until the 2D crossplots section]
    
    # =============================================
    # UPDATED 2D CROSSPLOTS SECTION WITH TEMPLATES
    # =============================================
    st.header("2D Crossplots")
    fig2, ax2 = plt.subplots(nrows=1, ncols=4, figsize=(16, 4))
    
    def plot_odegaard_avseth_templates(ax):
        """Add Ødegaard & Avseth rock physics templates to crossplot"""
        # Shale trend
        shale_x = np.linspace(6000, 12000, 100)
        shale_y = 2.2 - 0.2*(shale_x - 6000)/6000
        ax.plot(shale_x, shale_y, 'k--', alpha=0.5, linewidth=1)
        
        # Sand trends
        for vpvs, color in [(1.6, 'blue'), (1.8, 'green'), (2.0, 'red')]:
            sand_x = np.linspace(4000, 12000, 100)
            sand_y = vpvs + 0.2*(sand_x - 4000)/8000
            ax.plot(sand_x, sand_y, linestyle='--', color=color, alpha=0.5, linewidth=1)
        
        # Add ellipses for different facies
        ellipses = [
            {'center': (8000, 1.8), 'width': 3000, 'height': 0.3, 'angle': 0, 'label': 'Brine Sand', 'color': 'blue'},
            {'center': (9000, 1.7), 'width': 2000, 'height': 0.2, 'angle': 0, 'label': 'Oil Sand', 'color': 'green'},
            {'center': (7000, 1.6), 'width': 2500, 'height': 0.25, 'angle': 0, 'label': 'Gas Sand', 'color': 'red'},
            {'center': (11000, 2.1), 'width': 4000, 'height': 0.2, 'angle': 0, 'label': 'Shale', 'color': 'gray'}
        ]
        
        for el in ellipses:
            ellipse = Ellipse(xy=el['center'], width=el['width'], height=el['height'],
                             angle=el['angle'], alpha=0.2, color=el['color'])
            ax.add_patch(ellipse)
            ax.text(el['center'][0], el['center'][1], el['label'], 
                   ha='center', va='center', fontsize=8)
    
    # Plot crossplots with optional templates
    for i, case in enumerate(['', 'B', 'O', 'G']):
        col = logs[f'LFC_{case}'] if case else logs.LFC_B
        ax2[i].scatter(logs[f'IP_FRM{case}'] if case else logs.IP, 
                      logs[f'VPVS_FRM{case}'] if case else logs.VPVS, 
                      20, col, marker='o', edgecolors='none', alpha=0.5, 
                      cmap=cmap_facies, vmin=0, vmax=4)
        
        if show_templates:
            plot_odegaard_avseth_templates(ax2[i])
        
        ax2[i].set_xlim(3000,16000)
        ax2[i].set_ylim(1.5,3)
        ax2[i].set_title('Original Data' if not case else f'FRM to {"Brine" if case=="B" else "Oil" if case=="O" else "Gas"}')
    
    plt.tight_layout()
    st.pyplot(fig2)
    
    # =============================================
    # FIXED UNCERTAINTY VISUALIZATION SECTION
    # =============================================
    if include_uncertainty and mc_results:
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
        
        # Plot uncertainty distributions with fixed colors
        st.subheader("Property Uncertainty Distributions")
        fig_unc, ax_unc = plt.subplots(2, 2, figsize=(12, 8))
        
        # Define colors for each property
        prop_colors = {
            'VP': 'blue',
            'VS': 'green',
            'IP': 'red',
            'VPVS': 'purple'
        }
        
        # Plot distributions with individual colors
        ax_unc[0,0].hist(mc_results['VP'], bins=30, color=prop_colors['VP'], alpha=0.7)
        ax_unc[0,0].set_xlabel('VP (m/s)')
        ax_unc[0,0].set_ylabel('Frequency')
        ax_unc[0,0].set_title('P-wave Velocity Distribution')
        
        ax_unc[0,1].hist(mc_results['VS'], bins=30, color=prop_colors['VS'], alpha=0.7)
        ax_unc[0,1].set_xlabel('VS (m/s)')
        ax_unc[0,1].set_title('S-wave Velocity Distribution')
        
        ax_unc[1,0].hist(mc_results['IP'], bins=30, color=prop_colors['IP'], alpha=0.7)
        ax_unc[1,0].set_xlabel('IP (m/s*g/cc)')
        ax_unc[1,0].set_ylabel('Frequency')
        ax_unc[1,0].set_title('Acoustic Impedance Distribution')
        
        ax_unc[1,1].hist(mc_results['VPVS'], bins=30, color=prop_colors['VPVS'], alpha=0.7)
        ax_unc[1,1].set_xlabel('Vp/Vs')
        ax_unc[1,1].set_title('Vp/Vs Ratio Distribution')
        
        plt.tight_layout()
        st.pyplot(fig_unc)
        
        # [Rest of the original code remains unchanged]
    
    return logs, mc_results

# [All remaining functions and main execution code stay exactly the same]

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
        
        # [Rest of the visualization code remains unchanged]
        
    except Exception as e:
        st.error(f"An error occurred during processing: {str(e)}")
