import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.colors as colors
from mpl_toolkits.mplot3d import Axes3D
from io import BytesIO
import base64

# Set page config
st.set_page_config(layout="wide", page_title="Enhanced Rock Physics & AVO Modeling")

# Title and description
st.title("Enhanced Rock Physics & AVO Modeling Tool")
st.markdown("""
This app performs advanced rock physics modeling and AVO analysis with multiple models and visualization options.
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
         "Contact Theory (Hertz-Mindlin)"],
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
    elif model_choice == "Contact Theory (Hertz-Mindlin)":
        coordination_number = st.slider("Coordination Number", 6, 12, 9)
        effective_pressure = st.slider("Effective Pressure (MPa)", 1, 50, 10)
    
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
    
    # Visualization options
    st.subheader("Visualization Options")
    selected_cmap = st.selectbox("Color Map", seismic_colormaps, index=0)
    show_3d_crossplot = st.checkbox("Show 3D Crossplot", value=False)
    show_histograms = st.checkbox("Show Histograms", value=True)
    
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
def frm(vp1, vs1, rho1, rho_f1, k_f1, rho_f2, k_f2, k0, phi):
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

def critical_porosity_model(vp1, vs1, rho1, rho_f1, k_f1, rho_f2, k_f2, k0, phi, phi_c):
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

# Wavelet function (replacement for pyavo)
def ricker_wavelet(frequency, length=0.128, dt=0.001):
    """Generate a Ricker wavelet"""
    t = np.linspace(-length/2, length/2, int(length/dt))
    y = (1 - 2*(np.pi**2)*(frequency**2)*(t**2)) * np.exp(-(np.pi**2)*(frequency**2)*(t**2))
    return t, y

# Main processing function with error handling
@handle_errors
def process_data(uploaded_file, model_choice, **kwargs):
    # Read and validate data
    logs = pd.read_csv(uploaded_file)
    required_columns = {'DEPTH', 'VP', 'VS', 'RHO', 'VSH', 'SW', 'PHI'}
    if not required_columns.issubset(logs.columns):
        missing = required_columns - set(logs.columns)
        raise ValueError(f"Missing required columns: {missing}")
    
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

    # Fluid mixtures
    water = logs.SW.values
    hc = 1 - logs.SW.values
    tmp, k_fl, tmp, tmp, tmp, tmp = vrh([water, hc], [k_b, k_o], [0, 0])
    rho_fl = water*rho_b + hc*rho_o

    # Select model function
    if model_choice == "Gassmann's Fluid Substitution":
        model_func = frm
    elif model_choice == "Critical Porosity Model (Nur)":
        model_func = lambda *args: critical_porosity_model(*args, phi_c=kwargs['critical_porosity'])
    elif model_choice == "Contact Theory (Hertz-Mindlin)":
        model_func = lambda *args: hertz_mindlin_model(*args, Cn=kwargs['coordination_number'], P=kwargs['effective_pressure'])

    # Apply selected model
    vpb, vsb, rhob, kb = model_func(logs.VP, logs.VS, logs.RHO, rho_fl, k_fl, rho_b, k_b, k0, logs.PHI)
    vpo, vso, rhoo, ko = model_func(logs.VP, logs.VS, logs.RHO, rho_fl, k_fl, rho_o, k_o, k0, logs.PHI)
    vpg, vsg, rhog, kg = model_func(logs.VP, logs.VS, logs.RHO, rho_fl, k_fl, rho_g, k_g, k0, logs.PHI)

    # Litho-fluid classification
    brine_sand = ((logs.VSH <= sand_cutoff) & (logs.SW >= 0.65))
    oil_sand = ((logs.VSH <= sand_cutoff) & (logs.SW < 0.65))
    shale_flag = (logs.VSH > sand_cutoff)

    # Add results to logs
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

    # LFC flags
    for case, val in [('B', 1), ('O', 2), ('G', 3)]:
        temp_lfc = np.zeros(np.shape(logs.VSH))
        temp_lfc[brine_sand.values | oil_sand.values] = val
        temp_lfc[shale_flag.values] = 4
        logs[f'LFC_{case}'] = temp_lfc

    return logs

# Download link generator
def get_table_download_link(df, filename="results.csv"):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">Download CSV File</a>'
    return href

# AVO modeling functions
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

# Main content area
if uploaded_file is not None:
    try:
        # Process data with selected model
        logs = process_data(
            uploaded_file, 
            model_choice,
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

        # Display logs
        ll = logs.loc[(logs.DEPTH>=ztop) & (logs.DEPTH<=zbot)]
        cluster = np.repeat(np.expand_dims(ll['LFC_B'].values,1), 100, 1)

        fig, ax = plt.subplots(nrows=1, ncols=4, figsize=(12, 8))
        ax[0].plot(ll.VSH, ll.DEPTH, '-g', label='Vsh')
        ax[0].plot(ll.SW, ll.DEPTH, '-b', label='Sw')
        ax[0].plot(ll.PHI, ll.DEPTH, '-k', label='phi')
        ax[1].plot(ll.IP_FRMG, ll.DEPTH, '-r', label='Gas')
        ax[1].plot(ll.IP_FRMB, ll.DEPTH, '-b', label='Brine')
        ax[1].plot(ll.IP, ll.DEPTH, '-', color='0.5', label='Original')
        ax[2].plot(ll.VPVS_FRMG, ll.DEPTH, '-r', label='Gas')
        ax[2].plot(ll.VPVS_FRMB, ll.DEPTH, '-b', label='Brine')
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

        # 3D Crossplot if enabled
        if show_3d_crossplot:
            st.header("3D Crossplot")
            fig3d = plt.figure(figsize=(10, 8))
            ax3d = fig3d.add_subplot(111, projection='3d')
            
            # Plot points for each fluid case
            for case, color in [('B', 'blue'), ('O', 'green'), ('G', 'red')]:
                mask = logs[f'LFC_{case}'] == int(case == 'B')*1 + int(case == 'O')*2 + int(case == 'G')*3
                ax3d.scatter(
                    logs.loc[mask, f'IP_FRM{case}'],
                    logs.loc[mask, f'VPVS_FRM{case}'],
                    logs.loc[mask, f'RHO_FRM{case}'],
                    c=color, label=case, alpha=0.5
                )
            
            ax3d.set_xlabel('IP (m/s*g/cc)')
            ax3d.set_ylabel('Vp/Vs')
            ax3d.set_zlabel('Density (g/cc)')
            ax3d.set_title('3D Rock Physics Crossplot')
            ax3d.legend()
            st.pyplot(fig3d)

        # Histograms if enabled
        if show_histograms:
            st.header("Property Distributions")
            fig_hist, ax_hist = plt.subplots(2, 2, figsize=(12, 8))
            
            # IP Histogram
            ax_hist[0,0].hist(logs.IP_FRMB, bins=30, alpha=0.5, label='Brine', color='blue')
            ax_hist[0,0].hist(logs.IP_FRMO, bins=30, alpha=0.5, label='Oil', color='green')
            ax_hist[0,0].hist(logs.IP_FRMG, bins=30, alpha=0.5, label='Gas', color='red')
            ax_hist[0,0].set_xlabel('IP (m/s*g/cc)')
            ax_hist[0,0].set_ylabel('Frequency')
            ax_hist[0,0].legend()
            
            # Vp/Vs Histogram
            ax_hist[0,1].hist(logs.VPVS_FRMB, bins=30, alpha=0.5, label='Brine', color='blue')
            ax_hist[0,1].hist(logs.VPVS_FRMO, bins=30, alpha=0.5, label='Oil', color='green')
            ax_hist[0,1].hist(logs.VPVS_FRMG, bins=30, alpha=0.5, label='Gas', color='red')
            ax_hist[0,1].set_xlabel('Vp/Vs')
            ax_hist[0,1].legend()
            
            # Density Histogram
            ax_hist[1,0].hist(logs.RHO_FRMB, bins=30, alpha=0.5, label='Brine', color='blue')
            ax_hist[1,0].hist(logs.RHO_FRMO, bins=30, alpha=0.5, label='Oil', color='green')
            ax_hist[1,0].hist(logs.RHO_FRMG, bins=30, alpha=0.5, label='Gas', color='red')
            ax_hist[1,0].set_xlabel('Density (g/cc)')
            ax_hist[1,0].set_ylabel('Frequency')
            ax_hist[1,0].legend()
            
            # LFC Histogram
            ax_hist[1,1].hist(logs.LFC_B, bins=[0,1,2,3,4,5], alpha=0.5, rwidth=0.8, align='left')
            ax_hist[1,1].set_xlabel('Litho-Fluid Class')
            ax_hist[1,1].set_xticks([0.5,1.5,2.5,3.5,4.5])
            ax_hist[1,1].set_xticklabels(['Undef','Brine','Oil','Gas','Shale'])
            
            plt.tight_layout()
            st.pyplot(fig_hist)

        # AVO Modeling
        st.header("AVO Modeling")
        
        # Define middle zone (using 40-60% of depth range by default)
        middle_top = ztop + (zbot - ztop) * 0.4
        middle_bot = ztop + (zbot - ztop) * 0.6
        
        # AVO cases
        cases = ['Brine', 'Oil', 'Gas']
        case_data = {
            'Brine': {'vp': 'VP_FRMB', 'vs': 'VS_FRMB', 'rho': 'RHO_FRMB', 'color': 'b'},
            'Oil': {'vp': 'VP_FRMO', 'vs': 'VS_FRMO', 'rho': 'RHO_FRMO', 'color': 'g'},
            'Gas': {'vp': 'VP_FRMG', 'vs': 'VS_FRMG', 'rho': 'RHO_FRMG', 'color': 'r'}
        }
        
        # Generate wavelet
        wlt_time, wlt_amp = ricker_wavelet(wavelet_freq)
        t_samp = np.arange(0, 0.5, 0.0001)  # Time samples
        t_middle = 0.2  # Fixed time for middle interface
        
        # Create figure for AVO results with wavelet plot
        fig3, (ax_wavelet, ax_avo) = plt.subplots(1, 2, figsize=(12, 5), gridspec_kw={'width_ratios': [1, 2]})
        
        # Wavelet plot
        ax_wavelet.plot(wlt_time, wlt_amp, color='purple', linewidth=2)
        ax_wavelet.fill_between(wlt_time, wlt_amp, color='purple', alpha=0.3)
        ax_wavelet.set_title(f"Wavelet ({wavelet_freq} Hz)")
        ax_wavelet.set_xlabel("Time (s)")
        ax_wavelet.set_ylabel("Amplitude")
        ax_wavelet.grid(True)
        
        # Add slider for reflection coefficient range
        rc_min, rc_max = st.slider(
            "Reflection Coefficient Range",
            -0.5, 0.5, (-0.2, 0.2),
            step=0.01,
            key='rc_range'
        )
        
        # Process each case for AVO curves
        angles = np.arange(min_angle, max_angle + 1, 1)
        
        for case in cases:
            # Upper zone properties (shale)
            vp_upper = logs.loc[(logs.DEPTH >= middle_top - (middle_bot-middle_top)), 
                               'VP'].values.mean()
            vs_upper = logs.loc[(logs.DEPTH >= middle_top - (middle_bot-middle_top)), 
                               'VS'].values.mean()
            rho_upper = logs.loc[(logs.DEPTH >= middle_top - (middle_bot-middle_top)), 
                                 'RHO'].values.mean()
            
            # Middle zone properties
            vp_middle = logs.loc[(logs.DEPTH >= middle_top) & (logs.DEPTH <= middle_bot), 
                                case_data[case]['vp']].values.mean()
            vs_middle = logs.loc[(logs.DEPTH >= middle_top) & (logs.DEPTH <= middle_bot), 
                                case_data[case]['vs']].values.mean()
            rho_middle = logs.loc[(logs.DEPTH >= middle_top) & (logs.DEPTH <= middle_bot), 
                                  case_data[case]['rho']].values.mean()
            
            # Calculate reflection coefficients
            rc = []
            for angle in angles:
                rc.append(calculate_reflection_coefficients(
                    vp_upper, vp_middle, vs_upper, vs_middle, rho_upper, rho_middle, angle
                ))
            
            # Plot AVO curve
            ax_avo.plot(angles, rc, f"{case_data[case]['color']}-", label=f"{case}")
        
        # Finalize AVO plot
        ax_avo.set_title("AVO Reflection Coefficients (Middle Interface)")
        ax_avo.set_xlabel("Angle (degrees)")
        ax_avo.set_ylabel("Reflection Coefficient")
        ax_avo.set_ylim(rc_min, rc_max)
        ax_avo.grid(True)
        ax_avo.legend()
        
        st.pyplot(fig3)

        # Synthetic gathers for middle interface
        st.header("Synthetic Seismic Gathers (Middle Interface)")
        
        # Add slider for time range
        time_min, time_max = st.slider(
            "Time Range for Synthetic Gathers (s)",
            0.0, 0.5, (0.15, 0.25),
            step=0.01,
            key='time_range'
        )
        
        fig4, ax4 = plt.subplots(1, 3, figsize=(18, 5))
        
        for idx, case in enumerate(cases):
            # Upper zone properties (shale)
            vp_upper = logs.loc[(logs.DEPTH >= middle_top - (middle_bot-middle_top)), 
                               'VP'].values.mean()
            vs_upper = logs.loc[(logs.DEPTH >= middle_top - (middle_bot-middle_top)), 
                               'VS'].values.mean()
            rho_upper = logs.loc[(logs.DEPTH >= middle_top - (middle_bot-middle_top)), 
                                 'RHO'].values.mean()
            
            # Middle zone properties
            vp_middle = logs.loc[(logs.DEPTH >= middle_top) & (logs.DEPTH <= middle_bot), 
                                case_data[case]['vp']].values.mean()
            vs_middle = logs.loc[(logs.DEPTH >= middle_top) & (logs.DEPTH <= middle_bot), 
                                case_data[case]['vs']].values.mean()
            rho_middle = logs.loc[(logs.DEPTH >= middle_top) & (logs.DEPTH <= middle_bot), 
                                  case_data[case]['rho']].values.mean()
            
            # Generate synthetic gather
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
            im = ax4[idx].imshow(syn_gather.T, aspect='auto', extent=extent,
                               cmap=selected_cmap, vmin=-np.max(np.abs(syn_gather)), 
                               vmax=np.max(np.abs(syn_gather)))
            
            # Add title and property annotations
            props_text = f"Vp: {vp_middle:.0f} m/s\nVs: {vs_middle:.0f} m/s\nRho: {rho_middle:.2f} g/cc"
            ax4[idx].text(0.05, 0.95, props_text, transform=ax4[idx].transAxes,
                         fontsize=9, verticalalignment='top', bbox=dict(facecolor='white', alpha=0.7))
            
            ax4[idx].set_title(f"{case} Case", fontweight='bold')
            ax4[idx].set_xlabel("Angle (degrees)")
            ax4[idx].set_ylabel("Time (s)")
            ax4[idx].set_ylim(time_max, time_min)  # Note: inverted because of imshow
            
            plt.colorbar(im, ax=ax4[idx], label='Amplitude')
        
        plt.tight_layout()
        st.pyplot(fig4)

             # Export functionality
        st.header("Export Results")
        st.markdown(get_table_download_link(logs), unsafe_allow_html=True)
        
        # Option to export specific plots
        plot_export_options = st.multiselect(
            "Select plots to export as PNG",
            ["Well Log Visualization", "2D Crossplots", "3D Crossplot", "Histograms", "AVO Analysis"]
        )
        
        if st.button("Export Selected Plots"):
            if not plot_export_options:
                st.warning("No plots selected for export")
                return
            
            export_results = []
            
            def export_plot(figure, plot_name, file_name):
                try:
                    buf = BytesIO()
                    figure.savefig(buf, format="png", dpi=300)
                    st.download_button(
                        label=f"Download {plot_name}",
                        data=buf.getvalue(),
                        file_name=file_name,
                        mime="image/png"
                    )
                    return f"✓ {plot_name} exported successfully"
                except Exception as e:
                    return f"✗ Failed to export {plot_name}: {str(e)}"
            
            # Export each selected plot
            for plot_name in plot_export_options:
                if plot_name == "Well Log Visualization":
                    export_results.append(export_plot(fig, plot_name, "well_log_visualization.png"))
                elif plot_name == "2D Crossplots":
                    export_results.append(export_plot(fig2, plot_name, "2d_crossplots.png"))
                elif plot_name == "3D Crossplot" and show_3d_crossplot:
                    export_results.append(export_plot(fig3d, plot_name, "3d_crossplot.png"))
                elif plot_name == "Histograms" and show_histograms:
                    export_results.append(export_plot(fig_hist, plot_name, "histograms.png"))
                elif plot_name == "AVO Analysis":
                    export_results.append(export_plot(fig3, plot_name, "avo_analysis.png"))
            
            # Display results
            st.write("\n".join(export_results))
            
            if all("✓" in result for result in export_results):
                st.success("All exports completed successfully!")
            else:
                st.warning("Some exports had errors")
