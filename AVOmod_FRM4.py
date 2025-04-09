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

    # Visualization (same as before)
    ccc = ['#B3B3B3','blue','green','red','#996633']
    cmap_facies = colors.ListedColormap(ccc[0:len(ccc)], 'indexed')

    # Display logs (same as before)
    ll = logs.loc[(logs.DEPTH>=upper_top) & (logs.DEPTH<=lower_bot)]
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
    cbar.set_ticks(range(0,1)); cbar.set_ticklabels('')

    for i in ax[:-1]:
        i.set_ylim(upper_top, lower_bot)
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

    # Crossplots (same as before)
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

    # AVO Modeling for Three Layers
    st.header("AVO Modeling (Three-Layer Model)")
    
    # Define layer names and depth ranges
    layers = {
        "Upper": {"top": upper_top, "bot": upper_bot},
        "Middle": {"top": middle_top, "bot": middle_bot},
        "Lower": {"top": lower_top, "bot": lower_bot}
    }
    
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
    t_samp = np.arange(0, 0.5, 0.0001)  # Time samples
    
    # Assign fixed times to interfaces (for synthetic gathers)
    t_upper = 0.1  # Upper interface time
    t_middle = 0.2  # Middle interface time
    t_lower = 0.3   # Lower interface time
    
    # Create figure for AVO results
    fig3, ax3 = plt.subplots(1, 2, figsize=(12, 6), gridspec_kw={'width_ratios': [1, 2]})
    
    # Wavelet plot
    ax3[0].plot(wlt_time, wlt_amp, color='purple', linewidth=2)
    ax3[0].fill_between(wlt_time, wlt_amp, color='purple', alpha=0.3)
    ax3[0].set_title(f"Wavelet ({wavelet_freq} Hz)", fontsize=10)
    ax3[0].set_xlabel("Time (s)", fontsize=8)
    ax3[0].set_ylabel("Amplitude", fontsize=8)
    ax3[0].grid(True)
    ax3[0].tick_params(axis='both', which='major', labelsize=8)
    
    # Process each case for AVO curves
    angles = np.arange(min_angle, max_angle + 1, 1)
    
    for case in cases:
        # Initialize lists to store RC for each interface
        rc_upper = []
        rc_middle = []
        rc_lower = []
        
        for angle in angles:
            # Upper interface: Shale (above) vs. Upper Zone
            vp_sh = logs.loc[logs.VSH > 0.5, 'VP'].values.mean()
            vs_sh = logs.loc[logs.VSH > 0.5, 'VS'].values.mean()
            rho_sh = logs.loc[logs.VSH > 0.5, 'RHO'].values.mean()
            
            vp_upper = logs.loc[(logs.DEPTH >= layers["Upper"]["top"]) & 
                               (logs.DEPTH <= layers["Upper"]["bot"]), 
                               case_data[case]['vp']].values.mean()
            vs_upper = logs.loc[(logs.DEPTH >= layers["Upper"]["top"]) & 
                               (logs.DEPTH <= layers["Upper"]["bot"]), 
                               case_data[case]['vs']].values.mean()
            rho_upper = logs.loc[(logs.DEPTH >= layers["Upper"]["top"]) & 
                                 (logs.DEPTH <= layers["Upper"]["bot"]), 
                                 case_data[case]['rho']].values.mean()
            
            rc_upper.append(calculate_reflection_coefficients(
                vp_sh, vp_upper, vs_sh, vs_upper, rho_sh, rho_upper, angle
            ))
            
            # Middle interface: Upper Zone vs. Middle Zone
            vp_middle = logs.loc[(logs.DEPTH >= layers["Middle"]["top"]) & 
                                (logs.DEPTH <= layers["Middle"]["bot"]), 
                                case_data[case]['vp']].values.mean()
            vs_middle = logs.loc[(logs.DEPTH >= layers["Middle"]["top"]) & 
                                (logs.DEPTH <= layers["Middle"]["bot"]), 
                                case_data[case]['vs']].values.mean()
            rho_middle = logs.loc[(logs.DEPTH >= layers["Middle"]["top"]) & 
                                  (logs.DEPTH <= layers["Middle"]["bot"]), 
                                  case_data[case]['rho']].values.mean()
            
            rc_middle.append(calculate_reflection_coefficients(
                vp_upper, vp_middle, vs_upper, vs_middle, rho_upper, rho_middle, angle
            ))
            
            # Lower interface: Middle Zone vs. Lower Zone
            vp_lower = logs.loc[(logs.DEPTH >= layers["Lower"]["top"]) & 
                               (logs.DEPTH <= layers["Lower"]["bot"]), 
                               case_data[case]['vp']].values.mean()
            vs_lower = logs.loc[(logs.DEPTH >= layers["Lower"]["top"]) & 
                               (logs.DEPTH <= layers["Lower"]["bot"]), 
                               case_data[case]['vs']].values.mean()
            rho_lower = logs.loc[(logs.DEPTH >= layers["Lower"]["top"]) & 
                                 (logs.DEPTH <= layers["Lower"]["bot"]), 
                                 case_data[case]['rho']].values.mean()
            
            rc_lower.append(calculate_reflection_coefficients(
                vp_middle, vp_lower, vs_middle, vs_lower, rho_middle, rho_lower, angle
            ))
        
        # Plot AVO curves for all interfaces
        ax3[1].plot(angles, rc_upper, f"{case_data[case]['color']}--", label=f"{case} (Upper)")
        ax3[1].plot(angles, rc_middle, f"{case_data[case]['color']}-.", label=f"{case} (Middle)")
        ax3[1].plot(angles, rc_lower, f"{case_data[case]['color']}-", label=f"{case} (Lower)")
    
    # Finalize AVO plot
    ax3[1].set_title("AVO Response (Three Interfaces)")
    ax3[1].set_xlabel("Angle (degrees)")
    ax3[1].set_ylabel("Reflection Coefficient")
    ax3[1].grid(True)
    ax3[1].legend(fontsize='small')
    
    st.pyplot(fig3)

    # Create synthetic gathers for each case and layer
    st.header("Synthetic Seismic Gathers (Three Interfaces)")
    
    fig4, ax4 = plt.subplots(3, 3, figsize=(18, 12))
    
    for row, case in enumerate(cases):
        for col, layer in enumerate(layers.keys()):
            # Get properties for the current layer
            if layer == "Upper":
                vp_layer = logs.loc[(logs.DEPTH >= layers["Upper"]["top"]) & 
                                   (logs.DEPTH <= layers["Upper"]["bot"]), 
                                   case_data[case]['vp']].values.mean()
                vs_layer = logs.loc[(logs.DEPTH >= layers["Upper"]["top"]) & 
                                   (logs.DEPTH <= layers["Upper"]["bot"]), 
                                   case_data[case]['vs']].values.mean()
                rho_layer = logs.loc[(logs.DEPTH >= layers["Upper"]["top"]) & 
                                     (logs.DEPTH <= layers["Upper"]["bot"]), 
                                     case_data[case]['rho']].values.mean()
                
                # Shale above upper layer
                vp_sh = logs.loc[logs.VSH > 0.5, 'VP'].values.mean()
                vs_sh = logs.loc[logs.VSH > 0.5, 'VS'].values.mean()
                rho_sh = logs.loc[logs.VSH > 0.5, 'RHO'].values.mean()
                
                # Generate synthetic gather for upper interface
                syn_gather = []
                for angle in angles:
                    rc = calculate_reflection_coefficients(
                        vp_sh, vp_layer, vs_sh, vs_layer, rho_sh, rho_layer, angle
                    )
                    
                    # Create reflectivity series
                    rc_series = np.zeros(len(t_samp))
                    idx_upper = np.argmin(np.abs(t_samp - t_upper))
                    rc_series[idx_upper] = rc
                    
                    # Convolve with wavelet
                    syn_trace = np.convolve(rc_series, wlt_amp, mode='same')
                    syn_gather.append(syn_trace)
                
            elif layer == "Middle":
                # Upper layer properties
                vp_upper = logs.loc[(logs.DEPTH >= layers["Upper"]["top"]) & 
                                   (logs.DEPTH <= layers["Upper"]["bot"]), 
                                   case_data[case]['vp']].values.mean()
                vs_upper = logs.loc[(logs.DEPTH >= layers["Upper"]["top"]) & 
                                   (logs.DEPTH <= layers["Upper"]["bot"]), 
                                   case_data[case]['vs']].values.mean()
                rho_upper = logs.loc[(logs.DEPTH >= layers["Upper"]["top"]) & 
                                     (logs.DEPTH <= layers["Upper"]["bot"]), 
                                     case_data[case]['rho']].values.mean()
                
                # Middle layer properties
                vp_middle = logs.loc[(logs.DEPTH >= layers["Middle"]["top"]) & 
                                    (logs.DEPTH <= layers["Middle"]["bot"]), 
                                    case_data[case]['vp']].values.mean()
                vs_middle = logs.loc[(logs.DEPTH >= layers["Middle"]["top"]) & 
                                    (logs.DEPTH <= layers["Middle"]["bot"]), 
                                    case_data[case]['vs']].values.mean()
                rho_middle = logs.loc[(logs.DEPTH >= layers["Middle"]["top"]) & 
                                      (logs.DEPTH <= layers["Middle"]["bot"]), 
                                      case_data[case]['rho']].values.mean()
                
                # Generate synthetic gather for middle interface
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
                
            elif layer == "Lower":
                # Middle layer properties
                vp_middle = logs.loc[(logs.DEPTH >= layers["Middle"]["top"]) & 
                                    (logs.DEPTH <= layers["Middle"]["bot"]), 
                                    case_data[case]['vp']].values.mean()
                vs_middle = logs.loc[(logs.DEPTH >= layers["Middle"]["top"]) & 
                                    (logs.DEPTH <= layers["Middle"]["bot"]), 
                                    case_data[case]['vs']].values.mean()
                rho_middle = logs.loc[(logs.DEPTH >= layers["Middle"]["top"]) & 
                                      (logs.DEPTH <= layers["Middle"]["bot"]), 
                                      case_data[case]['rho']].values.mean()
                
                # Lower layer properties
                vp_lower = logs.loc[(logs.DEPTH >= layers["Lower"]["top"]) & 
                                   (logs.DEPTH <= layers["Lower"]["bot"]), 
                                   case_data[case]['vp']].values.mean()
                vs_lower = logs.loc[(logs.DEPTH >= layers["Lower"]["top"]) & 
                                   (logs.DEPTH <= layers["Lower"]["bot"]), 
                                   case_data[case]['vs']].values.mean()
                rho_lower = logs.loc[(logs.DEPTH >= layers["Lower"]["top"]) & 
                                     (logs.DEPTH <= layers["Lower"]["bot"]), 
                                     case_data[case]['rho']].values.mean()
                
                # Generate synthetic gather for lower interface
                syn_gather = []
                for angle in angles:
                    rc = calculate_reflection_coefficients(
                        vp_middle, vp_lower, vs_middle, vs_lower, rho_middle, rho_lower, angle
                    )
                    
                    rc_series = np.zeros(len(t_samp))
                    idx_lower = np.argmin(np.abs(t_samp - t_lower))
                    rc_series[idx_lower] = rc
                    
                    syn_trace = np.convolve(rc_series, wlt_amp, mode='same')
                    syn_gather.append(syn_trace)
            
            syn_gather = np.array(syn_gather)
            
            # Plot the gather
            extent = [angles[0], angles[-1], t_samp[-1], t_samp[0]]
            im = ax4[row, col].imshow(syn_gather.T, aspect='auto', extent=extent,
                                     cmap=selected_cmap, vmin=-np.max(np.abs(syn_gather)), 
                                     vmax=np.max(np.abs(syn_gather)))
            
            # Add title and labels
            ax4[row, col].set_title(f"{case} Case - {layer} Interface", fontsize=10)
            ax4[row, col].set_xlabel("Angle (degrees)", fontsize=8)
            ax4[row, col].set_ylabel("Time (s)", fontsize=8)
            ax4[row, col].set_ylim(0.35, 0.05)  # Focus on the interface time
            
            # Add colorbar
            plt.colorbar(im, ax=ax4[row, col], label='Amplitude')
    
    plt.tight_layout()
    st.pyplot(fig4)

else:
    st.info("Please upload a CSV file to begin analysis")
