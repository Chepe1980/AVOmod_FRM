import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.colors as colors
from pyavo.seismodel import wavelet
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from io import StringIO

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

    # Theory documentation
    with st.expander("Theory & Help"):
        st.markdown("""
        ## Rock Physics Theory
        - **Gassmann's Equations**: Used for fluid substitution
        - **AVO Modeling**: Based on Aki-Richards approximation
        - **VRH Averaging**: Voigt-Reuss-Hill average for mineral mixing
        """)

# Initialize session state for selected points
if 'selected_points' not in st.session_state:
    st.session_state.selected_points = []

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

# Fluid substitution function
def frm(vp1, vs1, rho1, rho_f1, k_f1, rho_f2, k_f2, k0, phi):
    # Add safety check
    phi = np.clip(phi, 1e-6, 1.0)  # Prevent zero porosity
    
    vp1 = vp1/1000.
    vs1 = vs1/1000.
    mu1 = rho1*vs1**2.
    k_s1 = rho1*vp1**2 - (4./3.)*mu1

    # The dry rock bulk modulus
    kdry = (k_s1*((phi*k0)/k_f1+1-phi)-k0)/((phi*k0)/k_f1+(k_s1/k0)-1-phi)

    # Now we can apply Gassmann to get the new values
    k_s2 = kdry + (1-(kdry/k0))**2/((phi/k_f2)+((1-phi)/k0)-(kdry/k0**2))
    rho2 = rho1-phi*rho_f1+phi*rho_f2
    mu2 = mu1
    vp2 = np.sqrt((k_s2+(4./3)*mu2)/rho2)
    vs2 = np.sqrt(mu2/rho2)

    return vp2*1000, vs2*1000, rho2, k_s2

# Main processing function
@st.cache_data
def process_data(logs, rho_qz, k_qz, mu_qz, rho_sh, k_sh, mu_sh, 
                rho_b, k_b, rho_o, k_o, rho_g, k_g, sand_cutoff):
    try:
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

        # Fluid substitution
        vpb, vsb, rhob, kb = frm(logs.VP, logs.VS, logs.RHO, rho_fl, k_fl, rho_b, k_b, k0, logs.PHI)
        vpo, vso, rhoo, ko = frm(logs.VP, logs.VS, logs.RHO, rho_fl, k_fl, rho_o, k_o, k0, logs.PHI)
        vpg, vsg, rhog, kg = frm(logs.VP, logs.VS, logs.RHO, rho_fl, k_fl, rho_g, k_g, k0, logs.PHI)

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
    
    except Exception as e:
        st.error(f"Error processing data: {str(e)}")
        return None

# Calculate reflection coefficients
@st.cache_data
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
        # Read data with validation
        logs = pd.read_csv(uploaded_file)
        
        # Validate required columns
        required_cols = ['DEPTH', 'VP', 'VS', 'RHO', 'VSH', 'SW', 'PHI']
        if not all(col in logs.columns for col in required_cols):
            st.error(f"Missing required columns. File must contain: {', '.join(required_cols)}")
            st.stop()
            
        # Process data
        logs = process_data(logs, rho_qz, k_qz, mu_qz, rho_sh, k_sh, mu_sh,
                           rho_b, k_b, rho_o, k_o, rho_g, k_g, sand_cutoff)
        
        if logs is None:
            st.stop()

        # Depth range selection
        st.header("Well Log Visualization")
        ztop, zbot = st.slider(
            "Select Depth Range", 
            float(logs.DEPTH.min()), 
            float(logs.DEPTH.max()), 
            (float(logs.DEPTH.min()), float(logs.DEPTH.max()))
        )
        
        # Filter data based on depth range
        ll = logs.loc[(logs.DEPTH>=ztop) & (logs.DEPTH<=zbot)].copy()
        
        # Create facies color map
        ccc = ['#B3B3B3','blue','green','red','#996633']
        facies_names = ['undef', 'brine', 'oil', 'gas', 'shale']
        ll['Facies'] = ll['LFC_B'].map({0: 'undef', 1: 'brine', 2: 'oil', 3: 'gas', 4: 'shale'})
        
        # Create log plot using Plotly
        fig_logs = make_subplots(rows=1, cols=4, 
                                subplot_titles=("Vcl/phi/Sw", "Ip", "Vp/Vs", "LFC"),
                                shared_yaxes=True,
                                horizontal_spacing=0.02)
        
        # Track 1: VSH, SW, PHI
        fig_logs.add_trace(go.Scatter(x=ll.VSH, y=ll.DEPTH, 
                                    name='VSH', line=dict(color='green')),
                         row=1, col=1)
        fig_logs.add_trace(go.Scatter(x=ll.SW, y=ll.DEPTH, 
                                    name='SW', line=dict(color='blue')),
                         row=1, col=1)
        fig_logs.add_trace(go.Scatter(x=ll.PHI, y=ll.DEPTH, 
                                    name='PHI', line=dict(color='black')),
                         row=1, col=1)
        
        # Track 2: IP
        fig_logs.add_trace(go.Scatter(x=ll.IP_FRMG, y=ll.DEPTH, 
                                    name='Gas', line=dict(color='red')),
                         row=1, col=2)
        fig_logs.add_trace(go.Scatter(x=ll.IP_FRMB, y=ll.DEPTH, 
                                    name='Brine', line=dict(color='blue')),
                         row=1, col=2)
        fig_logs.add_trace(go.Scatter(x=ll.IP, y=ll.DEPTH, 
                                    name='Original', line=dict(color='gray')),
                         row=1, col=2)
        
        # Track 3: Vp/Vs
        fig_logs.add_trace(go.Scatter(x=ll.VPVS_FRMG, y=ll.DEPTH, 
                                    name='Gas', line=dict(color='red')),
                         row=1, col=3)
        fig_logs.add_trace(go.Scatter(x=ll.VPVS_FRMB, y=ll.DEPTH, 
                                    name='Brine', line=dict(color='blue')),
                         row=1, col=3)
        fig_logs.add_trace(go.Scatter(x=ll.VPVS, y=ll.DEPTH, 
                                    name='Original', line=dict(color='gray')),
                         row=1, col=3)
        
        # Track 4: LFC
        for facies, color in zip(facies_names, ccc):
            subset = ll[ll['Facies'] == facies]
            fig_logs.add_trace(go.Scatter(x=[0]*len(subset), y=subset.DEPTH,
                              mode='markers', marker=dict(color=color, size=5),
                              name=facies, showlegend=False),
                           row=1, col=4)
        
        # Update layout
        fig_logs.update_layout(height=800, width=1200,
                             yaxis=dict(title='Depth', autorange='reversed'),
                             hovermode='y unified')
        
        fig_logs.update_xaxes(title_text="Vcl/phi/Sw", row=1, col=1, range=[-0.1, 1.1])
        fig_logs.update_xaxes(title_text="Ip [m/s*g/cc]", row=1, col=2, range=[6000, 15000])
        fig_logs.update_xaxes(title_text="Vp/Vs", row=1, col=3, range=[1.5, 2])
        fig_logs.update_xaxes(title_text="LFC", row=1, col=4, showticklabels=False)
        
        st.plotly_chart(fig_logs, use_container_width=True)

        # Crossplots with Plotly selection
        st.header("Crossplots with Interactive Selection")
        
        # Create crossplot figure
        fig_cross = make_subplots(rows=1, cols=4,
                                 subplot_titles=('Original Data', 'FRM to Brine', 
                                                'FRM to Oil', 'FRM to Gas'),
                                 shared_yaxes=True, shared_xaxes=True)
        
        # Original data crossplot
        fig1 = px.scatter(ll, x='IP', y='VPVS', color='Facies',
                         color_discrete_sequence=ccc,
                         hover_data=['DEPTH', 'VSH', 'PHI', 'SW'],
                         title='Original Data')
        
        # Brine FRM crossplot
        fig2 = px.scatter(ll, x='IP_FRMB', y='VPVS_FRMB', color='Facies',
                         color_discrete_sequence=ccc,
                         hover_data=['DEPTH', 'VSH', 'PHI', 'SW'],
                         title='FRM to Brine')
        
        # Oil FRM crossplot
        fig3 = px.scatter(ll, x='IP_FRMO', y='VPVS_FRMO', color='Facies',
                         color_discrete_sequence=ccc,
                         hover_data=['DEPTH', 'VSH', 'PHI', 'SW'],
                         title='FRM to Oil')
        
        # Gas FRM crossplot
        fig4 = px.scatter(ll, x='IP_FRMG', y='VPVS_FRMG', color='Facies',
                         color_discrete_sequence=ccc,
                         hover_data=['DEPTH', 'VSH', 'PHI', 'SW'],
                         title='FRM to Gas')
        
        # Combine all plots
        for trace in fig1.data:
            fig_cross.add_trace(trace, row=1, col=1)
        for trace in fig2.data:
            fig_cross.add_trace(trace, row=1, col=2)
        for trace in fig3.data:
            fig_cross.add_trace(trace, row=1, col=3)
        for trace in fig4.data:
            fig_cross.add_trace(trace, row=1, col=4)
        
        # Update layout
        fig_cross.update_layout(height=500, width=1200,
                              xaxis1=dict(title='IP', range=[3000, 16000]),
                              xaxis2=dict(title='IP', range=[3000, 16000]),
                              xaxis3=dict(title='IP', range=[3000, 16000]),
                              xaxis4=dict(title='IP', range=[3000, 16000]),
                              yaxis1=dict(title='Vp/Vs', range=[1.5, 3]),
                              showlegend=False)
        
        # Add selection functionality
        fig_cross.update_layout(
            dragmode='lasso',
            clickmode='event+select'
        )
        
        st.plotly_chart(fig_cross, use_container_width=True)
        
        # Handle selection
        selected_points = st.session_state.get('selected_points', [])
        selection_event = st.session_state.get('selection_event', None)
        
        if selection_event:
            selected_indices = selection_event['points']
            selected_points = [point['pointIndex'] for point in selected_indices]
            st.session_state.selected_points = selected_points
            st.session_state.selection_event = None
            
        if selected_points:
            st.write(f"{len(selected_points)} points selected")
            selected_data = ll.iloc[selected_points]
            st.dataframe(selected_data[['DEPTH', 'VSH', 'PHI', 'SW', 'VP', 'VS', 'RHO']])
        
        # AVO Modeling
        st.header("AVO Modeling")
        
        # Generate wavelet
        wlt_time, wlt_amp = wavelet.ricker(sample_rate=0.0001, length=0.128, c_freq=wavelet_freq)
        t_samp = np.arange(0, 0.5, 0.0001)  # Time samples
        t_middle = 0.2  # Fixed time for middle interface
        
        # Use middle of selected depth range for AVO analysis
        middle_depth = ztop + (zbot - ztop)/2
        window_size = (zbot - ztop) * 0.1  # 10% of the depth range
        
        # Get the indices for the middle zone
        middle_zone = logs.loc[
            (logs.DEPTH >= middle_depth - window_size) & 
            (logs.DEPTH <= middle_depth + window_size)
        ]
        
        # AVO cases
        cases = ['Brine', 'Oil', 'Gas']
        case_data = {
            'Brine': {'vp': 'VP_FRMB', 'vs': 'VS_FRMB', 'rho': 'RHO_FRMB', 'color': 'blue'},
            'Oil': {'vp': 'VP_FRMO', 'vs': 'VS_FRMO', 'rho': 'RHO_FRMO', 'color': 'green'},
            'Gas': {'vp': 'VP_FRMG', 'vs': 'VS_FRMG', 'rho': 'RHO_FRMG', 'color': 'red'}
        }
        
        # Create figure for AVO results with wavelet plot
        fig_avo = make_subplots(rows=1, cols=2, column_widths=[0.3, 0.7],
                              subplot_titles=(f"Wavelet ({wavelet_freq} Hz)", 
                                            "AVO Reflection Coefficients"))
        
        # Wavelet plot
        fig_avo.add_trace(go.Scatter(x=wlt_time, y=wlt_amp, 
                                   line=dict(color='purple', width=2),
                                   fill='tozeroy', fillcolor='rgba(128,0,128,0.3)',
                                   name='Wavelet'),
                        row=1, col=1)
        
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
            # Upper zone properties (shale) - using data just above middle zone
            upper_zone = logs.loc[
                (logs.DEPTH >= middle_depth - 2*window_size) & 
                (logs.DEPTH < middle_depth - window_size)
            ]
            
            vp_upper = upper_zone['VP'].mean()
            vs_upper = upper_zone['VS'].mean()
            rho_upper = upper_zone['RHO'].mean()
            
            # Middle zone properties (averaged)
            vp_middle = middle_zone[case_data[case]['vp']].mean()
            vs_middle = middle_zone[case_data[case]['vs']].mean()
            rho_middle = middle_zone[case_data[case]['rho']].mean()
            
            # Calculate reflection coefficients
            rc = []
            for angle in angles:
                rc.append(calculate_reflection_coefficients(
                    vp_upper, vp_middle, vs_upper, vs_middle, rho_upper, rho_middle, angle
                ))
            
            # Plot AVO curve
            fig_avo.add_trace(go.Scatter(x=angles, y=rc, 
                                       line=dict(color=case_data[case]['color']),
                                       name=f"{case}",
                                       showlegend=True),
                            row=1, col=2)
        
        # Update AVO plot layout
        fig_avo.update_layout(height=500, width=1200,
                            yaxis2=dict(title='Reflection Coefficient', range=[rc_min, rc_max]),
                            xaxis2=dict(title='Angle (degrees)'),
                            xaxis1=dict(title='Time (s)'),
                            yaxis1=dict(title='Amplitude'),
                            legend=dict(x=0.8, y=0.9))
        
        # Add property annotations
        props_text = (f"Depth: {middle_depth:.1f}-{middle_depth+window_size:.1f}m<br>"
                     f"Shale Vp: {vp_upper:.0f} m/s, Vs: {vs_upper:.0f} m/s, Rho: {rho_upper:.2f} g/cc")
        
        fig_avo.add_annotation(
            text=props_text,
            align='left',
            showarrow=False,
            xref='paper', yref='paper',
            x=0.05, y=0.95,
            bgcolor='white',
            row=1, col=2
        )
        
        st.plotly_chart(fig_avo, use_container_width=True)

        # Synthetic gathers for middle interface
        st.header("Synthetic Seismic Gathers (Middle Interface)")
        
        # Add slider for time range
        time_min, time_max = st.slider(
            "Time Range for Synthetic Gathers (s)",
            0.0, 0.5, (0.15, 0.25),
            step=0.01,
            key='time_range'
        )
        
        # Create synthetic gathers figure
        fig_synth = make_subplots(rows=1, cols=3,
                                subplot_titles=('Brine Case', 'Oil Case', 'Gas Case'),
                                shared_yaxes=True, shared_xaxes=True)
        
        for idx, case in enumerate(cases):
            # Upper zone properties (shale)
            upper_zone = logs.loc[
                (logs.DEPTH >= middle_depth - 2*window_size) & 
                (logs.DEPTH < middle_depth - window_size)
            ]
            
            vp_upper = upper_zone['VP'].mean()
            vs_upper = upper_zone['VS'].mean()
            rho_upper = upper_zone['RHO'].mean()
            
            # Middle zone properties
            vp_middle = middle_zone[case_data[case]['vp']].mean()
            vs_middle = middle_zone[case_data[case]['vs']].mean()
            rho_middle = middle_zone[case_data[case]['rho']].mean()
            
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
            fig_synth.add_trace(go.Heatmap(
                z=syn_gather.T,
                x=angles,
                y=t_samp,
                colorscale=selected_cmap,
                zmid=0,
                colorbar=dict(title='Amplitude', x=0.3 + idx*0.3),
                name=case
            ), row=1, col=idx+1)
            
            # Add property annotations
            props_text = (f"Vp: {vp_middle:.0f} m/s<br>"
                         f"Vs: {vs_middle:.0f} m/s<br>"
                         f"Rho: {rho_middle:.2f} g/cc")
            
            fig_synth.add_annotation(
                text=props_text,
                align='left',
                showarrow=False,
                xref=f'x{idx+1}', yref=f'y{idx+1}',
                x=0.05, y=0.95,
                xanchor='left', yanchor='top',
                bgcolor='white',
                row=1, col=idx+1
            )
        
        # Update layout
        fig_synth.update_layout(height=500, width=1200,
                              yaxis=dict(title='Time (s)', range=[time_max, time_min]),
                              xaxis1=dict(title='Angle (degrees)'),
                              xaxis2=dict(title='Angle (degrees)'),
                              xaxis3=dict(title='Angle (degrees)'))
        
        st.plotly_chart(fig_synth, use_container_width=True)
        
        # Export functionality
        st.header("Export Results")
        if st.button("Generate Export"):
            output = logs.to_csv(index=False)
            st.download_button(
                label="Download Processed Data",
                data=output,
                file_name="rock_physics_results.csv",
                mime="text/csv"
            )
    
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")

else:
    st.info("Please upload a CSV file to begin analysis")
