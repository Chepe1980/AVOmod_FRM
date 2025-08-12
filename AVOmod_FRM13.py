import streamlit as st
import numpy as np
import pandas as pd
from pyavo.seismodel import wavelet
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import matplotlib.pyplot as plt

# Set page config
st.set_page_config(layout="wide", page_title="Rock Physics & AVO Modeling")

# Title and description
st.title("Rock Physics & AVO Modeling Tool")
st.markdown("""
This app performs rock physics modeling and AVO analysis for brine, oil, and gas scenarios.
""")

# Available colormaps with Plotly equivalents
seismic_colormaps = {
    'seismic': 'RdBu',
    'RdBu': 'RdBu',
    'bwr': 'RdBu_r',
    'coolwarm': 'RdBu',
    'viridis': 'Viridis',
    'plasma': 'Plasma'
}

# Initialize session state for selected points
if 'selected_points' not in st.session_state:
    st.session_state.selected_points = []
if 'selection_event' not in st.session_state:
    st.session_state.selection_event = None

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
    phi = np.clip(phi, 1e-6, 1.0)
    vp1 = vp1/1000.
    vs1 = vs1/1000.
    mu1 = rho1*vs1**2.
    k_s1 = rho1*vp1**2 - (4./3.)*mu1

    kdry = (k_s1*((phi*k0)/k_f1+1-phi)-k0)/((phi*k0)/k_f1+(k_s1/k0)-1-phi)
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
        shale = logs.VSH.values
        sand = 1 - shale - logs.PHI.values
        shaleN = shale/(shale+sand)
        sandN = sand/(shale+sand)
        k_u, k_l, mu_u, mu_l, k0, mu0 = vrh([shaleN, sandN], [k_sh, k_qz], [mu_sh, mu_qz])

        water = logs.SW.values
        hc = 1 - logs.SW.values
        tmp, k_fl, tmp, tmp, tmp, tmp = vrh([water, hc], [k_b, k_o], [0, 0])
        rho_fl = water*rho_b + hc*rho_o

        vpb, vsb, rhob, kb = frm(logs.VP, logs.VS, logs.RHO, rho_fl, k_fl, rho_b, k_b, k0, logs.PHI)
        vpo, vso, rhoo, ko = frm(logs.VP, logs.VS, logs.RHO, rho_fl, k_fl, rho_o, k_o, k0, logs.PHI)
        vpg, vsg, rhog, kg = frm(logs.VP, logs.VS, logs.RHO, rho_fl, k_fl, rho_g, k_g, k0, logs.PHI)

        brine_sand = ((logs.VSH <= sand_cutoff) & (logs.SW >= 0.65))
        oil_sand = ((logs.VSH <= sand_cutoff) & (logs.SW < 0.65))
        shale_flag = (logs.VSH > sand_cutoff)

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

# Selection update function
def update_selection(trace, points, selector):
    st.session_state.selected_points = points.point_inds
    st.session_state.selection_event = {'points': points}

# Sidebar for input parameters
with st.sidebar:
    st.header("Mineral and Fluid Properties")
    
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
    
    st.subheader("AVO Modeling Parameters")
    min_angle = st.slider("Minimum Angle (deg)", 0, 10, 0)
    max_angle = st.slider("Maximum Angle (deg)", 30, 50, 45)
    wavelet_freq = st.slider("Wavelet Frequency (Hz)", 20, 80, 50)
    sand_cutoff = st.slider("Sand Cutoff (VSH)", 0.0, 0.3, 0.12, step=0.01)
    
    st.subheader("Seismic Display Options")
    col1, col2 = st.columns(2)
    with col1:
        selected_cmap = st.selectbox("Color Map", list(seismic_colormaps.keys()), index=0)
    with col2:
        show_wiggle = st.checkbox("Show Wiggle Traces", value=False)
    
    st.subheader("Well Log Data")
    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

    with st.expander("Theory & Help"):
        st.markdown("""
        ## Rock Physics Theory
        - **Gassmann's Equations**: Used for fluid substitution
        - **AVO Modeling**: Based on Aki-Richards approximation
        - **VRH Averaging**: Voigt-Reuss-Hill average for mineral mixing
        """)

# Main content area
if uploaded_file is not None:
    try:
        logs = pd.read_csv(uploaded_file)
        
        required_cols = ['DEPTH', 'VP', 'VS', 'RHO', 'VSH', 'SW', 'PHI']
        if not all(col in logs.columns for col in required_cols):
            st.error(f"Missing required columns. File must contain: {', '.join(required_cols)}")
            st.stop()
            
        logs = process_data(logs, rho_qz, k_qz, mu_qz, rho_sh, k_sh, mu_sh,
                          rho_b, k_b, rho_o, k_o, rho_g, k_g, sand_cutoff)
        
        if logs is None:
            st.stop()

        st.header("Well Log Visualization")
        ztop, zbot = st.slider(
            "Select Depth Range", 
            float(logs.DEPTH.min()), 
            float(logs.DEPTH.max()), 
            (float(logs.DEPTH.min()), float(logs.DEPTH.max()))
        )
        
        ll = logs.loc[(logs.DEPTH>=ztop) & (logs.DEPTH<=zbot)].copy()
        
        ccc = ['#B3B3B3','blue','green','red','#996633']
        facies_names = ['undef', 'brine', 'oil', 'gas', 'shale']
        ll['Facies'] = ll['LFC_B'].map({0: 'undef', 1: 'brine', 2: 'oil', 3: 'gas', 4: 'shale'})
        
        fig_logs = make_subplots(rows=1, cols=4, 
                               subplot_titles=("Vcl/phi/Sw", "Ip", "Vp/Vs", "LFC"),
                               shared_yaxes=True,
                               horizontal_spacing=0.02)
        
        # Add log traces...
        
        # Highlight selected points in log view
        if st.session_state.selected_points:
            selected_data = ll.iloc[st.session_state.selected_points]
            
            for col, x_col in enumerate(['VSH', 'IP', 'VPVS', 'LFC_B'], start=1):
                fig_logs.add_trace(go.Scatter(
                    x=selected_data[x_col],
                    y=selected_data['DEPTH'],
                    mode='markers',
                    marker=dict(
                        color='yellow',
                        size=10,
                        line=dict(color='black', width=1)
                    ),
                    name='Selected',
                    showlegend=(col == 1),
                    hoverinfo='none'
                ), row=1, col=col)
        
        fig_logs.update_layout(height=800, width=1200,
                             yaxis=dict(title='Depth', autorange='reversed'),
                             hovermode='y unified')
        
        # Add lasso selection to first trace
        fig_logs.data[0].on_selection(update_selection)
        st.plotly_chart(fig_logs, use_container_width=True)

        st.header("Crossplots with Interactive Selection")
        fig_cross = make_subplots(rows=1, cols=4,
                                subplot_titles=('Original Data', 'FRM to Brine', 
                                              'FRM to Oil', 'FRM to Gas'),
                                shared_yaxes=True, shared_xaxes=True)
        
        # Create crossplots with selection enabled
        for i, (x_col, y_col) in enumerate(zip(
            ['IP', 'IP_FRMB', 'IP_FRMO', 'IP_FRMG'],
            ['VPVS', 'VPVS_FRMB', 'VPVS_FRMO', 'VPVS_FRMG']
        )):
            fig = px.scatter(ll, x=x_col, y=y_col, color='Facies',
                           color_discrete_sequence=ccc,
                           hover_data=['DEPTH', 'VSH', 'PHI', 'SW'])
            
            fig.update_traces(
                selected=dict(marker=dict(color='yellow', size=10, line=dict(width=1, color='black'))),
                unselected=dict(marker=dict(opacity=0.3)),
                selector=dict(mode='markers')
            )
            
            for trace in fig.data:
                trace.update(selectedpoints=st.session_state.selected_points)
                fig_cross.add_trace(trace, row=1, col=i+1)
                trace.on_selection(update_selection)
        
        fig_cross.update_layout(
            dragmode='lasso',
            clickmode='event+select',
            height=500,
            width=1200,
            showlegend=False
        )
        
        # Update axes ranges
        for i in range(1,5):
            fig_cross.update_xaxes(range=[3000, 16000], row=1, col=i)
        fig_cross.update_yaxes(range=[1.5, 3], row=1, col=1)
        
        st.plotly_chart(fig_cross, use_container_width=True)
        
        # Display selected points
        if st.session_state.selected_points:
            st.write(f"{len(st.session_state.selected_points)} points selected")
            selected_data = ll.iloc[st.session_state.selected_points]
            st.dataframe(selected_data[['DEPTH', 'VSH', 'PHI', 'SW', 'VP', 'VS', 'RHO']])

        st.header("AVO Modeling")
        # AVO modeling code remains the same...

        st.header("Synthetic Seismic Gathers (Middle Interface)")
        time_min, time_max = st.slider(
            "Time Range for Synthetic Gathers (s)",
            0.0, 0.5, (0.15, 0.25),
            step=0.01,
            key='time_range'
        )
        
        # Create figure with colorbars outside
        fig_synth = make_subplots(
            rows=1, cols=3,
            subplot_titles=('Brine Case', 'Oil Case', 'Gas Case'),
            shared_yaxes=True, 
            shared_xaxes=True,
            horizontal_spacing=0.1
        )
        
        # Generate synthetic gathers
        for idx, case in enumerate(cases):
            # Generate synthetic data...
            
            # Add heatmap with external colorbar
            fig_synth.add_trace(go.Heatmap(
                z=syn_gather.T,
                x=angles,
                y=t_samp,
                colorscale=seismic_colormaps[selected_cmap],
                zmid=0,
                colorbar=dict(
                    title='Amplitude',
                    x=1.0 if idx == 2 else 0.33*idx + 0.33,
                    len=0.25,
                    y=0.5
                ),
                hoverongaps=False,
                showscale=True
            ), row=1, col=idx+1)
            
            # Add wiggle traces if enabled
            if show_wiggle:
                for i, angle in enumerate(angles):
                    if i % 5 == 0:  # Show every 5th trace for clarity
                        fig_synth.add_trace(go.Scatter(
                            x=syn_gather[i] + angle,
                            y=t_samp,
                            mode='lines',
                            line=dict(color='black', width=1),
                            showlegend=False,
                            hoverinfo='none'
                        ), row=1, col=idx+1)
        
        fig_synth.update_layout(
            height=500,
            width=1200,
            yaxis=dict(title='Time (s)', range=[time_max, time_min]),
            xaxis1=dict(title='Angle (degrees)'),
            xaxis2=dict(title='Angle (degrees)'),
            xaxis3=dict(title='Angle (degrees)')
        )
        
        st.plotly_chart(fig_synth, use_container_width=True)
        
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
