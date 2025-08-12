import streamlit as st
import numpy as np
import pandas as pd
from pyavo.seismodel import wavelet
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px

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

def vrh(volumes, k, mu):
    """Voigt-Reuss-Hill average for mineral mixing"""
    f = np.array(volumes).T
    k = np.resize(np.array(k), np.shape(f))
    mu = np.resize(np.array(mu), np.shape(f))

    # Voigt bounds
    k_u = np.sum(f*k, axis=1)
    mu_u = np.sum(f*mu, axis=1)
    
    # Reuss bounds
    k_l = 1. / np.sum(f/k, axis=1)
    mu_l = 1. / np.sum(f/mu, axis=1)
    
    # Hill average
    k0 = (k_u + k_l)/2.
    mu0 = (mu_u + mu_l)/2.
    
    return k_u, k_l, mu_u, mu_l, k0, mu0

def frm(vp1, vs1, rho1, rho_f1, k_f1, rho_f2, k_f2, k0, phi):
    """Gassmann fluid substitution"""
    # Convert units and calculate moduli
    vp1 = vp1/1000.
    vs1 = vs1/1000.
    mu1 = rho1 * vs1**2
    k_s1 = rho1 * vp1**2 - (4./3.) * mu1
    
    # Calculate dry rock bulk modulus
    kdry = (k_s1*((phi*k0)/k_f1 + 1 - phi) - k0) / ((phi*k0)/k_f1 + (k_s1/k0) - 1 - phi)
    
    # Apply Gassmann's equation
    k_s2 = kdry + (1 - (kdry/k0))**2 / ((phi/k_f2) + ((1 - phi)/k0) - (kdry/k0**2))
    rho2 = rho1 - phi*rho_f1 + phi*rho_f2
    mu2 = mu1  # Shear modulus doesn't change with fluid substitution
    
    # Calculate new velocities
    vp2 = np.sqrt((k_s2 + (4./3)*mu2)/rho2) * 1000  # Convert back to m/s
    vs2 = np.sqrt(mu2/rho2) * 1000  # Convert back to m/s
    
    return vp2, vs2, rho2, k_s2

@st.cache_data
def process_data(logs, rho_qz, k_qz, mu_qz, rho_sh, k_sh, mu_sh, 
                rho_b, k_b, rho_o, k_o, rho_g, k_g, sand_cutoff):
    """Process well log data through rock physics workflow"""
    try:
        # Calculate mineral properties using VRH
        shale = logs.VSH.values
        sand = 1 - shale - logs.PHI.values
        shaleN = shale/(shale + sand + 1e-10)  # Avoid division by zero
        sandN = sand/(shale + sand + 1e-10)
        
        _, _, _, _, k0, mu0 = vrh([shaleN, sandN], [k_sh, k_qz], [mu_sh, mu_qz])
        
        # Calculate fluid properties
        water = logs.SW.values
        hc = 1 - logs.SW.values
        _, k_fl, _, _, _, _ = vrh([water, hc], [k_b, k_o], [0, 0])
        rho_fl = water*rho_b + hc*rho_o
        
        # Perform fluid substitution for all cases
        results = {}
        for case, (rho_f, k_f) in zip(['B', 'O', 'G'], 
                                      [(rho_b, k_b), (rho_o, k_o), (rho_g, k_g)]):
            vp, vs, rho, _ = frm(logs.VP, logs.VS, logs.RHO, rho_fl, k_fl, rho_f, k_f, k0, logs.PHI)
            
            # Store results
            results[case] = {
                'VP': vp,
                'VS': vs,
                'RHO': rho,
                'IP': vp * rho,
                'IS': vs * rho,
                'VPVS': vp / vs
            }
        
        # Create litho-fluid classification
        brine_sand = ((logs.VSH <= sand_cutoff) & (logs.SW >= 0.65))
        oil_sand = ((logs.VSH <= sand_cutoff) & (logs.SW < 0.65))
        shale_flag = (logs.VSH > sand_cutoff)
        
        # Add results to logs dataframe
        for case in ['B', 'O', 'G']:
            for prop in ['VP', 'VS', 'RHO', 'IP', 'IS', 'VPVS']:
                logs[f'{prop}_FRM{case}'] = logs[prop] if prop in logs else np.nan
                logs.loc[brine_sand | oil_sand, f'{prop}_FRM{case}'] = results[case][prop][brine_sand | oil_sand]
            
            # Add LFC flags
            temp_lfc = np.zeros(len(logs))
            temp_lfc[brine_sand] = 1
            temp_lfc[oil_sand] = 2
            temp_lfc[shale_flag] = 4
            logs[f'LFC_{case}'] = temp_lfc
        
        return logs
    
    except Exception as e:
        st.error(f"Error processing data: {str(e)}")
        return None

def calculate_reflection_coefficients(vp1, vp2, vs1, vs2, rho1, rho2, angle):
    """Aki-Richards approximation for reflection coefficients"""
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
    
    return a*(dvp/vp_avg) + b*(dvs/vs_avg) + c*(drho/rho_avg)

def update_selection(trace, points, selector):
    """Update selected points in session state"""
    st.session_state.selected_points = points.point_inds
    st.session_state.selection_event = {'points': points}

# Sidebar for input parameters
with st.sidebar:
    st.header("Input Parameters")
    
    with st.expander("Mineral Properties"):
        col1, col2 = st.columns(2)
        with col1:
            rho_qz = st.number_input("Quartz Density (g/cc)", 2.65, 3.0, 2.65, 0.01)
            k_qz = st.number_input("Quartz Bulk Modulus (GPa)", 30.0, 50.0, 37.0, 0.1)
            mu_qz = st.number_input("Quartz Shear Modulus (GPa)", 40.0, 50.0, 44.0, 0.1)
        with col2:
            rho_sh = st.number_input("Shale Density (g/cc)", 2.5, 3.0, 2.81, 0.01)
            k_sh = st.number_input("Shale Bulk Modulus (GPa)", 10.0, 20.0, 15.0, 0.1)
            mu_sh = st.number_input("Shale Shear Modulus (GPa)", 1.0, 10.0, 5.0, 0.1)
    
    with st.expander("Fluid Properties"):
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("**Brine**")
            rho_b = st.number_input("Density (g/cc)", 1.0, 1.2, 1.09, 0.01)
            k_b = st.number_input("Bulk Modulus (GPa)", 2.0, 3.5, 2.8, 0.1)
        with col2:
            st.markdown("**Oil**")
            rho_o = st.number_input("Density (g/cc)", 0.7, 0.9, 0.78, 0.01)
            k_o = st.number_input("Bulk Modulus (GPa)", 0.5, 1.5, 0.94, 0.1)
        with col3:
            st.markdown("**Gas**")
            rho_g = st.number_input("Density (g/cc)", 0.1, 0.3, 0.25, 0.01)
            k_g = st.number_input("Bulk Modulus (GPa)", 0.01, 0.1, 0.06, 0.01)
    
    with st.expander("Modeling Parameters"):
        min_angle = st.slider("Minimum Angle (deg)", 0, 10, 0)
        max_angle = st.slider("Maximum Angle (deg)", 30, 50, 45)
        wavelet_freq = st.slider("Wavelet Frequency (Hz)", 20, 80, 50)
        sand_cutoff = st.slider("Sand Cutoff (VSH)", 0.0, 0.3, 0.12, 0.01)
    
    with st.expander("Display Options"):
        selected_cmap = st.selectbox("Color Map", list(seismic_colormaps.keys()), 0)
        show_wiggle = st.checkbox("Show Wiggle Traces", False)
    
    uploaded_file = st.file_uploader("Upload Well Log CSV", type=["csv"])

# Main app
if uploaded_file is not None:
    try:
        logs = pd.read_csv(uploaded_file)
        required_cols = ['DEPTH', 'VP', 'VS', 'RHO', 'VSH', 'SW', 'PHI']
        
        if not all(col in logs.columns for col in required_cols):
            st.error(f"Missing required columns: {', '.join(required_cols)}")
            st.stop()
            
        logs = process_data(logs, rho_qz, k_qz, mu_qz, rho_sh, k_sh, mu_sh,
                          rho_b, k_b, rho_o, k_o, rho_g, k_g, sand_cutoff)
        
        if logs is None:
            st.stop()

        # Well Log Visualization
        st.header("Well Log Visualization")
        depth_range = st.slider("Depth Range", float(logs.DEPTH.min()), 
                               float(logs.DEPTH.max()), 
                               (float(logs.DEPTH.min()), float(logs.DEPTH.max())))
        
        log_data = logs[(logs.DEPTH >= depth_range[0]) & (logs.DEPTH <= depth_range[1])].copy()
        log_data['Facies'] = log_data['LFC_B'].map({0: 'undef', 1: 'brine', 2: 'oil', 3: 'gas', 4: 'shale'})
        
        fig_logs = make_subplots(rows=1, cols=4, shared_yaxes=True,
                               subplot_titles=("Vcl/phi/Sw", "Ip", "Vp/Vs", "LFC"))
        
        # Track 1: VSH, SW, PHI
        fig_logs.add_trace(go.Scatter(x=log_data.VSH, y=log_data.DEPTH, name='VSH', 
                                    line=dict(color='green')), row=1, col=1)
        fig_logs.add_trace(go.Scatter(x=log_data.SW, y=log_data.DEPTH, name='SW', 
                                    line=dict(color='blue')), row=1, col=1)
        fig_logs.add_trace(go.Scatter(x=log_data.PHI, y=log_data.DEPTH, name='PHI', 
                                    line=dict(color='black')), row=1, col=1)
        
        # Track 2: IP
        for case, color in zip(['G', 'B', ''], ['red', 'blue', 'gray']):
            col = f'IP_FRM{case}' if case else 'IP'
            name = 'Gas' if case == 'G' else 'Brine' if case == 'B' else 'Original'
            fig_logs.add_trace(go.Scatter(x=log_data[col], y=log_data.DEPTH, 
                                        name=name, line=dict(color=color)), row=1, col=2)
        
        # Track 3: Vp/Vs
        for case, color in zip(['G', 'B', ''], ['red', 'blue', 'gray']):
            col = f'VPVS_FRM{case}' if case else 'VPVS'
            name = 'Gas' if case == 'G' else 'Brine' if case == 'B' else 'Original'
            fig_logs.add_trace(go.Scatter(x=log_data[col], y=log_data.DEPTH, 
                                        name=name, line=dict(color=color)), row=1, col=3)
        
        # Track 4: LFC
        colors = ['#B3B3B3','blue','green','red','#996633']
        for facies, color in zip(['undef', 'brine', 'oil', 'gas', 'shale'], colors):
            subset = log_data[log_data['Facies'] == facies]
            fig_logs.add_trace(go.Scatter(x=[0]*len(subset), y=subset.DEPTH,
                                        mode='markers', marker=dict(color=color, size=5),
                                        name=facies, showlegend=False), row=1, col=4)
        
        # Highlight selected points
        if st.session_state.selected_points:
            selected_data = log_data.iloc[st.session_state.selected_points]
            for col, x_col in enumerate(['VSH', 'IP', 'VPVS', 'LFC_B'], 1):
                fig_logs.add_trace(go.Scatter(
                    x=selected_data[x_col], y=selected_data.DEPTH,
                    mode='markers', marker=dict(color='yellow', size=10),
                    name='Selected', showlegend=(col==1), hoverinfo='none'
                ), row=1, col=col)
        
        fig_logs.update_layout(height=800, yaxis=dict(title='Depth', autorange='reversed'),
                             hovermode='y unified')
        fig_logs.update_xaxes(title_text="Vcl/phi/Sw", row=1, col=1, range=[-0.1, 1.1])
        fig_logs.update_xaxes(title_text="Ip [m/s*g/cc]", row=1, col=2, range=[6000, 15000])
        fig_logs.update_xaxes(title_text="Vp/Vs", row=1, col=3, range=[1.5, 2])
        fig_logs.update_xaxes(title_text="LFC", row=1, col=4, showticklabels=False)
        
        if len(fig_logs.data) > 0:
            fig_logs.data[0].on_selection(update_selection)
        st.plotly_chart(fig_logs, use_container_width=True)

        # Crossplots
        st.header("Crossplots with Interactive Selection")
        fig_cross = make_subplots(rows=1, cols=4, shared_yaxes=True,
                                subplot_titles=('Original', 'Brine', 'Oil', 'Gas'))
        
        for i, case in enumerate(['', 'B', 'O', 'G']):
            x_col = f'IP_FRM{case}' if case else 'IP'
            y_col = f'VPVS_FRM{case}' if case else 'VPVS'
            
            fig = px.scatter(log_data, x=x_col, y=y_col, color='Facies',
                           color_discrete_map={'undef':'#B3B3B3', 'brine':'blue', 
                                             'oil':'green', 'gas':'red', 'shale':'#996633'},
                           hover_data=['DEPTH', 'VSH', 'PHI', 'SW'])
            
            fig.update_traces(
                selected=dict(marker=dict(color='yellow', size=10)),
                unselected=dict(marker=dict(opacity=0.3)),
                selector=dict(mode='markers')
            )
            
            for trace in fig.data:
                trace.update(selectedpoints=st.session_state.selected_points)
                fig_cross.add_trace(trace, row=1, col=i+1)
                trace.on_selection(update_selection)
        
        fig_cross.update_layout(
            dragmode='lasso', clickmode='event+select', height=500, showlegend=False
        )
        fig_cross.update_xaxes(range=[3000, 16000])
        fig_cross.update_yaxes(range=[1.5, 3], row=1, col=1)
        
        st.plotly_chart(fig_cross, use_container_width=True)
        
        # Display selected points
        if st.session_state.selected_points:
            st.write(f"{len(st.session_state.selected_points)} points selected")
            st.dataframe(log_data.iloc[st.session_state.selected_points]
                        [['DEPTH', 'VSH', 'PHI', 'SW', 'VP', 'VS', 'RHO']])

        # AVO Modeling
        st.header("AVO Modeling")
        wlt_time, wlt_amp = wavelet.ricker(0.0001, 0.128, wavelet_freq)
        
        middle_depth = np.mean(depth_range)
        window_size = (depth_range[1] - depth_range[0]) * 0.1
        middle_zone = logs[(logs.DEPTH >= middle_depth - window_size) & 
                          (logs.DEPTH <= middle_depth + window_size)]
        upper_zone = logs[(logs.DEPTH >= middle_depth - 2*window_size) & 
                         (logs.DEPTH < middle_depth - window_size)]
        
        # Get average properties
        vp_upper = upper_zone['VP'].mean()
        vs_upper = upper_zone['VS'].mean()
        rho_upper = upper_zone['RHO'].mean()
        
        fig_avo = make_subplots(rows=1, cols=2, column_widths=[0.3, 0.7],
                              subplot_titles=(f"Wavelet ({wavelet_freq} Hz)", "AVO Curves"))
        
        # Wavelet plot
        fig_avo.add_trace(go.Scatter(x=wlt_time, y=wlt_amp, name='Wavelet',
                                   line=dict(color='purple', width=2),
                                   fill='tozeroy', fillcolor='rgba(128,0,128,0.3)'),
                        row=1, col=1)
        
        # AVO curves
        angles = np.arange(min_angle, max_angle + 1)
        rc_min, rc_max = st.slider("Reflection Coefficient Range", -0.5, 0.5, (-0.2, 0.2), 0.01)
        
        for case, color in zip(['B', 'O', 'G'], ['blue', 'green', 'red']):
            vp_middle = middle_zone[f'VP_FRM{case}'].mean()
            vs_middle = middle_zone[f'VS_FRM{case}'].mean()
            rho_middle = middle_zone[f'RHO_FRM{case}'].mean()
            
            rc = [calculate_reflection_coefficients(vp_upper, vp_middle, vs_upper, vs_middle,
                                                  rho_upper, rho_middle, angle) for angle in angles]
            
            fig_avo.add_trace(go.Scatter(x=angles, y=rc, name=case,
                                       line=dict(color=color), mode='lines+markers',
                                       marker=dict(size=8)), row=1, col=2)
        
        fig_avo.update_layout(height=500, legend=dict(x=0.8, y=0.9))
        fig_avo.update_yaxes(range=[rc_min, rc_max], row=1, col=2, title='Reflection Coefficient')
        fig_avo.update_xaxes(title='Angle (deg)', row=1, col=2)
        fig_avo.update_xaxes(title='Time (s)', row=1, col=1)
        fig_avo.update_yaxes(title='Amplitude', row=1, col=1)
        
        st.plotly_chart(fig_avo, use_container_width=True)

        # Synthetic Seismic Gathers
        st.header("Synthetic Seismic Gathers")
        t_samp = np.arange(0, 0.5, 0.0001)
        t_middle = 0.2
        time_range = st.slider("Time Range (s)", 0.0, 0.5, (0.15, 0.25), 0.01)
        
        fig_synth = make_subplots(rows=1, cols=3, shared_yaxes=True,
                                subplot_titles=('Brine', 'Oil', 'Gas'))
        
        for idx, case in enumerate(['B', 'O', 'G']):
            vp_middle = middle_zone[f'VP_FRM{case}'].mean()
            vs_middle = middle_zone[f'VS_FRM{case}'].mean()
            rho_middle = middle_zone[f'RHO_FRM{case}'].mean()
            
            syn_gather = []
            for angle in angles:
                rc = calculate_reflection_coefficients(vp_upper, vp_middle, vs_upper, vs_middle,
                                                      rho_upper, rho_middle, angle)
                
                rc_series = np.zeros(len(t_samp))
                rc_series[np.argmin(np.abs(t_samp - t_middle))] = rc
                syn_trace = np.convolve(rc_series, wlt_amp, mode='same')
                syn_gather.append(syn_trace)
            
            syn_gather = np.array(syn_gather)
            
            # Heatmap
            fig_synth.add_trace(go.Heatmap(
                z=syn_gather.T, x=angles, y=t_samp,
                colorscale=seismic_colormaps[selected_cmap], zmid=0,
                colorbar=dict(title='Amplitude', x=1.0 if idx==2 else 0.33*idx+0.33, len=0.25, y=0.5),
                hoverongaps=False, showscale=True
            ), row=1, col=idx+1)
            
            # Wiggle traces
            if show_wiggle:
                for i, angle in enumerate(angles):
                    if i % 5 == 0:  # Show every 5th trace
                        fig_synth.add_trace(go.Scatter(
                            x=syn_gather[i]*5000 + angle, y=t_samp,
                            mode='lines', line=dict(color='black', width=1),
                            showlegend=False, hoverinfo='none'
                        ), row=1, col=idx+1)
        
        fig_synth.update_layout(height=500, yaxis=dict(title='Time (s)', range=time_range[::-1]))
        fig_synth.update_xaxes(title='Angle (deg)')
        
        st.plotly_chart(fig_synth, use_container_width=True)
        
        # Export Results
        st.header("Export Results")
        if st.button("Generate Export"):
            csv = logs.to_csv(index=False)
            st.download_button(
                "Download Processed Data",
                data=csv,
                file_name="rock_physics_results.csv",
                mime="text/csv"
            )
    
    except Exception as e:
        st.error(f"Error: {str(e)}")
else:
    st.info("Please upload a well log CSV file to begin analysis")
