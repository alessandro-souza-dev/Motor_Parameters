#!/usr/bin/python

# -*- coding: utf-8 -*-

"""

SPE Moto — Induction Motor Parameter Estimation Tool

Streamlit + Plotly UI  (replaces PyQt5 main.py)

"""



import json

import os

import io



import numpy as np

import streamlit as st  # type: ignore

import plotly.graph_objects as go  # type: ignore

from plotly.subplots import make_subplots  # type: ignore



import globals

import saveload

from common_calcs import calc_pqt, get_torque, get_torque_sc

from descent import nr_solver, lm_solver, dnr_solver, nr_solver_sc

from genetic import ga_solver

from hybrid import hy_solver

from lab_calcs import (

    double_cage_curves,

    double_cage_performance,

    double_cage_summary,

    estimate_single_cage_parameters,

    fit_double_cage_parameters,

    load_point_summary,

    refine_single_cage_parameters,

    rated_torque,

    single_cage_curves,

    single_cage_performance,

    single_cage_summary,

    synchronous_speed,

)



BASE_DIR = os.path.dirname(os.path.abspath(__file__))

LAB_REPORT_DEFAULTS = {

    "manufacturer": "ALSTOM",

    "serial_number": "00003",

    "manufacturing_year": "2002",

    "rated_power_w": 4_772_710.0,

    "rated_voltage_v": 13_200.0,

    "rated_current_a": 238.8,

    "pole_count": 2,

    "stator_connection": "WYE",

    "frequency_hz": 60.0,

    "rated_speed_rpm": 3572.96,

    "rated_pf": 0.9020,

    "rated_eff_pct": 96.48,

    "blocked_temp_c": 54.0,


    "resistance_input_mode": "line_to_line",

    "resistance_measurement_temp_c": 18.0,

    "blocked_resistance_ohm": 0.35700,

    "blocked_voltage_v": 2604.0,

    "blocked_current_a": 248.0,

    "blocked_power_w": 167_900.0,

    "blocked_frequency_hz": 60.0,

    "starting_current_a": 1167.0,

    "no_load_temp_c": 37.0,

    "no_load_voltage_v": 13_200.0,

    "no_load_current_a": 43.6,

    "no_load_power_w": 74_300.0,

    "fw_loss_w": 43_200.0,

}





def resource_path(*parts):

    return os.path.join(BASE_DIR, *parts)





# ─────────────────────────────────────────────

#  Session-state bootstrap

# ─────────────────────────────────────────────

def init_session():

    if not hasattr(globals, 'motor_data'):

        globals.init()

    if "globals_ready" not in st.session_state:

        st.session_state.globals_ready = True



    for key, default in {

        "results": None,          # dict with z, iter, err, conv

        "lab_results": None,      # dict from build_lab_*_results

    }.items():

        if key not in st.session_state:

            st.session_state[key] = default





# ─────────────────────────────────────────────

#  Page config

# ─────────────────────────────────────────────

st.set_page_config(

    page_title="SPE Moto",

    page_icon=resource_path("icons", "motor.png"),

    layout="wide",

    initial_sidebar_state="collapsed",

)



# ─── minimal global CSS ───────────────────────

st.markdown("""

<style>

    /* tighten default padding */

    .block-container { padding-top: 1.2rem; padding-bottom: 1rem; }

    /* card-style group boxes */

    .moto-card {

        background: #f0f4fa;

        border: 1px solid #c8d8ee;

        border-radius: 8px;

        padding: 1rem 1.2rem 0.6rem 1.2rem;

        margin-bottom: 0.8rem;

    }

    .moto-card h4 { margin: 0 0 0.6rem 0; color: #1a3a5c; font-size: 0.95rem; }

    /* result badges */

    .result-ok  { color: #1a7a3a; font-weight: 700; }

    .result-bad { color: #b22222; font-weight: 700; }

</style>

""", unsafe_allow_html=True)





init_session()



# ─────────────────────────────────────────────

#  Helper: status bar replacement

# ─────────────────────────────────────────────

_status_placeholder = None





def status(msg: str):

    if _status_placeholder:

        _status_placeholder.caption(f"ℹ️ {msg}")





# ─────────────────────────────────────────────

#  Header

# ─────────────────────────────────────────────

col_logo, col_title = st.columns([1, 10])

logo_path = resource_path("icons", "motor.png")

if os.path.exists(logo_path):

    col_logo.image(logo_path, width=48)

col_title.markdown("## SPE Moto — Induction Motor Parameter Estimation")



tab_est, tab_lab = st.tabs(["⚡ Parameter Estimation", "🔬 Laboratory Tests"])





# ═══════════════════════════════════════════════════════════

#  TAB 1 — Parameter Estimation

# ═══════════════════════════════════════════════════════════

with tab_est:



    # ── File load / save ──────────────────────────────────

    with st.expander("📂 Open / Save motor file (.mto)", expanded=False):

        col_up, col_dn = st.columns(2)

        with col_up:

            uploaded = st.file_uploader("Open .mto file", type=["mto"], key="mto_upload")

            if uploaded is not None:

                tmp_path = resource_path("library", "_tmp_upload.mto")

                with open(tmp_path, "wb") as f:

                    f.write(uploaded.getbuffer())

                saveload.load_file(tmp_path)

                os.remove(tmp_path)

                st.success("File loaded.")

                st.rerun()



        with col_dn:

            if st.button("💾 Save current data as .mto"):

                tmp_path = resource_path("library", "_tmp_save.mto")

                saveload.save_file(tmp_path)

                with open(tmp_path, "rb") as f:

                    buf = f.read()

                os.remove(tmp_path)

                st.download_button(

                    "⬇️ Download .mto",

                    data=buf,

                    file_name=f"{globals.motor_data['description']}.mto",

                    mime="text/plain",

                )



    # ── Layout: Motor | Model | Settings | Results ────────

    col_motor, col_model, col_settings, col_results = st.columns([2.2, 1.4, 2.2, 2.2])



    # ── Motor data ────────────────────────────────────────

    with col_motor:

        st.markdown('<div class="moto-card"><h4>Motor</h4>', unsafe_allow_html=True)

        desc = st.text_input("Description", value=globals.motor_data["description"], key="t_desc")

        c1, c2 = st.columns(2)

        sync_speed = c1.number_input("Sync speed (rpm)", value=float(globals.motor_data["sync_speed"]),

                                     step=1.0, format="%.1f", key="t_sync")

        rated_speed = c2.number_input("Rated speed (rpm)", value=float(globals.motor_data["rated_speed"]),

                                      step=1.0, format="%.1f", key="t_rated")

        c3, c4 = st.columns(2)

        rated_pf = c3.number_input("Rated pf", value=float(globals.motor_data["rated_pf"]),

                                   step=0.01, format="%.4f", key="t_pf")

        rated_eff = c4.number_input("Rated eff (pu)", value=float(globals.motor_data["rated_eff"]),

                                    step=0.01, format="%.4f", key="t_eff")

        c5, c6, c7 = st.columns(3)

        T_b = c5.number_input("T_b (T/Tn)", value=float(globals.motor_data["T_b"]),

                              step=0.1, format="%.2f", key="t_Tb")

        T_lr = c6.number_input("T_lr (T/Tn)", value=float(globals.motor_data["T_lr"]),

                               step=0.1, format="%.2f", key="t_Tlr")

        I_lr = c7.number_input("I_lr (pu)", value=float(globals.motor_data["I_lr"]),

                               step=0.1, format="%.2f", key="t_Ilr")

        st.markdown("</div>", unsafe_allow_html=True)



        # push to globals immediately

        globals.motor_data["description"] = desc

        globals.motor_data["sync_speed"] = sync_speed

        globals.motor_data["rated_speed"] = rated_speed

        globals.motor_data["rated_pf"] = rated_pf

        globals.motor_data["rated_eff"] = rated_eff

        globals.motor_data["T_b"] = T_b

        globals.motor_data["T_lr"] = T_lr

        globals.motor_data["I_lr"] = I_lr



    # ── Model ─────────────────────────────────────────────

    with col_model:

        st.markdown('<div class="moto-card"><h4>Model</h4>', unsafe_allow_html=True)

        model_choice = st.radio("Model", ["Single cage", "Double cage"], index=1, key="r_model")

        img_name = "single_cage.png" if model_choice == "Single cage" else "dbl_cage.png"

        img_path = resource_path("images", img_name)

        if os.path.exists(img_path):

            st.image(img_path, use_column_width=True)

        st.markdown("</div>", unsafe_allow_html=True)



    # ── Algorithm settings ────────────────────────────────

    with col_settings:

        st.markdown('<div class="moto-card"><h4>Settings</h4>', unsafe_allow_html=True)



        algo_options_double = [

            "Newton-Raphson", "Levenberg-Marquardt", "Damped Newton-Raphson",

            "Genetic Algorithm", "Hybrid GA-NR", "Hybrid GA-LM", "Hybrid GA-DNR",

        ]

        algo_options_single = ["Newton-Raphson"]



        algo_options = algo_options_single if model_choice == "Single cage" else algo_options_double

        algo = st.selectbox("Algorithm", algo_options, key="sel_algo")



        is_ga = algo in ("Genetic Algorithm", "Hybrid GA-NR", "Hybrid GA-LM", "Hybrid GA-DNR")



        c1, c2 = st.columns(2)

        max_iter = c1.number_input("Max iterations", value=int(globals.algo_data["max_iter"]),

                                   step=1, min_value=1, key="t_maxiter")

        conv_err = c2.number_input("Convergence criterion", value=float(globals.algo_data["conv_err"]),

                                   format="%.2e", step=1e-6, key="t_converr")



        if not is_ga:

            c3, c4 = st.columns(2)

            k_r = c3.number_input("k_r", value=float(globals.algo_data["k_r"]),

                                  step=0.1, format="%.4f", key="t_kr")

            k_x = c4.number_input("k_x", value=float(globals.algo_data["k_x"]),

                                  step=0.1, format="%.4f", key="t_kx")

            globals.algo_data["k_r"] = k_r

            globals.algo_data["k_x"] = k_x

        else:

            c3, c4 = st.columns(2)

            n_gen = c3.number_input("Max generations", value=int(globals.algo_data["n_gen"]),

                                    step=1, min_value=1, key="t_ngen")

            pop = c4.number_input("Population", value=int(globals.algo_data["pop"]),

                                  step=1, min_value=2, key="t_pop")

            c5, c6, c7 = st.columns(3)

            n_r = c5.number_input("Mating pool", value=int(globals.algo_data["n_r"]),

                                  step=1, min_value=1, key="t_nr")

            n_e = c6.number_input("Elite", value=int(globals.algo_data["n_e"]),

                                  step=1, min_value=0, key="t_ne")

            c_f = c7.number_input("Crossover frac.", value=float(globals.algo_data["c_f"]),

                                  step=0.05, format="%.2f", key="t_cf")

            globals.algo_data["n_gen"] = n_gen

            globals.algo_data["pop"] = pop

            globals.algo_data["n_r"] = n_r

            globals.algo_data["n_e"] = n_e

            globals.algo_data["c_f"] = c_f



        globals.algo_data["max_iter"] = max_iter

        globals.algo_data["conv_err"] = conv_err



        st.markdown("</div>", unsafe_allow_html=True)



        calculate_btn = st.button("▶ Calculate", type="primary", use_container_width=True)



    # ── Results ───────────────────────────────────────────

    with col_results:

        st.markdown('<div class="moto-card"><h4>Results</h4>', unsafe_allow_html=True)

        res = st.session_state.results



        def _show(label, key, res):

            val = ""

            if res and key in res:

                v = res[key]

                val = str(np.round(v, 5)) if v is not None else ""

            st.text_input(label, value=val, disabled=True, key=f"res_{key}")



        c1, c2 = st.columns(2)

        with c1:

            _show("R_s", "Rs", res)

            _show("X_s", "Xs", res)

            _show("X_m", "Xm", res)

            _show("R_c", "Rc", res)

        with c2:

            _show("R_r1", "Rr1", res)

            if model_choice == "Double cage":

                _show("X_r1", "Xr1", res)

                _show("R_r2", "Rr2", res)

                _show("X_r2", "Xr2", res)



        if res:

            conv_val = res.get("conv", 0)

            err_val = res.get("err", 1.0)

            iter_val = res.get("iter", 0)

            tag = '<span class="result-ok">Yes ✔</span>' if conv_val == 1 else '<span class="result-bad">No ✘</span>'

            st.markdown(f"**Converged:** {tag}", unsafe_allow_html=True)

            st.markdown(f"**Squared error:** `{np.round(err_val, 9)}`")

            st.markdown(f"**Iterations:** `{iter_val}`")



        st.markdown("</div>", unsafe_allow_html=True)



    # ── Calculate ─────────────────────────────────────────

    if calculate_btn:

        sf = (globals.motor_data["sync_speed"] - globals.motor_data["rated_speed"]) / globals.motor_data["sync_speed"]

        try:

            with st.spinner("Calculating…"):

                if model_choice == "Single cage":

                    p = [sf, globals.motor_data["rated_eff"], globals.motor_data["rated_pf"], globals.motor_data["T_b"]]

                    z, it, err, conv = nr_solver_sc(p, 0, globals.algo_data["k_x"], globals.algo_data["k_r"],

                                                    globals.algo_data["max_iter"], globals.algo_data["conv_err"])

                    st.session_state.results = dict(

                        Rs=z[0], Xs=z[1], Xm=z[2], Rr1=z[3], Rc=z[4], Xr1=z[5],

                        Rr2=None, Xr2=None, iter=it, err=err, conv=conv, z=z, model="single"

                    )

                else:

                    p = [sf, globals.motor_data["rated_eff"], globals.motor_data["rated_pf"],

                         globals.motor_data["T_b"], globals.motor_data["T_lr"], globals.motor_data["I_lr"]]

                    if algo == "Newton-Raphson":

                        z, it, err, conv = nr_solver(p, 0, globals.algo_data["k_x"], globals.algo_data["k_r"],

                                                     globals.algo_data["max_iter"], globals.algo_data["conv_err"])

                    elif algo == "Levenberg-Marquardt":

                        z, it, err, conv = lm_solver(p, 0, globals.algo_data["k_x"], globals.algo_data["k_r"],

                                                     1e-7, 5.0, globals.algo_data["max_iter"], globals.algo_data["conv_err"])

                    elif algo == "Damped Newton-Raphson":

                        z, it, err, conv = dnr_solver(p, 0, globals.algo_data["k_x"], globals.algo_data["k_r"],

                                                      1e-7, globals.algo_data["max_iter"], globals.algo_data["conv_err"])

                    elif algo == "Genetic Algorithm":

                        z, it, err, conv = ga_solver(None, p, globals.algo_data["pop"], globals.algo_data["n_r"],

                                                     globals.algo_data["n_e"], globals.algo_data["c_f"],

                                                     globals.algo_data["n_gen"], globals.algo_data["conv_err"])

                    elif algo == "Hybrid GA-NR":

                        z, it, err, conv = hy_solver(None, "NR", p, globals.algo_data["pop"], globals.algo_data["n_r"],

                                                     globals.algo_data["n_e"], globals.algo_data["c_f"],

                                                     globals.algo_data["n_gen"], globals.algo_data["conv_err"])

                    elif algo == "Hybrid GA-LM":

                        z, it, err, conv = hy_solver(None, "LM", p, globals.algo_data["pop"], globals.algo_data["n_r"],

                                                     globals.algo_data["n_e"], globals.algo_data["c_f"],

                                                     globals.algo_data["n_gen"], globals.algo_data["conv_err"])

                    else:

                        z, it, err, conv = hy_solver(None, "DNR", p, globals.algo_data["pop"], globals.algo_data["n_r"],

                                                     globals.algo_data["n_e"], globals.algo_data["c_f"],

                                                     globals.algo_data["n_gen"], globals.algo_data["conv_err"])

                    st.session_state.results = dict(

                        Rs=z[0], Xs=z[1], Xm=z[2], Rr1=z[3], Xr1=z[4], Rr2=z[5], Xr2=z[6], Rc=z[7],

                        iter=it, err=err, conv=conv, z=z, model="double"

                    )

            st.rerun()

        except Exception as exc:

            st.error(f"Calculation error: {exc}")



    # ── Plot curves ───────────────────────────────────────

    res = st.session_state.results

    if res and res.get("err", 1.0) < 1.0:

        if st.button("📈 Plot torque/current curves", use_container_width=True):

            z = res["z"]

            sf = (globals.motor_data["sync_speed"] - globals.motor_data["rated_speed"]) / globals.motor_data["sync_speed"]

            T_rtd = globals.motor_data["rated_eff"] * globals.motor_data["rated_pf"] / (1 - sf)



            n_pts = 1001

            speed = np.array([i / 1000 * globals.motor_data["sync_speed"] for i in range(n_pts)])

            Tm = np.zeros(n_pts)

            Im = np.zeros(n_pts)

            for n in range(n_pts - 1):

                slip = 1 - n / 1000

                if res["model"] == "single":

                    Ti, Ii = get_torque_sc(slip, z)

                else:

                    Ti, Ii = get_torque(slip, z)

                Tm[n] = Ti / T_rtd

                Im[n] = np.abs(Ii)



            fig = make_subplots(rows=2, cols=1,

                                subplot_titles=("Torque–Speed", "Current–Speed"),

                                vertical_spacing=0.12)

            fig.add_trace(go.Scatter(x=speed, y=Tm, name="Torque (T/Tn)",

                                     line=dict(color="#1f77b4")), row=1, col=1)

            fig.add_trace(go.Scatter(x=speed, y=Im, name="Current (pu)",

                                     line=dict(color="#d62728")), row=2, col=1)

            fig.update_xaxes(title_text="Speed (rpm)")

            fig.update_yaxes(title_text="Torque (T/Tn)", row=1, col=1)

            fig.update_yaxes(title_text="Current (pu)", row=2, col=1)

            fig.update_layout(height=520, template="plotly_white",

                              title_text=f"{globals.motor_data['description']} — Parameter Estimation Curves")

            st.plotly_chart(fig, use_container_width=True)





# ═══════════════════════════════════════════════════════════

#  TAB 2 — Laboratory Tests

# ═══════════════════════════════════════════════════════════

with tab_lab:



    # ── Controls row ──────────────────────────────────────

    ctl1, ctl2, ctl3 = st.columns([1.5, 2, 1])

    with ctl1:

        lab_model = st.radio("Model", ["Single cage", "Double cage"], index=1,

                             horizontal=True, key="lab_model")

    with ctl2:

        if lab_model == "Single cage":

            lab_algo = st.selectbox("Algorithm", ["Chapman Direct"], disabled=True, key="lab_algo")

        else:

            lab_algo = st.selectbox("Algorithm", [

                "Newton-Raphson", "Levenberg-Marquardt", "Damped Newton-Raphson",

                "Genetic Algorithm", "Hybrid GA-NR", "Hybrid GA-LM", "Hybrid GA-DNR",

            ], key="lab_algo")

    with ctl3:

        if lab_model == "Single cage":

            st.markdown('<div class="moto-card"><h4>Hipóteses</h4>', unsafe_allow_html=True)

            lab_sc_variant = st.selectbox(

                "Circuito",

                ["Sem Rc (Chapman)", "Com Rc"],

                key="lab_sc_variant",

            )

            lab_kx = st.number_input(

                "k_x (Xr/Xs)",

                min_value=0.01,

                value=1.0,

                step=0.01,

                format="%.4f",

                key="lab_sc_kx",

            )

            st.caption("Restrição linear usada no modelo direto: Xr = k_x * Xs.")

            st.markdown("</div>", unsafe_allow_html=True)

        else:

            lab_sc_variant = "Com Rc"

            lab_kx = 1.0



    # ── Input forms ───────────────────────────────────────

    col_obj, col_meas = st.columns([1, 1.15])



    with col_obj:

        st.markdown('<div class="moto-card"><h4>Dados do Objeto Sob Teste</h4>', unsafe_allow_html=True)

        lab_manufacturer = st.text_input("Fabricante", value=LAB_REPORT_DEFAULTS["manufacturer"], key="lab_manufacturer")

        lab_serial = st.text_input("Nr Serie", value=LAB_REPORT_DEFAULTS["serial_number"], key="lab_serial")

        lab_year = st.text_input("Ano Fabricação", value=LAB_REPORT_DEFAULTS["manufacturing_year"], key="lab_year")

        c1, c2 = st.columns(2)

        lab_power_w = c1.number_input("Potência [W]", min_value=0.0, value=LAB_REPORT_DEFAULTS["rated_power_w"],

                          step=100.0, key="lab_power")

        lab_voltage_v = c2.number_input("Tensão [V]", min_value=0.0, value=LAB_REPORT_DEFAULTS["rated_voltage_v"],

                        step=1.0, key="lab_voltage")

        c3, c4 = st.columns(2)

        lab_current_a = c3.number_input("Corrente [A]", min_value=0.0, value=LAB_REPORT_DEFAULTS["rated_current_a"],

                        step=0.1, key="lab_current")

        lab_poles = c4.number_input("Nr Polos", min_value=2, step=2, value=LAB_REPORT_DEFAULTS["pole_count"], key="lab_poles")

        lab_connection = st.selectbox(

            "Ligação do estator",

            options=["WYE", "DELTA"],

            index=0 if LAB_REPORT_DEFAULTS["stator_connection"] == "WYE" else 1,

            format_func=lambda value: "Estrela (WYE)" if value == "WYE" else "Triângulo (DELTA)",

            key="lab_connection",

        )

        c5, c6 = st.columns(2)

        lab_freq_hz = c5.number_input("Frequência [Hz]", min_value=0.0, value=LAB_REPORT_DEFAULTS["frequency_hz"], step=1.0, key="lab_freq")

        lab_speed_rpm = c6.number_input("Rotação [rpm]", min_value=0.0, value=LAB_REPORT_DEFAULTS["rated_speed_rpm"],

                        step=1.0, key="lab_speed")

        c7, c8 = st.columns(2)

        lab_pf_target = c7.number_input("FP medido", min_value=0.0, max_value=1.0,

                        value=LAB_REPORT_DEFAULTS["rated_pf"],

                        step=0.01, format="%.4f", key="lab_pf_target")

        lab_eff_target_pct = c8.number_input("Rendimento medido [%]", min_value=0.0, max_value=100.0,

                             value=LAB_REPORT_DEFAULTS["rated_eff_pct"],

                             step=0.1, format="%.2f", key="lab_eff_target")

        st.caption("Opcional: use os valores medidos de plena carga quando o relatório trouxer esse resultado.")

        st.markdown("</div>", unsafe_allow_html=True)



    with col_meas:

        st.markdown('<div class="moto-card"><h4>Rotor Bloqueado</h4>', unsafe_allow_html=True)

        b1, b2 = st.columns(2)

        lab_b_temp = b1.number_input("Temperatura [C]", value=LAB_REPORT_DEFAULTS["blocked_temp_c"], step=0.1, key="lab_b_temp")

        lab_b_res = b2.number_input("Resistência [Ω]", min_value=0.0, value=LAB_REPORT_DEFAULTS["blocked_resistance_ohm"],

                        step=0.001, format="%.4f", key="lab_b_res")

        b_mode_1, b_mode_2 = st.columns(2)

        lab_res_mode = b_mode_1.selectbox(

            "Resistência informada como",

            options=["line_to_line", "phase_winding", "phase_equivalent"],

            index=["line_to_line", "phase_winding", "phase_equivalent"].index(LAB_REPORT_DEFAULTS["resistance_input_mode"]),

            format_func=lambda value: {

                "line_to_line": "Entre duas linhas",

                "phase_winding": "Fase do enrolamento",

                "phase_equivalent": "Fase equivalente do circuito",

            }[value],

            key="lab_res_mode",

        )

        lab_res_temp = b_mode_2.number_input(

            "Temp. medição resistência [C]",

            value=LAB_REPORT_DEFAULTS["resistance_measurement_temp_c"],

            step=0.1,

            key="lab_res_temp",

        )

        if lab_res_mode == "line_to_line":

            st.caption("A resistência entre duas linhas é convertida para Rs equivalente por fase usando Rs = Rlinha/2.")

        elif lab_res_mode == "phase_winding" and lab_connection == "DELTA":

            st.caption("Para fase medida diretamente em delta, o modelo converte para equivalente estrela com Rs = Rfase/3.")

        elif lab_res_mode == "phase_winding":

            st.caption("Para fase medida diretamente em estrela, o valor informado já é a resistência por fase do estator.")

        else:

            st.caption("Use esta opção quando o valor informado já for o Rs equivalente por fase do circuito do modelo.")

        b3, b4 = st.columns(2)

        lab_b_volt = b3.number_input("Tensão [V]", min_value=0.0, value=LAB_REPORT_DEFAULTS["blocked_voltage_v"],

                         step=1.0, key="lab_b_volt")

        lab_b_curr = b4.number_input("Corrente [A]", min_value=0.0, value=LAB_REPORT_DEFAULTS["blocked_current_a"],

                         step=0.1, key="lab_b_curr")

        b5, b6 = st.columns(2)

        lab_b_pow = b5.number_input("Potência [W]", min_value=0.0, value=LAB_REPORT_DEFAULTS["blocked_power_w"],

                        step=10.0, key="lab_b_pow")

        lab_b_freq = b6.number_input("Frequência [Hz]", min_value=0.0, value=LAB_REPORT_DEFAULTS["blocked_frequency_hz"],

                         step=1.0, key="lab_b_freq")

        lab_start_curr = st.number_input("Corrente de partida [A]", min_value=0.0,

                         value=LAB_REPORT_DEFAULTS["starting_current_a"], step=1.0,

                         key="lab_start_curr")

        st.markdown("</div>", unsafe_allow_html=True)



        st.markdown('<div class="moto-card"><h4>Em Vazio</h4>', unsafe_allow_html=True)

        n1, n2 = st.columns(2)

        lab_nl_temp = n1.number_input("Temperatura [C]", value=LAB_REPORT_DEFAULTS["no_load_temp_c"], step=0.1, key="lab_nl_temp")

        lab_nl_volt = n2.number_input("Tensão [V]", min_value=0.0, value=LAB_REPORT_DEFAULTS["no_load_voltage_v"],

                          step=1.0, key="lab_nl_volt")

        n3, n4 = st.columns(2)

        lab_nl_curr = n3.number_input("Corrente [A]", min_value=0.0, value=LAB_REPORT_DEFAULTS["no_load_current_a"],

                          step=0.1, key="lab_nl_curr")

        lab_nl_pow = n4.number_input("Potência [W]", min_value=0.0, value=LAB_REPORT_DEFAULTS["no_load_power_w"],

                         step=10.0, key="lab_nl_pow")

        lab_fw_w = st.number_input("Perdas atrito/vent. [W]", min_value=0.0, value=LAB_REPORT_DEFAULTS["fw_loss_w"],

                       step=10.0, key="lab_fw")

        if lab_model == "Single cage" and lab_sc_variant == "Sem Rc (Chapman)":

            st.caption("Sem Rc, o ensaio a vazio fornece a perda rotacional total; esta entrada não é usada.")

        st.markdown("</div>", unsafe_allow_html=True)



    # ── Calculate button ──────────────────────────────────

    lab_calc_btn = st.button("▶ Calcular Parâmetros de Laboratório", type="primary",

                             use_container_width=True, key="lab_calc_btn")



    if lab_calc_btn:

        def _as_float(value):

            return 0.0 if value is None else float(value)

        def _as_int(value):

            return 0 if value is None else int(value)

        lab_data = {

            'manufacturer': lab_manufacturer,

            'serial_number': lab_serial,

            'manufacturing_year': lab_year,

            'rated_power_w': _as_float(lab_power_w),

            'rated_voltage_v': _as_float(lab_voltage_v),

            'rated_current_a': _as_float(lab_current_a),

            'pole_count': _as_int(lab_poles),

            'stator_connection': lab_connection,

            'frequency_hz': _as_float(lab_freq_hz),

            'rated_speed_rpm': _as_float(lab_speed_rpm),

            'blocked_temp_c': _as_float(lab_b_temp),

            'blocked_resistance_ohm': _as_float(lab_b_res),

            'resistance_input_mode': lab_res_mode,

            'resistance_measurement_temp_c': _as_float(lab_res_temp),

            'blocked_voltage_v': _as_float(lab_b_volt),

            'blocked_current_a': _as_float(lab_b_curr),

            'blocked_power_w': _as_float(lab_b_pow),

            'blocked_frequency_hz': _as_float(lab_b_freq),

            'no_load_temp_c': _as_float(lab_nl_temp),

            'no_load_voltage_v': _as_float(lab_nl_volt),

            'no_load_current_a': _as_float(lab_nl_curr),

            'no_load_power_w': _as_float(lab_nl_pow),

            'fw_loss_w': _as_float(lab_fw_w),

            'rated_current_target_a': _as_float(lab_current_a),

            'rated_power_target_w': _as_float(lab_power_w),

            'rated_pf_target': _as_float(lab_pf_target),

            'rated_eff_target': _as_float(lab_eff_target_pct) / 100.0 if _as_float(lab_eff_target_pct) > 0.0 else 0.0,

            'starting_current_target_a': _as_float(lab_start_curr),

            'no_load_current_target_a': _as_float(lab_nl_curr),

            'no_load_power_target_w': _as_float(lab_nl_pow),

            'include_core_loss': lab_sc_variant == "Com Rc",

            'reactance_ratio': float(lab_kx),

            'fit_reactance_ratio': lab_model == "Single cage" and (

                _as_float(lab_pf_target) > 0.0 or _as_float(lab_eff_target_pct) > 0.0 or _as_float(lab_start_curr) > 0.0

            ),

        }

        measured_fit = (

            lab_data['rated_pf_target'] > 0.0 or

            lab_data['rated_eff_target'] > 0.0 or

            lab_data['starting_current_target_a'] > 0.0

        )

        try:

            with st.spinner("Calculando…"):

                single_params = estimate_single_cage_parameters(lab_data)

                single_fit_error = 0.0

                if lab_model == "Single cage" and measured_fit:

                    single_params, single_fit_error = refine_single_cage_parameters(lab_data, single_params)

                single_summary = single_cage_summary(lab_data, single_params)

                single_curves = single_cage_curves(lab_data, single_params)

                single_load_pts = load_point_summary(

                    lab_data,

                    lambda slip: single_cage_performance(lab_data, single_params, slip)

                )



                if lab_model == "Single cage":

                    params = single_params

                    summ = single_summary

                    curves = single_curves

                    load_pts = single_load_pts

                    st.session_state.lab_results = {

                        'mode': 'single',

                        'model_name': 'Single cage',

                        'algorithm': (

                            'Chapman Direct + Measured Fit (com Rc)'

                            if measured_fit and params['Rc'] is not None else

                            'Chapman Direct + Measured Fit (sem Rc)'

                            if measured_fit else

                            'Chapman Direct (com Rc)'

                            if params['Rc'] is not None else

                            'Chapman Direct (sem Rc)'

                        ),

                        'input_data': lab_data,

                        'params': {

                            'Rs': params['Rs'], 'Xs': params['Xs'], 'Xm': params['Xm'],

                            'Rr1': params['Rr1'], 'Xr1': params['Xr1'],

                            'Rr2': None, 'Xr2': None, 'Rc': params['Rc'],

                        },

                        'summary': {

                            'locked_current_a': summ['locked_rotor']['current_a'],

                            'locked_torque_nm': summ['locked_rotor']['torque_nm'],

                            'breakdown_torque_nm': summ['breakdown']['torque_nm'],

                            'rated_eff_pct': summ['rated']['efficiency'] * 100.0,

                            'rated_pf': summ['rated']['power_factor'],

                            'converged': 'Yes' if measured_fit else 'Direct', 'error': float(single_fit_error),

                        },

                        'curves': curves,

                        'load_points': load_pts,

                    }

                else:

                    if measured_fit:

                        params, fit_error = fit_double_cage_parameters(lab_data)

                        summ = double_cage_summary(lab_data, params)

                        curves = double_cage_curves(lab_data, params)

                        load_pts = load_point_summary(

                            lab_data,

                            lambda slip: double_cage_performance(lab_data, params, slip)

                        )

                        st.session_state.lab_results = {

                            'mode': 'double',

                            'model_name': 'Double cage',

                            'algorithm': 'Measured Test Fit',

                            'input_data': lab_data,

                            'params': {

                                'Rs': float(params['Rs']), 'Xs': float(params['Xs']), 'Xm': float(params['Xm']),

                                'Rr1': float(params['Rr1']), 'Xr1': float(params['Xr1']),

                                'Rr2': float(params['Rr2']), 'Xr2': float(params['Xr2']), 'Rc': float(params['Rc']),

                            },

                            'summary': {

                                'locked_current_a': summ['locked_rotor']['current_a'],

                                'locked_torque_nm': summ['locked_rotor']['torque_nm'],

                                'breakdown_torque_nm': summ['breakdown']['torque_nm'],

                                'rated_eff_pct': summ['rated']['efficiency'] * 100.0,

                                'rated_pf': summ['rated']['power_factor'],

                                'converged': 'Yes',

                                'error': float(fit_error),

                            },

                            'curves': curves,

                            'load_points': load_pts,

                        }

                    else:

                        targets = single_summary['targets']

                        p = [targets['sf'], targets['rated_eff'], targets['rated_pf'],

                             targets['T_b'], targets['T_lr'], targets['I_lr']]



                        if lab_algo == 'Newton-Raphson':

                            vec, iterations, error, converged = nr_solver(

                                p, 0, globals.algo_data['k_x'], globals.algo_data['k_r'],

                                globals.algo_data['max_iter'], globals.algo_data['conv_err'])

                        elif lab_algo == 'Levenberg-Marquardt':

                            vec, iterations, error, converged = lm_solver(

                                p, 0, globals.algo_data['k_x'], globals.algo_data['k_r'],

                                1e-7, 5.0, globals.algo_data['max_iter'], globals.algo_data['conv_err'])

                        elif lab_algo == 'Damped Newton-Raphson':

                            vec, iterations, error, converged = dnr_solver(

                                p, 0, globals.algo_data['k_x'], globals.algo_data['k_r'],

                                1e-7, globals.algo_data['max_iter'], globals.algo_data['conv_err'])

                        elif lab_algo == 'Genetic Algorithm':

                            vec, iterations, error, converged = ga_solver(

                                None, p, globals.algo_data['pop'], globals.algo_data['n_r'],

                                globals.algo_data['n_e'], globals.algo_data['c_f'],

                                globals.algo_data['n_gen'], globals.algo_data['conv_err'])

                        elif lab_algo == 'Hybrid GA-NR':

                            vec, iterations, error, converged = hy_solver(

                                None, 'NR', p, globals.algo_data['pop'], globals.algo_data['n_r'],

                                globals.algo_data['n_e'], globals.algo_data['c_f'],

                                globals.algo_data['n_gen'], globals.algo_data['conv_err'])

                        elif lab_algo == 'Hybrid GA-LM':

                            vec, iterations, error, converged = hy_solver(

                                None, 'LM', p, globals.algo_data['pop'], globals.algo_data['n_r'],

                                globals.algo_data['n_e'], globals.algo_data['c_f'],

                                globals.algo_data['n_gen'], globals.algo_data['conv_err'])

                        else:

                            vec, iterations, error, converged = hy_solver(

                                None, 'DNR', p, globals.algo_data['pop'], globals.algo_data['n_r'],

                                globals.algo_data['n_e'], globals.algo_data['c_f'],

                                globals.algo_data['n_gen'], globals.algo_data['conv_err'])



                        # build double-cage results (mirroring build_lab_double_results)

                        vec = np.abs(vec)

                        torque_base = targets['rated_eff'] * targets['rated_pf'] / (1.0 - targets['sf'])

                        sync_rpm = synchronous_speed(lab_data['frequency_hz'], lab_data['pole_count'])

                        rated_torque_nm = rated_torque(lab_data['rated_power_w'], lab_data['rated_speed_rpm'])



                        def _perf(slip):

                            slip = min(max(slip, 1e-4), 1.0)

                            torque_pu, current_nc = get_torque(slip, vec)

                            core_current = 1.0 / complex(vec[7], 0.0)

                            total_current = abs(current_nc + core_current)

                            pqt = calc_pqt(slip, vec)

                            pin_pu = pqt[0] / pqt[5] if pqt[5] > 0 else 0.0

                            denom = np.sqrt(pin_pu ** 2 + pqt[1] ** 2)

                            pf = pin_pu / denom if denom > 0 else 0.0

                            return {

                                'slip': slip,

                                'speed_rpm': sync_rpm * (1.0 - slip),

                                'torque_nm': (torque_pu / torque_base) * rated_torque_nm,

                                'current_a': total_current * lab_data['rated_current_a'],

                                'power_factor': pf,

                                'efficiency': pqt[5],

                            }



                        slips = np.linspace(1.0, 1e-4, 500)

                        pts = [_perf(float(s)) for s in slips]

                        load_pts = load_point_summary(lab_data, _perf)

                        breakdown_pt = max(pts, key=lambda p: p['torque_nm'])

                        locked_pt = _perf(1.0)

                        rated_pt = _perf(targets['sf'])



                        st.session_state.lab_results = {

                            'mode': 'double',

                            'model_name': 'Double cage',

                            'algorithm': lab_algo,

                            'input_data': lab_data,

                            'params': {

                                'Rs': float(vec[0]), 'Xs': float(vec[1]), 'Xm': float(vec[2]),

                                'Rr1': float(vec[3]), 'Xr1': float(vec[4]),

                                'Rr2': float(vec[5]), 'Xr2': float(vec[6]), 'Rc': float(vec[7]),

                            },

                            'summary': {

                                'locked_current_a': locked_pt['current_a'],

                                'locked_torque_nm': locked_pt['torque_nm'],

                                'breakdown_torque_nm': breakdown_pt['torque_nm'],

                                'rated_eff_pct': rated_pt['efficiency'] * 100.0,

                                'rated_pf': rated_pt['power_factor'],

                                'converged': 'Yes' if converged == 1 else 'No',

                                'error': float(error),

                            },

                            'curves': {

                                'slip': np.array([p['slip'] for p in pts]),

                                'speed_rpm': np.array([p['speed_rpm'] for p in pts]),

                                'torque_nm': np.array([p['torque_nm'] for p in pts]),

                                'current_a': np.array([p['current_a'] for p in pts]),

                            },

                            'load_points': load_pts,

                            'iterations': iterations,

                        }

            st.rerun()

        except Exception as exc:

            st.error(f"Erro no cálculo: {exc}")



    # ── Results display ───────────────────────────────────

    lab_res = st.session_state.lab_results

    if lab_res:

        params = lab_res['params']

        summ = lab_res['summary']

        curves = lab_res['curves']

        load_pts = lab_res['load_points']



        st.divider()

        rc1, rc2 = st.columns(2)



        with rc1:

            st.markdown('<div class="moto-card"><h4>Parâmetros Estimados</h4>', unsafe_allow_html=True)

            p_rows = [("Rs [Ω]", params['Rs']), ("Xs [Ω]", params['Xs']),

                      ("Xm [Ω]", params['Xm']), ("Rr1 [Ω]", params['Rr1']),

                      ("Xr1 [Ω]", params['Xr1']), ("Rr2", params['Rr2']),

                      ("Xr2", params['Xr2']), ("Rc [Ω]", params['Rc'])]

            for lbl, val in p_rows:

                v = "" if val is None else f"{np.round(val, 6)}"

                st.text_input(lbl, value=v, disabled=True, key=f"labr_{lbl}")

            st.markdown("</div>", unsafe_allow_html=True)



        with rc2:

            st.markdown('<div class="moto-card"><h4>Desempenho Estimado</h4>', unsafe_allow_html=True)

            st.text_input("Corrente de Partida [A]", value=f"{np.round(summ['locked_current_a'], 3)}",

                          disabled=True, key="labr_ilr")

            st.text_input("Conjugado de Partida [Nm]", value=f"{np.round(summ['locked_torque_nm'], 3)}",

                          disabled=True, key="labr_tlr")

            st.text_input("Conjugado Máximo [Nm]", value=f"{np.round(summ['breakdown_torque_nm'], 3)}",

                          disabled=True, key="labr_tb")

            st.text_input("Rendimento [%]", value=f"{np.round(summ['rated_eff_pct'], 3)}",

                          disabled=True, key="labr_eff")

            st.text_input("Fator de Potência", value=f"{np.round(summ['rated_pf'], 4)}",

                          disabled=True, key="labr_pf")

            st.text_input("Convergiu?", value=str(summ['converged']), disabled=True, key="labr_conv")

            st.text_input("Erro", value=f"{np.round(summ['error'], 8)}", disabled=True, key="labr_err")

            st.markdown("</div>", unsafe_allow_html=True)



        # ── Load-point table ──────────────────────────────

        st.markdown("**Tabela de Pontos de Carga**")

        lp_data = {

            "Carga (%)":   [f"{int(lp['load_fraction']*100)}" for lp in load_pts],

            "Rotação (rpm)": [np.round(lp['speed_rpm'], 1) for lp in load_pts],

            "Conjugado (Nm)": [np.round(lp['torque_nm'], 2) for lp in load_pts],

            "Corrente (A)": [np.round(lp['current_a'], 3) for lp in load_pts],

            "FP":          [np.round(lp['power_factor'], 4) for lp in load_pts],

            "Rendimento (%)": [np.round(lp['efficiency'] * 100, 2) for lp in load_pts],

        }

        st.table(lp_data)



        # ── Plots ─────────────────────────────────────────

        if st.button("📊 Plotar Curvas de Laboratório", use_container_width=True):

            fig = make_subplots(

                rows=2, cols=2,

                subplot_titles=(

                    "Conjugado × Rotação", "Conjugado × Escorregamento",

                    "Corrente × Rotação", "Rendimento e FP × Carga",

                ),

                vertical_spacing=0.15, horizontal_spacing=0.1,

            )

            fig.add_trace(go.Scatter(x=curves['speed_rpm'].tolist(), y=curves['torque_nm'].tolist(),

                                     name="Conjugado", line=dict(color="#1f77b4")), row=1, col=1)

            fig.add_trace(go.Scatter(x=curves['slip'].tolist(), y=curves['torque_nm'].tolist(),

                                     name="Conjugado (s)", line=dict(color="#ff7f0e")), row=1, col=2)

            fig.add_trace(go.Scatter(x=curves['speed_rpm'].tolist(), y=curves['current_a'].tolist(),

                                     name="Corrente", line=dict(color="#2ca02c")), row=2, col=1)



            load_pct = [lp['load_fraction'] * 100.0 for lp in load_pts]

            eff_pct  = [lp['efficiency'] * 100.0 for lp in load_pts]

            pf_vals  = [lp['power_factor'] for lp in load_pts]

            fig.add_trace(go.Scatter(x=load_pct, y=eff_pct, name="Rendimento (%)",

                                     mode="lines+markers", line=dict(color="#9467bd")), row=2, col=2)

            fig.add_trace(go.Scatter(x=load_pct, y=pf_vals, name="Fator de Potência",

                                     mode="lines+markers", line=dict(color="#8c564b")), row=2, col=2)



            fig.update_xaxes(title_text="Rotação (rpm)", row=1, col=1)

            fig.update_xaxes(title_text="Escorregamento (pu)", row=1, col=2)

            fig.update_xaxes(title_text="Rotação (rpm)", row=2, col=1)

            fig.update_xaxes(title_text="Carga (%)", row=2, col=2)

            fig.update_yaxes(title_text="Conjugado (Nm)", row=1, col=1)

            fig.update_yaxes(title_text="Conjugado (Nm)", row=1, col=2)

            fig.update_yaxes(title_text="Corrente (A)", row=2, col=1)

            fig.update_yaxes(title_text="Valor", row=2, col=2)

            fig.update_layout(height=600, template="plotly_white",

                              title_text=f"Resultados — {lab_res['model_name']} ({lab_res['algorithm']})")

            st.plotly_chart(fig, use_container_width=True)



        # ── Save data ─────────────────────────────────────

        def _serializable():

            c = lab_res['curves']

            return {

                'model': lab_res['model_name'],

                'algorithm': lab_res['algorithm'],

                'input_data': lab_res['input_data'],

                'params': lab_res['params'],

                'summary': lab_res['summary'],

                'load_points': lab_res['load_points'],

                'curves': {

                    'slip': c['slip'].tolist(),

                    'speed_rpm': c['speed_rpm'].tolist(),

                    'torque_nm': c['torque_nm'].tolist(),

                    'current_a': c['current_a'].tolist(),

                },

            }



        json_bytes = json.dumps(_serializable(), indent=2).encode()

        st.download_button(

            "⬇️ Guardar dados (JSON)",

            data=json_bytes,

            file_name="lab_results.json",

            mime="application/json",

            use_container_width=True,

        )



