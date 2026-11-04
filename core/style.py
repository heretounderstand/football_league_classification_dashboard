"""Palette et mise en forme Plotly communes à toutes les pages.

Dix ligues = trop pour dix teintes distinctes : on encode le **pays par la couleur**
(5 teintes validées daltonisme, ordre fixe) et la **division par la texture**
(division 2 = hachures pour les barres, tirets pour les courbes).
"""
from __future__ import annotations

import plotly.graph_objects as go
import streamlit as st

COUNTRIES = ["Angleterre", "Espagne", "Italie", "Allemagne", "France"]
_COUNTRY_LIGHT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
_COUNTRY_DARK = ["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181"]

INK = {"light": {"primary": "#0b0b0b", "secondary": "#52514e", "muted": "#898781",
                 "grid": "#e1e0d9", "axis": "#c3c2b7", "accent": "#2a78d6", "neutral": "#c3c2b7"},
       "dark": {"primary": "#f0efec", "secondary": "#c3c2b7", "muted": "#898781",
                "grid": "#2c2c2a", "axis": "#383835", "accent": "#3987e5", "neutral": "#52514e"}}

# Rampe séquentielle bleue (clair -> foncé) et pôles divergents bleu / rouge.
SEQUENTIAL = ["#f4f8fd", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
DIVERGING = [[0.0, "#256abf"], [0.5, "#f0efec"], [1.0, "#d03b3b"]]
DIVERGING_DARK = [[0.0, "#3987e5"], [0.5, "#383835"], [1.0, "#e66767"]]

FONT = 'system-ui, -apple-system, "Segoe UI", sans-serif'


def mode() -> str:
    """Thème forcé en clair (.streamlit/config.toml) : st.context.theme suit le navigateur, pas
    le thème réellement appliqué, et provoquait des encres sombres sur fond clair. Les valeurs
    'dark' restent disponibles si l'on autorise un jour le thème sombre."""
    return "light"


def ink(role: str) -> str:
    return INK[mode()][role]


def country_color(country: str) -> str:
    pal = _COUNTRY_DARK if mode() == "dark" else _COUNTRY_LIGHT
    return pal[COUNTRIES.index(country)]


def league_color(league: str, leagues: dict) -> str:
    return country_color(leagues[league]["country"])


def tier_pattern(tier: int) -> str:
    return "" if tier == 1 else "/"


def tier_dash(tier: int) -> str:
    return "solid" if tier == 1 else "dash"


def diverging():
    return DIVERGING_DARK if mode() == "dark" else DIVERGING


def layout(fig: go.Figure, height: int = 420, left: int = 60, bottom: int = 50, top: int = 40,
           right: int = 16, **kw) -> go.Figure:
    """Marges explicites : l'auto-marge de Plotly n'est pas fiable dans Streamlit."""
    fig.update_layout(
        height=height,
        margin=dict(l=left, r=right, t=top, b=bottom),
        font=dict(family=FONT, size=13, color=ink("secondary")),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        hoverlabel=dict(font=dict(family=FONT)),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0,
                    title=None, font=dict(color=ink("secondary"))),
        bargap=0.25,
        **kw,
    )
    fig.update_xaxes(automargin=True, gridcolor=ink("grid"), linecolor=ink("axis"), zerolinecolor=ink("axis"),
                     tickfont=dict(color=ink("muted")), title_font=dict(color=ink("secondary")))
    fig.update_yaxes(automargin=True, gridcolor=ink("grid"), linecolor=ink("axis"), zerolinecolor=ink("axis"),
                     tickfont=dict(color=ink("muted")), title_font=dict(color=ink("secondary")))
    return fig


def show(fig: go.Figure) -> None:
    st.plotly_chart(fig, use_container_width=True, theme=None,
                    config={"displaylogo": False, "modeBarButtonsToRemove": ["lasso2d", "select2d"]})


def legend_note() -> None:
    st.caption("Couleur = pays · barres hachurées / courbes en tirets = division 2.")
