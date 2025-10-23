"""
Stock Paper Analysis Package
============================

Author: 김현우 (Hyunwoo Kim)
Email: hwkim3330@gmail.com

This package contains analysis scripts for the research paper:
"Calendar-based Clustering of Weekly Extremes: A Reanalysis"
"""

__version__ = "1.0.0"
__author__ = "김현우 (Hyunwoo Kim)"
__email__ = "hwkim3330@gmail.com"

from .wcsv_model import WCSVModel
from .g_test import g_test, analyze_weekly_extremes
from .simulation import WCSVSimulator, GARCHSimulator, compare_models

__all__ = [
    'WCSVModel',
    'g_test',
    'analyze_weekly_extremes',
    'WCSVSimulator',
    'GARCHSimulator',
    'compare_models'
]
