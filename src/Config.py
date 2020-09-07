import os, sys, inspect
import fnmatch
from datetime import datetime, timedelta
from collections import OrderedDict
import pandas as pd

base_path, currentdir = os.path.split(os.path.dirname(inspect.getfile(inspect.currentframe())))

class Logger(object):
    info = print
    warning = print
    error = print
    critical = print


class Config(object):

    QDEBUG = True

    NAME = dict(
        FULL  = "Well Logs Prediction",
        SHORT = "LP",
    )

    FILES = dict(
        DATA_LOCAL          = "data_local",
        DATA                = currentdir + os.sep + "data",
        MODELS              = "models",
        OUTPUT_PATH         = r"D:\02-Project\Geoscience\Logs_Prediction\data_local\processed_data",
        WELL_LOGS_MERGE     = "well_logs_df",
    )

    # Fields
    FIELDS = OrderedDict(
        Sumandak = dict(
            abbr        = "SMDK",
            start_date  = datetime(2006, 2, 1),
            description = "Sumandak field is discovered in 2004. It is located in east of Malaysia, offshort Sabah.",
        ),
    )

    ANALYSIS_CONFIG = dict(
        FILE_EXT        = "las",
        NORM_FILE_EXT   = [".csv", ".pickle", ".dlis", ".xlsx"],
        ANALYSIS_COLS   = ['GR_PEP', 'CALI_PEP', 'RT_PEP', 'RHOB_PEP', 'NPHI_PEP', 'DTS_PEP', 'VS_DTS_PEP', 'DT_PEP', 'VP_DT_PEP', 'VELOCITY_RATIO_PEP', 'AI_PEP', 
                        'PHIE_PEP', 'SWE_PEP', 'VSAND_PEP', 'VSHALE_PEP', 'VCALC_PEP', 'VCLB_PEP', 'VCLD_PEP', 'VCLW_PEP', 'VSILT_PEP', 'VDOLO_PEP',
                        'VWATER_PEP', 'VOIL_PEP', 'VGAS_PEP', 'WELL', 'XWELL', 'YWELL'],
        HLV_HEIGHT      = 400,
        HLV_WIDTH       = 200,
        TEST_ALPHA      = 0.05,
        SAMPLE_SIZE     = 30,
        RHOB_SOURCE     = "rhob",
        VP_SOURCE       = "vp",
        VS_SOURCE       = "vs",
        ROLLING_SIZE    = 3,
        INTERPOLATE_DIR = "both",
    )

    MODELLING_CONFIG = dict(
        ALGORITHMS          = ["XGBR", "XGBR_tuned", "LGBMR", "LGBMR_tuned", "RFR", "RFR_tuned"],
        KNN_NEIGHBOUR       = 5,
        IMPUTE_THRESHOLD    = 0.80,
        ENCODING_ALG        = "Ordinal",
        RANDOM_STATE        = 0,
        SPLIT_RATIO         = 0.20,
        METRIC_BEST         = "R2_Score",
        METRIC_BEST_THRES   = 0.70,
        MISSING_FILL        = "ffill",
        IMPUTE_ALGORITHMS   = ["RFR", "RFR_tuned"],
        MODEL_BY_TYPE       = ["WELL", "WELLTYPE", "FIELD", "STRUCTURE", "CLUSTERING"],
        METRIC_EVAL_TYPE    = "test", # ["test", "cv"]
        CV_FOLD_TYPE        = "kf", # ["kf", "loo"]
        FEATURE_ENGINEERING = True, # True, False
    )

    METRIC_THRES_PLOT = OrderedDict(
        R2_Score            = (-1.0, 1.0),
        R2_Score_Adjusted   = (-1.0, 1.0),
        MAPE                = (0, 100),
        MAE                 = (0, 100),
        RMSE                = (0, 100),
    )

    _COLUMN_RENAME = OrderedDict(
        LOGS = {
            "rhob"          : "RHOB",
            "Sonic"         : "VP",
            "Shear"         : "VS",
            "Caliper Log"   : "CALIPER",
        },

        ELASTIC = {
            "Acoustic Impedance"    : "AI",
            "Shear Impedance"       : "SI",
            "Active Well"           : "ACTIVE_WELL",
        }
    )

    VARS = OrderedDict(
        Logs = [
            dict(var="GR_PEP",              unit="api",         min=35,     max=140,    impute="",      predictive=True ),
            dict(var="CALI_PEP",            unit="m",           min=0,      max=10,     impute="",      predictive=True ),
            dict(var="RT_PEP",              unit="Ohmm",        min=0,      max=33,     impute="",      predictive=True ),
            dict(var="RHOB_PEP",            unit="g/cm3",       min=1.75,   max=2.95,   impute="",      predictive=True ),
            dict(var="NPHI_PEP",            unit="%",           min=0,      max=0.60,   impute="",      predictive=True ),
            dict(var="DTS_PEP",             unit="us/ft",       min=210,    max=660,    impute="",      predictive=True ),
            dict(var="VS_DTS_PEP",          unit="m/s",         min=460,    max=1417,   impute="",      predictive=True ),
            dict(var="DT_PEP",              unit="us/ft",       min=119,    max=178,    impute="",      predictive=True ),
            dict(var="VP_DT_PEP",           unit="m/s",         min=1715,   max=2560,   impute="",      predictive=True ),
            dict(var="VELOCITY_RATIO_PEP",  unit=None,          min=1.27,   max=3.73,   impute="",      predictive=True ),
            dict(var="AI_PEP",              unit="g/cm3_m/s",   min=3524,   max=5563,   impute="",      predictive=True ),
            dict(var="PHIE_PEP",            unit="%",           min=0,      max=1,      impute="",      predictive=True ),
            dict(var="SWE_PEP",             unit="%",           min=0,      max=1,      impute="",      predictive=True ),
            dict(var="VSAND_PEP",           unit="%",           min=0,      max=1,      impute="",      predictive=True ),
            dict(var="VSHALE_PEP",          unit="%",           min=0,      max=1,      impute="",      predictive=True ),
            dict(var="VCALC_PEP",           unit="%",           min=0,      max=1,      impute="",      predictive=True ),
            dict(var="VCLB_PEP",            unit="%",           min=0,      max=1,      impute="",      predictive=True ),
            dict(var="VCLD_PEP",            unit="%",           min=0,      max=1,      impute="",      predictive=True ),
            dict(var="VCLW_PEP",            unit="%",           min=0,      max=1,      impute="",      predictive=True ),
            dict(var="VSILT_PEP",           unit="%",           min=0,      max=1,      impute="",      predictive=True ),
            dict(var="VDOLO_PEP",           unit="%",           min=0,      max=1,      impute="",      predictive=True ),
            dict(var="VWATER_PEP",          unit="%",           min=0,      max=1,      impute="",      predictive=True ),
            dict(var="VOIL_PEP",            unit="%",           min=0,      max=1,      impute="",      predictive=True ),
            dict(var="VGAS_PEP",            unit="%",           min=0,      max=1,      impute="",      predictive=True ),
            dict(var="VOIL_PEP",            unit="%",           min=0,      max=1,      impute="",      predictive=True ),
        ]       
    )