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

    _DATABASE = dict(
        PetronasIO_PRD = dict(
            SERVER  = "10.14.162.82",
            PORT    = 1433,
            UID     = "gd_user",
            PWD     = "ZAQ!2wsx",
            COLUMNS = dict(
                VI_COMPLETION_ALL_en_US = "ITEM_ID ITEM_NAME ACTIVE TYPE PRODUCT STATUS OPERATED LIFT_TYPE DATA_STATUS FIELD LATITUDE BOT_X BOT_Y",
                VI_PI_INPUT_H_en_US     = "ITEM_NAME START_DATETIME THP CHP GLIR FLT LIQ_RATE WATER_RATE GAS_RATE THT FLP CHOKE_1 CHT BHT BHP PDGT PDGP",  
            )
        ),
        DART = dict(
            START_DATE      = "2009-01-01",
            END_DATE        = "2020-06-19",
            AUTHORIZATION   = "",
            DATA_TYPES      = ["PWSI", "PWT", "PWP", "WPI", "WSA"],
            FIELDS          = ["SUMANDAK"],
        )
    )

    NAME = dict(
        FULL  = "Well Logs Prediction",
        SHORT = "LP",
    )

    QDEBUG = True

    FILES = dict(
        DATA_LOCAL              = "data_local",
        DATA                    = currentdir + os.sep + "data",
        MODELS                  = "models",
        WELL_LOGS_INPUT_FILE    = "well_logs_data",
        MERGED_DATA             = "merged_data",
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
        LAS_FILE_EXT    = [".las", ".LAS", ""],
        NORM_FILE_EXT   = [".csv", ".pickle", ".dlis", ".xlsx"],
        TEST_ALPHA      = 0.05,
        SAMPLE_SIZE     = 30,
        RHOB_SOURCE     = "rhob",
        VP_SOURCE       = "vp",
        VS_SOURCE       = "vs",
        COLUMNS_ROLLING = ["RHOB", "Vp", "Vs"],
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
            dict(var="RHOB",    unit="g/cc",    min=0,      max=2.65,   impute="",      modelled=True),
            dict(var="VP",      unit="g/cc",    min=0,      max=2.65,   impute="",      predictive=True),
            dict(var="VS",      unit="g/cc",    min=0,      max=2.65,   impute="",      predictive=True),
            dict(var="GR",      unit="g/cc",    min=0,      max=2.65,   impute="",      predictive=True),
            dict(var="CALIPER", unit="g/cc",    min=0,      max=2.65,   impute="",      predictive=True),
            dict(var="PHIE",    unit="g/cc",    min=0,      max=2.65,   impute="",      predictive=True),
            dict(var="PHIT",    unit="g/cc",    min=0,      max=2.65,   impute="",      predictive=True),
        ]
    )