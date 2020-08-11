import pickle, math, csv, 
import os, importlib, requests, json
from datetime import datetime, timedelta
import pandas as pd
from pandas.io.json import json_normalize
import numpy as np
import numpy.polynomial.polynomial as poly
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib import gridspec
from matplotlib.gridspec import GridSpec
import seaborn as sns
import mpld3
from mpld3 import plugins
import fnmatch
import calender
from Ipython.display import display, Markdown, clear_output, HTML
import joypy
import rfpimp

import lasio
import welly

from src.Config import Config
from src.analysis.feature_engineer import Feature_Engineer

class Logger(object):
    info = print
    error = print
    warning = print
    critical = print


class Analysis(Feature_Engineer):
    data = {}

    def __init__(self, field=["*"], suffix="", logger=Logger()):
        self.logger = logger
        self.suffix = suffix
        self.field = field
    

    @staticmethod
    def vars(types=None, wc_vars=[], qpredictive=False):
        """ Return list of variable names """
        if types == None:
            types = [V for V in Config.VARS]
        selected_vars = []
        for t in types:
            for d in Config.VARS[t]:
                if qpredictive and d.get("predictive", False):
                    pass
                elif len(wc_vars) != 0:
                    selected_vars.extend(fnmatch.filter(wc_vars, d["var"]))
                else:
                    selected_vars.append(d["var"])

            return list(set(selected_vars))

    
    def read_file(self, fname, date=""):
        try:
            fname = "{}.csv".format(os.path.join(Config.FILES["DATA_LOCAL"], fname))
            data = pd.read_csv(fname, thousands=",")
            if data.size == 0:
                self.logger.critical("no data found in file '{}'",format(fname))
                if self.logger == print:
                    exit()
        except FileNotFoundError:
            self.logger.critical("file '{}' is not found ...".format(fname))
            if self.logger == print:
                exit()
        except:
            raise

        if data != "":
            data[date] = pd.to_datetime(data[date])
            data.set_index(date, inplace=True)

        return data


    def read_las(self, in_dir, out_dir):
        well_ext = self.ANALYSIS_CONFIG["LAS_FILE_EXT"]
        norm_ext = self.ANALYSIS_CONFIG["NORM_FILE_EXT"]

        total_rows = 0

        logs_df = pd.DataFrame()
        for root, dirs, files in os.walk(in_dir):
            for filename in files:
                if filename.endswith(tuple(norm_ext)):
                    continue
                elif filename.endswith(tuple(well_ext)):
                    las_file = lasio.read(os.path.join(root, filename))
                    df_temp = las_file.df()
                    try:
                        df_temp["WELL"] = las_file.well.WELL.value
                    except:
                        df_temp["WELL"] = np.nan

                    total_rows += len(df_temp)
                    logs_df = pd.concat([logs_df, df_temp], axis=0, sort=True)

        if logs_df.size == 0:
            self.logger.critical("no data found in file '{}'",format(logs_df))
            if self.logger == print:
                exit()

        fname = os.path.join(self.FILES["DATA_LOCAL"], "{}{}.csv".format(self.FILES["WELL_LOGS_INPUT_FILE"], self.suffix))
        self.logger.info("  Saving well logs data to file '{}' ...".format(fname))
        logs_df.to_csv(fname)
        
        return logs_df

    
    def get_well_logs_data(self):
        self.logger.info(  "Converting well logs LAS files to CSV files for further analyis ...")
        read_las

        fname = "well_logs"
        self.logger.info("  Loading well logs data (in .csv) from {}{}.csv".format(self.FILES["WELL_LOGS_INPUT_FILE"], self.suffix))
        self.data[fname] = self.read_file(fname)