import pickle, math, csv, glob
import os, importlib, requests, json
from datetime import datetime, timedelta
import pandas as pd
from pandas.io.json import json_normalize
import numpy as np
import numpy.polynomial.polynomial as poly

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib import gridspec
from matplotlib.gridspec import GridSpec
import seaborn as sns
import mpld3
from mpld3 import plugins
import fnmatch
import calender
from IPython.display import display, Markdown, clear_output, HTML
import joypy
import rfpimp

import lasio
from welly import Well
import squarify

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

    
    def read_file(self, fname, ext=self.ANALYSIS_CONFIG["FILE_EXT"], date=""):
        try:
            fname = "{}.{}".format(os.path.join(Config.FILES["DATA_LOCAL"], fname, ext))
            if ext == "csv":
                data = pd.read_csv(fname, thousands=",")
            elif ext == "las":
                data = lasio.read(os.path.splitext(os.path.basename(fname))[0])
            else:
                self.logger.error("invalid file extension - {} ...".format(ext))

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

 
    # def read_las(self, in_dir, out_dir):
    #     well_ext = self.ANALYSIS_CONFIG["LAS_FILE_EXT"]
    #     norm_ext = self.ANALYSIS_CONFIG["NORM_FILE_EXT"]

    #     total_rows = 0

    #     logs_df = pd.DataFrame()
    #     for root, dirs, files in os.walk(in_dir):
    #         for filename in files:
    #             if filename.endswith(tuple(norm_ext)):
    #                 continue
    #             elif filename.endswith(tuple(well_ext)):
    #                 las_file = lasio.read(os.path.join(root, filename))
    #                 df_temp = las_file.df()
    #                 try:
    #                     df_temp["WELL"] = las_file.well.WELL.value
    #                 except:
    #                     df_temp["WELL"] = np.nan

    #                 total_rows += len(df_temp)
    #                 logs_df = pd.concat([logs_df, df_temp], axis=0, sort=True)

    #     if logs_df.size == 0:
    #         self.logger.critical("no data found in file '{}'",format(logs_df))
    #         if self.logger == print:
    #             exit()

    #     fname = os.path.join(self.FILES["DATA_LOCAL"], "{}{}.csv".format(self.FILES["WELL_LOGS_INPUT_FILE"], self.suffix))
    #     self.logger.info("  Saving well logs data to file '{}' ...".format(fname))
    #     logs_df.to_csv(fname)
        
    #     return logs_df


    def get_data(self):
        self.logger.info(" Reading ...")
        self.logger.info("   Las files ...")

        las_files = []
        for las in self.ANALYSIS_CONFIG["LAS_FILE_EXT"]:
            las_files.extend(glob.glob(files))

        for files in las_files:
            self.logger.info("    reading las file - {}".format(os.path.splitext(os.path.basename(files)))
            well_name = os.path.splitext(os.path.basename(files))[0]
            data


    def logs_data_quantity_plot(self, df, plot_title, xaxis, yaxis, source='pgps'):
        """Horizontal stack plot

        Visualize the amount of datapoints of each logs data in different wells

        Parameters
        ----------
        df : str
            Las file dataframe; lasFile_impt_df
        plot_title : str
            Title of visualization plot
        xaxis : str
            Label of x-axis
        yaxis : str
            Label of y-axis
        source : str, optional
            Select the well logs to be display, choose between "PGPS" or "raw"
            "PGPS" - 'WELL','DTC', 'DTCOM', 'DTSH', 'GR', 'NPHI','RHOB'
            "raw" - 'WELL','CALI', 'DENS', 'GR', 'DT', 'DTC', 'DTS', 'PERMH', 'PHIE', 'PHIT', 'SWT', 'SWE', 'VSAND', 'VCLW'
        
        Returns
        -------
        fig : object
            Chart
        """
        fig, ax = plt.subplots(figsize=(20,15))
        if source=='PGPS':
            impt_logs = ['WELL','DTC', 'DTCOM', 'DTSH', 'GR', 'NPHI','RHOB']
            sub_df = df[impt_logs]
            sub_groupby_df = sub_df.groupby('WELL').count()

            ax = sub_groupby_df.plot.barh(stacked = True, ax=ax)
            ax.set_title(plot_title, fontsize=18, weight='bold')
            ax.set_xlabel(xaxis, fontsize=16)
            ax.set_ylabel(yaxis, fontsize=16)
            plt.xticks(fontsize=13)
            plt.yticks(fontsize=13)
            plt.legend(prop={'size':13}, title='Logs')
        elif source=='raw':
            impt_logs = ['WELL','CALI', 'DENS', 'GR', 'DT', 'DTC', 'DTS', 'PERMH', 'PHIE', 'PHIT', 'SWT', 'SWE', 'VSAND', 'VCLW']
            sub_df = df[impt_logs]
            sub_groupby_df = sub_df.groupby('WELL').count()

            ax = sub_groupby_df.plot.barh(stacked = True, ax=ax)
            ax.set_title(plot_title, fontsize=18, weight='bold')
            ax.set_xlabel(xaxis, fontsize=16)
            ax.set_ylabel(yaxis, fontsize=16)
            plt.xticks(fontsize=13)
            plt.yticks(fontsize=13)
            plt.legend(prop={'size':13}, title='Logs')
        
        return fig


    def treemap(self, df, col, labels, plot_title):
        """Treemap plot for well logs data

        Visualization that split the area of chart to display the value of datapoints presence in dataframe.  
        A column called "LABELS" will be created that show the wells with number of datapoints below it.

        Parameters
        ----------
        df : str
            Well logs dataframe; lasFile_impt_df
        col : str
            Column that have no missing values, to acquire the count of datapoints
        labels : str
            String columns
        plot_title : str
            Title of the diagram
        
        Returns
        -------
        fig : object
            Chart
        """
        temp_df = df.groupby('WELL').count()
        temp_df = temp_df.reset_index()
        temp_df['LABELS']  = temp_df['WELL'].str.replace('SUMANDAK', 'SMDK') + '\n' + temp_df['DEPTH'].astype(str)
        temp_df['DEPTH'] = temp_df['DEPTH'].astype(int)
        temp_df = temp_df.sort_values(by = ['DEPTH'], ascending=False)
        temp_df = temp_df.reset_index(drop = True)
        
        norm = matplotlib.colors.Normalize(vmin=min(temp_df[col]), vmax=max(temp_df[col]))
        colors = [matplotlib.cm.Greens(norm(value)) for value in temp_df[col]]
        
        fig = plt.gcf()
        ax = fig.add_subplot()
        fig.set_size_inches(30, 20)

        squarify.plot(sizes=temp_df[col], label=temp_df[labels],
                      color=colors, alpha=0.7, text_kwargs={'fontsize':16})
        plt.title(plot_title + '\n' + 'Total Observations: {}'.format(str(df.shape[0])),fontsize=23,fontweight="bold")
        plt.axis('off')
        plt.gca().invert_yaxis()
        plt.show()
        
        return fig


    def box_plot_min_max(self, df, var, var_key, min_max=True):
        """Boxplot of well log data

        Detect the outliers of data with the minimum and maximum value defined at config files

        Parameters
        ----------
        df : str
            Well logs dataframe
        var : str
            Feature of dataframe
        min_max : boolean (default=True)
            Option to display the vertical line of the feature defined in Config
                - If True, display min and max
                - If False, do not display min and max
        
        Returns
        -------
        fig : object
            Boxplot with threshold limit in it
        """
        fig, ax = plt.subplots(figsize=(10,1))
        sns.boxplot(x=df[var], ax=ax)
        min_var = None
        max_var = None
        
        for vtype in self.VARS:
            if vtype == var_key:
                for v in self.VARS[vtype]:
                    if v["var"]== var:
                        min_var = v["min"]
                        max_var = v["max"] 
                        unit = v["unit"]
                        break
                if min_var != None:
                    break

        if min_max and min_var != None:
            paired_fence = dict(zip([min_var, max_var], ['r','r']))
            for key, value in paired_fence.items():
                plt.axvline(x=key, label='{}'.format(key), c=value, linestyle='solid', linewidth=4)
                ax.text(x=min_var, y=0.2, s='{}'.format(min_var), ha='center', rotation='vertical', backgroundcolor = 'white', fontsize = 8)
                ax.text(x=max_var, y=0.2, s='{}'.format(max_var), ha='center', rotation='vertical', backgroundcolor = 'white', fontsize = 8)
        
        plt.title('Boxplot of {} ({})'.format(var, unit), size = 8, weight='bold')
        plt.xlabel('{}'.format(var), size = 9)
        plt.xticks(fontsize=8)
        plt.yticks(fontsize=8)
        plt.show()
        
        return fig

    
    def triple_combo_plot(self, df, well, top_depth, bottom_depth, title):
        """Well logs display

        Vertical time series plot showing the well logs data in correspond to depth

        Parameters
        ----------
        df : str
            Well logs dataframe
        top_depth : str
            The shallowest depth of logs 
        bottom_depth : str
            The deepest depth of logs
        title : str
            Title of diagram
        
        Returns
        -------
        fig : object
            Chart on well logs analysis
        """
        sub_df = df.loc[df['WELL'] == well]
        
        logs=sub_df[(sub_df['DEPTH'] >= top_depth) & (sub_df['DEPTH'] <= bottom_depth)]
        fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(6,10), sharey=True)
        fig.suptitle("{}".format(title), fontsize=22)
        fig.subplots_adjust(top=0.81, wspace=0.9)

        for axes in ax:
            axes.set_ylim (top_depth, bottom_depth.max())
            axes.invert_yaxis()
            axes.yaxis.grid(True)
            axes.get_xaxis().set_visible(False)
            
        try:
            ax01=ax[0].twiny()
            ax01.set_xlim(0,150)
            ax01.plot(logs['GR'], logs['DEPTH'], label='GR[api]', color='green') 
            ax01.spines['top'].set_position(('outward',20))
            ax01.set_xlabel('GR[api]',color='green')    
            ax01.tick_params(axis='x', colors='green')
        except:
            pass
        
        try:
            ax11=ax[1].twiny()
            ax11.set_xlim(logs['NPHI'].min(),logs['NPHI'].max())
            ax11.invert_xaxis()
            ax11.plot(logs['NPHI'], logs['DEPTH'], label='NPHI[%]', color='green') 
            ax11.spines['top'].set_position(('outward',20))
            ax11.set_xlabel('NPHI[%]', color='green')    
            ax11.tick_params(axis='x', colors='green')
        except:
            pass
        
        try:
            ax12=ax[1].twiny()
            ax12.set_xlim(1.95,2.95)
            ax12.plot(logs['RHOB'], logs['DEPTH'] ,label='RHOB[g/cc]', color='red') 
            ax12.spines['top'].set_position(('outward',60))
            ax12.set_xlabel('RHOB[g/cc]',color='red')
            ax12.tick_params(axis='x', colors='red')
        except:
            pass

        return fig


    def mask_outside_thres(self, df, vtype):
        """Masking of values below and above threshold

        Setting the minimum and maximum values of features to follow argument set in config file

        Parameters
        ----------
        df : str
            Any dataframe
        vtype : str
            Variables dictionary from config file
        
        Returns
        -------
        df : object
            Dataframe with value on columns being masked
        """
        for v in self.VARS[vtype]:
            if v["min"] != None and v['var'] in df:
               df.mask(df[v['var']] < v["min"], np.nan, inplace=True) 
            if v["max"] != None and v['var'] in df:
               df.mask(df[v['var']] > v["max"], np.nan, inplace=True)
        
        return df