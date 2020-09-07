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


class Analysis(Config):
    data = {}

    def __init__(self, field=["*"], suffix="", logger=Logger()):
        self.logger = logger
        self.suffix = suffix
        self.field = field
    

    @staticmethod
    def vars(types=None, wc_vars=[], qpredictive=False):
        """ return list of variable names"""
        if types==None:
            types = [V for V in Config.VARS]
        selected_vars = []
        for t in types:
            for d in Config.VARS[t]:
                if qpredictive and d.get('predictive', False):
                    pass
                elif len(wc_vars) != 0: 
                    selected_vars.extend(fnmatch.filter(wc_vars, d['var']))
                else:
                    selected_vars.append(d['var'])
        return list(set(selected_vars))

    
    def read_file(self, fname, ext=Config.ANALYSIS_CONFIG["FILE_EXT"], date=""):
        try:
            fname = "{}.{}".format(os.path.join(self.FILES["DATA_LOCAL"], fname, ext))
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


    def get_data(self):
        self.logger.info(" Reading well logs files...")

        las_files = []
        las_files.extend(glob.glob(os.path.join(self.FILES["DATA_LOCAL"], "*{}".format(self.ANALYSIS_CONFIG["FILE_EXT"]))))

        self.data["{}".format(self.FILES["WELL_LOGS_MERGE"])] = pd.DataFrame()
        for files in las_files:
            self.logger.info("  Reading {}".format(os.path.splitext(os.path.basename(files))[0]))
            well_name = os.path.splitext(os.path.basename(files))[0]
            well_las = lasio.read("{}".format(files))
            self.data["{}".format(well_name)] = well_las.df()
            self.data["{}".format(well_name)]["WELL"] = well_las.well.WELL.value
            self.data["{}".format(well_name)]["XWELL"] = well_las.well.XWELL.value
            self.data["{}".format(well_name)]["YWELL"] = well_las.well.YWELL.value

            if self.data["{}".format(well_name)].size == 0:
                self.logger.critical("no data found in the LAS file: {}".format(well_name))

            self.data["{}".format(self.FILES["WELL_LOGS_MERGE"])] = pd.concat([self.data["{}".format(self.FILES["WELL_LOGS_MERGE"])], self.data["{}".format(well_name)]], axis=0, sort=True)

        self.data["{}".format(self.FILES["WELL_LOGS_MERGE"])] = self.data["{}".format(self.FILES["WELL_LOGS_MERGE"])][['WELL', 'XWELL', 'YWELL'] + analysis.vars(['Logs'])]

        fname = os.path.join(self.FILES["OUTPUT_PATH"], "{}{}.csv".format(self.FILES["WELL_LOGS_MERGE"], self.suffix))
        self.logger.info("  Saving merge well logs data in CSV format into local machine - '{}'...".format(fname))
        self.data["{}".format(self.FILES["WELL_LOGS_MERGE"])].to_csv(fname)

        self.logger.info("  done ...")


    # # Data Processing
    def get_numeric(self, df, depth_curve_name, x_coord, y_coord):
        curve_list = list(df.columns[df.dtypes.values==np.dtype('float64')])
        curve_list = [log for log in curve_list if log not in [depth_curve_name, x_coord, y_coord]]
        return curve_list 

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


    # # Exploratory Data Analysis
    def rename_data_type(self, types):
        """Convert the python data types to string

        Data types in pandas dataframe is based on:
        1. float64
        2. int64
        3. datetime64[ns]
        4. object

        Parameters
        ----------
        types : str
            "Types" column in categorical dataframe 

        Returns
        -------
        - If the data type is 'float64' or 'int64', return 'Number'  
        - If the data type is 'datetime64[ns]', return 'Date'  
        - If the data type is 'object', return 'String'  
        - Else, return 'No Valid'
        """
        if ('float64' in types) or ('int64' in types):
            return 'Number'
        elif ('datetime64[ns]' in types):
            return 'Date'
        elif ('object' in types):
            return 'String'
        else:
            return 'No Valid'


    def descriptive_data(self, df):
        """Acquire the description on dataframe

        Acquire the summary on the dataframe,
        and to be displayed in "Data Summary".

        Parameters
        ----------
        df : str
            Any dataframe

        Returns
        -------
        descriptive_df : object
            Dataframe on information the input dataframe
        """
        descriptive_info = {'Well Name: ': df['WELL'].unique().tolist(),
                            'No. of Variables':int(len(df.columns)), 
                            'No. of Observations':int(df.shape[0]),
                            }

        descriptive_df = pd.DataFrame(descriptive_info.items(), columns=['Descriptions', 'Values']).set_index('Descriptions')
        descriptive_df.columns.names = ['Data Statistics']
        return descriptive_df

    
    def data_type_analysis(self, df):
        """Acquire the data types in a dataframe

        Acquire the data types presence in a dataframe,
        and to be displayed in "Data Summary".

        Parameters
        ----------
        df : str
            Any dataframe
        """
        categorical_df = pd.DataFrame(df.reset_index(inplace=False).dtypes.value_counts())
        categorical_df.reset_index(inplace=True)
        categorical_df = categorical_df.rename(columns={'index':'Types', 0:'Values'})
        categorical_df['Types'] = categorical_df['Types'].astype(str)
        categorical_df['Types'] = categorical_df['Types'].apply(lambda x: self.rename_data_type(x))
        categorical_df = categorical_df.set_index('Types')
        categorical_df.columns.names = ['Variables']
        return categorical_df


    def grid_df_display(self, list_dfs, rows=1, cols=2, fill='cols'):
        """Display multiple tables side by side in jupyter notebook

        Descriptive table and Data Type table will be shown
        side by side in "Data Summary" in analysis.

        Parameters
        ----------
        list_dfs : array-like
            Multiple dataframes, you can put in a list on how many dataframe you want to see side by side
        rows : int
            Number of rows the tables to be displayed (default=1).
        cols : int 
            Number of columns the tables to be displayed (default=2).
        fills : str
            - If "cols", grid to display will be focused on columns.
            - If "rows", grid to display will be focused on rows. (default="cols")

        Returns
        -------
        dfs : object
            Dataframes displayed side by side on dataframe summary 
        """
        html_table = "<table style = 'width: 100%; border: 0px'> {content} </table>"
        html_row = "<tr style = 'border:0px'> {content} </tr>"
        html_cell = "<td style='width: {width}%; vertical-align: top; border: 0px'> {{content}} </td>"
        html_cell = html_cell.format(width=5000)

        cells = [ html_cell.format(content=df.to_html()) for df in list_dfs[:rows*cols] ]
        cells += cols * [html_cell.format(content="")]

        if fill == 'rows':
            grid = [ html_row.format(content="".join(cells[i:i+cols])) for i in range(0,rows*cols,cols)]

        if fill == 'cols': 
            grid = [ html_row.format(content="".join(cells[i:rows*cols:rows])) for i in range(0,rows)]
            
        dfs = display(HTML(html_table.format(content="".join(grid))))
        return dfs

    
    def distribution_plot_summary(self, df, col1, col2):
        """Variables summary with time-series and histogram
   

        Parameters
        ----------
        df : str
            Any dataframe
        col : str
            Columns in input dataframe

        Returns
        -------
        fig : object
            Variables summary plot on missing values, time-series and histogram
        """
        plt.style.use('seaborn-notebook')
        
        fig = plt.figure(figsize=(20, 6))
        spec = GridSpec(nrows=2, ncols=2)

        ax0 = fig.add_subplot(spec[0, :])
        ax0 = plt.plot(df.index, df[col1], '.')
        ax0 = plt.xlabel('DATE', fontsize=14)
        ax0 = plt.ylabel(col1, fontsize=14)
        ax0 = plt.grid(True)

        try:
            ax1 = fig.add_subplot(spec[1, 0])
            ax1 = sns.distplot(df[col1], hist=True, kde=True, 
                            bins=int(20), color = 'darkblue')
            ax1.set_xlabel(col1, fontsize=14)
            ax1.set_ylabel('Density', fontsize=14)
            ax1.grid(True)
        except:
            pass

        ax2 = fig.add_subplot(spec[1, 1])

        ax2 = plt.scatter(df[col1], df[col2],s=10)
        ax2 = plt.xlabel(col1, fontsize=11)
        ax2 = plt.ylabel(col2, fontsize=11)
        ax2 = plt.grid(True)
        
        plt.show()
        
        return fig

    
    def curve_plot(self, logs, df, depth_name, height=Config.ANALYSIS_CONFIG["HLV_HEIGHT"], width=Config.ANALYSIS_CONFIG["HLV_WIDTH"]):
        if logs == "GR_PEP":
            fig = df.hvplot(x=depth_name, y=logs, invert=True, flip_yaxis=True, shared_axes=True,
                            height=height, width=width).opts(fontsize={'labels': 10,'xticks': 9, 'yticks': 9}).opts(color='green')
        elif logs == "CALI_PEP":
            fig = df.hvplot(x=depth_name, y=logs, invert=True, flip_yaxis=True, shared_axes=True,
                            height=height, width=width).opts(fontsize={'labels': 10,'xticks': 9, 'yticks': 9}).opts(color='darkblue', line_dash='dashed')
        elif logs == "RT_PEP":
            fig = df.hvplot(x=depth_name, y=logs, invert=True, flip_yaxis=True, shared_axes=True,
                            height=height, width=width).opts(fontsize={'labels': 10,'xticks': 9, 'yticks': 9}).opts(color='darkgreen', line_dash='dotted')
        elif logs == "RHOB_PEP":
            fig = df.hvplot(x=depth_name, y=logs, invert=True, flip_yaxis=True, shared_axes=True,
                            height=height, width=width).opts(fontsize={'labels': 10,'xticks': 9, 'yticks': 9}).opts(color='red', line_dash='dashed')
        else:
            fig = df.hvplot(x=depth_name, y=logs, invert=True, flip_yaxis=True, shared_axes=True,
                            height=height, width=width).opts(fontsize={'labels': 10,'xticks': 9, 'yticks': 9})
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
