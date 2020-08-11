import numpy as np
import pandas as pd
import math, os
import datetime
from datetime import datetime
import fnmatch

from src.Config import Config

class Logger(Config):
    info = print
    critical = print
    error = print
    warning = print


class Feature_Engineer(Config):

    def percentile(self, n):
        def percentile_(x):
            return np.percentile(x, n)
        percentile_._name__ = "percentile_%s" % n
        return percentile_

    
    def groupby_log(self, data, cols):
        log_df = {}
        log_unit_df = pd.DataFrame()
        logs_list = cols

        for log in logs_list:
            pivot_df = pd.DataFrame(columns=["DATE", "WELL", "UNITS", f"{log}_MEAN", f"{log}_MAX", f"{log}_MIN", f"{log}_STD", f"{log}_PER25", f"{log}_PER75"])

            for unit, unit_data in data.groupby("UNIT"):
                log_data_operation = unit_data.groupby(["STRING", "UNIT"]).agg(["mean", "max", "min", "std", self.percentile(25), self.percentile(75)])[[log]].reset_index()
                wells = data.loc[data["UNIT"]==unit]["STRING"].values
                log_data_operation = pd.DataFrame(columns=["STRING", "UNIT", f"{log}_MEAN"])

                for well, well_data in self.data["well_test"].loc[self.data["well_test"]["STRING"].isin(strings)].groupby("STRING"):
                    mean_list   = []
                    max_list    = []
                    min_list    = []
                    std_list    = []
                    per25_list  = []
                    per75_list  = []

                    mean_value  = log_data_operation[log_data_operation["STRING"]==well].values[0][0]
                    max_value   = log_data_operation[log_data_operation["STRING"]==well].values[0][1]
                    min_value   = log_data_operation[log_data_operation["STRING"]==well].values[0][2]
                    std_value   = log_data_operation[log_data_operation["STRING"]==well].values[0][3]
                    per25_value = log_data_operation[log_data_operation["STRING"]==well].values[0][4]
                    per75_value = log_data_operation[log_data_operation["STRING"]==well].values[0][5]

                    for element in range(len(well_data)):
                        mean_list.append(mean_value)
                        max_list.append(max_value)
                        min_list.append(min_value)
                        std_list.append(std_value)
                        per25_list.append(per25_value)
                        per75_list.append(per75_value)

                    well_data[f"{log}_MEAN"]    = mean_list
                    well_data[f"{log}_MEAN"]    = well_data[f"{log}_MEAN"].astype(float)
                    well_data[f"{log}_MAX"]     = max_list
                    well_data[f"{log}_MAX"]     = well_data[f"{log}_MAX"].astype(float)
                    well_data[f"{log}_MIN"]     = min_list
                    well_data[f"{log}_MIN"]     = well_data[f"{log}_MIN"].astype(float)
                    well_data[f"{log}_STD"]     = std_list
                    well_data[f"{log}_STD"]     = well_data[f"{log}_STD"].astype(float)
                    well_data[f"{log}_PER25"]   = per25_list
                    well_data[f"{log}_PER25"]   = well_data[f"{log}_PER25"].astype(float)
                    well_data[f"{log}_PER75"]   = per75_list
                    well_data[f"{log}_PER75"]   = well_data[f"{log}_PER75"].astype(float)

                    log_data_operation = pd.concat([log_data_operation,
                                                    pd.DataFrame({"STRING"      : well_data["STRING"],
                                                                 "UNIT"         : unit,
                                                                 f"{log}_MEAN"  : well_data[f"{log}_MEAN"],
                                                                 f"{log}_MAX"   : well_data[f"{log}_MAX"],
                                                                 f"{log}_MIN"   : well_data[f"{log}_MIN"],
                                                                 f"{log}_STD"   : well_data[f"{log}_STD"],
                                                                 f"{log}_PER25" : well_data[f"{log}_PER25"],
                                                                 f"{log}_PER75" : well_data[f"{log}_PER75"],
                                                    })])
                    log_df[log] = pd.DataFrame(log_data_operation)
                    log_df[log] = log_df[log].reset_index().rename(columns={"index": "DATE"})
                    log_df[log] = log_df[log].sort_values(by="STRING").rename(columns={"UNIT1.0": f"{log}_{unit}"})
                    log_unit_df = log_unit_df.append(log_df[log])

            pivot_df = log_unit_df.pivot_table(index=["DATE, STRING"], columns="UNIT", aggfunc="mean")
            pivot_df = pivot_df.reset_index()
            pivot_df.columns = ["_".join(pair) if pair[1] != "" else pair[0] for pair in pivot_df.columns]

            return pivot_df


    def create_well_interval(self, data, res_data):
        interval_df = pd.DataFrame()
        for well, interval_data in res_data.groupby("WELL"):
            for i, row in interval_data.iterrows():
                temp_df = df.loc[(df['WELL']==row[0]) &
                                    (df["DEPTH"] > row[4]) &
                                    (df["DEPTH"] < row[5])]
                temp_df["UNIT"] = row[1]
                interval_df = interval_df.append(temp_df, ignore_index=True)

        return interval_df

    
    def elastic_properties(self, data):
        """genearate features on elastic properties from well logs

        with the presence of rhob, nphi in logs data, elastic properties like
        vp, vs, ai, si, young modulus, poisson ratio, bulk modulus, shear modulus, 
        vp/vs will be generated from mudrock / han / castagna.

        parameters
        ----------
        data : str
            well logs dataframe
        
        returns
        -------
        data : object
            create columns on elastic modulus of reservoir unit
        """
        # acquire vp from density logs - gardner equation
        data['vp_cal']          = ((data['dens']/0.31)**(1/0.25))*0.3048

        # acquire vs from vp - mudrock / han / castagna
        data['vs_mudrock']      = (data['vp_cal']*0.86) - 1.17
        data['vs_han']          = (data['vp_cal']*0.79) - 0.79
        data['vs_castagna']     = (data['vp_cal']*0.80) - 0.86

        # elastic properties of lithology
        data['ai_gardner']      = data['dens']*data['vp_cal']

        data['si_han']          = data['dens']*data['vs_han']
        data['young_han']       = (data['dens']*(data['vs_castagna']**2))*(((3*(data['vp_cal']**2))-(4*(data['vs_castagna']**2)))/((data['vp_cal']**2)-(data['vs_castagna']**2)))
        data['ps_han']          = (data['vp_cal']**2-2*(data['vs_castagna']**2))/(2*((data['vp_cal']**2)-(data['vs_castagna']**2)))
        data['bulk_han']        = data['dens']*((data['vp_cal']**2)-(4/3*(data['vs_castagna']**2)))
        data['shear_han']       = data['dens']*(data['vs_castagna']**2)
        data['vpvs_han']        = data['vp_cal']/data['vs_castagna']**2

        data['si_castagna']     = data['dens']*data['vs_castagna']
        data['young_castagna']  = (data['dens']*(data['vs_castagna']**2))*(((3*(data['vp_cal']**2))-(4*(data['vs_castagna']**2)))/((data['vp_cal']**2)-(data['vs_castagna']**2)))
        data['ps_castagna']     = (data['vp_cal']**2-2*(data['vs_castagna']**2))/(2*((data['vp_cal']**2)-(data['vs_castagna']**2)))
        data['bulk_castagna']   = data['dens']*((data['vp_cal']**2)-(4/3*(data['vs_castagna']**2)))
        data['shear_castagna']  = data['dens']*(data['vs_castagna']**2)
        data['vpvs_castagna']   = data['vp_cal']/data['vs_castagna']**2

        return data