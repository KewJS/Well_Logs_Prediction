import math
import numpy as np
from numpy.random import randn
from numpy import exp
import pandas as pd
import datetime as dt
from itertools import repeat
from collections import OrderedDict
from IPython.display import display, Markdown, HTML

import scipy.stats as stats
import scipy.optimize
import scipy.spatial
from scipy.linalg import toeplitz
from scipy.stats import ttest_ind, ttest_rel, ttest_1samp, chi2, chi2_contingency, t, sem, rankdata, norm, kurtosis
from scipy.stats import shapiro, boxcox, levene
import statsmodels.api as sm
from statsmodels.formula.api import ols

from src.Config import Config


class Logger(object):
    info = print
    critical = print
    error = print
    warning = print
    debug = print


class Statistic_Analysis(Config):
    def __init__(self, strings=None, suffix='', logger=Logger(), y_var='SAND_COUNT'):
        self.logger = logger
        self.suffix = suffix
        self.strings = strings
        self.y_var = y_var

    
    @staticmethod
    def _kurt(x, normal=True):
        n = x.shape[0]
        m = np.mean(x)

        kurt = np.sum(((x-m)**4.0 / n) / np.sqrt(np.var(x))**4.0) - (3.0 * normal)

        return kurt


    def kurtosis(self, x, axis=0):
        """Compute the kurtosis

        The kurtosis for a normal distribution is 3. For this reason, some sources use the following  
        definition of kurtosis (often referred to as "excess kurtosis"):

        Kurtosis is typically defined as:

        .. math::
            Kurt(x_0, \cdots, x_{n-1}) = \large{\frac{1}{n} \sum^{n-1}_{j=0} \large[\frac{x_j - \bar{x}}{\sigma}
            \large]^4 \large} - 3

        The :math:`-3` term is applied so a normal distribution will have a 0 kurtosis value (mesokurtic).  

        Positive kurtosis indicates a "heavy-tailed" distribution and negative kurtosis indicates a "light-tailed" distribution.

        Parameters
        ----------
        x : array-like
            One or two-dimensional array of data.
        axis : int {0, 1}
            Specifies which axis of the data to compute the kurtosis. The default is 0 (column-wise in a 2d-array). Cannot
            be greater than 1.

        Example
        -------
        >>> kurtosis([5, 2, 4, 5, 6, 2, 3])
        -1.4515532544378704
        >>> kurtosis([[5, 2, 4, 5, 6, 2, 3], [4, 6, 4, 3, 2, 6, 7]], axis=1)
        array([-1.45155325, -1.32230624])

        Returns
        -------
        k : float or array-like
            If x is one-dimensional, the kurtosis of the data is returned as a float. If x is two-dimensional, the
            calculated kurtosis along the specified axis is returned as a numpy array of floats.
        """
        if axis > 1:
            raise ValueError("axis must be 0 (row-wise) or 1 (column-wise)")

        if not isinstance(x, np.ndarray):
            raise ValueError("array cannot have more than two dimensions")

        k = np.apply_along_axis(Statistic_Analysis._kurt(x), axis, x)

        if k.shape == ():
            k = float()

        return k

    
    def qq_quantile_plot(self, data, var, title):
        """Check for whether samples used in parametric test are in normally distributed using graphical method

        We evaluate the normality of data using inference method:
            - Graphical Method: Q-Q quantile plot

        Q-Q quantile plot is a graphical technique for determining if two datasets come from sample populatons with a common distribution (normally distributed).  
        The idealized samples are divided into groups called quantiles. Each data points in the sample is paired with a similar member from the idealized ditribution  
        at the sample cumulative distribution. 

        A perfect match for the ditribution will be shown by a line of dots on a 45-degree anfle from the bottom left of the plot to the top right.  
        Deviation by the dots from the line shows a deviation from the expected distribution.

        Parameters
        ----------
        data : object
            
        """
        sample_data = data[var]

        fig, ax = plt.subplots(figsize=(8,6))
        fig = qqplot(sample_data, line="s")
        plt.title(title, weight="bold")
        plt.show()
        return fig

    
    def shapiro_wilk_test(self, data, var=None):
        """Check for normal distribution between groups

        We evaluate the normality of data using inference method:
            - Inference Method: Shapiro-Wilk test

        Shapiro-Wilk test evaluates a data sample and quantifies how likely the data was drawn from Gaussian Distribution.  
        The test gives us a \code

        Shapiro-Wilk test is typically defined as ``W`` value, where small value indicates that our sample is not normally distributed  
        (rejecting our null hypothesis). ``W`` is defined as:

        .. math::
            W = \frac{(\sum_{i=1}^n a_i x_(i))^2}{\sum_{i=1}^n (x_i-\bar{x})^2}

        where:   
        :math:`x_i` term is the ordered random sample values  
        :math:`a_i` term is the constant generated from the covariances, variances and means of the sample size (size, :math:`n`) from a normally distributed sample

        Null & Alternate hypothesis:
            - :math:`H_0`: Samples are normally distributed
            - :math:`H_1`: Samples are non-normally distributed

        Parameters
        ----------
        data : object
            Dataframe that has the interested column to be performed statistical analysis.
        var : array-like, optional (default=None)
            Column from the dataframe to be performed Shapiro-Wilk test. 
            If the input **data** is an array-like object, leave the option default (None).

        Example
        -------
        The way to perform normality test. We pass in an array from a datafrme based on the interested column to test on normality test.

        >>> professor_salary = [139750, 173200, 79750, 11500, 141500,
        ...                     103450, 124750, 137000, 89565, 102580]
        >>> wtest, p_value = shapiro_wilk_test(professor_salary)
        >>> wtest = 0.0869
        >>> p_value = 0.934
        >>> Sample data does not look Gaussian (fail to reject H0)
d
        Returns
        -------
        wtest: float
            W-statistics value from Shapiro-Wilk test
        p_value: float
            P-value for the test
        """
        if var != None:
            sample_data = data[var]
        else:
            sample_data = data
            
        wtest, p_value = shapiro(sample_data)
        if p_value > Config.ANALYSIS_CONFIG["TEST_ALPHA"]:
            info = "Sample looks Gaussian (fail to reject H0)"
        else:
            info = "Sample does not look Gaussian (reject H0)"

        sample_statistics = {
                "Test Description"      : "Shapiro-Wilk Test",
                "P-Value"               : p_value,
                "Levene's Statistic"    : wtest, 
                "Test Results"          : info
            }
        return sample_statistics

    
    def levene_test(self, data, center="Mean", var=None):
        """Check for homogeneity of variance between groups

        Levene's test is a statistical procedure for testing equality of variances (also sometimes called homoscedasticity or homogeneity of variances)  
        between two or more sample populations.

        Levene's test is typically defined as ``W`` value, where small value indicates that at least one sample has different variance compared to population  
        (rejecting our null hypothesis). ``W`` is defined as:

        .. math::
            W = \frac{(N-k)}{(k-1)} \frac{\sum_{i=1}^k n_i(Z_i - Z_..)^2}{\sum_{i=1}^k \sum_{j=1}^{n_i} (Z_{ij} - Z_i.)^2}

        where:   
        :math:`k` term is the number of groups
        :math:`n_i` term is the number of samples belonging to the :math:`i-th` group
        :math:`N` term is the total number of samples
        :math:`Y_{ij}` term is the :math:`j-th` observation from the :math:`i-th` group

        Null & Alternative hypothesis:
            - :math:`H_0`: All of the :math:`k` sample populations have equal variances
            - :math:`H_1`: At least one of the :math:`k` sample population variances are not equal

        Parameters
        ----------
        data : object
            Dataframe that has the interested column to be performed statistical analysis.
        center : : {‘mean’, ‘median’, ‘trimmed’}, optional
            Which function of the data to use in the test. The default is ‘median’.
                - 'median' : Recommended for skewed (non-normal) distributions.
                - 'mean' : : Recommended for symmetric, moderate-tailed distributions.
                - 'trimmed' : Recommended for heavy-tailed distributions.
        var : array-like, optional (default=None)
            The sample data, possibly with different lengths.
            If the input **data** is an array-like object, leave the option default (None).

        Example
        -------
        The way to perform homogeneity of variance test. We pass in an array from a datafrme based on the interested column to test on homogeneity of variance.

        >>> col1, col2, col3 = list(range(1, 100)), list(range(50, 78)), list(range(115, 139))
        >>> wtest, p_value = leven(col1,col2,col3,center="mean")
        >>> wtest = 0.0869
        >>> p_value = 0.934
        >>> Sample data does not look Gaussian (fail to reject H0)

        Returns
        -------
        wtest: float
            W-statistics value from Levene's test
        p_value: float
            P-value for the test
        """
        if var != None:
            sample_data = data[var]
        else:
            sample_data = data
            
        wtest, p_value = levene(sample_data, center=center)
        if p_value > Config.ANALYSIS_CONFIG["TEST_ALPHA"]:
            info = "Samples have equal variance (fail to reject H0)"
        else:
            info = "At least one of the sample has different variance from the rest (reject H0)"

        sample_statistics = {
                "Test Description"      : "Levene's Test",
                "P-Value"               : p_value,
                "Levene's Statistic"    : wtest, 
                "Test Results"          : info
            }
        return sample_statistics


    def t_test(self, data, var, y1, y2=None, group=None, var_equal=True, paired=False, sample_size=self.ANALYSIS_CONFIG["SAMPLE_SIZE"]):
        """T-Test on means  between samples

        There are 2 type of T-tests, which is one sample T-Test or two sample T-Test. 

        One Sample T-test is when when we are comparing the sample mean to a known mean like population mean. When we want to compare means between 2 samples,  
        we use paired sample test. There are 2 types of paired sample test, one is comparing sample means coming from 2 different groups, known as  
        Independent Paired Sample T-Test. On the other hand, when comparing 2 samples coming from same groups, we call it as Dependent Paired Sample T-Test.  
        T-test is only applicable for 2 different samples only. 

        Null & Alternate hypothesis:
            - :math:`H_0`: Means between 2 samples are the same
            - :math:`H_1`: Means between 2 samples are not the same

        Assumptions in T-Test:
            - Residuals (experimental error) are normally distributed (Shapiro Wilks Test)
            - Homogeneity of variances (variances are equal between treatment groups) (Levene or Bartlett Test)
            - Observations are sampled independently from each other

        Parameters
        ----------
        data : object
            Dataframe to acquire the population mean and standard deviation for one-sample t-tests
        var : string
            Column name from dataframe for t-tests
        y1 : array-like
            One-dimensional array-like object (list, numpy array, pandas DataFrame or pandas Series) containing
            the observed sample values.
        y2 : array-like, optional
            One-dimensional array-like object (list, numpy array, pandas DataFrame or pandas Series) containing
            the observed sample values. Not necessary to include when performing one-sample t-tests.
        group : array-like or None
            The corresponding group vector denoting group sample membership. Will return :code:`None` if not passed.
        var_equal : bool, optional
            If True, the two samples are assumed to have equal variances and Student's t-test is performed.
            Defaults to False, which performs Welch's t-test for unequal sample variances.
        paired : bool, optional
            If True, performs a paired t-test.

        Raises
        ------
        ValueError
            If :code:`paired` is True and a second sample, :code:`y2` is not passed.
        ValueError
            If :code:`paired` is True and the number of sample observations in :code:`y1` and :code:`y2` are not equal.

        Notes
        -----
        Welch's t-test is an adaption of Student's t test and is more performant when the
        sample variances and size are unequal. The test still depends on the assumption of
        the underlying population distributions being normally distributed.
        Welch's t test is defined as:

        .. math::
            t = \frac{\bar{X_1} - \bar{X_2}}{\sqrt{\frac{s_{1}^{2}}{N_1} + \frac{s_{2}^{2}}{N_2}}}

        where:
        :math:`\bar{X}` is the sample mean, :math:`s^2` is the sample variance, :math:`n` is the sample size
        
        If the :code:`var_equal` argument is True, Student's t-test is used, which assumes the two samples
        have equal variance. The t statistic is computed as:

        .. math::
            t = \frac{\bar{X}_1 - \bar{X}_2}{s_p \sqrt{\frac{1}{n_1} + \frac{1}{n_2}}

        where:

        .. math::
            s_p = \sqrt{\frac{(n_1 - 1)s^2_{X_1} + (n_2 - 1)s^2_{X_2}}{n_1 + n_2 - 2}

        Examples
        --------
        Similar to other inference methods, there are generally two ways of performing a t-test. The first is to pass
        a group vector with the :code:`group` parameter and the corresponding observation vector as below.
        The data used in this example is a subset of the professor salary dataset found in Fox and
        Weisberg (2011).
        >>> professor_discipline = ['B', 'B', 'B', 'B', 'B',
        ...                         'A', 'A', 'A', 'A', 'A']
        >>> professor_salary = [139750, 173200, 79750, 11500, 141500,
        ...                     103450, 124750, 137000, 89565, 102580]
        >>> ttest = t_test(professor_salary, group=professor_discipline)
        >>> print(ttest)
            {'Sample 1 Mean': 111469.0,
            'Sample 2 Mean': 109140.0,
            'p-value': 0.9342936060799869,
            't-statistic': 0.08695024086399619,
            'test description': "Two-Sample Welch's t-test"}

        The other approach is to pass each group sample vector similar to the below.
        >>> sal_a = [139750, 173200, 79750, 11500, 141500]
        >>> sal_b = [103450, 124750, 137000, 89565, 102580]
        >>> ttest2 = t_test(sal_a, sal_b)
        >>> print(ttest)
            {'Sample 1 Mean': 109140.0,
            'Sample 2 Mean': 111469.0,
            'p-value': 0.9342936060799869,
            't-statistic': -0.08695024086399619,
            'test description': "Two-Sample Welch's t-test"}

        Returns
        -------
        sample_statistics : dict
            Dictionary contains the statistical analysis on t-tests
        """
        self.pop_mean = data[var].mean()
        self.pop_std = data[var].std()

        self.group = group
        self.paired = paired

        if self.paired and y2 is None:
            self.log.error("Second sample is missing for paired T-Tests ...")

        if var_equal:
            self.method = "Student's T-Test"
            self.var_equal = var_equal
        else:
            self.method = "Welch's T-Test"
            self.var_equal = False

        if self.paired == False and y2 is None:
            test_description = "One Sample T-Test"
            sample_1 = data.loc[(data[group]==y1) & (data[var].notnull())]
            self.sample_1_mean = sample_1[var].mean()
            self.sample_1_std = sample_1[var].std()
            self.ttest, self.p_value = ttest_1samp(sample_1[var].values, popmean=self.pop_mean)
            if p_value > Config.ANALYSIS_CONFIG["TEST_ALPHA"]:
                self.info = "Accept null hypothesis that the means are equal between sample and population ... \
                            Interpretation: The P-value obtained from 1-Sample T-Test analysis is not significant (P>0.05), \
                            and therefore, we conclude that there are no significant differences between samples."
            else:
                self.info = "Reject null hypothesis that the means are equal between sample and population ... \
                            Interpretation: The P-value obtained from 1-Sample T-Test analysis is significant (P<0.05), \
                            and therefore, we conclude that there are significant differences between samples."
            self.sample_statistics = {
                "Test Description"      : "One-Sample T-Test",
                "No. of Observations"   : int(len(sample_1)),
                "Population Mean"       : self.pop_mean,
                "Sample Mean"           : self.sample_1_mean,
                "P-Value"               : self.p_value,
                "T-Statistic"           : self.ttest, 
                "Test Results"          : self.info
            }

        elif self.paired == True and y2 is not None:
            sample_1 = data.loc[(data[group]==y1) & (data[group].notnull())]
            sample_1 = sample_1.loc[sample_1[var].notnull()]
            self.sample_1_mean = sample_1[var].mean()
            self.sample_1_std = sample_1[var].std()

            sample_2 = data.loc[(data[group]==y2) & (data[group].notnull())]
            sample_2 = sample_2.loc[sample_2[var].notnull()]
            self.sample_2_mean = sample_2[var].mean()
            self.sample_2_std = sample_2[var].std()

            sample_1, sample_2 = sample_1.sample(n=sample_size), sample_2.sample(n=sample_size)
            if len(sample_1) != len(sample_2):
                self.logger.error("Paired samples must have the same number of observations ...")

            self.ttest, self.p_value = ttest_ind(sample_1[var], sample_2[var])
            if p_value > Config.ANALYSIS_CONFIG["TEST_ALPHA"]:
                self.info = "Accept null hypothesis that the means are equal between sample and population ... \
                            Interpretation: The P-value obtained from 2-Sample T-Test analysis is not significant (P>0.05), \
                            and therefore, we conclude that there are no significant differences between samples."
            else:
                self.info = "Reject null hypothesis that the means are equal between samples ... \
                            Interpretation: The P-value obtained from 2-Sample T-Test analysis is significant (P<0.05), \
                            and therefore, we conclude that there are significant differences between samples."
            self.sample_statistics = {
                "Test Description"      : "One-Sample T-Test",
                "No. of Observations"   : int(len(sample_1)),
                "Sample 1 Mean"         : self.sample_1_mean,
                "Sample 2 Mean"         : self.sample_2_mean,
                "P-Value"               : self.p_value,
                "T-Statistic"           : self.ttest, 
                "Test Results"          : self.info
            }

        return

    
    def anova(self, data, *args, y_var, type="one", var_equal=True):
        """Analysis of variance on one independent variable

        One-Way ANOVA is used to compare 2 means from 2 independent (unrelated) groups using F-distribution.  
        With the null hypothesis for the test is that  2  means are equal. Therefore, a significant results means that the two means are unequal.

        How ANOVA works:
            - Check sample sizes: equal number of observations in each grou
            - Calculate Mean Square (MS) for each group (Sum of Square of Group / DOG); DOF is degree of freedom for the samples
            - Calculate the Mean Square Error (MSE) (Sum of Square of Error / DOF of residuals)
            - Calculate the F-Value (MS of Group / Mean Square Error (MSE))

        Null & Alternate hypothesis:
            - :math:`H_0`: Groups means are equal (no variation in means of groups)
            - :math:`H_1`: At least, one group mean is different from other groups

        Assumptions in ANOVA:
            - Residuals (experimental error) are normally distributed (Shapiro Wilks Test)
            - Homogeneity of variances (variances are equal between treatment groups) (Levene or Bartlett Test)
            - Observations are sampled independently from each other

        Example
        -------


        Return
        ------
        """
        sample_data = data[var]

        if type == "one":
            anova_model = ols('{y} ~ C({x})'.format(y=y_var, x=args), data=sample_data).fit()
            self.anova_table = sm.stats.anova_lm(anova_model, typ=1)
            if self.anova_table["PR(>F)"][0] > self.ANALYSIS_CONFIG["TEST_ALPHA"]:
                self.info = "Accept null hypothesis that the means are equal between samples ... \
                            Interpretation: The P-value obtained from One-Way ANOVA is not significant (P>0.05), \
                            and therefore, we conclude that there are no significant differences between samples."
            else:
                self.info = "Reject null hypothesis that the means are equal between samples ... \
                            Interpretation: The P-value obtained from One-Way ANOVA is significant (P<0.05), \
                            and therefore, we conclude that there are significant differences between samples."
        elif type == "two":
            anova_model = ols('{y} ~ C({x})'.format(y=y_var, x=var), data=sample_data).fit()
            self.anova_table = sm.stats.anova_lm(anova_model, typ=1)
            if self.anova_table["PR(>F)"][0] > self.ANALYSIS_CONFIG["TEST_ALPHA"]:
                self.info = "Accept null hypothesis that the means are equal between samples ... \
                            Interpretation: The P-value obtained from One-Way ANOVA is not significant (P>0.05), \
                            and therefore, we conclude that there are no significant differences between samples."
            else:
                self.info = "Reject null hypothesis that the means are equal between samples ... \
                            Interpretation: The P-value obtained from One-Way ANOVA is significant (P<0.05), \
                            and therefore, we conclude that there are significant differences between samples."


        return sample_df

    
    def chi_squared(self, data, y1, y2):
        """Performs the Chi-square test of independence of variables

        Chi-Squared is to study the relationship between 2 categorical variables, to check is there any relationship between them.  
        In statistic, there are 2 types of variables, numerical (countable) variables and non-numerical variables (categorical) variables.  
        The Chi-Square statistic is a single number that tells you how much difference exists between our observed counts and  
        the counts we would expect if there were no relationship at all in the population. 

        Chi-Squared statistic used in Chi-Squared test is defined as:

        .. math::
            x^2_c = \sum\frac{(O_i - E_i)^2}{E_i}

        where:
        :math:`c` term is the degree of freedom
        :math:`O` term is the observed value
        :math:`E` expected value

        Null & Alternative hypothesis:
            - :math:`H_0`: There are no relationship between 2 categorical samples
            - :math:`H_1`: There is a relationship presence between 2 categorical samples

        Parameters
        ----------
        data : object
            Dataframe that contain the categorical variables
        y1 : array-like
            One-dimensional array-like object (list, numpy array, pandas DataFrame or pandas Series) containing
            the observed sample values.
        y2 : array-like, optional
            One-dimensional array-like object (list, numpy array, pandas DataFrame or pandas Series) containing
            the observed sample values.

        Examples
        --------
        The first is to pass a dataframe with 2 different categorical group vector .
        The data used in this example is a subset of the data in Sand Advisor project on SAND_COUNT & WC.
        >>> chi_statistic = chi_squared(model_df, 'SAND_COUNT_CLASS', 'WC_CLASS')
        >>> print(chi_statistic)
            {'Test Description': 'Chi-Squared Test',
            'P-Value': 0.00033203456800745546,
            'T-Statistic': 20.896189593657517,
            'Test Results': 'Reject null hypothesis that there are no relationship between the categorical variables ...}

        Returns
        -------
        sample_statistics : dict
            Dictionary contains the statistical analysis on chi-squared tests.
        """
        self.count_data = pd.crosstab(data[y1], data[y2])
    
        observed_values = self.count_data.values
        chi_val = stats.chi2_contingency(self.count_data)
        expected_value = chi_val[3]
        
        no_of_rows = self.count_data.shape[0]
        no_of_cols = self.count_data.shape[1]
        dof = (no_of_rows-1) * (no_of_cols-1)
        
        chi_square = sum([(o-e)**2.0 / e for o, e in zip(observed_values, expected_value)])
        self.chi_square_stat = chi_square[0] + chi_square[1]
        self.p_value = 1-chi2.cdf(x=self.chi_square_statistic, df=dof)
        if self.p_value > Config.ANALYSIS_CONFIG["TEST_ALPHA"]:
            self.info = "Accept null hypothesis that there are no relationship between the categorical variables ... \
                        Interpretation: The P-value obtained from Chi-Squared Test analysis is not significant (P>0.05), \
                        and therefore, we conclude that there are no significant differences between samples."
        else:
            self.info = "Reject null hypothesis that there are no relationship between the categorical variables ... \
                        Interpretation: The P-value obtained from 2-Sample T-Test analysis is significant (P<0.05), \
                        and therefore, we conclude that there are significant differences between samples."
        self.sample_statistics = {
                "Test Description"      : "Chi-Squared Test",
                "P-Value"               : self.p_value,
                "T-Statistic"           : self.chi_square_stat, 
                "Test Results"          : self.info
            }

        return


    @staticmethod
    def _normalized(data, var):
        """Normalize data distribution to normal distributed

        Apply box-cox power transformation on dataset, transforming them into normally distribution. Box-cox transformation requires input data to be positive. 
        When log transformation is applied to non-normal distribution data, it tries to expand the differences between smaller values because 
        the slope for the logarithmic function is steeper for smaller values whereas the differences between larger values can be reduced 
        because the log distribution for large values has a moderate slope.
        
        Box-cox transformation can be defined as:

        .. math::
            y(\lambda) = \left\{
                \begin{array}\\
                    (y^\lambda - 1) / \lambda & \mbox{if } \ \lambda \neq 0; \\
                    log y & \mbox{if } \ \lambda = 0
                \end{array}
            \right.

        Box-cox transformation only cares about computing the value :math:`\lambda`, which varies from -5 to 5. 

        Example
        -------
        
        Return
        ------

        """
        boxcox_data, boxcox_lambda = boxcox(data[var])

        return boxcox_data, boxcox_lambda