
from benchmark.utils import generate_columns_definition

pk_cols_definition = (
    "Drug name, is text, the name of drug mentioned in the paper.\n "
    "Analyte, is text, either the primary drug, its metabolite, or another drug on which the primary drug acts..\n "
    "Specimen, is text, what is the specimen, like 'blood', 'breast milk', 'cord blood', and so on..\n "
    "Population, Describe the patient age distribution, including categories such as 'pediatric,' 'adults,' 'old adults,' 'maternal,' 'fetal,' 'neonate,' etc..\n "
    "Pregnancy stage, is text, What pregnancy stages of patients mentioned in the paper, like 'postpartum', 'before pregnancy', '1st trimester' and so on. If not mentioned, please label as 'N/A',.\n "
    "Subject N,  the number of subjects that correspond to the specific parameter. .\n "
    "Parameter type, is text, the type of parameter, like 'concentration after the first dose', 'concentration after the second dose', 'clearance', 'Total area under curve' and so on..\n "
    "Value, is a number, the value of parameter.\n "
    "Unit, the unit of the value.\n "
    "Summary statistics, the statistics method to summary the data, like 'geometric mean', 'arithmetic mean'.\n "
    "Variation value, is a number, the number that corresponds to the specific variation..\n "
    "Variation type, the variability measure (describes how spread out the data is) associated with the specific parameter, e.g., standard deviation (SD), CV%..\n "
    "Interval type, is text, specifies the type of interval that is being used to describe uncertainty or variability around a measure or estimate, like '95% CI', 'range' and so on..\n "
    "Lower limit, is a number, the lower bounds of the interval.\n "
    "High limit, is a number, the higher bounds of the interval.\n "
    "P value, The p-value is a number, calculated from a statistical test, that describes the likelihood of a particular set of observations if the null hypothesis were true; varies depending on the study, and therefore it may not always be reported."
)
pe_cols_definition = (
    "Characteristic/risk factor lists the characteristics or risk factors associated with the outcomes in the study population, for example, age, race, year or time interval of interest, location etc..\n "
    "Exposure lists the main exposure of interest associated with the outcomes, which will primarily be the drug name, treatment, or therapy that is studied in the paper..\n "
    "Outcomes is A measure(s) of interest that is correlated with the exposure and is reported as the primary or secondary outcome(s) in the PE papers. These could be pregnancy, obstetric or birth related outcomes that are associated with the drug or exposure of interest. They could also include general disease or health conditions, health related events, or deaths..\n "
    "Statistics is the measure used to describe numerical values (ex: mean with standard deviations (SD), or median with range or interquartile range (IQR), absolute/relative risks, risk ratios, odds ratios, hazard ratios etc.).\n "
    "Value includes the numerical value(s) associated with their respective PE outcomes.\n "
    "Unit will includes the unit of measurement associated with the values of the PE outcomes, for example, grams, weeks, days etc..\n "
    "Variablility statistic lists the name of the statistic used to indicate the spread of data and is related to the main statistical method used for reporting (Column E), for example, standard deviations (SD), or range or interquartile range (IQR)..\n "
    "Variablility value includes the numerical value(s) reported for the variability statistic. .\n "
    "Interval type list the name of the interval which is expected to typically contain the parameter being estimated, for example, range, IQR, confidence intervals etc..\n "
    "Interval low is the lower end value reported based on the interval type. .\n "
    "Interval high is theupper end value reported based on the interval type..\n "
    "P value includes the numerical value associated with the outcomes and the statistical test conducted. Not always reported in the studies."
)

def test_generate_pk():
    res = generate_columns_definition("pk")
    assert res == pk_cols_definition

def test_generate_pe():
    res = generate_columns_definition("pe")
    assert res == pe_cols_definition

