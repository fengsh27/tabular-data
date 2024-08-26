
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
    "Characteristic/risk factor describes the particular or general characteristics of the study population related to the study's outcomes..\n "
    "Exposure refers to Any factor that may be associated with an outcome of interest (drugs, medical conditions, medications etc.).\n "
    "Outcomes is A measure(s) of interest that an investigator(s) considers to be the most important among the many outcomes that are to be examined in the study; however, in the study outcomes sheet, this is where all measures as outlined in the article must be well elaborated to include their reported statistics..\n "
    "Statistics is the measure used to describe numerical values (ex: adjusted odds ratio, multivariate, hazard ratio, risk ratio, incident rate, frequency, occurrence).\n "
    "Value.\n "
    "Unit.\n "
    "Variablility statistic means how spread out the data is, based on the study type (ex: standard deviation).\n "
    "Variablility value.\n "
    "Interval type is a scale the examines the difference usually between two numbers (range, interquartile range, 95% confidence interval).\n "
    "Interval low is the lower end value reported based on the interval type. .\n "
    "Interval high is theupper end value reported based on the interval type..\n "
    "P value is a number, calculated from a statistical test, that describes the likelihood of a particular set of observations if the null hypothesis were true; varies depending on the study, and therefore it may not always be reported."
)

def test_generate_pk():
    res = generate_columns_definition("pk")
    assert res == pk_cols_definition

def test_generate_pe():
    res = generate_columns_definition("pe")
    assert res == pe_cols_definition

