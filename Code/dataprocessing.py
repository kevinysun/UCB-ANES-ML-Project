#import pyreadstat as prs
import numpy as np
import pandas as pd
import scipy as sp
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.impute import KNNImputer
from sklearn.preprocessing import MinMaxScaler


def fit_prepare(data, labels, drop_thresh = 0.4, cont = []):
    """
    df: pandas dataframe
    drop_thresh: proportion of missing values threshold, if over this we drop the row
    cont: list of continuous variables to not use one-hot encoding with
    y: labels
    outputs: encoded ndarray to fit on


    ISSUE: need to somehow be able to identify columns
    """
    df = data.copy()
    pshape = df.shape
    df.insert(0, "y", labels.values)
    df = (df.mask(df < 0, np.nan)
                .dropna(axis = 0, thresh = np.round(drop_thresh * pshape[1]))
                )

    y_out = df['y']
    
    df = df.drop(columns = ['y'])
    scaler = MinMaxScaler()
    df = pd.DataFrame(scaler.fit_transform(df), columns = df.columns)  
    
    #knn imputer
    imputer = KNNImputer(n_neighbors=10, weights='uniform')


    df_imputed = pd.DataFrame(imputer.fit_transform(df),
                                    columns = df.columns)
    #rescale

    df_imputed = scaler.inverse_transform(df_imputed)
    df_imputed = pd.DataFrame(df_imputed, columns = df.columns)
    # knn imputer gives decimals, so we round to integers
    # ISSUE: continuous columns?
    # we will pass a list of column names so just ignore those
    # create df_cat which drops non-categorical
    if not cont:
        df_cat = df_imputed.copy()
    else:
        df_cat = df_imputed.drop(cont, axis=1) 

    #cat_vars = df_cat.columns
    df_cat = np.round(df_cat).astype(np.int32)

    if cont:
        df_cont = df_imputed[cont].to_numpy()
    
    #one-hot encoding
    encoder = OneHotEncoder()
    encoder.fit(df_cat)
    df_enc = encoder.transform(df_cat).A

    if not cont:
        df_prepared = df_enc
    else:
        df_prepared = np.hstack((df_cont, df_enc))

    return df_prepared, encoder, y_out


def load_data():
    '''
    load and initial processing of data
    separate outcome variable
    also returns feature info (question id and description)
    '''
    # filter only by those who have voted for R or D
    data = pd.read_csv("Data/anes_timeseries_2016.csv", low_memory=False)
    featureinfo = pd.read_csv("Codebook/anes_timeseries_2016_varlist.csv")
    # methodology = pd.read_csv("data/anes_timeseries_2016_methodology.csv")
    data = data[data["V162034a"].isin([1,2])]
    y = data["V162034a"]
    data = data.drop(columns = ["V162034a"])
    return data, y, featureinfo


def data_subset(data, featureinfo, featurelist=None):
    '''
    subset loaded data based on predetermined groups

    featurelist: list of QIDs as strings
    '''
    if featurelist is None:
        featurelist = data.columns

    data_subset = data[featurelist]
    features_subset = featureinfo[featureinfo['QID'].isin(featurelist)]
    return data_subset, features_subset


def subset_social_financial(data, featureinfo):
    '''
    subset our data based on social and financial questions
    '''
    sf_QIDs = ['V162098',  #Feeling thermometer: LABOR UNIONS
                    'V162099',  #Feeling thermometer: POOR PEOPLE
                    'V162100', # Feeling thermometer: BIG BUSINESS
                    'V162105', # Feeling thermometer: RICH PEOPLE
                    'V161361x', #Pre income summary
                    'V162128',  #Think of self as belonging to class
                    'V162129',  #Is R working or middle class
                    'V162130',  #If R had to choose working/ middle class
                    'V162131',  #Average or upper working/ middle class
                    'V162132',  #R social class
                    'V162133',  #Is R upper middle, middle, lower middle class
                    'V162134',  #How much opportunity in America to get ahead
                    'V162135',  #Economic mobility compared to 20 yrs ago
                    'V162136',  #How much easier/harder is econ mobility compared to 20 yrs ago
                    'V162136x', #SUMMARY -  Economic mobility easier/harder compared to 20 yrs ago
                    'V162137',  #What is current unemployment rate
                    'V162138',  #What is minimum wage in R state
                    'V162139',  #Importance of reducing deficit
                    'V162140',  #Does R favor or oppose tax on millionaires
                    'V162165',  #Worry about financial situation
                    'V162166',  #Able to make housing payments
                    'V162167',  #Anyone lost jobs
                    'V162180',  #Should gov do more or less to regulate banks
                    'V162180a', #How much more or less should gov do to regulate banks
                    'V162180x', #SUMMARY-  Gov should do more/less to regulate banks
                    'V162183',  #Govt bigger because too involved OR bigger problems
                    'V162184',  #Need strong govt for complex problems OR free market
                    'V162185',  #Less govt better OR more that govt should be doing
                    'V162186',  #Regulation of Business
                    'V162192',  #Should the minimum wage be raised
                    'V162268',  #Immigrants are generally good for America's economy
                    'V162276',  #Gov should take measures to reduce differences in income levels
                    'V162280'  #CSES: State of economy
                ]
    sf_data, sf_features = data_subset(data, featureinfo, featurelist=sf_QIDs)
    sf_cont = ['V162098',  #Feeling thermometer: LABOR UNIONS
                'V162099',  #Feeling thermometer: POOR PEOPLE
                'V162100', # Feeling thermometer: BIG BUSINESS
                'V162105'] # Feeling thermometer: RICH PEOPLE
    return sf_data, sf_features, sf_cont


def subset_community(data, featureinfo):
    '''
    subset our data based on Community Social and Political Activity
    '''
    community_QIDs = ['V162141',  #How often bought or boycotted product or service for pol/soc reason
                    'V162174',  #Ever discuss politics with family or friends
                    'V162174a', #Days in past week discussed politics
                    'V162194',  #Number of organizations in which R is a member
                    'V162195',  #Has R done community work in past 12 months 
                    'V162196',  #Did R attend meeting on school/community issue past 12 months
                    'V162197',  #Has R done any volunteer work in past 12 months
                    'V162198',  #Has R contacted elected federal official in past 12 months
                    'V162200',  #Has R contacted non- elected federal official in past 12 months
                    'V162202',  #Has R contacted elected local official in past 12 months
                    'V162204'  #Has R contacted non- elected local official in past 12 months
                    ]
    community_data, community_features = data_subset(data, featureinfo, featurelist=community_QIDs)
    community_cont = []
    return community_data, community_features, community_cont


def subset_polit_sys(data, featureinfo):
    '''
    subset our data based on view of political system
    '''
    QIDs = ['V162215', # [STD] Publ officials don't care what people think
                    'V162216',  #[STD] Have no say about what govt does
                    'V162217',  #[REV] Politics/govt too complicated to understand
                    'V162218',  #[REV] Good understanding of political issues
                    'V162219',  #Electoral integrity: are votes counted fairly
                    'V162220',  #Electoral integrity: do the rich buy elections
                    'V162234',  #Does R favor or oppose limits on campaign spending
                    'V162235',  #How much does Cong pass laws that benefit contributor organizations
                    'V162236',  #How much does Cong pass laws that benefit contributor individuals
                    'V162256',  #R's interest in politics
                    'V162257',  #R follows politics in media
                    'V162258',  #R understands most important political issues
                    'V162259',  #Compromise in politics is selling out on one's principles
                    'V162260',  #Most politicians do not care about the people
                    'V162261',  #Most politicians are trustworty
                    'V162262',  #Politicians are the main problem in the U.S.
                    'V162263',  #Strong leader is good for U.S. even if bends rules to get things done
                    'V162264',  #People not politicians should make most important policy decisions
                    'V162265',  #Most politicians only care about interests of rich and powerful
                    'V162267',  #The will of the majority should always prevail
                    'V162275',  #How widespread is corruption among politicians in U.S.
                    'V162290'  #CSES: Satisfied with way democracy works in the U.S
                    ]
    polisys_data, polisys_features = data_subset(data, featureinfo, featurelist=QIDs)
    polisys_cont = []
    return polisys_data, polisys_features, polisys_cont


def subset_pers(data, featureinfo):
    '''
    subset our data based on personality
    '''
    QIDs = ['V162207', # Agree/disagree: world is changing and we should adjust
            'V162208',  #Agree/disagree: newer lifestyles breaking down society
            'V162209',  ##Agree/disagree: be more tolerant of other moral standards
            'V162210',  #Agree/disagree: more emphasis on traditional family values
            'V162239',  #Child trait more important: independence or respect
            'V162240',  #Child trait more important: curiosity or good manners
            'V162241',  #Child trait more important: obedience or self- reliance
            'V162242',  #Child trait more important: considerate or well- behaved
            'V162243',  #Society should make sure everyone has equal opportunity
            'V162244',  #We'd be better off if worried less about equality
            'V162245',  #Not a big problem if some have more chance in life
            'V162246',  #If people were treated more fairly would be fewer probs
            'V162248',  #R likes to have strong opinions even when not personally involved
            'V162249',  #R forms opinions about everything
            'V162250',  #Important for R to hold strong opinions
            'V162251',  #It bothers R to remain neutral
            'V162252',  #R has many more opinions than the average person
            'V162253',  #R would rather have strong opinion than no opinion
            'V162253x', #SUMMARY-  Need to Evaluate score
            'V162333',  #FTF CASI/WEB: TIPI extraverted, enthusiastic
            'V162334',  #FTF CASI/WEB: TIPI critical, quarrelsome
            'V162335',  #FTF CASI/WEB: TIPI dependable, self- disciplined
            'V162336',  #FTF CASI/WEB: TIPI anxious, easily upset
            'V162337',  #FTF CASI/WEB: TIPI open to new experiences
            'V162338',  #FTF CASI/WEB: TIPI reserved, quiet
            'V162339',  #FTF CASI/WEB: TIPI sympathetic, warm
            'V162340',  #FTF CASI/WEB: TIPI disorganized, careless
            'V162341',  #FTF CASI/WEB: TIPI calm, emotionally stable
            'V162342',  #FTF CASI/WEB: TIPI conventional, uncreative
            'V162343',  #FTF CASI/WEB: How hard is it for R to control temper
            'V162344'  #FTF CASI/WEB: When provoked, how likely for R to hit someone
                    ]
    pers_data, pers_features = data_subset(data, featureinfo, featurelist=QIDs)
    pers_cont = ['V162333',  #FTF CASI/WEB: TIPI extraverted, enthusiastic
            'V162334',  #FTF CASI/WEB: TIPI critical, quarrelsome
            'V162335',  #FTF CASI/WEB: TIPI dependable, self- disciplined
            'V162336',  #FTF CASI/WEB: TIPI anxious, easily upset
            'V162337',  #FTF CASI/WEB: TIPI open to new experiences
            'V162338',  #FTF CASI/WEB: TIPI reserved, quiet
            'V162339',  #FTF CASI/WEB: TIPI sympathetic, warm
            'V162340',  #FTF CASI/WEB: TIPI disorganized, careless
            'V162341',  #FTF CASI/WEB: TIPI calm, emotionally stable
            'V162342',  #FTF CASI/WEB: TIPI conventional, uncreative
            'V162343',  #FTF CASI/WEB: How hard is it for R to control temper
            'V162344'  #FTF CASI/WEB: When provoked, how likely for R to hit someone
                    ]
    return pers_data, pers_features, pers_cont


def subset_gender(data, featureinfo):
    '''
    subset our data based on gender-related questions
    '''
    QIDs = ['V162149',  #Does R favor or oppose requiring equal pay for men and women
            'V162150',  #How much favor or oppose requiring equal pay for men and women
            'V162150x', #SUMMARY-  Favor/oppose equal pay for men and women
            'V162227',  #How important that more women get elected
            'V162228',  #Easier or harder for working mother to bond with child
            'V162229a', #How much easier for working mother to bond with child
            'V162229b', #How much harder for working mother to bond with child
            'V162229x', #SUMMARY-  Working mother's bond with child
            'V162230',  #Better if man works and woman takes care of home
            'V162230a', #How much better if man works and woman at home
            'V162230b', #How much worse if man works and woman at home
            'V162230x', #SUMMARY-  Better if man works and woman takes care of home
            'V162231',  #Media pay more attention to discrimination
            'V162231a', #How much more attn should media pay to discrim against women
            'V162231b', #How much less attn should media pay to discrim against women
            'V162231x', #SUMMARY-  How much attn media should pay to discrim against women
            'V162232',  #Do women demanding equality seek special favors
            'V162233',  #Do women complaining about discrim cause more problems
            'V162362',  #FTF CASI/WEB: Discrimination in the U.S. against Women
            'V162363'   #FTF CASI/WEB: Discrimination in the U.S. against Men
                    ]
    gender_data, gender_features = data_subset(data, featureinfo, featurelist=QIDs)
    gender_cont = []
    return gender_data, gender_features, gender_cont


def subset_hcsci(data, featureinfo):
    '''
    subset our data based on healthcare/science
    '''
    QIDs = ['V162112',  #Feeling thermometer: SCIENTISTS
            'V162142',  #Health Care Law effect on health care services
            'V162143',  #Health Care Law effect on number insured
            'V162144',  #Health Care Law effect on cost of health care
            'V162145',  #Health Care Law effect on cost of R's health care
            'V162146',  #Does R favor or oppose vaccines in schools
            'V162147',  #How much favor or oppose vaccines in schools
            'V162147x', #SUMMARY-  Favor/oppose vaccines in schools
            'V162161',  #Health benefits of vaccinations outweigh risks
            'V162162',  #Vaccinations benefit/risk strength
            'V162162x', #SUMMARY-  Benefits/risks of vaccinations
            'V162163',  #Put off checkup and vaccines
            'V162164',  #Will you pay all costs
            'V162193',  #Increase or decrease gov spending to help people pay for health care
            'V162193a', #How much favor increase/decrease gov help paying for health care
            'V162193x'  #SUMMARY-  Increase/decrease gov spending for health care
                    ]
    hcsci_data, hcsci_features = data_subset(data, featureinfo, featurelist=QIDs)
    hcsci_cont = ['V162112'  #Feeling thermometer: SCIENTISTS
                    ]
    return hcsci_data, hcsci_features, hcsci_cont


def subset_natlism(data, featureinfo):
    '''
    subset our data based on National Identity
    '''
    QIDs = ['V162123',  #Better if rest of world more like America
            'V162124',  #How does R feel to see American flag
            'V162125',  #How good/bad does R feel to see American flag
            'V162168',  #Country needs free thinkers
            'V162169',  #Country would be great by getting rid of rotten apples
            'V162170',  #Country needs strong leader to take us back to true path
            'V162266',  ##Minorities should adapt to to customs/traditions of U.S
            'V162269',  #America's culture is generally harmed by immigrants
            'V162271',  #To be truly American important to have been born in U.S.
            'V162272',  #To be truly American important to have American ancestry 
            'V162273',  #To be truly American important to speak English
            'V162274',  #To be truly American important to follow America's customs/traditions
            'V162332',  #FTF CASI/WEB: How important is being American to identity
            'V162355',  #FTF CASI/WEB: Stereotype: Muslims patriotic
            'V162356'   #FTF CASI/WEB: Stereotype: Christians patriotic
                    ]
    natl_data, natl_features = data_subset(data, featureinfo, featurelist=QIDs)
    natl_cont = []
    return natl_data, natl_features, natl_cont


def subset_intl(data, featureinfo):
    '''
    subset our data based on  International relations
    '''
    QIDs = [
        'V162152a', #Does R favor or oppose limits on foreign imports [REV]
        'V162152b', #Does R favor or oppose limits on foreign imports [STD]
        'V162153',  #Is U.S. too supportive of Israel or not supportive enough
        'V162154a', #How much U.S. support Israel in conflict w/Palestinians [ISR 1st]
        'V162154b', #How much U.S. support Palestinians in conflict w/Israel [ISR 1st]
        'V162155a', #How much U.S. support Palestinians in conflict w/Israel [PAL 1st] 
        'V162155b', #How much U.S. support Israel in conflict w/Palestinians [PAL 1st] 
        'V162155x', #SUMMARY-  How much should U.S. support Israelis
        'V162156x', #SUMMARY-  How much should U.S. support Palestinians
        'V162157',  #What should immigration levels be
        'V162158',  #How likely immigration will take away jobs
        'V162159',  #China military threat
        'V162160',  #How worried about terrorist attack next 12 months
        'V162176',  #Does R favor or oppose free trade agreements w/other countries
        'V162176a', #How strongly favor/oppose free trade agreements w/other countries
        'V162176x', #SUMMARY-  Favor/oppose free trade agreements
        'V162177',  #Should govt encourage/discourage outsourcing
        'V162294',  #DHS: How worried about terrorist attack in next 12 months
        'V162295',  #DHS: Favor or oppose torture for suspected terrorists
        'V162295a', #DHS: How much favor torture for suspected terrorists
        'V162295b', #DHS: How much oppose torture for suspected terrorists
        'V162295x', #SUMMARY-  Favor/oppose torture for suspected terrorists
        'V162313'   #FTF CASI/WEB: Feeling thermometer: ILLEGAL IMMIGRANTS
        ]
    intl_data, intl_features = data_subset(data, featureinfo, featurelist=QIDs)
    intl_cont = ['V162313' #FTF CASI/WEB: Feeling thermometer: ILLEGAL IMMIGRANTS
                ]
    return intl_data, intl_features, intl_cont


def subset_race(data, featureinfo):
    '''
    subset our data based on racial questions
    '''
    QIDs = [
        'V162211',  #Agree/disagree: blacks shd work way up w/o special favors
        'V162212',  ##Agree/disagree: past slavery make more diff for blacks
        'V162213',  #Agree/disagree: blacks have gotten less than deserve
        'V162214',  #Agree/disagree: blacks must try harder to get ahead
        'V162221',  #How important that more Hispanics get elected
        'V162224',  #Hisp R: life be affected by what happens to Hispanics
        'V162225',  #Black R: life be affected by what happens to blacks
        'V162226',  #Asian R: life be affected by what happens to Asians
        'V162238a', #Strength favor preferential hiring/promotion of blacks
        'V162238b', #Strength oppose preferential hiring/promotion blacks
        'V162238x', #SUMMARY-  Favor preferential hiring and promotion of blacks
        'V162310',  #FTF CASI/WEB: Feeling thermometer: ASIAN- AMERICANS
        'V162311',  #FTF CASI/WEB: Feeling thermometer: HISPANICS
        'V162312',  #FTF CASI/WEB: Feeling thermometer: BLACKS
        'V162314',  #FTF CASI/WEB: Feeling thermometer: WHITES
        'V162316',  #FTF CASI/WEB: How important whites work together to change laws unfair to whites
        'V162317',  #FTF CASI/WEB: How likely whites unable to find job b/c employers hire minorities
        'V162318',  #FTF CASI/WEB: Federal gov treats blacks or whites better
        'V162319',  #FTF CASI/WEB: How much federal gov treats blacks or whites better
        'V162320',  #FTF CASI/WEB: Police treat blacks or whites better
        'V162321',  #FTF CASI/WEB: How much police treat blacks or whites better
        'V162322',  #FTF CASI/WEB: How much influence do whites have in U.S. politics
        'V162323',  #FTF CASI/WEB: How much influence do blacks have in U.S. politics
        'V162324',  #FTF CASI/WEB: How much influence do Hispanics have in U.S. politics
        'V162325',  #FTF CASI/WEB: How much influence do Asian- Americans have in U.S. politics
        'V162345',  #FTF CASI/WEB: Stereotype: Whites hardworking
        'V162346',  #FTF CASI/WEB: Stereotype: Blacks hardworking
        'V162347',  #FTF CASI/WEB: Stereotype: Hispanics hardworking
        'V162348',  #FTF CASI/WEB: Stereotype: Asians hardworking
        'V162349',  #FTF CASI/WEB: Stereotype: Whites violent
        'V162350',  #FTF CASI/WEB: Stereotype: Blacks violent
        'V162351',  #FTF CASI/WEB: Stereotype: Hispanics violent
        'V162352',  #FTF CASI/WEB: Stereotype: Asians violent
        'V162357',  #FTF CASI/WEB: Discrimination in the U.S. against Blacks
        'V162358',  #FTF CASI/WEB: Discrimination in the U.S. against Hispanics
        'V162359',  #FTF CASI/WEB: Discrimination in the U.S. against AsianAmericans
        'V162360'   #FTF CASI/WEB: Discrimination in the U.S. against Whites
        ]
    race_data, race_features = data_subset(data, featureinfo, featurelist=QIDs)
    race_cont = ['V162310',  #FTF CASI/WEB: Feeling thermometer: ASIAN- AMERICANS
                'V162311',  #FTF CASI/WEB: Feeling thermometer: HISPANICS
                'V162312',  #FTF CASI/WEB: Feeling thermometer: BLACKS
                'V162314',  #FTF CASI/WEB: Feeling thermometer: WHITES
                ]
    return race_data, race_features, race_cont


def subset_other(data, featureinfo):
    '''
    subset our data based on racial questions
    '''
    QIDs = [
        'V162151',  #Changes in security at public places
        'V162178',  #Has increase in govt wiretap powers gone too far
        'V162179',  ##Should marijuana be legal
        'V162254',  #Did the U.S. government know about 9/11 in advance
        'V162270',  #Immigrants increase crime rates in the U.S.
        'V162296a', #FTF CASI/WEB: WEB ONLY: R has any living sons or daughters
        'V162296b', #FTF CASI/WEB: FTF ONLY: R has any living sons or daughters (2nd mention -  order)
        'V162296c', #FTF CASI/WEB: WEB ONLY: R has any living sons or daughters
        'V162296x', #FTF CASI/WEB: FTF/WEB: SUMMARY-  R has living sons or daughters
        'V162297',  #FTF CASI/WEB: In past 12 months any family members stopped/questioned by police
        'V162298',  #FTF CASI/WEB: Has R ever been arrested
        'V162367',  #FTF CASI/WEB: How much discrimination has R faced personal
        'V162368',  #FTF CASI/WEB: R rate own skintone
        'V162369',  #FTF CASI/WEB: Discrimination due to skintone
        'V162370'   #FTF CASI/WEB: Facebook account used recently
        ]
    other_data, other_features = data_subset(data, featureinfo, featurelist=QIDs)
    other_cont = []
    return other_data, other_features, other_cont
