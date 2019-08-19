
# coding: utf-8

# In[ ]:


#Alex Ram
#Stat 287 - Final Project
#12/07/2018


# In[ ]:


#Imports

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats.stats import pearsonr
get_ipython().magic('matplotlib inline')


# In[ ]:


#loading data

happiness_df = pd.read_csv("data/happiness.csv")
changes_in_hap_df = pd.read_csv("data/change_hap.csv")
foreign_local_hap_df = pd.read_csv("data/foreign_local_hap.csv")
hap_factors_df = pd.read_csv("data/hap_factors.csv")
migration_rates_df = pd.read_csv("data/UNdata_migration.csv")
urban_rural_pop_df = pd.read_csv("data/urban-and-rural-population.csv")


# In[ ]:


#visualize data first

print("Ten happiest countries and their migration rates")
for country in happiness_df["Country"][0:10]:
    x = migration_rates_df['Country or Area'] == country
    recent, old = migration_rates_df['Value'][x]
    print(country + " -> " + str(recent))

print("Ten least happy countries and their migration rates")
for country in happiness_df["Country"][-11:-1]:
    x = migration_rates_df['Country or Area'] == country
    recent, old = migration_rates_df['Value'][x]
    print(country + " -> " + str(recent))


# In[ ]:


#Shapping datasets

#first, for happiness and migration rates now (hapScore_migRate)
countries = happiness_df['Country']
hapScore_migRate = np.zeros((len(countries), 2),dtype=float)

count = 0
for country in countries:
    for i in range(len(migration_rates_df['Country or Area'])):
        if country == migration_rates_df['Country or Area'][i] and migration_rates_df['Year(s)'][i] == '2015-2020':
            hapScore_migRate[count][0] = happiness_df['Happiness score'][count]
            hapScore_migRate[count][1] = migration_rates_df['Value'][i]
    count+=1

missing_migs = []
for i in range(len(countries)):
    if hapScore_migRate[i][1] == 0:
        missing_migs.append(i)
        
#print(countries[missing_migs]) 
#If uncommented you see a few 0-values for countries. Originally there were 28 (many because of misssing data). 
#I had to either find migration data for missing countries/missing values for existing countries in the dataset. 
#Sometimes it was just matching the names of countries. This was all done by altering the .csv files.
#There is the case where migration = 0, which only happened with the three countries shown by the print statement.

#print(hapScore_migRate) #to visualize the array produced above

#second, for change in happiness and change in migraiton rates (changeIn_hap_mig)
change_countries = changes_in_hap_df['Country']
changeIn_hap_mig = np.zeros((len(change_countries), 2),dtype=float)

count = 0
for country in change_countries:
    for i in range(len(migration_rates_df['Country or Area'])):
        if country == migration_rates_df['Country or Area'][i] and migration_rates_df['Year(s)'][i] == '2015-2020':
            changeIn_hap_mig[count][0] = changes_in_hap_df['Changes in happiness scores (2008/2010 - 2015/2017)'][count]
            #the first value is 2015-2020 and right bellow that is the same countries 2005-2010
            migRate2005 = migration_rates_df['Value'][i+1]
            migRate2015 = migration_rates_df['Value'][i]
            sign = 1
            if migRate2005 > migRate2015:
                sign = -1 
            migRate_change = sign * max(migRate2005 - migRate2015, migRate2015 - migRate2005)
            #to check that the change isn't 0 so the logic is sound - there is none.
            if migRate_change == 0:
                print(migration_rates_df['Country or Area'][i])
            if changeIn_hap_mig[count][1] == 0:
                changeIn_hap_mig[count][1] = migRate_change
    count+=1
    
#print(changeIn_hap_mig) #to visualize the array produced above

#third, difference in local vs foreign born happiness levels and migration rates (forVSloc_mig)
forVSloc_countries = foreign_local_hap_df['Country']
forVSloc_mig = np.zeros((len(forVSloc_countries), 2),dtype=float)

count = 0
for country in forVSloc_countries:
    for i in range(len(migration_rates_df['Country or Area'])):
        if country == migration_rates_df['Country or Area'][i] and migration_rates_df['Year(s)'][i] == '2015-2020':
            forHap = foreign_local_hap_df['Average happiness of foreign born'][count]
            locHap = foreign_local_hap_df['Average happiness of locally born'][count]
            #positive means foreign born are happier than local born, negative means vice versa
            forVSloc_mig[count][0] = forHap - locHap
            forVSloc_mig[count][1] = migration_rates_df['Value'][i]
    count+=1
#don't need to worry about missing values at this point because it's been taken care of for these countries above    

#print(forVSloc_mig) #to visualize the array produced above

#fourth, is migration related to any factor used to calculate happiness strongly?
allFactors_countries = hap_factors_df['country']
#for now replace nans with 0, but delete them later
hap_factors_df.fillna(0, inplace=True)

dystopia_mig = np.zeros((len(allFactors_countries), 2),dtype=float)
GDP_mig = np.zeros((len(allFactors_countries), 2),dtype=float)
lifeExp_mig = np.zeros((len(allFactors_countries), 2),dtype=float)
socialSup_mig = np.zeros((len(allFactors_countries), 2),dtype=float)
freedom_mig = np.zeros((len(allFactors_countries), 2),dtype=float)
generosity_mig = np.zeros((len(allFactors_countries), 2),dtype=float)
corruption_mig = np.zeros((len(allFactors_countries), 2),dtype=float)

count = 0
for country in allFactors_countries:
    for i in range(len(migration_rates_df['Country or Area'])):
        if country == migration_rates_df['Country or Area'][i] and migration_rates_df['Year(s)'][i] == '2015-2020':
            dystopia_mig[count][0] = hap_factors_df['Life ladder, 2015-2017'][count]
            GDP_mig[count][0] = hap_factors_df['GDP per person, 2015-2017'][count]
            lifeExp_mig[count][0] = hap_factors_df['Healthy life expectancy, 2015-2017'][count]
            socialSup_mig[count][0] = hap_factors_df['Social support, 2015-2017'][count]
            freedom_mig[count][0] = hap_factors_df['Freedom to make life choices, 2015-2017'][count]
            generosity_mig[count][0] = hap_factors_df['Generosity, 2015-2017, without adjustment for GDP per person'][count]
            corruption_mig[count][0] = hap_factors_df['Perceptions of corruption, 2015-2017'][count]
            dystopia_mig[count][1] = migration_rates_df['Value'][i]
            GDP_mig[count][1] = migration_rates_df['Value'][i]
            lifeExp_mig[count][1] = migration_rates_df['Value'][i]
            socialSup_mig[count][1] = migration_rates_df['Value'][i]
            freedom_mig[count][1] = migration_rates_df['Value'][i]
            generosity_mig[count][1] = migration_rates_df['Value'][i]
            corruption_mig[count][1] = migration_rates_df['Value'][i]       
    count+=1
    
#print(allFactors_mig) #to visualize the arrays produced above

#remove 0 value rows from factor arrays
def removeZeroRows(data):
    for i in range(len(data)-1,-1,-1):
        if data[i,0] == 0:
            data = np.delete(data, (i), axis=0)
    return data

dystopia_mig = removeZeroRows(dystopia_mig)
GDP_mig = removeZeroRows(GDP_mig)
lifeExp_mig = removeZeroRows(lifeExp_mig)
socialSup_mig = removeZeroRows(socialSup_mig)
freedom_mig = removeZeroRows(freedom_mig)
generosity_mig = removeZeroRows(generosity_mig)
corruption_mig = removeZeroRows(corruption_mig)

#fifth, use urban-rural population data and relate that to happiness scores
countries = happiness_df['Country']
hapScore_urbanRural = np.zeros((len(countries), 2),dtype=float)

count = 0
for country in countries:
    for i in range(len(urban_rural_pop_df['Entity'])):
        if country == urban_rural_pop_df['Entity'][i] and urban_rural_pop_df['Year'][i] == 2010:
            hapScore_urbanRural[count][0] = happiness_df['Happiness score'][count]
            change2010 = urban_rural_pop_df['Urban population'][i] / (urban_rural_pop_df['Rural population'][i] + 1)
            change2016 = urban_rural_pop_df['Urban population'][i+6] / (urban_rural_pop_df['Rural population'][i+6] + 1)
            sign = 1
            if change2010 > change2016:
                sign = -1
            rural_to_urban_mig = sign * max(change2010 - change2016, change2016 - change2010)
            hapScore_urbanRural[count][1] = rural_to_urban_mig
    count+=1

missing_migs = []
for i in range(len(countries)):
    if hapScore_urbanRural[i][1] == 0:
        missing_migs.append(i)
        
#print(countries[missing_migs]) # to see countries with missing data in urban-rural-pop

#this time i deleted these rows because it was much harder to find data for these coutries
missing_migs.reverse()
for i in missing_migs:
    hapScore_urbanRural = np.delete(hapScore_urbanRural, (i), axis=0)


# In[ ]:


#Create some histograms to visualize migration rates and happiness of regions

#collect indecies of countries in each region
regions = dict()
for i in range(len(hap_factors_df)):
    region = hap_factors_df['Region indicator'][i]
    if regions.get(region) == None:
        regions[region] = [i]
    else:
        regions[region].append(i)

#normalize data
hapScore_migRate_norm = np.zeros((len(countries), 2),dtype=float)
for i in range(2):
    hapScore_migRate_norm[:,i] = hapScore_migRate[:,i] / np.linalg.norm(hapScore_migRate[:,i])
    
#plot
for region in regions:
    fig = plt.figure()
    plt.title(region)
    plt.xlabel("Normalized rate or score")
    plt.ylabel("Countries")
    hapScores = []
    migRates = []
    for i in regions[region]:
        hapScores.append(hapScore_migRate_norm[i,0])
        migRates.append(hapScore_migRate_norm[i,1])
        
        
    bins = np.linspace(min(min(hapScores),min(migRates)), max(max(hapScores),max(migRates)), 30)
            
    plt.hist(hapScores, bins, alpha=0.5, label="Happiness Score", facecolor='blue')
    plt.hist(migRates, bins, alpha=0.5, label="Migration Rate", facecolor='orange')
    plt.legend(loc='best', fontsize=8)
    plt.axvline(x=np.mean(hapScore_migRate_norm[:,0]), linestyle='dashed', color='blue')
    plt.axvline(x=0, linestyle='dashed', color='orange')
    plt.show()
    fig.savefig("stats/"+region+'.png')


# In[ ]:


#preprocessing

#remove outliers
def removeOutliers(data):
    #find outliers
    m = 2
    outliers_index = []
    for column in range(len(data[0])):
        mean = np.mean(data[:,column])
        std = np.std(data[:,column])
        for i in range(len(data)):          
            if abs(data[i,column] - mean) > m * std:
                outliers_index.append(i)
    
    #remove rows with outliers
    outliers_index = sorted(outliers_index, reverse=True)
    for i in outliers_index:
        data = np.delete(data, (i), axis=0)
        
    return data

hapScore_migRate = removeOutliers(hapScore_migRate)

changeIn_hap_mig = removeOutliers(changeIn_hap_mig)

forVSloc_mig = removeOutliers(forVSloc_mig)

dystopia_mig = removeOutliers(dystopia_mig)
GDP_mig = removeOutliers(GDP_mig)
lifeExp_mig = removeOutliers(lifeExp_mig)
socialSup_mig = removeOutliers(socialSup_mig)
freedom_mig = removeOutliers(freedom_mig)
generosity_mig = removeOutliers(generosity_mig)
corruption_mig = removeOutliers(corruption_mig)

hapScore_urbanRural = removeOutliers(hapScore_urbanRural)
hapScore_urbanRural = removeOutliers(hapScore_urbanRural)


# In[ ]:


#Graphs

def createGraph(data, title, xlabel, ylabel):
    fig = plt.figure()
    x = data[:,0]
    y = data[:,1]
    plt.scatter(x, y)
    plt.plot(np.unique(x), np.poly1d(np.polyfit(x, y, 1))(np.unique(x)))
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()
    c, p = pearsonr(x, y)
    print("Correlation coefficient: " + str(c) + " p-value: " + str(p))
    fig.savefig('stats/'+title+'.png')
    
createGraph(hapScore_migRate, "Migration Rates VS Happiness Scores", "Happiness scores", "Migration rates")

createGraph(changeIn_hap_mig, "Change in Migration Rates VS Change in Happiness Scores (2007 - 2018)", "Change in Happiness scores", "Change in Migration rates")

createGraph(forVSloc_mig, "Migration Rates VS Foreign and Local Born Happiness", "(Foreign minus Local) Born Happiness Score", "Migration rates")

createGraph(dystopia_mig, "Migration Rates VS Life Ladder Reports", "Dystopia Score", "Migration rates")

createGraph(GDP_mig, "Migration Rates VS GDP", "GDP", "Migration rates")

createGraph(lifeExp_mig, "Migration Rates VS Life Expectancy", "Life Expectancy", "Migration rates")

createGraph(socialSup_mig, "Migration Rates VS Social Support", "Social Support", "Migration rates")

createGraph(freedom_mig, "Migration Rates VS Freedom", "Freedom", "Migration rates")

createGraph(generosity_mig, "Migration Rates VS Generosity", "Generosity", "Migration rates")

createGraph(corruption_mig, "Migration Rates VS Corruption", "corruption", "Migration rates")

createGraph(hapScore_urbanRural, "Rural to Urban Migration Rates VS Happiness Scores", "Happiness scores", "Rural to Urban Migration Rates")

