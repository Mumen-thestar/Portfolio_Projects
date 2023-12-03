## 2. Introduction to the Data ##

happiness2015= pd.read_csv('World_Happiness_2015.csv')

first_5= happiness2015.head(5)
first_5.info()

## 3. Using Loops to Aggregate Data ##

import pandas as pd

# Assuming you have a DataFrame named df with a 'Region' column and a 'Happiness Score' column
# For example:
# df = pd.DataFrame({'Region': ['A', 'A', 'B', 'B', 'C', 'C'],
#                    'Happiness Score': [5, 6, 7, 8, 9, 10]})

# Create an empty dictionary to store mean happiness scores
mean_happiness = {}

# Get unique values from the 'Region' column
unique_regions = happiness2015['Region'].unique()

# Iterate over unique regions
for region in unique_regions:
    # Assign rows belonging to the current region to a variable
    region_group = happiness2015[happiness2015['Region'] == region]
    
    # Calculate the mean happiness score for the current region
    mean_score = region_group['Happiness Score'].mean()
    
    # Assign the mean value to the mean_happiness dictionary
    mean_happiness[region] = mean_score

# Print or use the mean_happiness dictionary as needed
print(mean_happiness)

## 5. Creating GroupBy Objects ##

grouped = happiness2015.groupby('Region')
aus_nz =  grouped.get_group('Australia and New Zealand')

## 6. Exploring GroupBy Objects ##

grouped = happiness2015.groupby('Region')


north_america = happiness2015.iloc[[4,14]]
na_group = grouped.get_group('North America')
equal = north_america == na_group

## 7. Common Aggregation Methods with Groupby ##

grouped = happiness2015.groupby('Region')
means = grouped.mean()

## 8. Aggregating Specific Columns with Groupby ##

grouped = happiness2015.groupby('Region')
grouped = happiness2015.groupby('Region')


happy_grouped = grouped['Happiness Score']
happy_mean = happy_grouped.mean()

## 9. Introduction to the Agg() Method ##

import numpy as np
grouped= happiness2015.groupby('Region')
happy_grouped = grouped['Happiness Score']
def dif(group):
    return (group.max() - group.mean())
happy_mean_max = happy_grouped.agg([np.mean, np.max])
mean_max_dif = happy_grouped.agg(dif)

## 11. Aggregation with Pivot Tables ##

pv_happiness = happiness2015.pivot_table(values='Happiness Score', index='Region', aggfunc=np.mean, margins=True)
pv_happiness.plot(kind='barh', xlim=(0,10), title='Mean Happiness Scores by Region', legend=False)
plt.show()
world_mean_happiness = happiness2015['Happiness Score'].mean()

## 12. Aggregating Multiple Columns and Functions with Pivot Tables ##

grouped_by_region = happiness2015.groupby('Region')
grouped = grouped_by_region['Happiness Score','Family']
happy_family_stats = grouped.agg([np.min, np.max, np.mean])
pv_happy_family_stats = happiness2015.pivot_table(['Happiness Score', 'Family'], 'Region', aggfunc=[np.min, np.max, np.mean], margins=True)