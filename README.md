# Earthquake Predictor
## Annie Rumbles
[DrivenData competition](https://www.drivendata.org/competitions/57/nepal-earthquake/page/134/): ***"Based on aspects of building location and construction, your goal is to predict the level of damage to buildings caused by the 2015 Gorkha earthquake in Nepal."***

### Table of Contents
- [The Data](##the-data)
- [EDA](##exploratory-data-analysis)
- [Principal Component Analysis](###principal-component-analysis)
- [Modelling](##modelling)
- [Results](##results)
- [Analysis](##analysis)
- [Next Steps](##next-steps)
- [Citations](##citations)
 
## The Data
The data was collected through surveys by Kathmandu Living Labs and the Central Bureau of Statistics, which works under the National Planning Commission Secretariat of Nepal. This survey is one of the largest post-disaster datasets ever collected, containing valuable information on earthquake impacts, household conditions, and socio-economic-demographic statistics<sup>1</sup>.

## Exploratory Data Analysis

Imbalanced classes:

![](images/classbalances.png)

![](images/scaled_structure_counts.png)

![](images/structure_types_damage_counts.png)

### Principal Component Analysis

![](images/pca_best.gif)

## Modelling

Features included: `'age', 'area_percentage', 'height_percentage', 'land_surface_condition',
                           'foundation_type', 'position', 'has_superstructure_adobe_mud', 
                           'has_superstructure_mud_mortar_stone', 'has_superstructure_stone_flag',
                           'has_superstructure_cement_mortar_stone', 'has_superstructure_mud_mortar_brick',
                           'has_superstructure_cement_mortar_brick', 'has_superstructure_timber',
                           'has_superstructure_bamboo', 'has_superstructure_rc_non_engineered',
                            'has_superstructure_rc_engineered', 'has_superstructure_other', 'damage_grade'`

### Random Forest

My first random forest included: n_estimators=100, max_depth=4, n_jobs=-2, random_state=9, class_weight='balanced'

## Results

|  Model  |  Micro Averaged F1-Score  |
|---------|---------------------------|
|   RF    |            .40            |

## Analysis

## Next Steps

## Citations
<sup>1</sup>[Richter's Predictor: Modeling Earthquake Damage](https://www.drivendata.org/competitions/57/nepal-earthquake/page/134/)
