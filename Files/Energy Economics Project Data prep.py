import os
import pandas as pd
import matplotlib.pyplot as plt
import plotly_express as px
import plotly.offline as pyo


directory = "/Users/buzunonat/Desktop/Weather data"
os.chdir(directory)
gathered_data=pd.DataFrame()
gathered_data=pd.read_csv("./Delaware.csv",skiprows=3)[["Year","Month"]]

states=pd.read_excel("/Users/buzunonat/Desktop/ratio renew final data.xlsx",sheet_name="Hard Copy")
nuclear=pd.read_excel("/Users/buzunonat/Desktop/nuclear capacity.xlsx")
nuclear_production=pd.read_excel("/Users/buzunonat/Desktop/nuclear power production.xlsx")

residential_energy=pd.read_excel("/Users/buzunonat/Desktop/industrial share.xlsx",sheet_name="Residential Sector")
commertial_energy=pd.read_excel("/Users/buzunonat/Desktop/industrial share.xlsx",sheet_name="Commercial Sector")
industrial_energy=pd.read_excel("/Users/buzunonat/Desktop/industrial share.xlsx",sheet_name="Industrial Sector")
transport_energy=pd.read_excel("/Users/buzunonat/Desktop/industrial share.xlsx",sheet_name="Transportation Sector")
total_energy=pd.read_excel("/Users/buzunonat/Desktop/industrial share.xlsx",sheet_name="Total Consumption")


#####For Temperature Variance create a dataset from individual state data
for root,dirs,files in os.walk(directory):
    for file in files:
       if file.endswith(".csv"):
           print(file)
           a=pd.read_csv(file,skiprows=3)
           gathered_data[file]=a.Value
#group
temp_var=gathered_data.groupby("Year").var().reset_index()
temp_var.columns=temp_var.columns.str.replace('.csv', '')



del temp_var["Month"]
#shape adjustment
temp_var=temp_var.set_index("Year")
temp_var=temp_var.T
temp_var=temp_var.reset_index()

temp_var = temp_var.rename(columns={'index': 'States'})

state_names_to_index=nuclear["U.S. State"]
#matching
temp=[]
for i in range(49):
    temp.append(temp_var.States[i] in (state_names_to_index).tolist())

temperature_var=temp_var[temp]

#melt
melted_temp=pd.melt(temperature_var, id_vars =['States'], value_vars =temperature_var.columns)
# index for 2002-2022
melted_temp=melted_temp[(melted_temp.Year>2002)&(melted_temp.Year<2022)]


###### For the ratio of renewables
temp=[]
for i in range(50):
    temp.append(states.States[i] in (state_names_to_index).tolist())

states=states[temp]


ratio_of_renew=pd.melt(states, id_vars =['States'], value_vars =states.columns)
ratio_of_renew=ratio_of_renew[ratio_of_renew.variable>2002]

###### For the nuclear capacity data
final_nuclear=pd.melt(nuclear, id_vars =['U.S. State'], value_vars =nuclear.columns)
final_nuclear=final_nuclear[final_nuclear.variable<2022]

### merging the datas
final_nuclear.columns=['States', 'Year', 'Nuclear']
ratio_of_renew.columns=['States', 'Year', 'Renew_ratio']
melted_temp.columns=['States', 'Year', 'Temp_var']
merged_df = pd.merge(final_nuclear, ratio_of_renew, on=['States', 'Year'], how='inner')
merged_df = pd.merge(merged_df, melted_temp, on=['States', 'Year'], how='inner')
merged_df["Nuclear"] /=1000
merged_df["Renew_ratio"]*=100
#### merged_df is the final one needed for the regressions
# To see the regressions and the model refer to the R file

###############


##### synthetic controls data generation & merging

gdp_state= pd.read_csv("/Users/buzunonat/Desktop/syntetic control/SAGDP1__ALL_AREAS_1997_2022.csv")

gdp_state=gdp_state[gdp_state.Description=="Real GDP (millions of chained 2012 dollars)  "]



drob = ['GeoFIPS', 'Region', 'TableName', 'LineCode', 'IndustryClassification', 'Description', 'Unit']
gdp_state.drop(columns=drob, inplace=True)
#copying
gdp_state2=gdp_state.copy(deep=True)
gdp_state2.set_index('GeoName', inplace=True)
gdp_state2["2022"]=gdp_state2["2022"].astype("float64")
gdp_state3=(gdp_state2.T.diff()*100/gdp_state2.T).T

gdp_state3=gdp_state3.reset_index()

# Repeating the same stuff for similar dataets
temp_state=[]

for i in range(len((gdp_state3["GeoName"].to_list()))):
    temp_state.append(gdp_state3["GeoName"].to_list()[i] in (state_names_to_index).tolist())

gdp_state3=gdp_state3[temp_state]

gdp_melt2=pd.melt(gdp_state3, id_vars =['GeoName'], value_vars =gdp_state3.columns)


gdp_melt2["variable"]=gdp_melt2["variable"].astype("int")

gdp_melt2=gdp_melt2.query("variable<2022&variable>2002")
gdp_melt2.columns=["States","Year","GDP"]










# same process
temp_state=[]

for i in range(len((gdp_state["GeoName"].to_list()))):
    temp_state.append(gdp_state["GeoName"].to_list()[i] in (state_names_to_index).tolist())

gdp_state=gdp_state[temp_state]

gdp_melt=pd.melt(gdp_state, id_vars =['GeoName'], value_vars =gdp_state3.columns)


gdp_melt["variable"]=gdp_melt["variable"].astype("int")

gdp_melt=gdp_melt.query("variable<2022&variable>2002")
gdp_melt.columns=["States","Year","GDP"]


emission=pd.read_excel("/Users/buzunonat/Desktop/syntetic control/table7.xlsx",skiprows=5)

temp_state=[]

for i in range(len((emission["State"].to_list()))):
    temp_state.append(emission["State"].to_list()[i] in (state_names_to_index).tolist())


emission=emission[temp_state]
emission.drop(columns=["Absolute","Absolute.1","Percent","Percent.1"], inplace=True)

emission_melt=pd.melt(emission, id_vars =['State'], value_vars =emission.columns)

emission_melt=emission_melt.query("variable<2022&variable>2002")

emission_melt.columns=['States', 'Year', 'Emissions']
emission_melt.reset_index(inplace=True)
gdp_melt.reset_index(inplace=True)
emission_melt.drop(columns="index",inplace=True)
gdp_melt.drop(columns="index",inplace=True)


latitude=pd.read_clipboard() # go to this link and copy it to your clipboard.
#https://developers.google.com/public-data/docs/canonical/states_csv
# Pandas will read it (couldnt download it)
latitude=latitude[["name","latitude","longitude"]]

temp_state=[]

for i in range(len((latitude["name"].to_list()))):
    temp_state.append(latitude["name"].to_list()[i] in (state_names_to_index).tolist())


latitude=latitude[temp_state]

merged_df["lat"]=0.00
merged_df["long"]=0.00

for name in merged_df["States"]:
    merged_df.loc[merged_df["States"] == name,"lat"] =latitude[latitude.name==name]["latitude"].values[0]
    merged_df.loc[merged_df["States"] == name,"long"] = latitude[latitude.name == name]["longitude"].values[0]





#################
#  Very manual work after this point ###
#################

merged_df = pd.merge(merged_df, emission_melt, on=['States', 'Year'], how='inner')
merged_df = pd.merge(merged_df, gdp_melt, on=['States', 'Year'], how='inner')
merged_df = pd.merge(merged_df, gdp_melt2, on=['States', 'Year'], how='inner')
os.chdir("/Users/buzunonat/Desktop/syntetic control")
merged_df.to_csv("data_for_synthetic.csv")
####### additional variables like differences etc
merged_df["diff_nuclear"]=merged_df.groupby("States")["Nuclear"].diff()
merged_df["diff_renew"]=merged_df.groupby("States")["Renew_ratio"].diff()



##### nuclear power production
production_melt=pd.melt(nuclear_production, id_vars =['States'], value_vars =nuclear_production.columns)
production_melt["value"]/=10000000
production_melt.columns=["States","Year","Production"]
merged_df=pd.merge(merged_df,production_melt,on=["States","Year"],how="inner")
# ready for going to R and create models!


#### Adding coastal states as dummies

coastal_states=["Washington","California","Arizona","Texas","Louisiana","Alabama","Mississippi",
                 "Florida","Georgia","South Carolina","North Carolina","Virginia","New York","New Jersey",
                 "Connecticut","New Hampshire","Massachusetts"]

merged_df["Coastal"]=0
for name in coastal_states:
        merged_df.loc[merged_df.States==name,"Coastal"] = 1


# a lot of manual work bcs of the dataset issues
election2003=pd.read_excel("/Users/buzunonat/Desktop/President 2000-2020.xlsx",sheet_name="2000")
election2003=election2003[["STATE","Diff, Rep"]]


election2004=pd.read_excel("/Users/buzunonat/Desktop/President 2000-2020.xlsx",sheet_name="2004")
election2004=election2004[["STATE","Unnamed: 9"]]
election2004.columns=["STATE","Diff, Rep"]


election2008=pd.read_excel("/Users/buzunonat/Desktop/President 2000-2020.xlsx",sheet_name="2008")
election2008=election2008[["STATE","Unnamed: 9"]]
election2008.columns=["STATE","Diff, Rep"]

election2012=pd.read_excel("/Users/buzunonat/Desktop/President 2000-2020.xlsx",sheet_name="2012")
election2012=election2012[["STATE","Unnamed: 9"]]
election2012.columns=["STATE","Diff, Rep"]

election2016=pd.read_excel("/Users/buzunonat/Desktop/President 2000-2020.xlsx",sheet_name="2016")
election2016=election2016[["STATE","Unnamed: 11"]]
election2016.columns=["STATE","Diff, Rep"]
election2016.nunique()

election2020=pd.read_excel("/Users/buzunonat/Desktop/President 2000-2020.xlsx",sheet_name="2020")
election2020=election2020[["STATE","Unnamed: 11"]]
election2020.columns=["STATE","Diff, Rep"]
election2020.dropna(inplace=True)


# Deep copying bcs editing the years
election2005=election2004.copy(deep=True)
election2006=election2004.copy(deep=True)
election2007=election2004.copy(deep=True)
election2009=election2008.copy(deep=True)
election2010=election2008.copy(deep=True)
election2011=election2008.copy(deep=True)
election2013=election2012.copy(deep=True)
election2014=election2012.copy(deep=True)
election2015=election2012.copy(deep=True)
election2017=election2016.copy(deep=True)
election2018=election2016.copy(deep=True)
election2019=election2016.copy(deep=True)
election2021=election2020.copy(deep=True)

election2003["Year"]=2003
election2004["Year"]=2004
election2005["Year"]=2005
election2006["Year"]=2006
election2007["Year"]=2007
election2008["Year"]=2008
election2009["Year"]=2009
election2010["Year"]=2010
election2011["Year"]=2011
election2012["Year"]=2012
election2013["Year"]=2013
election2014["Year"]=2014
election2015["Year"]=2015
election2016["Year"]=2016
election2017["Year"]=2017
election2021["Year"]=2021
election2018["Year"]=2018
election2019["Year"]=2019
election2020["Year"]=2020

raw_election=pd.concat([election2003,
               election2004,
               election2005,
               election2006,
               election2007,
               election2008,
               election2009,
               election2010,
               election2011,
               election2012,
               election2013,
               election2014,
               election2015,
               election2016,
               election2017,
               election2018,
               election2019,
               election2020,
               election2021]).reset_index().drop(columns="index")


temp_elec=[]

for i in range(len((raw_election["STATE"].to_list()))):
    temp_elec.append(raw_election["STATE"].to_list()[i] in (state_names_to_index).tolist())

election_final=raw_election[temp_elec]
election_final.columns=["States","Vote_Diff","Year"]
merged_df=pd.merge(merged_df,election_final,on=["States","Year"],how="inner")



#Sectoral ratios of energy usage
residential_energy.drop(columns="State",inplace=True)
commertial_energy.drop(columns="State",inplace=True)
industrial_energy.drop(columns="State",inplace=True)
transport_energy.drop(columns="State",inplace=True)
total_energy.drop(columns="State",inplace=True)

temp_elec1=[]
temp_elec2=[]
temp_elec3=[]
temp_elec4=[]
temp_elec5=[]

for i in range(len((residential_energy["States"].to_list()))):
    temp_elec1.append(residential_energy["States"].to_list()[i] in (state_names_to_index).tolist())
    temp_elec2.append(commertial_energy["States"].to_list()[i] in (state_names_to_index).tolist())
    temp_elec3.append(industrial_energy["States"].to_list()[i] in (state_names_to_index).tolist())
    temp_elec4.append(transport_energy["States"].to_list()[i] in (state_names_to_index).tolist())
    temp_elec5.append(total_energy["States"].to_list()[i] in (state_names_to_index).tolist())


residential_energy=residential_energy[temp_elec1]
commertial_energy=commertial_energy[temp_elec2]
industrial_energy=industrial_energy[temp_elec3]
transport_energy=transport_energy[temp_elec4]
total_energy=total_energy[temp_elec5]

residential_melt=pd.melt(residential_energy, id_vars =['States'], value_vars =residential_energy.columns)
commertial_melt=pd.melt(commertial_energy, id_vars =['States'], value_vars =commertial_energy.columns)
industrial_melt=pd.melt(industrial_energy, id_vars =['States'], value_vars =industrial_energy.columns)
transport_melt=pd.melt(transport_energy, id_vars =['States'], value_vars =transport_energy.columns)
total_melt=pd.melt(total_energy, id_vars =['States'], value_vars =total_energy.columns)

residential_melt.columns=["States","Year","R_Energy"]
commertial_melt.columns=["States","Year","C_Energy"]
industrial_melt.columns=["States","Year","I_Energy"]
transport_melt.columns=["States","Year","Tr_Energy"]
total_melt.columns=["States","Year","T_Energy"]


residential_melt=residential_melt.query("Year<2022&Year>2002")
commertial_melt=commertial_melt.query("Year<2022&Year>2002")
industrial_melt=industrial_melt.query("Year<2022&Year>2002")
transport_melt=transport_melt.query("Year<2022&Year>2002")
total_melt=total_melt.query("Year<2022&Year>2002")


residential_melt.reset_index(drop=True,inplace=True)
commertial_melt.reset_index(drop=True,inplace=True)
industrial_melt.reset_index(drop=True,inplace=True)
transport_melt.reset_index(drop=True,inplace=True)
total_melt.reset_index(drop=True,inplace=True)


merged_df=pd.merge(merged_df,residential_melt,on=["States","Year"],how="inner")
merged_df=pd.merge(merged_df,commertial_melt,on=["States","Year"],how="inner")
merged_df=pd.merge(merged_df,industrial_melt,on=["States","Year"],how="inner")
merged_df=pd.merge(merged_df,transport_melt,on=["States","Year"],how="inner")
merged_df=pd.merge(merged_df,total_melt,on=["States","Year"],how="inner")

merged_df["Residential_ratio"]=merged_df["R_Energy"]/merged_df["T_Energy"]*100
merged_df["Commertial_ratio"]=merged_df["C_Energy"]/merged_df["T_Energy"]*100
merged_df["Industrial_ratio"]=merged_df["I_Energy"]/merged_df["T_Energy"]*100
merged_df["Transport_ratio"]=merged_df["Tr_Energy"]/merged_df["T_Energy"]*100



os.chdir("/Users/buzunonat/Desktop/syntetic control")
merged_df.to_csv("data_for_synthetic.csv")




##### 3d plots


#if you dont want to run all the things above just load it from drive
merged_df = pd.read_csv("/Users/buzunonat/Desktop/syntetic control/data_for_synthetic.csv")

plt.figure(figsize=(20, 20))
fig=px.scatter_3d(x=merged_df.Year, y=merged_df.Nuclear,
                  z=merged_df.Renew_ratio,color=merged_df.States)
fig.update_traces(marker_size = 5)
fig.update_traces(mode='markers+lines')
fig.update_layout(scene = dict(
                    xaxis_title='Year',
                    yaxis_title='Nuclear Capacity',
                    zaxis_title='Renewable Ratio'))

fig.write_html("last_plot.html")



#plot with differenced values

plt.figure(figsize=(20, 20))
fig=px.scatter_3d(x=merged_df.Year, y=merged_df.diff_nuclear,
                  z=merged_df.Renew_ratio,color=merged_df.States)
fig.update_traces(marker_size = 5)
fig.update_traces(mode='markers+lines')
fig.update_layout(scene = dict(
                    xaxis_title='Year',
                    yaxis_title='Nuclear Capacity',
                    zaxis_title='Renewable Ratio'))

fig.write_html("last_plot2.html")

