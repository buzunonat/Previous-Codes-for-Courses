library(readxl)
library(stargazer)
library(ggplot2)
library(dplyr)
library(openxlsx)
library(miceadds)
library(dplyr)
library(tidysynth)
library(readxl)
library(readr)



df= read_csv("/Users/buzunonat/Desktop/syntetic control/data_for_synthetic.csv")
df[1]=NULL

# plotting the correlation
corrplot::corrplot(cor(
  df[c("Nuclear","Renew_ratio","Temp_var","lat","long","Emissions",
       "GDP_y","Production","Coastal","Vote_Diff","Residential_ratio",
       "Industrial_ratio","Transport_ratio")]),
  type = "upper",method = "square",)


# creating a dummy for the votes

df$Vote_dummy<-ifelse(df$Vote_Diff>=0,"Republican","Democrat")
# assigning a neutral level 
df$Vote_dummy[(df$Vote_Diff<5 & df$Vote_Diff>-5)] <- "Neutral"




### -110 and -90 longitude
df$long_dummy=ifelse(df$long>-90,"East","West")
df$long_dummy[(df$long > (-110) & df$long < (-90))]<-"Center"

### latitude
df$lat_dummy=ifelse(df$lat>40,"Northern","Southern")


### shale fixed effects

#df$shale_FE<- ifelse(df$States %in% c("Michigan","Texas","Ohio","Alabama",
#                                      "Arkansas","Louisiana","Illinois",
#                                      "New York","Iowa"),"Shale","No Shale")
#

#regression for basic analysis
model1=lm(Renew_ratio~Nuclear+Temp_var+GDP_y+C(Year),data=df)
model2=lm(Renew_ratio~Nuclear+Temp_var+GDP_y+C(Vote_dummy)+C(Year),data=df)
model3=lm(Renew_ratio~Nuclear+Temp_var+GDP_y+C(lat_dummy)+C(long_dummy)+C(Year),data=df)
model4=lm(Renew_ratio~Nuclear+Temp_var+GDP_y+Transport_ratio+C(lat_dummy)+C(long_dummy)+C(Year),data=df)
# model with regular regression output
stargazer(model1,model2,model3,model4,type="latex",keep = c("Nuclear","Temp_var","GDP_y","shale_FE","Transport_ratio",
                                                            "Vote_dummy","lat_dummy","long_dummy","Constant"),
          font.size = "small",no.space = T,column.sep.width = "-5pt",omit.stat = c( "ser"))



#some visualisations
ggplot(df[df$States==c("Texas","California"),],aes(x=Year,y=T_Energy,col=States))+geom_point()+geom_line()
ggplot(df,aes(x=Year,y=GDP_x,col=States))+geom_point()+geom_line()

ggplot(df,aes(x=Nuclear,y=Renew_ratio,col=States))+geom_point()+
  geom_smooth(method = "lm")+ylab("ratio of renewable")


ggplot(df,aes(x=df$Nuclear,y=df$Renew_ratio))+geom_point()+
  geom_smooth(method = model1)+ylab("ratio of renewable")


########## synthetic controls


#states with big changes 5% (change this for every state)
#c("California","Iowa","Massachusetts","Nebraska",
#  "New Jersey","New York","Pennsylvania","Vermont",
#  "Wisconsin","Tennessee")


#states with big changes 10% (change this for every state)
#c("California","Iowa","Massachusetts","Nebraska",
#  "New Jersey","New York","Vermont","Mississippi",
#  "Wisconsin","Tennessee")



big_changed_states=c("California","Iowa","Massachusetts","Nebraska",
                       "New Jersey","New York","Vermont","Mississippi",
                       "Wisconsin","Tennessee")

df = subset(df, !(States%in% big_changed_states) )





synth_func<- function(intervention_year,state_of_interest){
synth<-
  
  df %>%
  
  # initial the synthetic control object
  synthetic_control(outcome = Renew_ratio, # outcome
                    unit = States, # unit index
                    time = Year, # time index 
                    i_unit = state_of_interest, # unit of interest
                    i_time = intervention_year, # time period when the intervention occurred
                    generate_placebos=F # no
  ) %>%
  
  # Generate the aggregate predictors
  generate_predictor(time_window = 2003:intervention_year, #date
                     # variables in the syntetic control creation
                    "Temperature Variance" = mean(Temp_var, na.rm = T), 
                     Emissions = mean(Emissions, na.rm = T),
                     GDP = mean(GDP_x, na.rm = T),
                     "Latitude"=mean(lat),
                     "Longitude"=mean(long),
                     "Vote Differential"=mean(Vote_Diff),
                     "Coastal State"=mean(Coastal),
                     "Residential Sector" =mean(Residential_ratio),
                     "Industrial Sector"=mean(Industrial_ratio),
                     #"Transport Sector"=mean(Transport_ratio), 
                     # excluding this one bcs its
                    #the complement of the other sectoral variables
                     "Commercial Sector"=mean(Commertial_ratio)
  ) %>%
  
  # average nuclear capacity from 2003 till intervention
  generate_predictor(time_window = 2003:intervention_year,
                     Nuclear = mean(Nuclear, na.rm = T)) %>%
  
  
  # Generate the fitted weights for the synthetic control
  generate_weights(optimization_window = 2003:intervention_year, # time to use in the optimization task
                   margin_ipop = .05,sigf_ipop = 5,bound_ipop = 7 # optimizer options
  ) %>%
  
  # Generate the synthetic control
  generate_control()
}



#FOR CALIFORNIA
df= read_csv("/Users/buzunonat/Desktop/syntetic control/data_for_synthetic.csv")
df[1]=NULL
# This Filter Is FOR CALİFORNİA
# excluding states that are neighboring or too closely related
big_changed_states=c("Iowa","Massachusetts","Nebraska",
                     "New Jersey","New York","Vermont","Mississippi",
                     "Wisconsin","Tennessee")
df = subset(df, !(States%in% big_changed_states) )
# also arizona
df = subset(df, !(States%in%c("Arizona")) )

california_synth=synth_func(2013,"California")

california_synth %>% plot_trends()

california_synth %>% plot_weights()

california_synth %>% plot_differences()





# FOR WİSCONSİN
df= read_csv("/Users/buzunonat/Desktop/syntetic control/data_for_synthetic.csv")
df[1]=NULL


big_changed_states=c("California","Iowa","Massachusetts","Nebraska",
                     "New Jersey","New York","Vermont","Mississippi",
                     "Tennessee")
df = subset(df, !(States%in% big_changed_states) )
df = subset(df, !(States%in%c("Minnesota","Michigan")) )

wisc_synth=synth_func(2013,"Wisconsin")

wisc_synth %>% plot_trends()
wisc_synth %>% plot_weights()
wisc_synth %>% plot_differences()
# FOR TENESSE
df= read_csv("/Users/buzunonat/Desktop/syntetic control/data_for_synthetic.csv")
df[1]=NULL



big_changed_states=c("California","Iowa","Massachusetts","Nebraska",
                     "New Jersey","New York","Vermont","Mississippi",
                     "Wisconsin")
df = subset(df, !(States%in% big_changed_states) )
df = subset(df, !(States%in%c("Alabama","Missouri","Georgia","Arkansas",
                              "North Carolina","South Carolina","Virginia") ))



tennesse_synth=synth_func(2015,"Tennessee")

tennesse_synth %>% plot_trends()
tennesse_synth %>% plot_weights()
tennesse_synth %>% plot_differences()















