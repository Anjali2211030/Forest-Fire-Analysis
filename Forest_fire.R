#Reading CSV files
forest_fire <- read.csv("D:/3rd Semester/Algerian_data_2.csv")
forest_fire

#Calculating number of columns and rows
nrow(forest_fire)
ncol(forest_fire)

#Installing ggplot
install.packages("ggplot2")
library(ggplot2)

#Preparation of scatter plot
ggplot(data = forest_fire) +
  geom_point(mapping = aes(x = Temperature  , y = RH, color = Classes))+
  geom_smooth(mapping = aes(x = Temperature  , y = RH))
  labs(title = "Rainfall Data",
    x = "Monthly",
    y = "Rainfall") +
  theme_minimal()

ggplot(data = forest_fire) +
  geom_point(mapping = aes(x = Temperature, y = FFMC, colour = Classes)) +
  geom_smooth(mapping = aes(x = Temperature  , y = FFMC))
  labs(title = "Effect of Temperature on FFMC" , 
       x = "Temperature" , 
       y = "FFMC")
  
ggplot(data = forest_fire) +
    geom_point(mapping = aes(x = Ws, y = ISI, colour = Classes)) +
    geom_smooth(mapping = aes(x = Ws  , y = ISI))
    labs(title = "Effect of Wind Speed on ISI" , 
       x = "Ws" , 
       y = "ISI")

ggplot(data = forest_fire) +
  geom_point(mapping = aes(x = RH, y = FWI, colour = Classes)) +
  geom_smooth(mapping = aes(x = RH  , y = FWI))
  labs(title = "Effect of Relative Humidity on FWI" , 
         x = "RH" , 
         y = "FWI")

#Preparation of Bar Graph
ggplot(data = forest_fire, aes(x = Month)) +
  geom_bar() +
  labs(title = "Frequency of Fires by Month",
       x = "Month",
       y = "Frequency") +
  theme_minimal()

ggplot(data = forest_fire, aes(x = Month)) +
  stat_count() +
  labs(title = "Frequency of Fires by Month",
       x = "Month",
       y = "Frequency") +
  theme_minimal()

ggplot(data = forest_fire) +
  geom_bar(mapping = aes(x = Month, y = Temperature), stat = "identity")

#abline
ggplot(data = forest_fire, mapping = aes(x = Ws, y = ISI)) +
  geom_point() + 
  geom_abline() +
  coord_fixed()

#Position Adjustment
ggplot(data = forest_fire) + 
  geom_bar(mapping = aes(x = Temperature, colour = Classes))
ggplot(data = forest_fire) + 
  geom_bar(mapping = aes(x = Temperature, fill = Classes))

#Coordinate System
ggplot(data = forest_fire, mapping = aes(x = Temperature, y = RH , fill = Classes)) + 
  geom_boxplot()
ggplot(data = forest_fire, mapping = aes(x = Temperature, y = RH , fill = Classes)) + 
  geom_boxplot() +
  coord_flip()

#Facets
ggplot(data = forest_fire) + 
  geom_point(mapping = aes(x = RH, y = FWI)) + 
  facet_wrap(~ Classes, nrow = 2)

ggplot(forest_fire) + 
  geom_bar(mapping = aes(x = Temperature))

ggplot(data = forest_fire) +
    geom_count(mapping = aes(x = Day, y = FWI))  

#Co variation
install.packages("corrplot")
library(coorplot)
forest_fire <- read.csv("D:/3rd Semester/Algerian_data_2.csv", header = TRUE, sep = ",")
forest_fire
names(forest_fire)
str(forest_fire)
selected_vars <- forest_fire[, c("Temperature", "RH", "Ws", "Month" , "Day")]
cor_matrix <- cor(selected_vars, use = "pairwise.complete.obs")
install.packages("corrplot")
library(corrplot)
corrplot(cor_matrix, method = "color", type = "upper", tl.cex = 0.7)
install.packages("ggplot2")
library(ggplot2)
correaltion_table <- as.data.frame(cor_matrix)
print(cor_matrix)

#QQ-Plot
library(ggplot2)
# Replace 'my_data' with your data set
forest_fire <- c(71,73,80,64,60,54,44,51,59,41,42,58,52,79,90,87,69,62,67,72,55,46,59,68,70,62,55,37,36,42,58,48,56,58,45,42,43,47,43,51,56,44,45,37,45,83,81,68,58,50,29,48,71,63,64,58,87,57,59,56,55,52,34,33,35,42,54,63,56,3,39,31,21,34,40,46,41,24,37,66,81,71,53,43,38,40,37,54,56,53,49,59,86,67,75,66,58,71,62,88,80,74,73,72,49,81,51,26,44,33,41,58,34,64,56,49,70,65,87,87,54,64)
# Create a QQ-plot
qqnorm(forest_fire, main = "QQ-Plot for Forest Fire", xlab = "Theoretical Quantiles", ylab = "Temperature")
qqline(forest_fire)
# Add a legend (optional)
legend("bottomright", legend = "Reference Line", lty = 1, col = "black")

#Time Series
install.packages("ggplot2")
library(ggplot2) 
data<- read.csv("D:/3rd Semester/Algerian_data_2.csv")
years = 2012
ggplot(data , aes(x=as.numeric(FFMC), y=Rain))+
geom_line()+
ylab('Rain')+
xlab('FFMC')+
theme_bw()

summary(forest_fire)
