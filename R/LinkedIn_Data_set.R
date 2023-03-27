library(tidyverse)
library(dplyr)
library(ggplot2) # for visualization
library(metan) # for correlation matrix
library(caret) # for machine learning
library(e1071) # for machine learning
library(car) # for Levene's test
library(e1071)
library(caTools) # for logistic regression

# Import the data set
data <- read_csv("LinkedIn Profile Data.csv")

# Missing values
sapply(data, function(x) sum(is.na(x)))

# Some variables are not useful and some for ethical reasons need to be dropped.

data <- data %>%
  select(-m_urn, -m_urn_id, -beauty, -beauty_female, 
         -beauty_male, -blur, -ethnicity, -skin_acne, 
         -skin_dark_circle, -skin_health, -skin_stain,
         -nationality, -african, -celtic_english, - east_asian,
         -european, -greek, -hispanic, -jewish, -muslim,
         -nordic, -south_asian)

# Data Visualization------------------------------------------------------------

# Bar chart: Variable glasses

by_glass <- data %>%
  group_by(glass) %>%
  summarise(c_id = n())

ggplot(by_glass,aes(x=reorder(glass,desc(c_id)), y = c_id))+
  geom_bar(stat="identity", fill="purple") +
  labs(x="Glasses",title="Profile picture with sunglasses, glasses or none") +
  theme(axis.text.x=element_text(angle=90,hjust=1,vjust=0.5))

# Bar chart: Variable no_of_promotions

by_promotion <- data %>%
  group_by(no_of_promotions) %>%
  summarise(c_id = n())

ggplot(by_promotion,aes(x=reorder(no_of_promotions,desc(c_id)), y = c_id))+
  geom_bar(stat="identity", fill="red") +
  labs(x="Promotions",title="Number of promotions") +
  theme(axis.text.x=element_text(angle=90,hjust=1,vjust=0.5))

# Bar chart: Variable no_of_previous_positions

by_previous_position <- data %>%
  group_by(no_of_previous_positions) %>%
  summarise(c_id = n()) %>%
  filter(no_of_previous_positions < 25)

ggplot(by_previous_position,aes(x=reorder(no_of_previous_positions,desc(c_id)), y = c_id))+
  geom_bar(stat="identity", fill="gold") + 
  labs(x="Previous position",title="Number of previous positions") +
  theme(axis.text.x=element_text(angle=90,hjust=1,vjust=0.5))

table(by_previous_position)

# Pie chart for gender

gender <- data %>%
  mutate(total = n()) %>%
  group_by(gender, total) %>%
  summarise(c_id = n(),
            pct = c_id / total) %>%
  distinct(gender, .keep_all = T)

ggplot(gender, aes(x="", y=pct, fill=gender)) +
  geom_bar(stat="identity", width=1) +
  coord_polar("y", start=0)

# Histogram for the variable age
# According to LinkedIn official site the age should be 16 and above.
df_age <- data %>%
  filter(age >= 16)

ggplot(df_age, aes(x=age, color = 'red')) + geom_histogram(binwidth = 1) 

# Box plot for age

ggplot(df_age,
       aes(y = age)) +
  geom_boxplot(color="tomato", fill="blue") +
  labs(title = "")

summary(df_age$age)

# Scatter plot for the variables emo_fear & emo_disgust

df_emot <- data %>%
  filter(emo_fear <= 100.00,
         emp_disgust <= 100.00)

ggplot(df_emot, aes(x=emo_fear, y=emp_disgust, color = 'red')) + geom_point(alpha = 50/100)

# correlation matrix heat map

corrl <- corr_coef(data)
plot(corrl)

# Descriptive statistics--------------------------------------------------------

# Data information about female and male accounts

females <- filter(data, gender=="Female")
males <- filter(data, gender=="Male")
paste("Number of female Profiles")
nrow(females)
paste("Number of male Profiles")
nrow(males)

# Who has the most followers, the females or the males?
paste("Mean followers for female accounts")
mean(females$n_followers)

paste("Mean followers for male accounts")
mean(males$n_followers)

# Who is smiling more on their profile pictures, females or males?
paste("Mean smile female accounts")
mean(females$smile)

paste("Mean smile male accounts")
mean(males$smile)

# The average age of each gender (include and the fake profiles,
# which are those below the age 15)
paste("Mean age for females")
mean(females$age)

paste("Mean age for males")
mean(males$age)

# standard deviation for each gender
summarise(females, sd=sd(age))
summarise(males, sd=sd(age))

par(mfrow = c(1,2))
hist(females$age, breaks = 40, 
     xlab="Mean age for the females", 
     main="Females", col = "purple")
hist(males$age, breaks = 40, 
     xlab="Mean age for the males", 
     main="males", col = "blue")

# min, max, mean, median, 1st and 3rd quartile for each gender
summary(females$age)

summary(males$age)

# T-test gender & age

# Null hypothesis: there is no difference in age mean between the  
# female and male. 
# Alternative hypothesis: there is a difference in age mean between the 
# female and male.

t.test(age ~ gender, data=data, var.equal = TRUE, paired = FALSE)

# (p < 0.5) is statistically significant
# p-value < 2.2e-16 (we reject the null hypothesis)
# t = 16.513

# Levene's test
# Null hypothesis: population variances are equal

leveneTest(data$age ~ data$gender, center=mean)
leveneTest(data$age ~ data$gender, center=median)

# p-value < 2.2e-16 (we reject the null hypothesis)

# Mean for the emoticon variables-----------------------------------------------

# The percentages of the emoticon that is used by the users

df_emo_anger <- data %>%
  filter(emo_anger <= 100)

mean(df_emo_anger$emo_anger)

df_emp_disgust <- data %>%
  filter(emp_disgust <= 100)

mean(df_emp_disgust$emp_disgust)

df_emo_fear <- data %>%
  filter(emo_fear <= 100)

mean(df_emo_fear$emo_fear)

mean(data$emo_happiness)

mean(data$emo_neutral)

mean(data$emo_sadness)

mean(data$emo_surprise)

# Model building ---------------------------------------------------------------

# Linear regression 
# Y: current_position_length 

data_1 <- data %>%
  select(no_of_promotions, age, current_position_length,
         avg_current_position_length)

#Split the data 80% -20%
set.seed(789)
trainIndex <- createDataPartition(data_1$current_position_length, p = .8, 
                                  list = FALSE, 
                                  times = 1)
train <- data_1[ trainIndex,]
test  <- data_1[-trainIndex,]

#Linear regression (predict current_position_length)
#Train the model using the training data

linear_model <- lm(
  current_position_length ~ ., data=train)

summary(linear_model)

plot(linear_model)

predictions <- predict(linear_model, test)
rmse <- sqrt(mean((test$current_position_length -predictions)^2))
rmse # 349.3227

# Simple linear regression
# If they are happy in their current position.

plot(data$emo_happiness, data$current_position_length)

#Split the data 70% - 30%
set.seed(123)
trainIndex <- createDataPartition(data$emo_happiness, p = .7, 
                                  list = FALSE, 
                                  times = 1)

train <- data[ trainIndex,]
test  <- data[-trainIndex,]

Simple_reg <- lm(emo_happiness ~ current_position_length, data=data)
summary(Simple_reg)

abline(Simple_reg, col="red", pch=25)

predictions <- predict(Simple_reg, test)
rmse <- sqrt(mean((test$emo_happiness -predictions)^2))
rmse # 41.32445

# Simple linear regression
# If the years of experience is a reason for someone to be promoted.

plot(data$no_of_promotions, data$avg_time_in_previous_position)

#Split the data 80% - 20%
set.seed(489)
trainIndex <- createDataPartition(data$no_of_promotions, p = .8, 
                                  list = FALSE, 
                                  times = 1)

train <- data[ trainIndex,]
test  <- data[-trainIndex,]

Simple_reg2 <- lm(no_of_promotions ~ avg_time_in_previous_position, data=data)
summary(Simple_reg2)

abline(Simple_reg2, col="red", pch=19)

predictions <- predict(Simple_reg2, test)
rmse <- sqrt(mean((test$no_of_promotions -predictions)^2))
rmse # 0.7577346

# Logistic regression
# Gender & emo_happiness
# Who is happier, the females or the males?

data_2 <- data %>%
  select(gender, emo_happiness)

data_2$gender = as.factor(if_else(data_2$gender == "Female", 1, 0))

set.seed(123)
trainIndex <- createDataPartition(data_2$gender, p = .8, 
                                  list = FALSE, 
                                  times = 1)
train <- data_2[ trainIndex,]
test  <- data_2[-trainIndex,]

#Logistic regression 
#Train the model using the training data

logistic_model <- glm(
  gender ~ emo_happiness, data=train, family="binomial")

summary(logistic_model)

#Run the test data through the model
predictions <- predict(logistic_model, test, type="response")

confusion_matrix <- table(Actual_value = test$gender, predicted_value = predictions > 0.5)

confusion_matrix #inaccurate model
