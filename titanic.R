#################################################################
#
#           Machine Learning Kaggle
#
# Autor: Roberto Haruo Lopes Fuzimoto
# Projeto: Neste desafio, pedimos que você complete a análise de que
# tipos de pessoas provavelmente sobrevivem. Em particular, pedimos 
# que você aplique as ferramentas de aprendizado de máquina para prever 
# quais passageiros sobreviveram à tragédia.
#
#
#################################################################

getwd()
setwd("/home/roberto/Documentos/git/R")

install.packages("funModeling")
install.packages("Amelia", lib = "~/myrlibrary")
install.packages("caret")
install.packages("ggplot2")
install.packages("dplyr")
install.packages("reshape")
install.packages("randomForest")
install.packages("readxl")
install.packages("e1071") # Naive Bayes
install.packages("tidyr")
install.packages("ggthemes")
install.packages("rpart")

# Carregando os pacotes
library(Amelia)
library(caret)
library(ggplot2) # Gráficos
library(dplyr) # data manipulation
library(reshape)
library(randomForest)
library(readxl)
library(e1071) # Naive Bayes
library(funModeling)
library(tidyr) # data manipulation
library(ggthemes)
library(rpart) #Classification and Regression Trees (CART)
library(rpart.plot) #Classification and Regression Trees (CART)

rfNews()

# carregando o arquivo
dataset <- read.csv("train.csv", sep=",")
dataset <- data.frame(dataset)
head(dataset)
#View(dataset)


# Resumo do dataset
str(dataset)
colnames(dataset)

# Removendo colunas 
dataset$PassengerId <- NULL
dataset$SibSp <- NULL
dataset$Parch <- NULL
dataset$Ticket <- NULL
dataset$Cabin <- NULL
dataset$Name <- NULL

colnames(dataset)
# RENOMEANDO AS COLUNAS
colnames(dataset)[1] <- "SOBREVIVEU"
colnames(dataset)[2] <- "CLASSE_SOCIAL"
colnames(dataset)[3] <- "SEXO"
colnames(dataset)[4] <- "IDADE"
colnames(dataset)[5] <- "PREÇO"
colnames(dataset)[6] <- "PORTO_DE_EMBARCAÇÃO"

# Ordenando as colunas
dataset <- dataset %>%
  select(CLASSE_SOCIAL, IDADE, SEXO, PREÇO, PORTO_DE_EMBARCAÇÃO, SOBREVIVEU)



# Transoformando as Variáveis em fator, numerico e character

#dataset$NOME <- as.character(dataset$NOME)


dataset$SOBREVIVEU <- as.factor(dataset$SOBREVIVEU)
levels(dataset$SOBREVIVEU) <- c("Não", "Sim")
head(dataset$SOBREVIVEU)

dataset$SEXO <- as.factor(dataset$SEXO)
levels(dataset$SEXO) <- c("Feminino", "Masculino")

dataset$CLASSE_SOCIAL <- as.factor(dataset$CLASSE_SOCIAL)
levels(dataset$CLASSE_SOCIAL)
levels(dataset$CLASSE_SOCIAL) <- c("Alta", "Média", "Baixa")

dataset$IDADE <- as.numeric(dataset$IDADE)
dataset$IDADE <- cut(dataset$IDADE, c(0, 30, 50, 100), labels = c("Jovem", "adulto", "Idoso"))

dataset$PORTO_DE_EMBARCAÇÃO <- as.factor(dataset$PORTO_DE_EMBARCAÇÃO)
levels(dataset$PORTO_DE_EMBARCAÇÃO) <- c(0,"Cherbourg","Southampton", "Queenstown")

#########################################################################
#Eclat Algotitmo utilizado para encontrar padrões nos data sets #########
#########################################################################
installed.packages("arules")
library(arules)

dataset$PREÇO <- as.factor(dataset$PREÇO)
regras <- eclat(dataset, parameter = list(supp = 0.1, maxlen = 5))
inspect(regras)
install.packages("aruleViz")
plot(regras, method="graph", control=list(type="items"))

# Verificando se a valores missing
sapply(dataset, function(x) sum(is.na(x)))
missmap(dataset, main = "Valores Missing Observados")
dataset <- na.omit(dataset)

# Verifcar valores missing
dataset <- na.omit(dataset)

#Medidas de posição
str(dataset)
mean(dataset$PREÇO)
median(dataset$PREÇO)

dataset$PREÇO <- as.numeric(dataset$PREÇO)
# Gráfico média de preço da passagem por Idade
ggplot(dataset) + stat_summary(aes(x = dataset$IDADE, y = dataset$PREÇO),
                               fun.y = mean, geom = "bar",
                               fill = "lightgreen", col = "grey50")

#Gráfico média de preço da passagem por CLasse Social
ggplot(dataset) + stat_summary(aes(x = dataset$CLASSE_SOCIAL, y = dataset$PREÇO),
                               fun.y = mean, geom = "bar",
                               fill = "lightblue", col = "grey50")

#Gráfico média de preço da passagem por Sexo
ggplot(dataset) + stat_summary(aes(x = dataset$SEXO, y = dataset$PREÇO),
                               fun.y = mean, geom = "bar",
                               fill = "lightblue", col = "grey50")

# Esses 3 gráficos nos mostram que pessoas mais velhas, de classe social alta e do sexo 
# feminino pagarama mais caro nas cabines do navio.



# Medidas de Dispersão
var(dataset$PREÇO)
sd(dataset$PREÇO)

# Coeficiente  de Variação
CV <- (sd(dataset$PREÇO) / var(dataset$PREÇO) ) * 100
CV

# Total de Sobreviventes ou Não
barplot(table(dataset$SOBREVIVEU))

# Plot da distribuição utilizando ggplot2
qplot(SOBREVIVEU, data = dataset, geom = "bar")
+ theme(axis.text.x = element_text(angle = 90, hjust = 1))

# Idade vs Sobreviveu
ggplot(dataset[1:714,],aes(IDADE, fill = (SOBREVIVEU))) + 
  geom_bar(stat = "count") + 
  theme_few() +
  xlab("Idade") +
  facet_grid(.~SEXO)+
  ylab("count") +
  scale_fill_discrete(name = "Sobreviveu") + 
  ggtitle("Idade vs Sobreviveu")

# Clase Social vc Sobreviveu
ggplot(dataset[1:714,], aes(CLASSE_SOCIAL, fill = (SOBREVIVEU))) +
  geom_bar(stat = "count") +
  theme_few() +
  xlab("Classe Social") +
  facet_grid(.~SEXO) +
  ylab("count") +
  scale_fill_discrete(name = "Sobreviveu") +
  ggtitle("Classe Social vs Sobrevivieu")

# Porto de Embarque vs Sobreviveu
ggplot(dataset[1:714,], aes(PORTO_DE_EMBARCAÇÃO, fill = (SOBREVIVEU))) +
  geom_bar(stat = "count") +
  xlab("Porto de Embaracação") +
  facet_grid(.~SEXO) +
  ylab("count") +
  scale_fill_discrete(name = "Sobreviveu") +
  ggtitle("Porto de Embarcação vs Sobreviveu")



#########################################################################
################## Random Forest Classification Model ###################
#########################################################################

# Set the seed
set.seed(12345)
#summary(dataset)
str(dataset)

# Contruindo o modelo Random Forest
rf_model <- randomForest(SOBREVIVEU ~ ., data = dataset)
print("Modelo contruido")
rf_model

#########################################################################
################## Naive Bayes ##########################################
#########################################################################
x = dataset[,-6]
y = dataset$SOBREVIVEU
nbmodel <- train(x,y,'nb',trControl = trainControl(method = 'cv',number = 10))
class(nbmodel)
summary(nbmodel)
print(nbmodel)
predict(nbmodel$finalModel,x)
table(predict(nbmodel$finalModel,x)$class,y)
naive_iris <- NaiveBayes(dataset$SOBREVIVEU ~ ., data = dataset)
plot(naive_iris)

#########################################################################
################## Classification and Regression Trees (CART)############
#########################################################################

set.seed(123)
treemodel <- rpart(SOBREVIVEU ~ ., data = dataset, control = rpart.control(cp = 0.0001))
printcp(treemodel)
bestcp <- treemodel$cptable[which.min(treemodel$cptable[,"xerror"]),"CP"]
bestcp
tree.pruned <- prune(treemodel, cp = bestcp)
tree.pruned



########################################################################
##############Carregando dataset de Teste###############################
########################################################################

data_teste <- read.csv("test.csv", sep = ",")
#data_teste <- data.frame(data_teste)
head(data_teste)
colnames(data_teste)

# Removendo colunas                     
data_teste$PassengerId <- NULL
data_teste$Name <- NULL
data_teste$SibSp <- NULL
data_teste$Parch <- NULL
data_teste$Parch <- NULL
data_teste$Ticket <- NULL
data_teste$Cabin <- NULL

# RENOMEANDO AS COLUNAS
colnames(data_teste)[1] <- "CLASSE_SOCIAL"
colnames(data_teste)[2] <- "SEXO"
colnames(data_teste)[3] <- "IDADE"
colnames(data_teste)[4] <- "PREÇO"
colnames(data_teste)[5] <- "PORTO_DE_EMBARCAÇÃO"


# Ordenando as colunas
data_teste <- data_teste %>%
  select(CLASSE_SOCIAL, IDADE, SEXO, PREÇO, PORTO_DE_EMBARCAÇÃO)


# Transoformando as Variáveis em fator, numerico e character

data_teste$CLASSE_SOCIAL <- as.factor(data_teste$CLASSE_SOCIAL)
levels(data_teste$CLASSE_SOCIAL)
levels(data_teste$CLASSE_SOCIAL) <- c("Alta", "Média", "Baixa")

data_teste$SEXO <- as.factor(data_teste$SEXO)
levels(data_teste$SEXO) <- c("Feminino", "Masculino")

data_teste$IDADE <- as.numeric(data_teste$IDADE)
data_teste$IDADE <- cut(data_teste$IDADE, c(0, 30, 50, 100), labels = c("Jovem", "adulto", "Idoso"))

data_teste$PORTO_DE_EMBARCAÇÃO <- as.factor(data_teste$PORTO_DE_EMBARCAÇÃO)
levels(data_teste$PORTO_DE_EMBARCAÇÃO) <- c(0, "Cherbourg", "Queenstown", "Southampton")

# Verificando se a valores missing
sapply(dataset, function(x) sum(is.na(x)))
missmap(dataset, main = "Valores Missing Observados")
dataset <- na.omit(dataset)

# Verifcar valores missing
data_teste <- na.omit(data_teste)

####################### Previsão Randon Forest #####################################
####################################################################################
predictionrf <- predict(rf_model, data_teste)
predictionrf <- data.frame(predictionrf)
predictionrf



data_teste['Previsão'] <- c(predictionrf)
data_teste
table(predictionrf)

rf_model

# Conferindo o erro do modelo 
plot(rf_model, ylim = c(0,0.36))
legend('topright', colnames(rf_model$err.rate), col = 1:3, fill = 1:3)

varImpPlot(rf_model)


# Obtendo as variaveis mais importantes
importance <- importance(rf_model)
varimportance <- data.frame(Variables = row.names(importance), importance = round(importance[,'MeanDecreaseGini'],2))

# Criando o rank de variaveis baseado na importância
rankImportance <- varimportance %>%
  mutate(Rank = paste0('#', dense_rank(desc(importance))))

# Usando o ggplot2 para visualizar a importância relativa das variaveis
ggplot(rankImportance, aes(x = reorder(Variables, importance), y = importance, fill = importance))+
  geom_bar(stat = 'identity')+
  geom_text(aes(x = Variables, y = 0.5, label = Rank), hjust = 0, vjust=0.55, size = 4, colour = 'red')+
  labs(x = 'Variables')+
  coord_flip()



# SVC
# Decision Tree
# AdaBoost
# Random Forest
# Extra Trees
# Gradient Boosting
# Multiple layer perceprton (neural network)
# KNN
# Logistic regression
# Linear Discriminant Analysis

















getwd()
setwd('/home/rfuzi/Documentos/Projetos/Kaggle/Titanic')

dados_treino <- read.csv('train.csv',sep = ',')
dados_treino <- 

head(dados_treino)
summary(dados_treino)
View(dados_treino)

