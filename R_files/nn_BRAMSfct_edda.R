# Autor: Fernandes, A. A.
library (ggplot2)
library(neuralnet) 
library(tictoc)  # biblioteca para medicao de tempo

tic("Total")
nrepet <- 10

for (hh in sprintf("%02.0f", seq(06,72,6))) {   # loop nas previsoes: 06, 12, 18, 24, 30, 36, ..., 72
#for (hh in "24"){
	files <- list.files(pattern = paste("dados_",hh,"fct_..\\.csv_fmt$", sep=""))  # cria lista dos arquivos particionados
    for (f in 1:length(files)){ # loop nos diversos arquivos(particionados) das previsoes
#	for (i in 1:1) { 
	  print (files[i])
	  tic(files[i])
	  
#	  set.seed(17) # fixa a semente para evitar randomizar cada teste
	  
	  tic("Dados")
	  
		data_complete <- read.csv(files[i], header=F) # abre o arquivo CSV em um dataframe

		data <- data_complete[,c(8, 12, 13, 14)] # extrai apenas os campos de interesse para o dataframe "data"
		colnames(data) <- c("psnm", "cape", "cine","pwt") # nomeia os campos de interesse no cabeçlho do dataframe
		
		td925 <- data_complete$V37 - 2*273.13 - (100 - data_complete$V44)/5.0 # calcula td925
		td800 <- data_complete$V38 - 2*273.13 - (100 - data_complete$V45)/5.0   # calcula td800
		td700 <- data_complete$V39 - 2*273.13 - (100 - data_complete$V46)/5.0 # calcula td700
		
		data[["totals"]] <- (data_complete$V37 - (2*273.13)) + td925 - 2.0 * (data_complete$V40 - (2*273.13)) # calcula TTs
		
		data[["k"]] <- (data_complete$V37-2*273.13) - (data_complete$V40-2*273.15) +   # calcula k index
		                          td800 - (data_complete$V39 - (2*273.13) - td700)
		
		data[["z925"]] <- data_complete[,30] # escreve no dataframe a variavel z925 como atributo
		data[["z800"]] <- data_complete[,31] # escreve no dataframe como atributo
		data[["z500"]] <- data_complete[,33] # escreve no dataframe como atributo
		
		data[["o925"]] <- data_complete[,51] # escreve no dataframe como atributo
		data[["o800"]] <- data_complete[,52] # escreve no dataframe como atributo
		data[["o500"]] <- data_complete[,54] # escreve no dataframe como atributo
		
		data[["d"]] <- data_complete[,64] # escreve no dataframe como atributo
		data <- data[sample(nrow(data), nrow(data)), ] # embaralha o dataframe data
		data <- data[1:3200,] # corta apenas as primeiras 2400 linhas do dataframe
		
#		pdf.dFO <- density (data[data$d>=0.01,]$d)
#		pdf.dNFO <- density (data[data$d<0.01,]$d)
#		
#		ggplot() +
#		  geom_density(data = data[data$d<0.01,], aes(x=d)) +
#		geom_density(data = data[data$d<0.01,], aes(x=d))
		
		index <- sample(1:nrow(data),round(0.80*nrow(data))) # embaralha novamente as 2400 linhas e corta em 80% os índices do dataframe
		train <- data[index,]  # filtra todos os dados indexados
		test <- data[-index,]  # filtra todos os dados que não foram indexados

		maxs <- apply(data, 2, max) # procura pelo maior valor de cada atributos
		mins <- apply(data, 2, min) # procura pelo menor valor de cada atributos
		scaled <- as.data.frame(scale(data, center = mins, scale = maxs - mins)) # normaliza os dados baseado nos mínimos e máximos
		train_ <- scaled[index,] # dados normalizados para treinamemto
		test_ <- scaled[-index,] # dados normalizados para testes
		
		n <- names(train_) # captura os nomes dos atributos de treinamento, para criar a fŕomula de inserção na rede neural
		f <- as.formula(paste("d ~", paste(n[!n %in% "d"], collapse = " + "))) # cria a fórmula da rede
		
		toc()
		
		tic("Treinamento")
		
		set.seed(17) # mantém a semente fixa em 17
		
		nn <- neuralnet(f,data=train_,hidden=c(12,2), # executa a rede com 2 camadas ocultas, com 12 e 2 neurônios, e treina com train_ normalizado
		                lifesign="full",  lifesign.step=10000, # nível de informação a ser impressa na tela
		                linear.output=T, # dado contínuo, não classificado
		                threshold=0.01, # valor de parada
		                stepmax=1e7, # passos máximos da rede
		                err.fct="sse", # tipo erro a ser calculado, sum of square error
		                act.fct = "tanh", rep=nrepet, 
		                algorithm = "rprop+") # função de ativação

		w <- nn$weights
		write.table(unlist(w), paste("../data_files/weights_", files[i], sep = ""))

		toc()
		
		tic ("Previsão")
		
		pr.nn <- compute(nn,test_[,1:ncol(test_[1,])-1], rep=nrepet) # faz a previsão com os dados de testes normalizados
		pr.nn_ <- pr.nn$net.result*(max(data$d)-min(data$d))+min(data$d) # retorna o resultado para os valores não normalizados
		
		#test.r <- (test_$d)*(max(data$d)-min(data$d))+min(data$d) # retorna os dados de output do teste que haviam sido normalizado. Mesmo que teste$d
		
		toc()
		
		MSE.nn <- sum((test$d - pr.nn_)^2)/nrow(test_)
		
#		print(paste(MSE.lm,MSE.nn))
		
		# plota o gráfico de dispressao
		plot(test$d, pr.nn_,col='red',main='Real vs predicted NN',pch=18,cex=0.7)
		abline(0,1,lwd=2)
		legend('bottomright',legend='NN',pch=18,col='red', bty='n')
		
		# calcula as tabelas de contingencia
		tabnn <- as.integer(pr.nn_>=0.01)     # PREVISAO:  FO=1, NFO=0
		tabts <- as.integer(test$d>=0.01)*2   # OBSERVADO: FO=2, NFO=0

		tab <- tabnn + tabts # soma previsao + observado, gerando 4 valores possiveis: 
		
		hits <- sum(as.integer(tab==3))   #previsto=1 + observado=2  -> ACERTO (hits==3)
		miss <- sum(as.integer(tab==2))   #previsto=0 + observado=2  -> ERRO   (missing==2)
		falm <- sum(as.integer(tab==1))   #previsto=1 + observado=0  -> ERRO   (false alarm==1)
		corj <- sum(as.integer(tab==0))   #previsto=0 + observado=0  -> ACERTO (Correct rejected==0 )

		# Imprime informacoes na tela
		print(sprintf("Treinamento: %d dados", length(train_[,1])))
		print(sprintf("Validação: %d dados", length(test_[,1])))
		print(sprintf("%s", files[i]))
		print(sprintf("%d  %d", hits, falm))
		print(sprintf("%d  %d", miss, corj))
	  print(sprintf("%s%03.1f%%", "ACC=", (hits+corj)/(hits+miss+falm+corj)*100))

		toc()
		print ("----------------------------------------------------")
      
		
	}
}

toc ()
