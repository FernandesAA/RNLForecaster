library(neuralnet)

setwd("~/STEPHAN/macaiba/")

#for (hh in sprintf("%02.0f", seq(0,72,6))) {
for (hh in "24"){
	files <- list.files(pattern = paste("dados_",hh,"fct_..\\.csv$", sep=""))
#    for (f in 1:length(files)){
	for (i in 1) {
		print (files[i])
		data_complete <- read.csv(files[i], header=F)
		
		data <- data_complete[,c(8, 12, 14)]
		
		colnames(data) <- c("psnm", "cape", "pwt")
		
		td925 <- data_complete$V37 - 2*273.13 - (100 - data_complete$V44)/5.0
		td800 <- data_complete$V38 - 2*273.13 - (100-data_complete$V45)/5.0
		td700 <- data_complete$V39 - 2*273.13 - (100 - data_complete$V46)/5.0
		
		data[["totals"]] <- (data_complete$V37 - (2*273.13)) + td925 - 2.0 * (data_complete$V40 - (2*273.13))
		
		data[["k"]] <- (data_complete$V37-2*273.13) - (data_complete$V40-2*273.15) + 
		                          td800 - (data_complete$V39 - (2*273.13) - td700)
		
		data[["z925"]] <- data_complete[,30]
		data[["z800"]] <- data_complete[,31]
		data[["z500"]] <- data_complete[,33]
		
		data[["o925"]] <- data_complete[,51]
		data[["o800"]] <- data_complete[,52]
		data[["o500"]] <- data_complete[,54]
		
		data[["d"]] <- data_complete[,64]
		
		index <- sample(1:nrow(data),round(0.80*nrow(data)))
		train <- data[index,]
		test <- data[-index,]
		lm.fit <- glm(d~., data=train)
		summary(lm.fit)
		pr.lm <- predict(lm.fit,test)
		MSE.lm <- sum((pr.lm - test$d)^2)/nrow(test)
		
		maxs <- apply(data, 2, max) 
		mins <- apply(data, 2, min)
		scaled <- as.data.frame(scale(data, center = mins, scale = maxs - mins))
		train_ <- scaled[index,]
		test_ <- scaled[-index,]
		
		n <- names(train_)
		f <- as.formula(paste("d ~", paste(n[!n %in% "d"], collapse = " + ")))
		nn <- neuralnet(f,data=train_,hidden=c(12,3), lifesign="full", linear.output=T, threshold=0.01, stepmax=250000)
		
		pr.nn <- compute(nn,test_[,1:ncol(test_[1,])-1])
		pr.nn_ <- pr.nn$net.result*(max(data$d)-min(data$d))+min(data$d)
		test.r <- (test_$d)*(max(data$d)-min(data$d))+min(data$d)
		
		MSE.nn <- sum((test.r - pr.nn_)^2)/nrow(test_)
		
		print(paste(MSE.lm,MSE.nn))
		
		plot(test$d, pr.nn_,col='red',main='Real vs predicted NN',pch=18,cex=0.7)
		abline(0,1,lwd=2)
		legend('bottomright',legend='NN',pch=18,col='red', bty='n')
		
		tabnn <- as.integer(pr.nn_>=0.01)
		tabts <- as.integer(test$V64>=0.01)*2   # OBSERVADO

		tab <- tabnn + tabts
		
		hits <- sum(as.integer(tab==3))
		miss <- sum(as.integer(tab==2))
		falm <- sum(as.integer(tab==1))
		corj <- sum(as.integer(tab==0))

		print (c(hits, miss))
		print (c(falm, corj))

		print ((hits+corj)/(hits+miss+falm+corj))
		
	}
}
