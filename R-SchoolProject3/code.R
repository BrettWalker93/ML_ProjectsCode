Question 1:

library(keras)
library(tensorflow)

data <- read.csv("nyt.csv")
df <- data[,-1]
df <- df[ , which(apply(df, 2, var) != 0)]
df.pca <- prcomp(df, center = T, scale = T)
df.latent.sem <- df.pca$rotation
#signif(sort(df.latent.sem[, 1], decreasing = TRUE)[1:30], 2)

x.std <- apply(df, 2, function(x){(x-mean(x))/sd(x)})
train_data <- array_reshape(array(c(unlist(x.std))), c(102, 4429))

enc_input = layer_input(shape = dim(train_data)[2])
enc_output = enc_input %>% layer_dense(units = 2, activation = "linear")
encoder = keras_model(enc_input, enc_output)

dec_input = layer_input(shape = 2)
dec_output = dec_input %>% layer_dense(units = dim(train_data)[2], activation = "linear")
decoder = keras_model(dec_input, dec_output)

aen_input = layer_input(shape = dim(train_data)[2])
aen_output = aen_input %>% encoder() %>% decoder()
aen = keras_model(aen_input, aen_output)

aen %>% compile(optimizer="rmsprop", loss="binary_crossentropy")
aen %>% fit(train_data,train_data, epochs=50, batch_size=2) 
w <- aen$get_weights()[[1]]

enc_input2 = layer_input(shape = dim(train_data)[2])
enc_output2 = enc_input2 %>% layer_dense(units = 2, activation = "relu")
encoder2 = keras_model(enc_input2, enc_output2)

dec_input2 = layer_input(shape = 2)
dec_output2 = dec_input2 %>% layer_dense(units = dim(train_data)[2], activation = "linear")
decoder2 = keras_model(dec_input2, dec_output2)

aen_input2 = layer_input(shape = dim(train_data)[2])
aen_output2 = aen_input2 %>% encoder2() %>% decoder2()
aen2 = keras_model(aen_input2, aen_output2) 

aen2 %>% compile(optimizer="rmsprop", loss="binary_crossentropy")
aen2 %>% fit(train_data,train_data, epochs=50, batch_size=2)
w2 <- aen2$get_weights()[[1]]

keepart <- data$class.labels == "art"
artdf <- x.std[keepart,]
artm <- matrix(unlist(artdf), ncol = 4429, byrow=TRUE)
artcoords <- artm %*% w2
artcoords <- pmax(artcoords, 0)
keepmusic <- data$class.labels == "music"
musicdf <- x.std[keepmusic,]
musicm <- matrix(unlist(musicdf), ncol=4429, byrow=TRUE)
musiccoords <- musicm %*% w2
musiccoords <- pmax(musiccoords, 0)

plot(df.pca$x[, 1:2], pch = ifelse(data[, "class.labels"] == "music", "m", "a"), col = ifelse(data[, "class.labels"] == "music", "blue", "red"))

signif(sort(w[,1], decreasing=TRUE)[1:30], 2)

signif(sort(df.latent.sem[, 1], decreasing = TRUE)[1:30], 2)

plot(artcoords[,1],artcoords[,2],col="red")
points(musiccoords[,1],musiccoords[,2],col="blue")

artcoords2 <- artm %*% w2
musiccoords2 <- musicm %*% w2
plot(artcoords2[,1],artcoords2[,2],col="red")
points(musiccoords2[,1],musiccoords2[,2],col="blue")

----------

Question 2:
#NOTE: change ncol to 5 for k = 5 and change for loop in gradient descent for q in 1:5
library(rsparse)
data("movielens100k")

mdf <- movielens100k
keeprows <- rowSums(mdf[1:943,] != 0) > 50
keepcols <- colSums(mdf[,1:1639] != 0) > 50
trimmed <- mdf[keeprows,keepcols]
x <- trimmed

ut <- matrix(runif(563*10)/200, ncol=10)

vt <- matrix(runif(626*10)/100, ncol=10)

delta <- function(ij) {

	xh <- xhat(ij)

	d <- x[ij[1],ij[2]] - xh

	return (d)
}

xhat <- function(ij) {

	i <- ij[1]
	j <- ij[2]
	
	x_ <- u[i,] %*% v[j,]
	x <- as.numeric(x_)
	if (x < 0) {
		return (0)
	}

	if (x > 5) {
		return (5)
	}
	
	return (x)

}

rate = 0.0146

go = TRUE
h <- 1

while (go) {
	for (i in 1:563) {
		cat(i, "\n")
		for (j in 1:626) {
			if (x[i,j] != 0) {
				xx <- x[i,j]				
				for (q in 1:10) {				
					lu = delta(c(i,j)) * vt[j,q]
					lv = delta(c(i,j)) * ut[i,q]
					ut[i,q] = ut[i,q] + rate * lu * xx
					vt[j,q] = vt[j,q] + rate * lv * xx		
				}
				
			}					
		}				
	}
	h <- h + 1
	if (h > 2) { go = FALSE }
}

user1 <- which(trimmed[1,] == 0)
rec1 <- rep(0, 395)
for (i in 1:395) {
	rec1[i] <- xhat(c(1, i))
}


user2 <- which(trimmed[2,] == 0)
rec2 <- rep(0, 395)
for (i in 1:395) {
	rec2[i] <- xhat(c(2, i))
}
mind1 <- order(-rec1)[1:5]
mind2 <- order(-rec2)[1:5]

names(user1[mind1])
names(user2[mind2])

----------

Question 3:

library(PerformanceAnalytics)
df <- USArrests
x.std <- apply(df, 2, function(x){(x-mean(x))/sd(x)})
chart.Correlation(x.std)
x.pca <- prcomp(x.std)
s<-cov(x.std)
s.eigen <- eigen(s)
x.svd <- svd(x.std)

#Eigenvalues:  
`r s.eigen$values`  
#Eigenvectors:  
`r s.eigen$vectors`  

#From D:  
#Eigenvalues (sdev squared):  
`r x.pca$sdev^2`  
#Eigenvectors:  
`r x.pca$rotation`  

#Covariance of scores:
`r round(cov(x.pca$x), 3)`

#Singular values:  
`r x.svd$d`  
#Matrix:  
`r x.svd$v`  

for (k in 1:4) {
    cat(svd(x.std, nu = k)$v, '\n')
    cat(prcomp(x.std, rank = k)$rotation, '\n')
}