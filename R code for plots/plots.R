###############################################################################
##### This R code is used for drawing the pictures presented in the paper #####
###############################################################################

library(ggplot2)

# function to import featured data
getValues <- function(valueType, algorithm, range, size, sparsity){
  temp1 <- X$valueType %in% valueType
  temp2 <- X$Algorithm %in% algorithm
  temp3 <- X$range %in% range
  temp4 <- X$size %in% size
  temp5 <- X$Sparsity %in% sparsity
  X[temp1 & temp2 & temp3 & temp4 & temp5,]
}

# function to create statistcs
summarizeCol <- function(dataToSum, fun){
  temp <- dataToSum[,-ncol(dataToSum)] 
  indices <- apply(as.matrix(temp), 1, paste, collapse="")
  indexes <- unique(indices)
  res <- data.frame()
  for(index in indexes){
    val <- fun(dataToSum[indices == index, ncol(dataToSum)])
    resTemp <- dataToSum[indices == index, -ncol(dataToSum)][1,]
    resTemp$val <- val
    res <- rbind(res, resTemp)
  }
  res
}

###################################
##### 1-hidden layer networks #####
###################################

# import dataset
X <- read.delim("1_hidden_layer_networks.csv", header = F, sep = ',', colClasses = c("factor", "factor", "factor", "numeric", "numeric", "numeric"))
colnames(X) <- c("valueType", "Algorithm", "range", "size", "Sparsity", "value")
levels(X$Algorithm) <- c("HR-2", "LBS", "LipOpt-3", "SHOR")
levels(X$range) <- c("Global", "Local")
X$Algorithm <- factor(X$Algorithm, levels = c("HR-2", "SHOR", "LipOpt-3", "LBS"))
X$range <- factor(X$range, c("Global", "Local"))

################################################################
### Figure 1: global/local upper bounds for (80,80) networks ###

# import data for upper bounds and for (80,80) networks
toPlot <- getValues("bounds", unique(X$Algorithm), unique(X$range), 80, unique(X$Sparsity))

# calculate 1st quartile, 2nd quartile (median), 3rd quartile
firstQ <- summarizeCol(toPlot, function(x){quantile(x)[2]})
medians <- summarizeCol(toPlot, median)
thirdQ <- summarizeCol(toPlot, function(x){quantile(x)[4]})

# add statistics to data
toPlot <- medians
colnames(toPlot)[6] <- "median"
toPlot$q1 <- firstQ$val
toPlot$q3 <- thirdQ$val

# draw the plot (fix colour, size and linetype for each algorithm)
colourPalette <- c("#FF0000", "#E69F00", "#56B4E9", "#CC79A7")
linetypePalette <- c("solid", "longdash", "twodash", "dotted")
pl <- ggplot(data = toPlot, aes(x = Sparsity, colour = Algorithm, linetype=Algorithm)) + 
  geom_line(aes(y = median), size = .5) + 
  geom_ribbon(aes(ymax=q3, ymin=q1, fill=Algorithm), alpha = 0.1, linetype = 0) + 
  geom_point(aes(y = median), size = .5) +
  facet_grid(.~range, scales = "free") + 
  scale_linetype_manual(values=linetypePalette) +
  scale_colour_manual(values=colourPalette) +
  scale_fill_manual(values=colourPalette) +
  ylab("Lipschitz Constant")+
  theme_bw()
pl

# save the plot
pdf("bound1.pdf", width = 5, height = 2)
print(pl)
dev.off()

####################################################################
##### Figure 2: global/local running time for (80,80) networks #####

# import data for running time and for (80,80) networks
toPlot <- getValues("times", c("HR-2", "SHOR", "LipOpt-3"), unique(X$range), 80, unique(X$Sparsity))

# calculate 1st quartile, 2nd quartile (median), 3rd quartile
firstQ <- summarizeCol(toPlot, function(x){quantile(x)[2]})
medians <- summarizeCol(toPlot, median)
thirdQ <- summarizeCol(toPlot, function(x){quantile(x)[4]})

# add statistics to data
toPlot <- medians
colnames(toPlot)[6] <- "median"
toPlot$q1 <- firstQ$val
toPlot$q3 <- thirdQ$val

# draw the plot (fix colour, size and linetype for each algorithm)
colourPalette <- c("#FF0000", "#E69F00", "#56B4E9")
linetypePalette <- c("solid", "longdash", "twodash")
pl <- ggplot(data = toPlot, aes(x = Sparsity, colour = Algorithm, linetype=Algorithm)) + 
  geom_line(aes(y = median), size = .5) + 
  geom_ribbon(aes(ymax=q3, ymin=q1, fill=Algorithm), alpha = 0.1, linetype = 0) + 
  geom_point(aes(y = median), size = .5) +
  facet_grid(.~range, scales = "free") + 
  scale_linetype_manual(values=linetypePalette) +
  scale_colour_manual(values=colourPalette) +
  scale_fill_manual(values=colourPalette) +
  ylab("Running Time") +
  scale_y_log10() + 
  theme_bw()
pl

# save the plot
pdf("time1.pdf", width = 5, height = 2)
print(pl)
dev.off()

########################################################################
##### Figure 3(a): global upper bounds for 1-hidden layer networks #####

# import data for global upper bounds
toPlot <- getValues("bounds", unique(X$Algorithm), c("Global"), unique(X$size), unique(X$Sparsity))

# calculate 1st quartile, 2nd quartile (median), 3rd quartile
firstQ <- summarizeCol(toPlot, function(x){quantile(x)[2]})
medians <- summarizeCol(toPlot, median)
thirdQ <- summarizeCol(toPlot, function(x){quantile(x)[4]})

# add statistics to data
toPlot <- medians
colnames(toPlot)[6] <- "median"
toPlot$q1 <- firstQ$val
toPlot$q3 <- thirdQ$val

# define legends
temp <- as.character(toPlot$size)
temp2 <- temp
temp2[] <- "("
temp3 <- temp
temp3[] <- ","
temp4 <- temp
temp4[] <- ") Global"
#
Temp <- cbind(temp2, temp, temp3, temp, temp4)
temp <- apply(Temp, 1, paste, collapse="")
toPlot$size <- temp

# draw the plot (fix colour, size and linetype for each algorithm)
colourPalette <- c("#FF0000", "#E69F00", "#56B4E9", "#CC79A7")
linetypePalette <- c("solid", "longdash", "twodash", "dotted")
pl <- ggplot(data = toPlot, aes(x = Sparsity, colour = Algorithm, linetype=Algorithm)) + 
  geom_line(aes(y = median), size = .5) + 
  geom_ribbon(aes(ymax=q3, ymin=q1, fill=Algorithm), alpha = 0.1, linetype = 0) + 
  geom_point(aes(y = median), size = .5) +
  facet_wrap(~size, scales = "free") + 
  scale_linetype_manual(values=linetypePalette) +
  scale_colour_manual(values=colourPalette) +
  scale_fill_manual(values=colourPalette) +
  ylab("Lipschitz Constant")+
  theme_bw()
pl

# save the plot
pdf("compare_lip_global.pdf", width = 5, height = 4)
print(pl)
dev.off()

########################################################################
##### Figure 3(b): global running time for 1-hidden layer networks #####

# import data for global running time
toPlot <- getValues("times", c("HR-2", "SHOR", "LipOpt-3"), c("Global"), unique(X$size), unique(X$Sparsity))

# calculate 1st quartile, 2nd quartile (median), 3rd quartile
firstQ <- summarizeCol(toPlot, function(x){quantile(x)[2]})
medians <- summarizeCol(toPlot, median)
thirdQ <- summarizeCol(toPlot, function(x){quantile(x)[4]})

# add statistics to data
toPlot <- medians
colnames(toPlot)[6] <- "median"
toPlot$q1 <- firstQ$val
toPlot$q3 <- thirdQ$val

# define legends
temp <- as.character(toPlot$size)
temp2 <- temp
temp2[] <- "("
temp3 <- temp
temp3[] <- ","
temp4 <- temp
temp4[] <- ") Global"
#
Temp <- cbind(temp2, temp, temp3, temp, temp4)
temp <- apply(Temp, 1, paste, collapse="")
toPlot$size <- temp

# draw the plot (fix colour, size and linetype for each algorithm)
colourPalette <- c("#FF0000", "#E69F00", "#56B4E9")
linetypePalette <- c("solid", "longdash", "twodash")
pl <- ggplot(data = toPlot, aes(x = Sparsity, colour = Algorithm, linetype=Algorithm)) + 
  geom_line(aes(y = median), size = .5) + 
  geom_ribbon(aes(ymax=q3, ymin=q1, fill=Algorithm), alpha = 0.1, linetype = 0) + 
  geom_point(aes(y = median), size = .5) +
  facet_wrap(~size, scales = "free") + 
  scale_linetype_manual(values=linetypePalette) +
  scale_colour_manual(values=colourPalette) +
  scale_fill_manual(values=colourPalette) +
  ylab("Runnging time")+
  scale_y_log10() + 
  theme_bw()
pl

# save the plot
pdf("time_lip_global.pdf", width = 5, height = 4)
print(pl)
dev.off()

#######################################################################
##### Figure 4(a): local upper bounds for 1-hidden layer networks #####

# import data for local upper bounds
toPlot <- getValues("bounds", unique(X$Algorithm), c("Local"), unique(X$size), unique(X$Sparsity))

# calculate 1st quartile, 2nd quartile (median), 3rd quartile
firstQ <- summarizeCol(toPlot, function(x){quantile(x)[2]})
medians <- summarizeCol(toPlot, median)
thirdQ <- summarizeCol(toPlot, function(x){quantile(x)[4]})

# add statistics to data
toPlot <- medians
colnames(toPlot)[6] <- "median"
toPlot$q1 <- firstQ$val
toPlot$q3 <- thirdQ$val

# define legends
temp <- as.character(toPlot$size)
temp2 <- temp
temp2[] <- "("
temp3 <- temp
temp3[] <- ","
temp4 <- temp
temp4[] <- ") Local"
#
Temp <- cbind(temp2, temp, temp3, temp, temp4)
temp <- apply(Temp, 1, paste, collapse="")
toPlot$size <- temp

# draw the plot (fix colour, size and linetype for each algorithm)
colourPalette <- c("#FF0000", "#E69F00", "#CC79A7")
linetypePalette <- c("solid", "longdash", "dotted")
pl <- ggplot(data = toPlot, aes(x = Sparsity, colour = Algorithm, linetype=Algorithm)) +
  geom_line(aes(y = median), size = .5) +
  geom_ribbon(aes(ymax=q3, ymin=q1, fill=Algorithm), alpha = 0.1, linetype = 0) +
  geom_point(aes(y = median), size = .5) +
  facet_wrap(~size, scales = "free") +
  scale_linetype_manual(values=linetypePalette) +
  scale_colour_manual(values=colourPalette) +
  scale_fill_manual(values=colourPalette) +
  ylab("Lipschitz constant")+
  theme_bw()
pl

# save the plot
pdf("compare_lip_local.pdf", width = 5, height = 4)
print(pl)
dev.off()

#######################################################################
##### Figure 4(b): local running time for 1-hidden layer networks #####

# import data for loal running time
toPlot <- getValues("times", c("HR-2", "SHOR", "LipOpt-3"), c("Local"), unique(X$size), unique(X$Sparsity))

# calculate 1st quartile, 2nd quartile (median), 3rd quartile
firstQ <- summarizeCol(toPlot, function(x){quantile(x)[2]})
medians <- summarizeCol(toPlot, median)
thirdQ <- summarizeCol(toPlot, function(x){quantile(x)[4]})

# add statistics to data
toPlot <- medians
colnames(toPlot)[6] <- "median"
toPlot$q1 <- firstQ$val
toPlot$q3 <- thirdQ$val

# define legends
temp <- as.character(toPlot$size)
temp2 <- temp
temp2[] <- "("
temp3 <- temp
temp3[] <- ","
temp4 <- temp
temp4[] <- ") Local"
#
Temp <- cbind(temp2, temp, temp3, temp, temp4)
temp <- apply(Temp, 1, paste, collapse="")
toPlot$size <- temp

# draw the plot (fix colour, size and linetype for each algorithm)
colourPalette <- c("#FF0000", "#E69F00")
linetypePalette <- c("solid", "longdash")
pl <- ggplot(data = toPlot, aes(x = Sparsity, colour = Algorithm, linetype=Algorithm)) + 
  geom_line(aes(y = median), size = .5) + 
  geom_ribbon(aes(ymax=q3, ymin=q1, fill=Algorithm), alpha = 0.1, linetype = 0) + 
  geom_point(aes(y = median), size = .5) +
  facet_wrap(~size, scales = "free") + 
  scale_linetype_manual(values=linetypePalette) +
  scale_colour_manual(values=colourPalette) +
  scale_fill_manual(values=colourPalette) +
  ylab("Running time")+
  scale_y_log10() + 
  theme_bw()
pl

# save the plot
pdf("time_lip_local.pdf", width = 5, height = 4)
print(pl)
dev.off()

###################################
##### 2-hidden layer networks #####
###################################

# import dataset
X <- read.delim("2_hidden_layer_networks.csv", header = F, sep = ',', colClasses = c("factor", "factor", "factor", "numeric", "numeric", "numeric"))
colnames(X) <- c("valueType", "Algorithm", "range", "size", "Sparsity", "value")
levels(X$Algorithm) <- c("HR-1", "HR-2", "HR-2-approx", "LBS", "LipOpt-3", "LipOpt-4", "SHOR-approx")
levels(X$range) <- c("Global", "Local")
X$Algorithm <- factor(X$Algorithm, levels = c("HR-2-approx", "HR-2", "HR-1", "SHOR-approx", "LipOpt-3", "LipOpt-4", "LBS"))
X$range <- factor(X$range, c("Global", "Local"))

########################################################################
##### Figure 5(a): global upper bounds for 2-hidden layer networks #####

# import data for global upper bounds
toPlot <- getValues("bounds", c("HR-2", "HR-1", "LipOpt-3", "LipOpt-4", "LBS"), c("Global"), 5 * 1:4, 2 * 1:5)

# calculate 1st quartile, 2nd quartile (median), 3rd quartile
firstQ <- summarizeCol(toPlot, function(x){quantile(x)[2]})
medians <- summarizeCol(toPlot, median)
thirdQ <- summarizeCol(toPlot, function(x){quantile(x)[4]})

# add statistics to data
toPlot <- medians
colnames(toPlot)[6] <- "median"
toPlot$q1 <- firstQ$val
toPlot$q3 <- thirdQ$val

# define legends
temp <- as.character(toPlot$size)
temp2 <- temp
temp2[] <- "("
temp3 <- temp
temp3[] <- ","
temp4 <- temp
temp4[] <- ",10) Global"
#
Temp <- cbind(temp2, temp, temp3, temp, temp4)
temp <- apply(Temp, 1, paste, collapse="")
toPlot$size <- temp
toPlot$size <- factor(toPlot$size, levels = c('(5,5,10) Global', '(10,10,10) Global', '(15,15,10) Global', '(20,20,10) Global'))

# draw the plot (fix colour, size and linetype for each algorithm)
colourPalette <- c("#FF0000", "#E69F00", "#56B4E9", "#009E73", "#CC79A7")
linetypePalette <- c("solid", "longdash", "twodash", "twodash", "dotted")
pl <- ggplot(data = toPlot, aes(x = Sparsity, colour = Algorithm, linetype=Algorithm)) + 
  geom_line(aes(y = median), size = .5) + 
  geom_ribbon(aes(ymax=q3, ymin=q1, fill=Algorithm), alpha = 0.1, linetype = 0) + 
  geom_point(aes(y = median), size = .5) +
  facet_wrap(~size, scales = "free") + 
  scale_linetype_manual(values=linetypePalette) +
  scale_colour_manual(values=colourPalette) +
  scale_fill_manual(values=colourPalette) +
  ylab("Lipschitz constant") +
  theme_bw()
pl

# save the plot
pdf("compare_lip_global_multi.pdf", width = 5, height = 4)
print(pl)
dev.off()

########################################################################
##### Figure 5(b): global running time for 2-hidden layer networks #####

# import data for global running time
toPlot <- getValues("times", c("HR-2", "HR-1", "LipOpt-3", "LipOpt-4"), c("Global"), 5 * 1:4, 2 * 1:5)

# calculate 1st quartile, 2nd quartile (median), 3rd quartile
firstQ <- summarizeCol(toPlot, function(x){quantile(x)[2]})
medians <- summarizeCol(toPlot, median)
thirdQ <- summarizeCol(toPlot, function(x){quantile(x)[4]})

# add statistics to data
toPlot <- medians
colnames(toPlot)[6] <- "median"
toPlot$q1 <- firstQ$val
toPlot$q3 <- thirdQ$val

# define legends
temp <- as.character(toPlot$size)
temp2 <- temp
temp2[] <- "("
temp3 <- temp
temp3[] <- ","
temp4 <- temp
temp4[] <- ",10) Global"
#
Temp <- cbind(temp2, temp, temp3, temp, temp4)
temp <- apply(Temp, 1, paste, collapse="")
toPlot$size <- temp
toPlot$size <- factor(toPlot$size, levels = c('(5,5,10) Global', '(10,10,10) Global', '(15,15,10) Global', '(20,20,10) Global'))

# draw the plot (fix colour, size and linetype for each algorithm)
colourPalette <- c("#FF0000", "#E69F00", "#56B4E9", "#009E73")
linetypePalette <- c("solid", "longdash", "twodash", "twodash")
pl <- ggplot(data = toPlot, aes(x = Sparsity, colour = Algorithm, linetype=Algorithm)) + 
  geom_line(aes(y = median), size = .5) + 
  geom_ribbon(aes(ymax=q3, ymin=q1, fill=Algorithm), alpha = 0.1, linetype = 0) + 
  geom_point(aes(y = median), size = .5) + 
  facet_wrap(~size, scales = "free") +
  scale_linetype_manual(values=linetypePalette) +
  scale_colour_manual(values=colourPalette) +
  scale_fill_manual(values=colourPalette) +
  ylab("Running Time") + 
  scale_y_log10() + 
  theme_bw()
pl

# save the plot
pdf("time_lip_global_multi.pdf", width = 5, height = 4)
print(pl)
dev.off()