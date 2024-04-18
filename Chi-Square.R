# Import the data
file_path <- "http://www.sthda.com/sthda/RDoc/data/housetasks.txt"
housetasks <- read.delim(file_path, row.names = 1)
# head(housetasks)
install.packages("gplots")
library("gplots")
# 1. convert the data as a table
dt <- as.table(as.matrix(housetasks))
# 2. Graph
balloonplot(t(dt), main ="housetasks", xlab ="", ylab="",
            label = FALSE, show.margins = FALSE)
library("graphics")
mosaicplot(dt, shade = TRUE, las=2,
           main = "housetasks")

install.packages("vcd")
library("vcd")
# plot just a subset of the table
assoc(head(dt, 5), shade = TRUE, las=3)

chisq <- chisq.test(housetasks)
chisq
# Observed counts
chisq$observed
# Expected counts
round(chisq$expected,2)

round(chisq$residuals, 3)
install.packages("corrplot")
library(corrplot)
corrplot(chisq$residuals, is.cor = FALSE)

# Contibution in percentage (%)
contrib <- 100*chisq$residuals^2/chisq$statistic
round(contrib, 3)

# Visualize the contribution
corrplot(contrib, is.cor = FALSE)
