from nltk.stem import PorterStemmer,RegexpStemmer
# words=["eating","eaten","eat","finally","finalize","goes","gone","going","writing","writes","programming","programs"]
# stemming=PorterStemmer()
# for word in words:
#     print(word+"------>"+stemming.stem(word))
regStemmer=RegexpStemmer("ing|s$|e$|ed$|es$")
print(regStemmer.stem("ingeating"))