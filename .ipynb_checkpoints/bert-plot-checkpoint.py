import pandas as pd
import matplotlib.pyplot

df = pd.read_csv("./bert-results.csv")
print(df.columns)




matplotlib.pyplot.figure()
matplotlib.pyplot.plot(df['Step'] , df['Training Loss'] , color = 'violet' , label = 'Training loss')
matplotlib.pyplot.plot(df['Step'] , df['Validation Loss'] , color = 'black' , label = 'Validation loss')
matplotlib.pyplot.plot(df['Step'] , df['Precision'] , color = 'yellow' , label = 'Precision')
matplotlib.pyplot.plot(df['Step'] , df['Recall'] , color = 'blue' , label = 'Recall')
matplotlib.pyplot.plot(df['Step'] , df['F1'] , color = 'violet' , label = 'F1 score')
matplotlib.pyplot.plot(df['Step'] , df['Accuracy'] , color = 'violet' , label = 'Accuracy')
matplotlib.pyplot.legend(loc = "lower left" , prop={'size': 8})
matplotlib.pyplot.xlabel("Step value")
matplotlib.pyplot.ylabel("Value")
matplotlib.pyplot.title("Variations in values of different metrices throughout model iterations" , size = 14)
matplotlib.pyplot.show()

matplotlib.pyplot.figure()
matplotlib.pyplot.plot(df['Step'] , df['Training Loss'] , color = 'violet' , label = 'Training loss')
matplotlib.pyplot.xlabel("Step value")
matplotlib.pyplot.ylabel("Value")
matplotlib.pyplot.title("Training loss value throughout model iterations" , size = 14)
matplotlib.pyplot.show()



matplotlib.pyplot.figure()
matplotlib.pyplot.plot(df['Step'] , df['Validation Loss'] , color = 'black' , label = 'Validation loss')
matplotlib.pyplot.xlabel("Step value")
matplotlib.pyplot.ylabel("Value")
matplotlib.pyplot.title("Validation loss value throughout model iterations" , size = 14)
matplotlib.pyplot.legend()
matplotlib.pyplot.show()


matplotlib.pyplot.figure()
matplotlib.pyplot.plot(df['Step'] , df['Precision'] , color = 'yellow' , label = 'Precision')
matplotlib.pyplot.xlabel("Step value")
matplotlib.pyplot.ylabel("Value")
matplotlib.pyplot.title("Precision throughout model iterations" , size = 14)
matplotlib.pyplot.legend()
matplotlib.pyplot.show()



matplotlib.pyplot.figure()
matplotlib.pyplot.plot(df['Step'] , df['Recall'] , color = 'blue' , label = 'Recall')
matplotlib.pyplot.xlabel("Step value")
matplotlib.pyplot.ylabel("Value")
matplotlib.pyplot.title("Recall throughout model iterations" , size = 14)
matplotlib.pyplot.legend()
matplotlib.pyplot.show()



matplotlib.pyplot.figure()
matplotlib.pyplot.plot(df['Step'] , df['F1'] , color = 'violet' , label = 'F1 score')
matplotlib.pyplot.xlabel("Step value")
matplotlib.pyplot.ylabel("Value")
matplotlib.pyplot.title("F1 score throughout model iterations" , size = 14)
matplotlib.pyplot.legend()
matplotlib.pyplot.show()



matplotlib.pyplot.figure()
matplotlib.pyplot.plot(df['Step'] , df['Accuracy'] , color = 'beige' , label = 'Accuracy')
matplotlib.pyplot.xlabel("Step value")
matplotlib.pyplot.ylabel("Value")
matplotlib.pyplot.title("Variations in accuracy throughout model iterations" , size = 14)
matplotlib.pyplot.legend()
matplotlib.pyplot.show()