import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df=pd.read_csv(r"C:\Users\YASH\Desktop\pandas\t20-world-cup-22.csv")

df.head()
df.isnull().sum()

##Number of Matches Won by teams in t20 World Cup 2022
sns.countplot(x=df.winner)
plt.xticks(rotation=90)

##Number of Matches won by runs or wickets
wp=df['won by'].value_counts()
plt.pie(wp,labels=wp.index,autopct='%.0f%%')

##Toss decisons in t20 world cup 2022
td=df['toss decision'].value_counts()
plt.pie(td,labels=td.index,autopct='%.0f%%')

##Top scorer in t20 world cup
sns.barplot(x=df['top scorer'],y=df['highest score'])
plt.xticks(rotation=90)

##Player of the Match Awards in t20 World Cup 2022
sns.countplot(x=df['player of the match'])
plt.xticks(rotation=90)

##Best Bowlers in t20 World Cup 2022
sns.countplot(x=df['best bowler'])
plt.xticks(rotation=90)












