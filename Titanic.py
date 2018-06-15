import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np

#Reading the csv Titanic data file.
CurrentDir = os.getcwd()
filename = os.path.join(CurrentDir,"titanic_data.csv")
titanic_df = pd.read_csv(filename)


#Calculating survival rate
Survival = titanic_df.groupby("Survived").size().reset_index(name="count")
Survival["count"] /= (Survival["count"].sum() / 100)
SurvivedPre=Survival["count"][0]
DiedPre=Survival["count"][1]

#Plotting survival rate with a pie chart
plt.suptitle("Titanic Survival Rate")
plt.pie([SurvivedPre,DiedPre], labels=["Survived","Died"],autopct = "%1.2f%%", shadow =True)
plt.show()

#Calculating survival rate by genders
SurviversGender_df = titanic_df.groupby(["Survived", "Sex"]).size().reset_index(name="count")
SurvivedGender_df = SurviversGender_df[0:2]
SurvivedGender = SurvivedGender_df["count"] / SurvivedGender_df["count"].sum() * 100
DiedGender_df = SurviversGender_df[2:4].reset_index()
DiedGender = DiedGender_df["count"] / DiedGender_df["count"].sum() * 100

#Plotting survival rate by genders, bar chart
p1 = plt.bar(np.arange(2),(SurvivedGender[0],DiedGender[0]), width=(0.3,0.3))
p2 = plt.bar(np.arange(2),(SurvivedGender[1],DiedGender[1]), width=(0.3,0.3), bottom=(SurvivedGender[0],DiedGender[0]))
#Adding values to bar
plt.text(s=round(SurvivedGender[0],2), x=-0.05, y=5)
plt.text(s=round(SurvivedGender[1],2), x=-0.05, y=60)
plt.text(s=round(DiedGender[0],2), x=0.95, y=30)
plt.text(s=round(DiedGender[1],2), x=0.95, y=80)

plt.title('Survival By Gender')
plt.xticks(np.arange(2), ('Survived', 'Died'))
plt.legend((p1[0], p2[0]), ('Female', 'Male'))
plt.show()

#Calculating ticket prices info
print("The average ticket price is: {0:.2f}".format(titanic_df["Fare"].mean()))
print("The maximum ticket price is: {0:.2f}".format(titanic_df["Fare"].max()))
print("The minimal ticket price is: {0:.2f}".format(titanic_df["Fare"].min()))

#Standardize Data Survival chances of families vs. singles

def StandardizeData(values):
    return ((values - values.mean()) / values.std())

titanic_df["relationship"] = titanic_df["Parch"] + titanic_df["SibSp"]
Relationships = titanic_df.groupby(["Survived", "relationship"]).size().reset_index(name="count")
Relationships["count"] = StandardizeData(Relationships["count"])
SurvivedRelationship = Relationships.loc[Relationships["Survived"] == 0][["relationship","count"]].set_index("relationship")
DiedRelationship = Relationships.loc[Relationships["Survived"] == 1][["relationship","count"]].set_index("relationship")

plt.plot(SurvivedRelationship,label = "Survived")
plt.plot(DiedRelationship,label = "Died")

plt.title('Standard Deviation of Survival Rate of Families vs. Singles')
plt.legend()
plt.show()





