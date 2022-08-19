#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()


# In[2]:


ppt=pd.read_csv(r"C:\Users\HP\Documents\Property_Price_Train.csv")
ppt


# In[3]:


ppt.describe(include="all")


# In[4]:


#ppt.head()


# In[5]:


#ppt.shape


# In[6]:


#ppt.columns


# In[7]:


ppt.info()


# # cleaning

# In[8]:


(pd.set_option("display.Max_rows",None))


# In[9]:


ppt.isnull().sum()


# In[10]:


for x in ppt.columns:
    if "Year" in x:
        ppt =  ppt.drop([x],axis=1)
        print(x,ppt.shape)


# In[11]:


for x in ["Lane_Type","Pool_Quality","Miscellaneous_Feature","Fence_Quality","Month_Sold","Lot_Configuration","Neighborhood","Sale_Type"]:
    ppt =  ppt.drop([x],axis=1)
    print(x,ppt.shape)


# In[12]:


#ppt.t_Extent.value_counts()Lo


# In[13]:


#d.Lane_Type.value_counts()


# In[14]:


#ppt.Brick_Veneer_Type.value_counts()


# In[15]:


#ppt.Pool_Area.value_counts()


# In[16]:


#ppt.Brick_Veneer_Area.value_counts()


# In[17]:


#ppt.Basement_Height.value_counts()


# In[18]:


#ppt.Basement_Condition.value_counts()


# In[19]:


#ppt.Basement_Condition.value_counts()


# In[20]:


#ppt.BsmtFinType1.value_counts()


# In[21]:


#ppt.BsmtFinType2.value_counts()


# In[22]:


#ppt.Electrical_System.value_counts()


# In[23]:


#ppt.Fireplace_Quality.value_counts()


# In[24]:


#ppt.Garage.value_counts()


# In[25]:


#ppt.Garage_Quality.value_counts() 


# In[26]:


#ppt.Garage_Condition.value_counts()


# In[27]:


#d.Pool_Quality.value_counts()  


# In[28]:


#d.Fence_Quality.value_counts()


# In[29]:


#d.Miscellaneous_Feature.value_counts()


# In[30]:


ppt.Lot_Extent=ppt.Lot_Extent.fillna(38.0)
ppt.Brick_Veneer_Type=ppt.Brick_Veneer_Type.fillna("BrkCmn")
ppt.Brick_Veneer_Area=ppt.Brick_Veneer_Area.fillna(0.0)
ppt.Basement_Height=ppt.Basement_Height.fillna("Fa")
ppt.Basement_Condition=ppt.Basement_Condition.fillna("Po")
ppt.Exposure_Level=ppt.Exposure_Level.fillna("Mn")
ppt.BsmtFinType1=ppt.BsmtFinType1.fillna("LwQ ")
ppt.BsmtFinType2=ppt.BsmtFinType2.fillna("GLQ")
ppt.Electrical_System=ppt.Electrical_System.fillna("Mix")
ppt.Fireplace_Quality=ppt.Fireplace_Quality.fillna("Po")
ppt.Garage=ppt.Garage.fillna("2Types")
ppt.Garage_Quality=ppt.Garage_Quality.fillna("Ex")
ppt.Garage_Condition=ppt.Garage_Condition.fillna("Ex")


# In[31]:


ppt.isnull().sum()


# In[32]:


ppt.dtypes


# # Data conversion Categorical to numerical

# In[33]:


from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()


# In[34]:


ppt.Zoning_Class=le.fit_transform(ppt.Zoning_Class)
ppt.Property_Shape=le.fit_transform(ppt.Property_Shape)
ppt.Land_Outline=le.fit_transform(ppt.Land_Outline)
ppt.Utility_Type=le.fit_transform(ppt.Utility_Type)
ppt.Property_Slope=le.fit_transform(ppt.Property_Slope)
ppt.Condition1=le.fit_transform(ppt.Condition1)
ppt.Condition2=le.fit_transform(ppt.Condition2)
ppt.House_Type=le.fit_transform(ppt.House_Type)
ppt.House_Design=le.fit_transform(ppt.House_Design)
ppt.Roof_Design=le.fit_transform(ppt.Roof_Design)
ppt.Roof_Quality=le.fit_transform(ppt.Roof_Quality)
ppt.Exterior1st=le.fit_transform(ppt.Exterior1st)
ppt.Exterior2nd=le.fit_transform(ppt.Exterior2nd)
ppt.Brick_Veneer_Type=le.fit_transform(ppt.Brick_Veneer_Type)
ppt.Exterior_Material=le.fit_transform(ppt.Exterior_Material)
ppt.Exterior_Condition=le.fit_transform(ppt.Exterior_Condition)
ppt.Foundation_Type=le.fit_transform(ppt.Foundation_Type)
ppt.Basement_Condition=le.fit_transform(ppt.Basement_Condition)
ppt.Basement_Height=le.fit_transform(ppt.Basement_Height)
ppt.Exposure_Level=le.fit_transform(ppt.Exposure_Level)
ppt.BsmtFinType1=le.fit_transform(ppt.BsmtFinType1)
ppt.BsmtFinType2=le.fit_transform(ppt.BsmtFinType2)
ppt.Heating_Type=le.fit_transform(ppt.Heating_Type)
ppt.Heating_Quality=le.fit_transform(ppt.Heating_Quality)
ppt.Air_Conditioning=le.fit_transform(ppt.Air_Conditioning)
ppt.Electrical_System=le.fit_transform(ppt.Electrical_System)
ppt.Kitchen_Quality=le.fit_transform(ppt.Kitchen_Quality)
ppt.Functional_Rate=le.fit_transform(ppt.Functional_Rate)
ppt.Fireplace_Quality=le.fit_transform(ppt.Fireplace_Quality)
ppt.Garage=le.fit_transform(ppt.Garage)
ppt.Garage_Quality=le.fit_transform(ppt.Garage_Quality)
ppt.Garage_Condition=le.fit_transform(ppt.Garage_Condition)
ppt.Pavedd_Drive=le.fit_transform(ppt.Pavedd_Drive)
ppt.Sale_Condition=le.fit_transform(ppt.Sale_Condition)
ppt.Road_Type=le.fit_transform(ppt.Road_Type)


# In[35]:


ppt.dtypes


# In[36]:


ppt.shape


# In[37]:


ppt.drop_duplicates(inplace=True)


# In[38]:


ppt.shape


# In[39]:


ppt1=ppt.copy()


# # Basic Model Building

# In[40]:


dr_x=ppt1.iloc[:,1:-1]
dr_y=ppt1.iloc[:,-1]


# In[41]:


import sklearn 
from sklearn.model_selection import train_test_split


# In[42]:


dr_xtrain,dr_xtest,dr_ytrain,dr_ytest=train_test_split(dr_x,dr_y,test_size=0.2,random_state=101)


# In[43]:


dr_xtrain.shape,dr_xtest.shape,dr_ytrain.shape,dr_ytest.shape


# In[44]:


from sklearn import linear_model 
ln=linear_model.LinearRegression()


# In[45]:


ln.fit(dr_xtrain,dr_ytrain)


# In[46]:


ln.intercept_


# In[47]:


ln.coef_


# In[48]:


rsq=ln.score(dr_xtrain,dr_ytrain)
rsq


# In[49]:


adjr=1-(((1-rsq)*(580-1))/(580-5-1))
adjr


# In[50]:


pred=ln.predict(dr_xtest)


# In[51]:


from sklearn import metrics


# In[52]:


ppt_mae=metrics.mean_absolute_error(dr_ytest,pred)
ppt_mae


# In[53]:


ppt_mse=metrics.mean_squared_error(dr_ytest,pred)
ppt_mse


# In[55]:


ppt_rmse=np.sqrt(ppt_mse)
ppt_rmse


# In[56]:


#MAPE  Mean Abosolute Error
error=dr_ytest-pred
error


# In[57]:


aerror=np.abs(error)
aerror


# In[58]:


mape=np.mean(aerror/dr_ytest)*100
mape


# In[59]:


accuracy=100-mape
accuracy


# In[60]:


ppt2=ppt.corr()
#ppt2


# In[61]:


plt.figure(figsize=(75,60))
heatmap=sns.heatmap(ppt2,linewidth=1,annot=True,cmap=plt.cm.Blues)
plt.title("Heatmap using Seaborn Method")
plt.show()


# # Removeing outlier

# In[62]:


ppt.boxplot(figsize=(20,10))


# In[63]:


ppt.boxplot(column='Building_Class')


# In[64]:


ppt.Building_Class.hist()


# In[65]:


ppt.Building_Class.describe()


# In[66]:


iqrB=ppt.Building_Class.quantile(0.75)-ppt.Building_Class.quantile(0.25)
iqrB


# In[68]:


lowerB=ppt['Building_Class'].quantile(0.25)-(1*iqrB)
upperB=ppt['Building_Class'].quantile(0.75)+(1*iqrB)
print(lowerB,upperB)


# In[70]:


new=ppt.copy()


# In[71]:


new.loc[new['Building_Class']>120,'Building_Class']=120


# In[72]:


ppt.boxplot(column='Lot_Extent')


# In[73]:


ppt.Lot_Extent.hist()


# In[74]:


ppt.Lot_Extent.describe()


# In[75]:


iqrLE=ppt.Lot_Extent.quantile(0.75)-ppt.Lot_Extent.quantile(0.25)
iqrLE


# In[76]:


lowerLE=ppt['Lot_Extent'].quantile(0.25)-(1*iqrLE)
upperLE=ppt['Lot_Extent'].quantile(0.75)+(1*iqrLE)
print(lowerLE,upperLE)


# In[77]:


ppt.loc[new['Lot_Extent']>112,'Lot_Extent']=112


# In[78]:


ppt.boxplot(column='Lot_Size')


# In[79]:


ppt.Lot_Size.hist()


# In[80]:


ppt.Lot_Size.describe()


# In[81]:


iqrLS=ppt.Lot_Size.quantile(0.75)-ppt.Lot_Size.quantile(0.25)
iqrLS


# In[82]:


lowerLS=ppt['Lot_Extent'].quantile(0.25)-(3*iqrLS)
upperLS=ppt['Lot_Extent'].quantile(0.75)+(3*iqrLS)
print(lowerLS,upperLS)


# In[83]:


new.loc[new['Lot_Size']>12241,'Lot_Size']=12241


# In[84]:


ppt.boxplot(column='House_Condition')


# In[85]:


ppt.House_Condition.hist()


# In[86]:


ppt.House_Condition.describe()


# In[87]:


iqrHC=ppt.House_Condition.quantile(0.75)-ppt.House_Condition.quantile(0.25)
iqrHC


# In[88]:


lowerHC=ppt['House_Condition'].quantile(0.25)-(1*iqrHC)
upperHC=ppt['House_Condition'].quantile(0.75)+(1*iqrHC)
print(lowerHC,upperHC)


# In[89]:


new.loc[new['House_Condition']>7,'House_Condition']=7


# In[90]:


ppt.boxplot(column='BsmtFinSF1')


# In[91]:


ppt.BsmtFinSF1.hist()


# In[92]:


ppt.BsmtFinSF1.describe()


# In[93]:


iqrBsF=ppt.BsmtFinSF1.quantile(0.75)-ppt.BsmtFinSF1.quantile(0.25)
iqrBsF


# In[94]:


lowerBsF=ppt['BsmtFinSF1'].quantile(0.25)-(1*iqrBsF)
upperBsF=ppt['BsmtFinSF1'].quantile(0.75)+(1*iqrBsF)
print(lowerBsF,upperBsF)


# In[95]:


new.loc[new['BsmtFinSF1']>1424,'BsmtFinSF1']=1424


# In[96]:


ppt.boxplot(column='BsmtFinSF2')


# In[97]:


ppt.BsmtFinSF2.hist()


# In[98]:


ppt.BsmtFinSF2.describe()


# In[99]:


iqrBsF2=ppt.BsmtFinSF2.quantile(0.75)-ppt.BsmtFinSF2.quantile(0.25)
iqrBsF2


# In[100]:


lowerBsF2=ppt['BsmtFinSF1'].quantile(0.25)-(1*iqrBsF2)
upperBsF2=ppt['BsmtFinSF1'].quantile(0.75)+(1*iqrBsF2)
print(lowerBsF2,upperBsF2)


# In[101]:


new.loc[new['BsmtFinSF2']>712,'BsmtFinSF2']=724


# In[102]:


ppt.boxplot(column='BsmtUnfSF')


# In[103]:


ppt.BsmtUnfSF.hist()


# In[104]:


ppt.BsmtUnfSF.describe()


# In[105]:


iqrBSF=ppt.BsmtUnfSF.quantile(0.75)-ppt.BsmtUnfSF.quantile(0.25)
iqrBSF


# In[106]:


lowerBSF=ppt['BsmtUnfSF'].quantile(0.25)-(1*iqrBSF)
upperBSF=ppt['BsmtUnfSF'].quantile(0.75)+(1*iqrBSF)
print(lowerBSF,upperBSF)


# In[107]:


new.loc[new['BsmtUnfSF']>1392,'BsmtUnfSF']=1392


# In[108]:


ppt.boxplot(column='Total_Basement_Area')


# In[109]:


ppt.Total_Basement_Area.hist()


# In[110]:


ppt.Total_Basement_Area.describe()


# In[111]:


iqrBA=ppt.Total_Basement_Area.quantile(0.75)-ppt.Total_Basement_Area.quantile(0.25)
iqrBA


# In[112]:


lowerBA=ppt['Total_Basement_Area'].quantile(0.25)-(1*iqrBA)
upperBA=ppt['Total_Basement_Area'].quantile(0.75)+(1*iqrBA)
print(lowerBA,upperBA)


# In[113]:


new.loc[new['Total_Basement_Area']>1801,'Total_Basement_Area']=1801


# In[114]:


ppt.boxplot(column='First_Floor_Area')


# In[115]:


ppt.First_Floor_Area.hist()


# In[116]:


ppt.First_Floor_Area.describe()


# In[117]:


iqrFFA=ppt.First_Floor_Area.quantile(0.75)-ppt.First_Floor_Area.quantile(0.25)
iqrFFA


# In[118]:


lowerFFA=ppt['First_Floor_Area'].quantile(0.25)-(1*iqrFFA)
upperFFA=ppt['First_Floor_Area'].quantile(0.75)+(1*iqrFFA)
print(lowerFFA,upperFFA)


# In[119]:


new.loc[new['First_Floor_Area']>1901,'First_Floor_Area']=1901


# In[120]:


ppt.boxplot(column='Second_Floor_Area')


# In[121]:


ppt.Second_Floor_Area.hist()


# In[122]:


ppt.Second_Floor_Area.describe()


# In[123]:


iqrFsA=ppt.Second_Floor_Area.quantile(0.75)-ppt.Second_Floor_Area.quantile(0.25)
iqrFsA


# In[124]:


lowerFsA=ppt['Second_Floor_Area'].quantile(0.25)-(1*iqrFsA)
upperFsA=ppt['Second_Floor_Area'].quantile(0.75)+(1*iqrFsA)
print(lowerFsA,upperFsA)


# In[125]:


new.loc[new['Second_Floor_Area']>1456,'Second_Floor_Area']=145


# In[126]:


ppt.boxplot(column='Grade_Living_Area')


# In[127]:


ppt.Grade_Living_Area.hist(bins=30)


# In[128]:


ppt.Grade_Living_Area.describe()


# In[129]:


iqrGLA=ppt.Grade_Living_Area.quantile(0.75)-ppt.Grade_Living_Area.quantile(0.25)
iqrGLA


# In[130]:


lowerGLA=ppt['Grade_Living_Area'].quantile(0.25)-(1*iqrGLA)
upperGLA=ppt['Grade_Living_Area'].quantile(0.75)+(1*iqrGLA)
print(lowerGLA,upperGLA)


# In[131]:


new.loc[new['Grade_Living_Area']>2426,'Grade_Living_Area']=2426


# In[132]:


ppt.boxplot(column='Bedroom_Above_Grade')


# In[133]:


ppt.Bedroom_Above_Grade.hist()


# In[134]:


ppt.Bedroom_Above_Grade.describe()


# In[136]:


upperBAG=ppt['Bedroom_Above_Grade'].mean()+2.5*ppt['Bedroom_Above_Grade'].std()
lowerBAG=ppt['Bedroom_Above_Grade'].mean()-2.5*ppt['Bedroom_Above_Grade'].std()
print(lowerBAG,upperBAG)


# In[137]:


new.loc[new['Bedroom_Above_Grade']>4.90,'Bedroom_Above_Grade']=4.90


# In[138]:


ppt.boxplot(column='Rooms_Above_Grade')


# In[139]:


ppt.Rooms_Above_Grade.hist(bins=10)


# In[140]:


ppt.Rooms_Above_Grade.describe()


# In[142]:


upperRAG=ppt['Rooms_Above_Grade'].mean()+2.5*ppt['Rooms_Above_Grade'].std()
lowerRAG=ppt['Rooms_Above_Grade'].mean()-2.5*ppt['Rooms_Above_Grade'].std()
print(lowerRAG,upperRAG)


# In[143]:


new.loc[new['Rooms_Above_Grade']>10.58,'Rooms_Above_Grade']=10.58


# In[144]:


ppt.boxplot(column='Garage_Area')


# In[145]:


ppt.Garage_Area.hist(bins=50)


# In[146]:


ppt.Garage_Area.describe()


# In[147]:


upperGA=ppt['Garage_Area'].mean()+2.5*ppt['Garage_Area'].std()
lowerGA=ppt['Garage_Area'].mean()-2.5*ppt['Garage_Area'].std()
print(lowerGA,upperGA)


# In[148]:


new.loc[new['Garage_Area']>997,'Garage_Area']=997


# In[149]:


ppt.boxplot(column='W_Deck_Area')


# In[150]:


ppt.W_Deck_Area.hist()


# In[151]:


ppt.W_Deck_Area.describe()


# In[152]:


upperGA=ppt['W_Deck_Area'].mean()+2.5*ppt['W_Deck_Area'].std()
lowerGA=ppt['W_Deck_Area'].mean()-2.5*ppt['W_Deck_Area'].std()
print(lowerGA,upperGA)


# In[153]:


new.loc[new['W_Deck_Area']>405.00,'W_Deck_Area']=405.00


# In[154]:


ppt.boxplot(column='Open_Lobby_Area')


# In[155]:


ppt.Open_Lobby_Area.hist()


# In[156]:


ppt.Open_Lobby_Area.describe()


# In[157]:


upperOLA=ppt['Open_Lobby_Area'].mean()+2.5*ppt['Open_Lobby_Area'].std()
lowerOLA=ppt['Open_Lobby_Area'].mean()-2.5*ppt['Open_Lobby_Area'].std()
print(lowerOLA,upperOLA)


# In[158]:


new.loc[new['Open_Lobby_Area']>216.48,'Open_Lobby_Area']=216.48


# In[159]:


ppt.boxplot(column='Open_Lobby_Area')


# In[160]:


ppt.Enclosed_Lobby_Area.hist()


# In[161]:


ppt.Enclosed_Lobby_Area.describe()


# In[162]:


uppereLA=ppt['Enclosed_Lobby_Area'].mean()+2.5*ppt['Enclosed_Lobby_Area'].std()
lowereLA=ppt['Enclosed_Lobby_Area'].mean()-2.5*ppt['Enclosed_Lobby_Area'].std()
print(lowereLA,uppereLA)


# In[163]:


new.loc[new['Enclosed_Lobby_Area']>177.96,'Enclosed_Lobby_Area']=177.96


# In[164]:


ppt.skew()


# In[166]:


new["Building_Class"]=np.sqrt(new["Building_Class"])


# In[167]:


new.Building_Class.skew()


# In[168]:


new["Lot_Extent"]=np.sqrt(new["Lot_Extent"])


# In[169]:


new.Lot_Extent.skew()


# In[170]:


new["Lot_Size"]=np.sqrt(new["Lot_Size"])


# In[172]:


new["Utility_Type"]=np.sqrt(new["Utility_Type"])


# In[173]:


new["Property_Slope"]=np.sqrt(new["Property_Slope"])


# In[174]:


new["Condition1"]=np.sqrt(new["Condition1"])


# In[175]:


new["Condition2"]=np.sqrt(new["Condition2"])


# In[176]:


new["House_Type"]=np.sqrt(new["House_Type"])


# In[177]:


new["Roof_Design"]=np.sqrt(new["Roof_Design"])


# In[178]:


new["Roof_Quality"]=np.sqrt(new["Roof_Quality"])


# In[179]:


new["Brick_Veneer_Area"]=np.sqrt(new["Brick_Veneer_Area"])


# In[180]:


new["BsmtFinSF2"]=np.sqrt(new["BsmtFinSF2"])


# In[181]:


new["Total_Basement_Area"]=np.sqrt(new["Total_Basement_Area"])


# In[182]:


new["Heating_Type"]=np.sqrt(new["Heating_Type"])


# In[183]:


new["First_Floor_Area"]=np.sqrt(new["First_Floor_Area"])


# In[184]:


new["Second_Floor_Area"]=np.sqrt(new["Second_Floor_Area"])


# In[185]:


new["Grade_Living_Area"]=np.sqrt(new["Grade_Living_Area"])


# In[186]:


new["Underground_Full_Bathroom"]=np.sqrt(new["Underground_Full_Bathroom"])


# In[187]:


new["Bedroom_Above_Grade"]=np.sqrt(new["Bedroom_Above_Grade"])


# In[189]:


new["Kitchen_Above_Grade"]=np.sqrt(new["Kitchen_Above_Grade"])


# In[190]:


new["Rooms_Above_Grade"]=np.sqrt(new["Rooms_Above_Grade"])


# In[191]:


new["Fireplaces"]=np.sqrt(new["Fireplaces"])


# In[192]:


new["Garage"]=np.sqrt(new["Garage"])


# In[193]:


new["Three_Season_Lobby_Area"]=np.sqrt(new["Three_Season_Lobby_Area"])


# In[194]:


new["Screen_Lobby_Area"]=np.sqrt(new["Screen_Lobby_Area"])


# In[195]:


new["Pool_Area"]=np.sqrt(new["Pool_Area"])


# In[196]:


new["Miscellaneous_Value"]=np.sqrt(new["Miscellaneous_Value"])


# In[197]:


new["Sale_Price"]=np.sqrt(new["Sale_Price"])


# In[198]:


##-


# In[199]:


new["Zoning_Class"]=np.square(new["Zoning_Class"])


# In[200]:


#new.Zoning_Class.skew()


# In[201]:


new["Road_Type"]=np.square(new["Road_Type"])


# In[202]:


new["Land_Outline"]=np.square(new["Land_Outline"])


# In[203]:


new["Brick_Veneer_Type"]=np.square(new["Brick_Veneer_Type"])


# In[204]:


new["Exterior_Material"]=np.square(new["Exterior_Material"])


# In[206]:


new["Exterior_Condition"]=np.square(new["Exterior_Condition"])


# In[207]:


new["Basement_Height"]=np.square(new["Basement_Height"])


# In[208]:


new["Basement_Condition"]=np.square(new["Basement_Condition"])


# In[209]:


new["Exposure_Level"]=np.square(new["Exposure_Level"])


# In[210]:


new["BsmtFinType2"]=np.square(new["BsmtFinType2"])


# In[211]:


new["Air_Conditioning"]=np.square(new["Air_Conditioning"])


# In[212]:


new["Electrical_System"]=np.square(new["Electrical_System"])


# In[213]:


new["Kitchen_Quality"]=np.square(new["Kitchen_Quality"])


# In[214]:


new["Functional_Rate"]=np.square(new["Functional_Rate"])


# In[215]:


new["Garage_Condition"]=np.square(new["Garage_Condition"])


# In[216]:


new["Pavedd_Drive"]=np.square(new["Pavedd_Drive"])


# In[217]:


new["W_Deck_Area"]=np.square(new["W_Deck_Area"])


# In[218]:


new["Sale_Condition"]=np.square(new["Sale_Condition"])


# In[219]:


#new.hist(grid=False,figsize=(25,16),bins=50)


# In[220]:


#new.isnull().sum()


# # Test data

# In[221]:


tt=pd.read_csv(r"C:\Users\HP\Documents\Property_Price_Test.csv")


# In[222]:


tt


# In[223]:


tt.shape


# In[224]:


tt.head()


# In[225]:


tt.isnull().sum()


# In[226]:


for x in tt.columns:
    if "Year" in x:
        tt =  tt.drop([x],axis=1)
        print(x,tt.shape)


# In[227]:


for x in ["Lane_Type","Pool_Quality","Miscellaneous_Feature","Fence_Quality","Month_Sold","Lot_Configuration","Neighborhood","Sale_Type"]:
    tt =  tt.drop([x],axis=1)
    print(x,tt.shape)


# In[228]:


tt.isnull().sum()


# In[229]:


#tt.Lot_Extent.value_counts()


# In[230]:


#tt.Zoning_Class.value_counts()


# In[231]:


#tt.Exterior1st.value_counts()


# In[232]:


#tt.Exterior2nd.value_counts()


# In[233]:


#tt.Basement_Height.value_counts()


# In[234]:


#tt.Basement_Condition.value_counts()


# In[235]:


#tt.Exposure_Level.value_counts()


# In[236]:


#tt.BsmtFinType1.value_counts()


# In[237]:


#tt.BsmtFinSF1.value_counts()


# In[238]:


#tt.BsmtFinType2.value_counts()


# In[239]:


#tt.BsmtFinType2.value_counts()


# In[240]:


#tt.BsmtFinSF2.value_counts()


# In[241]:


#tt.Total_Basement_Area.value_counts()


# In[242]:


#tt.Underground_Full_Bathroom.value_counts()


# In[243]:


#tt.Underground_Half_Bathroom.value_counts()


# In[244]:


#tt.Kitchen_Quality.value_counts()


# In[245]:


#tt.Functional_Rate.value_counts()


# In[246]:


#tt.Fireplace_Quality.value_counts()


# In[247]:


#tt.Garage.value_counts()


# In[248]:


#tt.Garage_Size.value_counts()


# In[249]:


#tt.Garage_Size.value_counts()


# In[250]:


#tt.Garage_Area.value_counts()


# In[251]:


#tt.Garage_Quality.value_counts()


# In[252]:


#tt.Garage_Condition.value_counts()


# In[253]:


#tt.Utility_Type.value_counts()


# In[254]:


#tt.Brick_Veneer_Type.value_counts()


# In[255]:


#tt.Brick_Veneer_Area.value_counts()


# In[256]:


#tt.BsmtFinSF1.value_counts()


# In[257]:


#tt.BsmtUnfSF.value_counts()


# In[258]:


tt.Lot_Extent=tt.Lot_Extent.fillna(25.0)
tt.Zoning_Class=tt.Zoning_Class.fillna("RHD ")
tt.Exterior1st=tt.Exterior1st.fillna("CB")
tt.Exterior2nd=tt.Exterior2nd.fillna("Stone")
tt.Basement_Height=tt.Basement_Height.fillna("Fa")
tt.Basement_Condition=tt.Basement_Condition.fillna("Po")
tt.Exposure_Level=tt.Exposure_Level.fillna("Mn")
tt.BsmtFinType1=tt.BsmtFinType1.fillna("LwQ ")
tt.BsmtFinType2=tt.BsmtFinType2.fillna("GLQ")
tt.BsmtFinSF2=tt.BsmtFinSF2.fillna(0.0)
tt.Total_Basement_Area=tt.Total_Basement_Area.fillna(0.0)
tt.Underground_Full_Bathroom=tt.Underground_Full_Bathroom.fillna(0.0)
tt.Underground_Half_Bathroom=tt.Underground_Half_Bathroom.fillna(0.0)
tt.Kitchen_Quality=tt.Kitchen_Quality.fillna("Fa")
tt.Functional_Rate=tt.Functional_Rate.fillna("MS")
tt.Fireplace_Quality=tt.Fireplace_Quality.fillna("Ex")
tt.Garage=tt.Garage.fillna("CarPort")
tt.Garage_Size=tt.Garage_Size.fillna(1.0)
tt.Garage_Area=tt.Garage_Area.fillna(0.0)
tt.Garage_Quality=tt.Garage_Quality.fillna("Po")
tt.Garage_Condition=tt.Garage_Condition.fillna("Ex")
tt.Utility_Type=tt.Utility_Type.fillna("AllPub")
tt.Brick_Veneer_Type=tt.Brick_Veneer_Type.fillna("BrkCmn")
tt.Brick_Veneer_Area=tt.Brick_Veneer_Area.fillna(0.0)
tt.BsmtFinSF1=tt.BsmtFinSF1.fillna(0.0)
tt.BsmtUnfSF=tt.BsmtUnfSF.fillna(0.0)


# In[259]:


tt.isnull().sum()


# In[260]:


tt.dtypes


# In[261]:


from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()


# # Data conversion categorical to numerical

# In[262]:


tt.Zoning_Class=le.fit_transform(tt.Zoning_Class)
tt.Road_Type=le.fit_transform(tt.Road_Type)
#tt.Lane_Type =le.fit_transform(tt..Lane_Type)
tt.Property_Shape=le.fit_transform(tt.Property_Shape)
tt.Land_Outline=le.fit_transform(tt.Land_Outline)
tt.Utility_Type=le.fit_transform(tt.Utility_Type)
#tt.Lot_Configuration=le.fit_transform(tt.Lot_Configuration)
tt.Property_Slope=le.fit_transform(tt.Property_Slope)
#tt.Neighborhood=le.fit_transform(tt.Neighborhood)
tt.Condition1=le.fit_transform(tt.Condition1)
tt.Condition2=le.fit_transform(tt.Condition2)
tt.House_Type=le.fit_transform(tt.House_Type)
tt.House_Design=le.fit_transform(tt.House_Design)
tt.Roof_Design=le.fit_transform(tt.Roof_Design)
tt.Roof_Quality=le.fit_transform(tt.Roof_Quality)
tt.Exterior1st=le.fit_transform(tt.Exterior1st)
tt.Exterior2nd=le.fit_transform(tt.Exterior2nd)
tt.Brick_Veneer_Type=le.fit_transform(tt.Brick_Veneer_Type)
tt.Exterior_Material=le.fit_transform(tt.Exterior_Material)
tt.Exterior_Condition=le.fit_transform(tt.Exterior_Condition)
tt.Foundation_Type=le.fit_transform(tt.Foundation_Type)
tt.Basement_Condition=le.fit_transform(tt.Basement_Condition)
tt.Basement_Height=le.fit_transform(tt.Basement_Height)
tt.Exposure_Level=le.fit_transform(tt.Exposure_Level)
tt.BsmtFinType1=le.fit_transform(tt.BsmtFinType1)
tt.BsmtFinType2=le.fit_transform(tt.BsmtFinType2)
tt.Heating_Type=le.fit_transform(tt.Heating_Type)
tt.Heating_Quality=le.fit_transform(tt.Heating_Quality)
tt.Air_Conditioning=le.fit_transform(tt.Air_Conditioning)
tt.Electrical_System=le.fit_transform(tt.Electrical_System)
tt.Kitchen_Quality=le.fit_transform(tt.Kitchen_Quality)
tt.Functional_Rate=le.fit_transform(tt.Functional_Rate)
tt.Fireplace_Quality=le.fit_transform(tt.Fireplace_Quality)
tt.Garage=le.fit_transform(tt.Garage)
tt.Garage_Quality=le.fit_transform(tt.Garage_Quality)
tt.Garage_Condition=le.fit_transform(tt.Garage_Condition)
tt.Pavedd_Drive=le.fit_transform(tt.Pavedd_Drive)
tt.Sale_Condition=le.fit_transform(tt.Sale_Condition)


# In[263]:


tt.dtypes


# # Model Building 1.Linear Model

# In[264]:


d_x=new.iloc[:,1:-1]
d_y=new.iloc[:,-1]


# In[265]:


import sklearn 
from sklearn.model_selection import train_test_split


# In[266]:


d_xtrain,d_xtest,d_ytrain,d_ytest=train_test_split(d_x,d_y,test_size=0.2,random_state=101)


# In[267]:


d_xtrain.shape,d_xtest.shape,d_ytrain.shape,d_ytest.shape


# In[268]:


from sklearn import linear_model 
ln=linear_model.LinearRegression()


# In[269]:


ln.fit(d_xtrain,d_ytrain)


# In[270]:


ln.coef_


# In[271]:


R_1=ln.score(d_xtrain,d_ytrain)
R_1


# In[272]:


adj_R_1=1-(((1-rsq)*(580-1))/(580-5-1))
adj_R_1


# In[273]:


pred1=ln.predict(d_xtest)
pred1


# In[274]:


from sklearn import metrics


# In[275]:


d_mae=metrics.mean_absolute_error(d_ytest,pred1)
d_mae


# In[276]:


MSE_1=metrics.mean_squared_error(d_ytest,pred1)
MSE_1


# In[277]:


d_rmse=np.sqrt(MSE_1)
d_rmse


# In[278]:


#MAPE  Mean Abosolute Error
error=d_ytest-pred1
error


# In[279]:


aerror=np.abs(error)
aerror


# In[280]:


mape=np.mean(aerror/d_ytest)*100
mape


# # 2.Lasso model

# In[281]:


from sklearn.linear_model import Lasso
lasso=Lasso()


# In[282]:


lasso.fit(d_xtrain,d_ytrain)


# In[284]:


pred2=lasso.predict(d_xtest)
#pred2


# In[285]:


lasso.coef_


# In[286]:


R_2=lasso.score(d_xtrain,d_ytrain)
R_2


# In[287]:


adj_R_2=1-(((1-R_2)*(580-1))/(580-5-1)) 
adj_R_2 


# In[288]:


error1=d_ytest-pred2
error


# In[289]:


aerror1=np.abs(error1)
aerror1


# In[290]:


mape_2=np.mean(aerror1/d_ytest)*100
mape_2


# In[291]:


MSE_2=metrics.mean_squared_error(d_ytest,pred2)
MSE_2


# # 3.Ridge Model

# In[292]:


from sklearn.linear_model import Ridge
ridge=Ridge()


# In[293]:


ridge.fit(d_xtrain,d_ytrain)


# In[294]:


pred3=ridge.predict(d_xtest)


# In[295]:


R_3=ridge.score(d_xtrain,d_ytrain)
R_3


# In[296]:


adj_R_3=1-(((1-R_3)*(580-1))/(580-5-1)) 
adj_R_3


# In[297]:


error2=d_ytest-pred3
error2


# In[298]:


aerror2=np.abs(error2)
aerror2


# In[299]:


mape_3=np.mean(aerror2/d_ytest)*100
mape_3


# In[300]:


MSE_3=metrics.mean_squared_error(d_ytest,pred3)
MSE_3


# In[301]:


l1=["linear","lasso","ridge"]
l2=[R_1,R_2,R_3]
l3=[adj_R_1,adj_R_2,adj_R_3]


# In[302]:


final_df=pd.DataFrame({"model_name":l1,"R-squre":l2,"ADJ_R":l3})
final_df


# In[ ]:




