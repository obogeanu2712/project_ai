from ucimlrepo import fetch_ucirepo 
  
# fetch dataset 
post_operative_patient = fetch_ucirepo(id=82) 
  
# data (as pandas dataframes) 
X = post_operative_patient.data.features 
y = post_operative_patient.data.targets 
  
# metadata 
print(post_operative_patient.metadata) 
  
# variable information 
print(post_operative_patient.variables) 
