This folder contains the real dataset - Electricity described by M. Harries and analysed by Gama. 

Description from MOA webpage (source http://moa.cms.waikato.ac.nz/datasets):
This data was collected from the Australian New South Wales Electricity Market. 
In this market, prices are not fixed and are affected by demand and supply of the market. 
They are set every five minutes. The ELEC dataset contains 45, 312 instances. 
The class label identifies the change of the price relative to a moving average of the last 24 hours. 

Original dataset source: http://www.inescporto.pt/~jgama/ales/ales_5.html

Files:
elec.tar contains the original dataset from Gama's page above (not in arff format).

electricity.arff contains the full original dataset created from elec.tar including ignored attributes (date, nswprice).

electricity-remove-ignored.arff contains the electricity dataset excluding ignored attribute columns (date, nswprice).
 
electricity-normalized.arff is the normalized version of the dataset, so that the numerical values are between 0 and 1 including ignored attributes (date, nswprice).