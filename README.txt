# ZenTrade
######################################################################################################
Description:
ZenTrade is a streamlined application that allows for traders with limited experience to easily access
different technical indicators and analyze markets more efficiently. This application will provide the user
with useful technical information and prioritize which indicators to keep track of. The application will
also make it easier for the user to analyze stocks in over-looked markets and improve the user’s portfolio
diversification. Additionally, the user will be able to create a custom portfolio to analyze potential gains.
This app will not make direct recommendations to the user nor will it serve as a brokerage account.
######################################################################################################
How To Run ZenTrade:
*Use file "ZenTrade.py" to run application. Make sure cutCsv.py and Stock_Data are in the same folder as ZenTrade.py
######################################################################################################
Necessary Modules (can be installed through the windows command prompt)

#GIU
pip install tkinter

#graphics
pip install matplotlib

#data manipulation & computations
pip install pandas
pip install numpy

#finance data source
pip install quandl

#advanced trading algorithms
pip install sklearn
pip install ta

#Others
pip install datetime


Optional: quandl.ApiConfig.api_key = "5hP1qvLqKeLFqAHnA9yG"

**Stock data may take a few seconds to download and process from online
#########################################################################################################
Other Information:
The Stock_Data folder will hold csv files of stock data that are downloaded from the quandl api.
