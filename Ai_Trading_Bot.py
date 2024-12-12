
from langchain_unstructured import UnstructuredLoader
import asyncio
from langchain_anthropic import ChatAnthropic
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_anthropic import AnthropicLLM
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
import MetaTrader5 as mt5
import time
import json
import requests
from bs4 import BeautifulSoup
import pandas as pd

# Define the URL
symbol= "EURUSDz"
api_key= "XXXXX"

url = "https://www.investing.com/technical/technical-analysis"  
id= 'id'
name= 'curr_table'


def get_suppots(url, id, name):

    # Step 1: Fetch the webpage
    response = requests.get(url)

    # Step 2: Parse the webpage content
    soup = BeautifulSoup(response.content, 'html.parser')

    # Step 3: Find the table
    table = soup.find('table', {id: name})  # Replace with actual table ID or class
    if not table:
        print("Table not found!")
        exit()

    # Step 4: Extract table headers
    headers = [header.text.strip() for header in table.find_all('th')]

    # Step 5: Extract table rows
    rows = []
    for row in table.find_all('tr')[1:]:  # Skip the header row
        cells = [cell.text.strip() for cell in row.find_all(['td', 'th'])]
        rows.append(cells)

    # Step 6: Create a Pandas DataFrame
    df = pd.DataFrame(rows, columns=headers)
    return df


# Async function to process the loader
async def load_documents():
    loader = UnstructuredLoader(web_url=url)
    docs = []
    async for doc in loader.alazy_load():
        docs.append(doc)
    return docs

#### Buy and sell functions #####

def place_buy(symbol, lot, sl, tp):
    symbol_info = mt5.symbol_info(symbol)
    if symbol_info is None:
        print(symbol, "not found, can not call order_check()")
        return None
    
    lot = lot
    point = mt5.symbol_info(symbol).point
    price = mt5.symbol_info_tick(symbol).ask
    deviation = 20
    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": symbol,
        "volume": lot,
        "type": mt5.ORDER_TYPE_BUY,
        "price": price,
        "sl": sl,
        "tp": tp,
        "deviation": deviation,
        "magic": 234000,
        "comment": "python script open",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC,
    }
    
    # send a trading request
    result = mt5.order_send(request)
    if result.retcode != mt5.TRADE_RETCODE_DONE:
        print(f"The order failed with {result.retcode}")
        return None
    
    return result.order

def place_sell(symbol, lot, sl, tp):
    symbol_info = mt5.symbol_info(symbol)
    if symbol_info is None:
        print(symbol, "not found, can not call order_check()")
        return None
    
    lot = lot
    point = mt5.symbol_info(symbol).point
    price = mt5.symbol_info_tick(symbol).ask
    deviation = 20
    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": symbol,
        "volume": lot,
        "type": mt5.ORDER_TYPE_SELL,
        "price": price,
        "sl":  sl,
        "tp": tp,
        "deviation": deviation,
        "magic": 234000,
        "comment": "python script open",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC,
    }
    
    # send a trading request
    result = mt5.order_send(request)
    if result.retcode != mt5.TRADE_RETCODE_DONE:
        print(f"The order failed with {result.retcode}")
        return None
    
    return result.order

# function to check the trades 
def check_trade_outcome(order_ticket):
    deals = mt5.history_deals_get(position=order_ticket)
    if deals:
        for deal in deals:
            if deal.reason == mt5.ORDER_REASON_SL:
                return True
            elif deal.reason == mt5.ORDER_REASON_TP:
                return True

    orders = mt5.history_orders_get(position=order_ticket)
    if orders:
        for order in orders:
            if order.reason == mt5.ORDER_REASON_SL:
                return True
            elif order.reason == mt5.ORDER_REASON_TP:
                return True

    return False


    
system_prompt =(
        '''You are an advanced trading assistant. You are provided with hourly technical analysis for the currency pair EUR/USD from Investing.com. The analysis includes the following:
Current Price: The latest market price of the pair.
Summary of Indicators: A consolidated overview of technical indicators such as moving averages, oscillators, and their recommended actions (e.g., buy, sell, or neutral).
Pivot Points: Key resistance and support levels, which serve as potential stop-loss (sl) or take-profit (tp) levels based on the trade direction.
Your task is to determine:

Direction: Whether to trade in the "Buy" or "Sell" direction based on the provided indicators and summary.
Current_Price: The current market price of the pair.
Stop Loss (sl): The best support level relative to the {start_price} if the trade direction is "Buy," or the best resistance level relativeto {start_price} if the direction is "Sell."
Take Profit (tp): The best resistance level relative to {start_price} if the trade direction is "Buy," or the best support level relative to {start_price} if the direction is "Sell."
the support levele and the the resistance levels are provided as {df} where the supports are labled as S3 S2 and S1, the resistant levles are
labeled as R1, R2 and R3. The sl or the tp choosen must not be sell than 10 pips from the {start_price} Also give a reason as to why you choose the tp and sl. 
You must always provide the response in the following JSON dictionary format:

    "Direction": "Sell",
    "Current_Price":{start_price} ,
    "sl": XXXXX,
    "tp" XXXXX:,
    "Reason":XXXXX
No additional text or explanations should be included in your response.
 "\n\n"
 "{context}"

'''
    )

def load_website(documents):
    llm = ChatAnthropic(model="claude-3-5-sonnet-20240620", api_key=api_key)
    openai_api="openai_apikey"
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(documents)
    vectorstore = InMemoryVectorStore.from_documents(
        documents=splits, embedding=OpenAIEmbeddings(api_key=openai_api)
    )

    retriever = vectorstore.as_retriever()

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "{input}"),
        ]
    )

    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)


    start_price=  mt5.symbol_info_tick(symbol).ask
    df= get_suppots(url=url, id=id, name=name)
    print(df)
    results = rag_chain.invoke({"start_price": start_price,
                        "df": df,
                        "input": "What trading action (Buy, Sell, Hold) is suggested based on the article,and also the key indicators for the signal pick the sl and tp using the support and the resistance levels",
                        })
    answer= json.loads(results['answer'])
    answer["sl"] = float(answer.get("sl", 0.0))  # Default to 0.0 if "sl" is missing
    answer["tp"] = float(answer.get("tp", 0.0))  # Default to 0.0 if "tp" is missing


    return answer




# Run the async function
if __name__ == "__main__":

    #initialize the bot 
    # display data on the MetaTrader 5 package
    print("MetaTrader5 package author: ",mt5.__author__)
    print("MetaTrader5 package version: ",mt5.__version__)
    
    # establish MetaTrader 5 connection to a specified trading account
    if not mt5.initialize(path="C:/Program Files/MetaTrader 5/terminal64.exe",login=812, server="ExnessKE-MT5Trial10",
                        password="password*"):
        print("initialize() failed, error code =",mt5.last_error())
        quit()

    
    order_ticket= None

    while True:
        positions=mt5.positions_get(symbol=symbol)

        if len(positions)==0:
            documents= asyncio.run(load_documents())
            print(documents)
            answer= load_website(documents= documents )
            print(answer)

            if order_ticket is None or order_ticket == 0:
                current_price = mt5.symbol_info_tick(symbol).ask 
                if answer['Direction']=="Buy":
                    order_ticket= place_buy(symbol=symbol, lot=0.01, sl= answer['sl'], tp= answer['tp']) # palce a buy order.
                    print(f"Buy order placed order_ticket {order_ticket} price % {current_price}")
                elif  answer['Direction']=="Sell":
                    order_ticket= place_sell(symbol=symbol, lot=0.01,sl= answer['sl'], tp= answer['tp'] ) # palce a sell order.
                    print(f"sell order placed order_ticket  {order_ticket} price % {current_price}")
                else:
                    print(f"No trading signal.")
            else:
                if check_trade_outcome(order_ticket):
                    order_ticket=None
                    
                    print("Order_ticket & start_price updated,")
            
        # else:
        #     print("We have an position")
        time.sleep(3)