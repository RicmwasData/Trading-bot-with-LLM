# AI Trading Bot

This project implements an advanced AI trading bot that leverages **MetaTrader 5**, **LangChain**, and **Anthropic's Claude** for dynamic trading based on technical analysis from **Investing.com**. The bot fetches technical analysis data, processes it to determine trading signals, and executes trades on the MetaTrader 5 platform.

## Features

- **Dynamic Trade Execution**: The bot determines whether to buy, sell, or hold based on market conditions.
- **Stop Loss (SL) & Take Profit (TP)**: Automatically calculates SL and TP levels using resistance and support levels from technical analysis.
- **Data Processing with LangChain**: Processes unstructured data using LangChain components and Anthropic's Claude model.
- **MetaTrader 5 Integration**: Executes trades and monitors positions on MetaTrader 5.

## Data Flow

Below is a visual representation of the bot's data flow:
![Data Flow](<Data_flow.PNG>)

## Prerequisites

- Python 3.8 or higher
- MetaTrader 5 installed on your machine
- Accounts for:
  - MetaTrader 5 trading platform
  - Anthropic API
  - OpenAI API (optional for embeddings)
- Required Python libraries (install with `pip`):
  - `MetaTrader5`
  - `requests`
  - `beautifulsoup4`
  - `pandas`
  - `langchain`

## Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/RicmwasData/Trading-bot-with-LLM.git
   cd ai-trading-bot
