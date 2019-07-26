import pandas as pd
import numpy as np
import datetime as dt
import pdb
# identify optimal stop loss
# uses as input: trade-level data with maxDowns and max runups
# sets take profit at 1.5 times the stop loss
# computation uses only price pnl and not rebates.

# Input - trades dataframe:
# maxGrossExp and maxDown expected to be +ve
# trades = trades.rename(columns={"max_liability":"maxGrossExp", "max_drawdown":"maxDrawdown", "max_runup":"maxRunUp", "PnL_t":"totalPnl"})


def normalise(x):
    return (x - min(x)) / (max(x) - min(x))

def getOptimalStopLossAndTakeProfit(trades):

    stopLoss = []
    vec = []
    left = []
    numTradesCutVector = []
    totalTimeInMarketLeft = []
    winLossRatioAll = []
    consistencyRatioAll = []
    pnlMetric = []
    medianDown = []
    
    nTrades = trades.shape[0]

    trades.totalPnlDollars = trades.totalPnl
    
    trades.totalPnl = trades.totalPnl * 100 / trades.maxGrossExp
    trades.maxDown = trades.maxDown * 100 / trades.maxGrossExp

    # loop over all possible values for the stop loss

    for sl in np.arange(min(trades.maxDown), max(trades.maxDown)+3, 2):
    
        # trades to the left of the SL line
        sub = trades.iloc[np.where(trades.maxDown <= sl)[0]]
        sub = sub[["date", "totalPnl"]]
        
        # trades to the right of the SL line.
        rightSub = trades.iloc[np.where(trades.maxDown > sl)[0]]
        if rightSub.shape[0] > 0:
            rightSub.totalPnl = -1*sl  # Truncate pnl of trades to the right at the SL line.
            rightSub = rightSub[["date", "totalPnl"]]

        pnlDf = pd.concat([sub, rightSub], axis = 0)
        pnlLeft = np.sum(trades.totalPnl[trades.maxDown <= sl])
        L = sum(trades.maxDown > sl)
        pnlRight = L * sl * -1

        # compute metrics for this level of SL

        pctWinners = len(trades.totalPnl[(trades.maxDown <= sl) & (trades.totalPnl > 0)]) / float(nTrades)
        winLossRatio = pctWinners / (1 - pctWinners)
        
        medianDown.append(np.nanmedian(np.concatenate((np.array(trades.maxDown[trades.maxDown <= sl]), sl*np.ones(L)), axis = 0)))

        winningTrades = pctWinners * nTrades
        avgWinningTrade = np.nanmean(trades.totalPnl[(trades.maxDown <= sl) & (trades.totalPnl > 0)])
        avgLosingTrade = np.nanmean(np.concatenate((np.array(trades.totalPnl[(trades.maxDown <= sl) & (trades.totalPnl < 0)]), -sl*np.ones(L)), axis = 0))

        consistencyRatio = (winningTrades / (nTrades - winningTrades)) * np.abs(avgWinningTrade / avgLosingTrade)
        consistencyRatioAll.append(consistencyRatio)

        winLossDays = pd.DataFrame(pnlDf.groupby(['date'])['totalPnl'].aggregate(sum))
        numWinningDays = sum(winLossDays.totalPnl > 0)
        numLosingDays = sum(winLossDays.totalPnl < 0)

        if (np.nansum(pnlDf.totalPnl) < 0) or (numWinningDays < numLosingDays):
            sign = -1.0
        else:
            sign = 1.0

        pnlMetric.append(np.nansum(pnlDf.totalPnl) * abs(numWinningDays-numLosingDays) * sign)

        winLossRatioAll.append(winLossRatio)

        totalPnl = pnlRight + pnlLeft 
        vec.append(totalPnl)
        numTradesCutVector.append(L)
        stopLoss.append(sl)
        left.append(pnlLeft)
        totalTimeInMarketLeft.append(np.nansum(trades.timeInMarketSeconds[trades.maxDown <= sl]))
        
    # make sure that the stop loss does not cut out more than 15% of the trades
    # choosing an optimal stop loss by optimising a combination of metrics.
    df = pd.DataFrame([vec, stopLoss, numTradesCutVector, left, totalTimeInMarketLeft, winLossRatioAll, consistencyRatioAll, pnlMetric, medianDown]).transpose()
    df.columns = ['vec', 'stopLoss', 'numTradesCutVector', 'left', 'totalTimeInMarketLeft', 'winLossRatioAll', 'consistencyRatioAll', 'pnlMetric', 'medianDown']
    df['optimalNo'] = normalise(df.consistencyRatioAll / df.medianDown) + normalise(df.pnlMetric)

    df.pctTradesCutVector = df.numTradesCutVector / nTrades
    slThreshold = np.amin(df.stopLoss[df.pctTradesCutVector <= 0.15])
    df = df.iloc[np.where(stopLoss >= slThreshold)]

    if df.shape[0] != 0:
        sl = float(max(df.stopLoss[df.optimalNo == np.nanmax(df.optimalNo)]))
    else:
        sl = float(max(trades.maxDown) + 1)
    
    tp = max(1.5*sl, np.amax(trades.maxRunUp))
    return [sl*-1, tp]