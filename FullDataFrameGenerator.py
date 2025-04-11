import pandas as pd
import numpy as np

years = ["05-06", "06-07", "07-08", "08-09", "09-10", "10-11", "11-12", "12-13", "13-14", "14-15", "15-16", "16-17", "17-18", "18-19", "19-20","20-21","21-22","22-23","23-24"]

def goalDifferentialToHomeWinOrLose(gd):
    if gd < 0:
        return False
    if gd > 0:
        return True

def modifyGameData(year):
    gameData = pd.read_csv(f"Datasets\\{year}\\GD{year}.csv")
    goalDifferentialsVector = gameData.loc[:, "GH"] - gameData.loc[:, "GV"]
    goalDifferentialsVector = goalDifferentialsVector.apply(goalDifferentialToHomeWinOrLose)
    yearVector = pd.DataFrame(index=np.arange(goalDifferentialsVector.shape[0]), columns=np.arange(1))
    yearVector.columns = ["year"]
    yearVector["year"] = year
    newDf = pd.concat([yearVector, gameData.loc[:, "Home"], gameData.loc[:, "Visitor"], goalDifferentialsVector], axis=1)
    newDf = newDf.rename(columns={0:"HomeWin"})
    return newDf

def concatGameData():
    dataframes = [None for x in years]
    for year in years:
        dataframes[years.index(year)] = modifyGameData(year)
    
    totalGameData = pd.concat(dataframes, ignore_index=True)
    totalGameData = totalGameData.dropna()
    return totalGameData

def modifyTeamData(year):
    allTeamStats = pd.read_csv(f"Datasets\\{year}\\TS{year}.csv")
    allTeamStats = allTeamStats.sort_values("TeamName", ignore_index=True)
    allTeamStats = allTeamStats.drop(f"S%", axis=1)
    allTeamStats = allTeamStats.drop(f"SV%", axis=1)

    yearVector = pd.DataFrame(index=np.arange(allTeamStats.shape[0]), columns=np.arange(1))
    yearVector.columns = ["year"]
    yearVector["year"] = year

    combinedTeamsData = pd.concat([yearVector, allTeamStats], axis=1)
    combinedTeamsData.drop_duplicates()
    return combinedTeamsData

def concatTeamData():
    dataframes = [None for x in years]
    for year in years:
        dataframes[years.index(year)] = modifyTeamData(year)
    
    totalTeamsData = pd.concat(dataframes, ignore_index=True)
    totalTeamsData = totalTeamsData.dropna()
    return totalTeamsData

def combineTeamDataAndGameData(gameData, teamsData):
    playingTeamsDataByGame = pd.DataFrame()

    for i in range(0, gameData.shape[0]):
        print(f"Game# {i}")
        game = gameData.iloc[i, :]
        gameDf = game.to_frame().T
        gameDf = gameDf.drop(columns=["year", "Home", "Visitor"])
        gameDf = gameDf.reset_index(drop=True)

        year = game.loc["year"]
        teamDataByYear = teamsData.loc[teamsData["year"] == year, :]

        homeTeam = game.loc["Home"]
        homeTeamData = teamDataByYear.loc[teamDataByYear["TeamName"] == homeTeam, :]
        homeTeamData.columns = [col + "_home" for col in homeTeamData.columns]
        homeTeamData = homeTeamData.drop(columns=["year_home"], axis=1)
        homeTeamData = homeTeamData.reset_index(drop=True)

        visTeam = game.loc["Visitor"]
        visTeamData = teamDataByYear.loc[teamDataByYear["TeamName"] == visTeam, :]
        visTeamData.columns = [col + "_visitor" for col in visTeamData.columns]
        visTeamData = visTeamData.drop("year_visitor", axis=1)
        visTeamData = visTeamData.reset_index(drop=True)

        playingTeamsDataCombined = pd.concat([homeTeamData, visTeamData, gameDf], axis=1)
        if playingTeamsDataByGame.empty:
            playingTeamsDataByGame = playingTeamsDataCombined
        else:
            playingTeamsDataByGame = pd.concat([playingTeamsDataByGame, playingTeamsDataCombined], ignore_index=True)
    
    playingTeamsDataByGame = playingTeamsDataByGame.drop(columns=["TeamName_home","TeamName_visitor"])

    playingTeamsDataByGame.to_csv("FinalDataSet.csv", index=False)
        
gd = concatGameData()
td = concatTeamData()
combineTeamDataAndGameData(gd, td)