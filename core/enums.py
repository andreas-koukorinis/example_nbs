from enum import IntEnum


class GsmCompetitions(IntEnum):
    NetherlandsED = 1
    PrimeraDivision = 7
    PremierLeague = 8
    Bundesliga1 = 9
    UEFAChampionsLeague = 10
    UEFAEuropaLeague = 18
    Bundesliga2 = 11
    ItalySerieA = 13
    FranceLigue1 = 16
    TurkishSuperLig = 19
    EuropeanChampionship = 25
    BrazilianSerieA = 26
    SwissSuperLeague = 27
    Allsvenskan = 28
    Tippeligaen = 29
    USAMLS = 33
    ScottishPrem = 43
    AustriaBL = 49
    PortugalPrimLiga = 63
    EnglishChampionship = 70
    GreeceSL = 107
    JapanJ1League = 109
    CopaAmerica = 288
    AustralianALeague = 283
    ChineseSuperLeague = 51
    LeagueOne = 15
    BelgiumFirstDivisionA = 24
    ArgentinaSuperliga = 87
    WorldCup = 72
    EnglishLeagueTwo = 32


GsmCompetitionInternalNameMap = {
    GsmCompetitions.NetherlandsED: 'NetED',
    GsmCompetitions.PrimeraDivision: 'SpaPr',
    GsmCompetitions.PremierLeague: 'EngPr',
    GsmCompetitions.Bundesliga1: 'GerBL1',
    GsmCompetitions.UEFAChampionsLeague: 'EurUCL',
    GsmCompetitions.UEFAEuropaLeague: 'EurUEL',
    GsmCompetitions.Bundesliga2: 'GerBL2',
    GsmCompetitions.ItalySerieA: 'ItaSA',
    GsmCompetitions.FranceLigue1: 'FraL1',
    GsmCompetitions.EuropeanChampionship: 'IntEuro',
    GsmCompetitions.BrazilianSerieA: 'BraSA',
    GsmCompetitions.SwissSuperLeague: 'SwiSL',
    GsmCompetitions.Allsvenskan: 'SweAV',
    GsmCompetitions.Tippeligaen: 'NorEs',
    GsmCompetitions.USAMLS: 'UsaMLS',
    GsmCompetitions.ScottishPrem: 'ScoPr',
    GsmCompetitions.AustriaBL: 'AutBL',
    GsmCompetitions.PortugalPrimLiga: 'PorPL',
    GsmCompetitions.EnglishChampionship: 'EngCh',
    GsmCompetitions.GreeceSL: 'GreSL',
    GsmCompetitions.JapanJ1League: 'JapJL1',
    GsmCompetitions.CopaAmerica: 'SAmCopaAm',
    GsmCompetitions.TurkishSuperLig: 'TurSL',
    GsmCompetitions.AustralianALeague: 'AusAL',
    GsmCompetitions.ChineseSuperLeague: 'ChiSL',
    GsmCompetitions.LeagueOne: 'EngL1',
    GsmCompetitions.BelgiumFirstDivisionA: 'BelPL',
    GsmCompetitions.ArgentinaSuperliga: 'ArgPr',
    GsmCompetitions.WorldCup: 'WorldCup',
    GsmCompetitions.EnglishLeagueTwo: 'EngL2'
}

InternalNameGsmCompetitionMap = {v: k for k, v in GsmCompetitionInternalNameMap.iteritems()}

GsmCompetitionAreaMap = {
    GsmCompetitions.Tippeligaen: 146,
    GsmCompetitions.Allsvenskan: 184,
    GsmCompetitions.JapanJ1League: 102,
    GsmCompetitions.AustriaBL: 20,
    GsmCompetitions.Bundesliga1: 80,
    GsmCompetitions.Bundesliga2: 80,
    GsmCompetitions.SwissSuperLeague: 185,
    GsmCompetitions.PremierLeague: 68,
    GsmCompetitions.EnglishChampionship: 68,
    GsmCompetitions.ScottishPrem: 165,
    GsmCompetitions.FranceLigue1: 76,
    GsmCompetitions.NetherlandsED: 138,
    GsmCompetitions.PortugalPrimLiga: 156,
    GsmCompetitions.PrimeraDivision: 176,
    GsmCompetitions.ItalySerieA: 100,
    GsmCompetitions.USAMLS: 203,
    GsmCompetitions.TurkishSuperLig: 196,
    GsmCompetitions.EuropeanChampionship: 7,
    GsmCompetitions.UEFAChampionsLeague: 7,
    GsmCompetitions.UEFAEuropaLeague: 7,
    GsmCompetitions.GreeceSL: 82,
    GsmCompetitions.CopaAmerica: 5,
    GsmCompetitions.AustralianALeague: 19,
    GsmCompetitions.ChineseSuperLeague: 49,
    GsmCompetitions.BrazilianSerieA: 35,
    GsmCompetitions.LeagueOne: 68,
    GsmCompetitions.BelgiumFirstDivisionA: 27,
    GsmCompetitions.ArgentinaSuperliga: 16,
    GsmCompetitions.WorldCup: 1,
    GsmCompetitions.EnglishLeagueTwo: 68
}


class EnetTournaments(IntEnum):
    WTA = 152
    GrandSlam = 153
    DavisCup = 155
    ATP = 156
    Challenger = 157
    SummerOlympics = 158
    FedCup = 159
    WTAChallenger = 9371


EnetTournamentTemplateInternalNameMap = {
    EnetTournaments.WTA: 'WTA',
    EnetTournaments.GrandSlam: 'Grand Slam',
    EnetTournaments.DavisCup: 'Davis Cup',
    EnetTournaments.ATP: 'ATP',
    EnetTournaments.Challenger: 'Challenger',
    EnetTournaments.FedCup: 'Fed Cup',
    EnetTournaments.WTAChallenger: 'WTA Challenger',
}

TennisTournamentAllowedNames = {
    'Grand Slam',
    'Davis Cup',
    'Challenger',
    'Fed Cup',
    'WTA',
    'WTA Challenger',
    'WTA Qualification',
    'ATP',
    'ATP Qualification',
    'Grand Slam Qualification',
}


class GsmSeasonNames(object):
    S2014 = '2014'
    S2013_2014 = '2013/2014'
    S2015 = '2015'
    S2014_2015 = '2014/2015'
    S2016 = '2016'
    S2015_2016 = '2015/2016'
