from sgmtradingcore.analytics.series_inspector.crypto_inspections import MissingRealtimeCryptoData

# Lsit of inspections and configs that will be run
INSPECTION_TO_RUN = [
    # list of (class, kwargs)
    ('MissingRealtimeCryptoData', {
        'tickers': [
            'BTCUSD.PERP.BMEX',
            'ETHUSD.PERP.BMEX',
            'BTCUSD.SPOT.GDAX',
        ],
        'data_types': [
            'lob',
            'trades'
        ]})
]

# list of classes
KNOWN_INSPECTIONS = [MissingRealtimeCryptoData]
